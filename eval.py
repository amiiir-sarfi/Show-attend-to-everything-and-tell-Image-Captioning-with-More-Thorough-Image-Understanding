import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from models import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from metrics.Bleu import Bleu
from metrics.CIDEr import Cider
from metrics.ROUGE_L import Rouge



temp = Encoder(forward_type= forward_type)
input_batch = torch.zeros([1,3,224,224])
with torch.no_grad():
    output = temp(input_batch)
del(temp)

encoder_dim = output.shape[-1]
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Parameters
# Data parameters
data_folder = ''  # folder with data files saved by create_input_files.py
data_name = ''  # base name shared by data files

checkpoint = ''
word_map_file = './WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
print('[+]',checkpoint['epoch'],' ',checkpoint['bleu-4'])


net = 'resnet'
fine_tune = None
vanilla = False

# Load model
checkpoint = torch.load(checkpoint)
decoder = DecoderWithAttention(attention_dim=attention_dim,
                                embed_dim=emb_dim,
                                decoder_dim=decoder_dim,
                                vocab_size=len(word_map),
                                encoder_dim=encoder_dim,
                                dropout=dropout,
                                embed_fine_tune=True)
decoder.load_state_dict(checkpoint['decoder'].state_dict())
decoder.eval()
encoder = checkpoint['encoder']
encoder = Encoder(forward_type=forward_type, net = net ,fine_tune = fine_tune, vanilla=vanilla)
encoder.load_state_dict(checkpoint['encoder'].state_dict())

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

gts = {}
res = {}

def evaluate(beam_size):
    """
    Evaluation
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    references = list()
    hypotheses = list()

    counter = 0
    # For each image
    # for i, (image, caps, caplens, allcaps) in enumerate(loader):
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
        seqs = k_prev_words  # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        while True:
            
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypo2 = [[w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]]
        hypotheses.append(hypo2[0])
        import numpy as np
        # print('[2]',np.array(hypo).shape)
        gts[counter] = img_captions
        res[counter] = hypo2
        counter+=1
        assert len(references) == len(hypotheses)

    my_bleu_calculator = Bleu(4)
    print('Bleu-1 to 4 sores are: ',my_bleu_calculator.compute_score(gts,res)[0])

    my_ref = {}
    my_gts = {}
    for j in res:
      my_ref[j] = [[rev_word_map[i] for i in res[j][0]]]
    for i in gts:
      my_gts[i] = [' '.join([rev_word_map[k] for k in j]) for j in gts[i]]
      
    rgs = Rouge()
    print('Rouge score is: ',rgs.compute_score(my_gts,my_ref)[0])
    cdr = Cider()
    print('CIDEr score is: ',cdr.compute_score(my_gts,my_ref)[0])

    return (references, hypotheses)


if __name__ == '__main__':
  beam_size = 5
  evaluate(beam_size)