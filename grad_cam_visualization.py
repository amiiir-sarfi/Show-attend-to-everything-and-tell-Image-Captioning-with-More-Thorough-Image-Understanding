from grad_cam import *
import matplotlib.cm as cm
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
# from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
data_folder = '/content/drive/My Drive/out/'  # folder with data files saved by create_input_files.py
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  # base name shared by data files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

layer =  12

# Training parameters
workers = 1  # for data-loading; right now, only 1 works with h5py

x = [30]

def main():


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    

    mymodel = models.alexnet(pretrained=True)
    print(mymodel.eval())
    mymodel.to(device)
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

      if i in x:
        imgs = imgs.to(device)
        imgs = F.interpolate(imgs, size=224)

        cam,target_class = visualize(imgs,  None, mymodel, target_layer=layer)
        cam_orig_alone = np.tile(np.expand_dims(cam12,0),[3,1,1])# normalized, 0,1


        for i in range(3):
          imgs[0,i,:,:] = (imgs[0,i,:,:]-imgs[0,i,:,:].min())/(imgs[0,i,:,:].max()-imgs[0,i,:,:].min())
        plt.imshow(np.transpose(imgs.squeeze(0).cpu(), (1, 2, 0)))
        plt.imshow(np.transpose(cam_orig_alone, (1, 2, 0)), alpha=0.7)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
        plt.savefig('gradcam_layer%d.png'%(layer))

        print('This is the %d\'th image which was from class: %d'%(i,target_class))
        
if __name__ == '__main__':
    main()