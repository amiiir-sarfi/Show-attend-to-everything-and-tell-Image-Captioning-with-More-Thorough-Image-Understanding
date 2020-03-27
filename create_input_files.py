from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    address = ''
    create_input_files(dataset='coco',
                       karpathy_json_path=address+'karpathy/dataset_coco.json',
                       image_folder=address,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=address+'out',
                       max_len=50)