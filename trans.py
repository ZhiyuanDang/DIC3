import argparse
import torch
import yaml
from termcolor import colored
from utils.common_config import get_val_dataset, get_train_transformations, get_val_dataloader,\
                                get_model
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.memory import MemoryBank 
from utils.utils import fill_memory_bank
from PIL import Image

FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
FLAGS.add_argument('--config_exp', help='Location of config file')
args = FLAGS.parse_args()

import matplotlib.pyplot as plt
import numpy as np

def main():

    # Read config file
    print(colored('Read config file {} ...'.format(args.config_exp), 'blue'))
    with open(args.config_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    config['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
    print(config)
    
     # Get dataset
    print(colored('Get validation dataset ...', 'blue'))
    transforms = get_train_transformations(config)
    dataset = get_val_dataset(config, transforms['standard'])
    dataset_aug = get_val_dataset(config, transforms['augment'])
    
    
    
    img = np.array(dataset.__getitem__(6123)['image'])
    #img = np.transpose(img, (1, 2, 0))
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    #plt.show()
    plt.savefig('./'+config['train_db_name']+'/'+config['train_db_name']+'_weak.png', dpi=500, bbox_inches='tight')
    
    img = np.array(dataset_aug.__getitem__(6123)['image'])
    img = np.transpose(img, (1, 2, 0))
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    #plt.show()
    plt.savefig('./'+config['train_db_name']+'/'+config['train_db_name']+'_strong.png', dpi=500, bbox_inches='tight')




if __name__ == "__main__":
    main() 