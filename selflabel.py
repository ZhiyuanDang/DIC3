"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np
import random
from utils.config import create_config
from utils.common_config import get_train_dataset, get_train_transformations,\
                                get_val_dataset, get_val_transformations,\
                                get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.ema import EMA
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.train_utils import selflabel_train
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='Self-labeling')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--gpus', default='', type=str,
                            help='available gpu list, leave empty to use cpu')
parser.add_argument('--seed', default=None, type=int,
                            help='random seed')
args = parser.parse_args()

def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    
    # print(model)
    # fix random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        print('Random seed will be fixed to %d' % args.seed)
    

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p, p['ccdc_model'])
    

    #print(model)
    # data parallel
    if len(args.gpus.split(',')) >= 1:
        print('Data parallel will be used for acceleration purpose')
        device_ids = [int(x) for x in args.gpus.split(',')]
        torch.cuda.set_device(f'cuda:{device_ids[0]}')
        model = torch.nn.DataParallel(model, device_ids)
    # model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    strong_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, strong_transforms,
                                        split='train', to_augmented_dataset=True) 
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataset = get_val_dataset(p, val_transforms) 
    val_dataloader = get_val_dataloader(p, val_dataset)
    print(colored('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)), 'yellow'))

    # Checkpoint
    if os.path.exists(p['selflabel_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['selflabel_checkpoint']), 'blue'))
        checkpoint = torch.load(p['selflabel_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])        
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['selflabel_checkpoint']), 'blue'))
        start_epoch = 0

    # EMA
    if p['use_ema']:
        ema = EMA(model, alpha=p['ema_alpha'])
    else:
        ema = None

    # Main loop
    print(colored('Starting main loop', 'blue'))
    
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Perform self-labeling 
        print('Train ...')
        selflabel_train(train_dataloader, model, criterion, optimizer, epoch, ema=ema)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        predictions = get_predictions(p, val_dataloader, model)
        clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False) 
        print(clustering_stats)
        
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, p['selflabel_checkpoint'])
        torch.save(model.module.state_dict(), p['selflabel_model'])
    
    # Evaluate and save the final model
    print(colored('Evaluate model at the end', 'blue'))
    predictions = get_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(0, predictions, 
                                class_names=val_dataset.classes,
                                compute_confusion_matrix=True,
                                confusion_matrix_file=os.path.join(p['selflabel_dir'], 'confusion_matrix.png'))
    print(clustering_stats)
    torch.save(model.module.state_dict(), p['selflabel_model'])


if __name__ == "__main__":
    main()
