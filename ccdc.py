"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import faiss
import numpy as np
import random
import os
import torch.nn.functional as F

from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
from utils.train_utils import scan_train
from utils.faiss_rerank import compute_jaccard_distance
from utils.faiss_utils import search_index_pytorch, search_raw_array_pytorch
import numpy as np
import time

FLAGS = argparse.ArgumentParser(description='ccdc Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--gpus', default='', type=str,
                            help='available gpu list, leave empty to use cpu')
FLAGS.add_argument('--seed', default=None, type=int,
                            help='random seed')

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    train_dataset = get_train_dataset(p, train_transformations, 
                                        split='train', to_neighbors_dataset = True)
    # val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset = True)
    val_dataset = get_train_dataset(p, val_transformations, 
                                        split='train', to_neighbors_dataset = True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))

    # fix random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        print('Random seed will be fixed to %d' % args.seed)
    
    #torch.cuda.set_device('cuda:2')
    
    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p, p['pretext_model'])
    centroids_feature = np.load(p['centroids_feature_path'])
    centroids_feature = torch.tensor(centroids_feature)
    model.cluster_head[0].weight.data = centroids_feature
    # centroids_feature_np = np.load(p['centroids_feature_path'], allow_pickle = True)
    # centroids_feature = torch.from_numpy(centroids_feature_np).cuda()
    # model.cluster_head[0].weight.data = centroids_feature

    # centroids_feature_under = torch.from_numpy(centroids_feature_np[1]).cuda()
    # model.under_cluster_head.weight.data = centroids_feature_under

    # centroids_feature_over = torch.from_numpy(centroids_feature_np[1]).cuda()
    # model.over_cluster_head.weight.data = centroids_feature_over

    print(model.cluster_head[0].weight)
    # exit()

    # data parallel
    if len(args.gpus.split(',')) >= 1:
        print('Data parallel will be used for acceleration purpose')
        device_ids = [int(x) for x in args.gpus.split(',')]
        torch.cuda.set_device(f'cuda:{device_ids[0]}')
        model = torch.nn.DataParallel(model, device_ids)

    # print(model)

    model = model.cuda()

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)
    
    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: CCDC will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p) 
    criterion.cuda()
    print(criterion)

    clustering_results = None

    # Checkpoint
    if os.path.exists(p['ccdc_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['ccdc_checkpoint']), 'blue'))
        checkpoint = torch.load(p['ccdc_checkpoint'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])        
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']
        # clustering_results = checkpoint['clustering_results']
        best_acc = checkpoint['best_acc']


    else:
        print(colored('No checkpoint file at {}'.format(p['ccdc_checkpoint']), 'blue'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None
        best_acc = 0.
    

    

    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        scan_train(train_dataloader, model, criterion, optimizer, epoch, p['update_cluster_head_only'])


        # Evaluate 
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)

        print('Evaluate based on CCDC loss ...')
        scan_stats = scan_evaluate(predictions)
        print(scan_stats)
        lowest_loss_head = scan_stats['lowest_loss_head']
        lowest_loss = scan_stats['lowest_loss']

        print('Evaluate with hungarian matching algorithm ...')
        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
        print(clustering_stats)    

        if best_acc < clustering_stats['ACC']:
            print('New lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
            print('Best ACC on validation set: %.4f -> %.4f' %(best_acc, clustering_stats['ACC']))
            print('Lowest loss head is %d' %(lowest_loss_head))
            best_loss = lowest_loss
            best_loss_head = lowest_loss_head
            best_acc = clustering_stats['ACC']
            torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['ccdc_model'])

        else:
            print('No new lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
            print('Lowest loss head is %d' %(best_loss_head)) 
            print('Current Best ACC on validation set: %.4f' % best_acc)
        
        if epoch%10 == 0:
            torch.save({'centroids': model.module.cluster_head[0].weight.data, 'model': model.module.state_dict()}, os.path.join(p['ccdc_dir'], 'centroidsandmodel'+str(epoch)+'.pth.tar'))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1, 'best_loss': best_loss, 'best_acc': best_acc, 'best_loss_head': best_loss_head, 'clustering_results': clustering_results},
                     p['ccdc_checkpoint'])
    
    # Evaluate and save the final model
    print(colored('Evaluate best model based on CCDC metric at the end', 'blue'))
    model_checkpoint = torch.load(p['ccdc_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    predictions = get_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(model_checkpoint['head'], predictions, 
                            class_names=val_dataset.dataset.classes, 
                            compute_confusion_matrix=True, 
                            confusion_matrix_file=os.path.join(p['ccdc_dir'], 'confusion_matrix.png'))
    print(clustering_stats)         


if __name__ == "__main__":
    main()
