"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x, withhead = True):
        if withhead:
          features = self.contrastive_head(self.backbone(x))
          features = F.normalize(features, dim = 1)
        else:
          features = self.backbone(x)
          features = F.normalize(features, dim = 1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, p, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters, bias = False) for _ in range(self.nheads)])
        # normalization layer for the representations z1 and z2
        # self.bn = nn.BatchNorm1d(self.backbone_dim, affine=False)
        # self.bn1 = nn.BatchNorm1d(nclusters, affine=False)
        # centroids_feature = np.load(p['centroids_feature_path'])
        # centroids_feature = torch.from_numpy(centroids_feature)
        # self.register_buffer('centroids_feature', centroids_feature)
        self.under_cluster_head = nn.Linear(self.backbone_dim, int(nclusters / 5), bias = False)
        self.over_cluster_head = nn.Linear(self.backbone_dim, int(nclusters * 5), bias = False)


    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            #out = [cluster_head(x) for cluster_head in self.cluster_head]
            x = F.normalize(x, dim = -1)
            # for cluster_head in self.cluster_head:
            #     weight = cluster_head.weight.data
            #     weight[weight<0] = 0
            
            out = [torch.mm(x, F.normalize(cluster_head.weight.t(), dim = -1)) for cluster_head in self.cluster_head]
            # out = [F.linear(x, F.normalize(cluster_head.weight, dim = -1)) for cluster_head in self.cluster_head]

        elif forward_pass == 'underhead':
            # out = self.under_cluster_head(x)
            x = F.normalize(x, dim = -1)
            out = torch.mm(x, F.normalize(self.under_cluster_head.weight.t(), dim = -1))
        
        elif forward_pass == 'overhead':
            # out = self.over_cluster_head(x)
            x = F.normalize(x, dim = -1)
            out = torch.mm(x, F.normalize(self.over_cluster_head.weight.t(), dim = -1))

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out
