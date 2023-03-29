"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn.functional as F
from .evaluate_utils import hungarian_evaluate_kmeans

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions.cpu(), self.features.cpu().t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1).cpu()
        retrieval = torch.gather(candidates, 1, yi).to(self.device)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_().to(self.device)
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):

        _, count = torch.unique(self.targets.cpu(), return_counts = True)
        print(count)
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        # index = faiss.IndexFlatL2(dim)
        
        # index = faiss.index_cpu_to_all_gpus(index)
        res = faiss.StandardGpuResources()
        cfg_faiss = faiss.GpuIndexFlatConfig()
        cfg_faiss.useFloat16 = False
        cfg_faiss.device = 0
        index = faiss.GpuIndexFlatL2(res, dim, cfg_faiss)
        
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
    
    def run_kmeans(self):
        import faiss
        x = F.normalize(self.features.cpu())
        x = x.numpy()
        
        # x = self.features.cpu().numpy()
        
        d = x.shape[1]
        k = self.C 
            # # intialize faiss clustering parameters
        clus = faiss.Clustering(d, k)
        clus.verbose = False
        clus.niter = 100
        clus.nredo = 5
        clus.seed = 1234
        clus.max_points_per_centroid = int(x.shape[0]/k)
        clus.min_points_per_centroid = int(x.shape[0]/k) #1
        clus.spherical = True

        res = faiss.StandardGpuResources()
        cfg_faiss = faiss.GpuIndexFlatConfig()
        cfg_faiss.useFloat16 = False
        cfg_faiss.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg_faiss)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        # print(I)
        label = I.squeeze()
        # get cluster centroids
        centroids_head = faiss.vector_to_array(clus.centroids).reshape(k, d)

        predictions=[{'predictions':torch.from_numpy(label).cuda(),'probabilities':torch.tensor(1), 'targets':self.targets}]
        clustering_stats = hungarian_evaluate_kmeans(0, predictions, k,
                                                compute_confusion_matrix=False)
        print(clustering_stats)
        # kmeans = faiss.Kmeans(d, k, niter=100, verbose=False, nredo=5, seed = 1234, max_points_per_centroid = int(x.shape[0]/k), gpu=True)
        # kmeans.train(x)
        # centroids_head = kmeans.centroids

        # #under
        # k = int(self.C / 5)
        # clus = faiss.Clustering(d, k)
        # clus.verbose = False
        # clus.niter = 100
        # clus.nredo = 5
        # clus.seed = 1234
        # clus.max_points_per_centroid = int(x.shape[0]/k)
        # clus.min_points_per_centroid = int(x.shape[0]/k) #1

        # res = faiss.StandardGpuResources()
        # cfg_faiss = faiss.GpuIndexFlatConfig()
        # cfg_faiss.useFloat16 = False
        # cfg_faiss.device = 0
        # index = faiss.GpuIndexFlatL2(res, d, cfg_faiss)  

        # clus.train(x, index)   

        # # D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        # # im2cluster = [int(n[0]) for n in I]
        # # # get cluster centroids
        # centroids_head_under = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # #over
        # k = int(self.C * 5)
        # kmeans = faiss.Kmeans(d, k, niter=100, verbose=False, nredo=5, seed = 1234, max_points_per_centroid = int(x.shape[0]/k), gpu=True)
        # kmeans.train(x)
        # centroids_head_over = kmeans.centroids
        
        # clus = faiss.Clustering(d, k)
        # clus.verbose = False
        # clus.niter = 100
        # clus.nredo = 5
        # clus.seed = 1234
        # clus.max_points_per_centroid = int(x.shape[0]/k)
        # clus.min_points_per_centroid = int(x.shape[0]/k) #1

        # res = faiss.StandardGpuResources()
        # cfg_faiss = faiss.GpuIndexFlatConfig()
        # cfg_faiss.useFloat16 = False
        # cfg_faiss.device = 0
        # index = faiss.GpuIndexFlatL2(res, d, cfg_faiss)  

        # clus.train(x, index)   

        # # D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        # # im2cluster = [int(n[0]) for n in I]
        # # # get cluster centroids
        # centroids_head_over = faiss.vector_to_array(clus.centroids).reshape(k, d)

        return centroids_head

    def run_test_kmeans(self, centroids):
        import faiss
        x = F.normalize(self.features.cpu())
        x = x.numpy()

        # x = self.features.cpu().numpy()
        
        d = x.shape[1]
        k = self.C 
        res = faiss.StandardGpuResources()
        cfg_faiss = faiss.GpuIndexFlatConfig()
        cfg_faiss.useFloat16 = False
        cfg_faiss.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg_faiss)
        
        index.add(centroids)
        _, I = index.search(x, 1) # Sample itself is included

        label = I.squeeze()

        predictions=[{'predictions':torch.from_numpy(label).cuda(),'probabilities':torch.tensor(1), 'targets':self.targets}]
        clustering_stats = hungarian_evaluate_kmeans(0, predictions, k,
                                                compute_confusion_matrix=False)
        print(clustering_stats)

