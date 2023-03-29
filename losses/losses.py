"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS=1e-8


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        if not (mask != 0).any():
            return 0
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        #+ margin_loss(anchors_weak, anchors_strong, neighbood = 20)
        
        return loss


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0, threshold = 0.8):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.threshold = threshold
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, input, input_aug):
        weak_anchors_prob = self.softmax(input) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        if not (mask != 0).any():
            return 0
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input_aug, mask.view(b, 1)).view(n, c)
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.long().unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
    


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0
        self.ce = nn.CrossEntropyLoss()
        self.t = 0.5
        self.confidence_ce = ConfidenceBasedCE(0.7, True)
        self.LSloss = LabelSmoothLoss(0.5)

    def forward(self, anchors, neighbors, anchor_augmented, index, model):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        # b1, n1 = under_anchors_output.size()
        # b2, n2 = over_anchors_output.size()

        anchors_prob = self.softmax(anchors)
        # anchors_prob = anchors
        positives_prob = self.softmax(neighbors)
        anchor_augmented_prob = self.softmax(anchor_augmented)
        # anchor_augmented_prob = anchor_augmented

        # anchor_augmented_prob = anchors_prob[initial_rank_index, :]

        # under_anchors_prob = self.softmax(under_anchors_output)
        # # under_anchors_prob = under_anchors_output

        # under_neighbors_prob = self.softmax(under_neighbors_output)
        # # under_neighbors_prob = under_neighbors_output

        # # under_neighbors_prob = under_anchors_prob[initial_rank_index, :]

        # over_anchors_prob = self.softmax(over_anchors_output)
        # # over_anchors_prob = over_anchors_output
        # over_neighbors_prob = self.softmax(over_neighbors_output)
        # over_neighbors_prob = over_neighbors_output
        # over_neighbors_prob = over_anchors_prob[initial_rank_index, :]

        # features_norm = F.normalize(anchors_features, dim = -1)
        # features_sim = torch.mm(features_norm, features_norm.t())
        # features_sim_norm = F.normalize(features_sim, dim = -1)
        # features_soft = self.softmax(features)

        anchors_prob_norm = F.normalize(anchors_prob, dim = -1)
        positives_prob_norm = F.normalize(positives_prob, dim = -1)
        anchor_augmented_prob_norm = F.normalize(anchor_augmented_prob, dim = -1)

        # anchors_sim = torch.mm(anchors_norm, anchors_norm.t())
        # anchors_sim_norm = F.normalize(anchors_sim, dim = -1)
        # anchors_sim_soft = self.softmax(anchors_sim)
        # features_aug_soft = self.softmax(features_aug)



        # similarity = torch.bmm(features_soft.view(b, 1, -1), features_aug_soft.view(b, -1, 1)).squeeze()
        # similarity = torch.mm(features_sim, anchors_sim.t())
        # similarity = torch.bmm(anchors_prob_c.t().view( n, 1, 2 * b), positives_prob_c.t().view( n, 2 * b, 1))
        # similarity = torch.einsum('cn,nc->c', [anchors_prob_c.t(), positives_prob_c])
        # similarity = torch.bmm(features_sim_norm.view( b, 1, -1), anchors_sim_norm.view( b, -1, 1))
        similarity = torch.bmm(anchors_prob.view( b, 1, n), anchor_augmented_prob.view( b, n, 1))
        similarity_neigh = torch.bmm(anchors_prob.view( b, 1, n), positives_prob.view( b, n, 1))

        similarity_neigh_norm = torch.bmm(anchors_prob_norm.view( b, 1, n), positives_prob_norm.view( b, n, 1))
        similarity_aug_norm = torch.bmm(anchors_prob_norm.view( b, 1, n), anchor_augmented_prob_norm.view( b, n, 1))
        # similarity_under = torch.bmm(under_anchors_prob.view( b1, 1, n1), under_neighbors_prob.view( b1, n1, 1))
        # similarity_over = torch.bmm(over_anchors_prob.view( b2, 1, n2), over_neighbors_prob.view( b2, n2, 1))
        # labels = torch.tensor(list(range(b))).cuda()
        # labels = torch.ones_like(similarity)


        # consistency_loss =  self.ce(similarity, labels)
        
        ones = torch.ones_like(similarity)
        
        # consistency_loss = self.bce(similarity_neigh, ones) 
        consistency_loss = - (similarity_neigh).mean()
        # consistency_loss = - (similarity_aug_norm).mean()
        # + self.confidence_ce(anchors, anchor_augmented)\
            # + self.bce(similarity_over, ones)  + self.confidence_ce(over_anchors_output, over_neighbors_output) 
        # positives_prob_augmented = anchors_prob[batch_index,:]

        # positives_prob = torch.cat([positives_prob, anchor_augmented_prob], dim=0)
        # positives_prob = torch.cat([positives_prob, positives_prob_augmented], dim=0)
        
        # anchors_prob = torch.cat([anchors_prob, anchors_prob], dim=0)
        # anchors_prob = anchors_prob.repeat(2, 1)

        # anchors_prob_s = F.normalize(anchors_prob, dim = 1)
        # positives_prob_s = F.normalize(positives_prob, dim = 1)
        # anchor_augmented_prob_s = F.normalize(anchor_augmented_prob, dim = 1)

        # weights_norm = F.normalize(weights, dim = 1)
        # global_weights = F.normalize(self.softmax(features), dim = -1)

        # empirical cross-correlation matrix
        # c = model.bn(weights) @ model.bn(weights).T 
        # c1 = model.bn1(weights.t()).T @ model.bn1(weights.t())

        # sum the cross-correlation matrix between all gpus
        # c.div_(n)
        # c1.div_(model.backbone_dim)
        # weights_norm = F.normalize(self.softmax(weights), dim = -1)
        # similarity = torch.mm(weights_norm, global_weights.t()) 
        # weights_soft = F.normalize(weights, dim = -1)
        # centroids_feature_soft = F.normalize(centroids_feature, dim = -1)
        # similarity = torch.mm(weights_soft, centroids_feature_soft.t()) 
        # similarity = torch.bmm(weights_soft.view( n, 1, -1), centroids_feature_soft.view( n, -1, 1))
        # ones = torch.ones_like(similarity)
        # labels = torch.tensor(list(range(n))).cuda()
        # centroid_loss =  off_diagonal(similarity).pow_(2).sum().div_(n) + self.confidence_ce(anchors, anchor_augmented)
        # consistency_loss = off_diagonal(c).pow_(2).sum().mul(1e-1) \
            # + self.confidence_ce(anchors, anchor_augmented)
        # consistency_loss =  self.ce(similarity, labels) \
            # + self.confidence_ce(anchors, anchor_augmented)

        # Similarity in output space
        # similarity = torch.bmm(anchors_prob.view(b, 1, n), anchor_augmented_prob.view(b, n, 1)).squeeze()
        # similarity = torch.mm(anchors_prob, positives_prob.t())
        # similarity = torch.einsum('nc,nc->n', [anchors_prob_s, positives_prob_s]).unsqueeze(-1)
        # ones = torch.ones_like(similarity)
        # eyes = torch.eye(b).repeat(2, 2).cuda()

        # zeros = torch.zeros_like(similarity).squeeze().long()
        # zeros = torch.zeros(similarity.shape[0], dtype=torch.long).cuda()
        # consistency_loss = self.ce(similarity, labels) \
            # + self.confidence_ce(anchors, anchor_augmented)
        
        # kmeans_loss = 0
        # if clustering_results is not None:
        #     # prototypes = F.normalize(prototypes, dim = 1)
        #     samples = torch.cat([anchors_prob, positives_prob], dim=0)
        #     # logits_global_kmeans = torch.mm(samples, prototypes.t())
        #     # logits_global_kmeans_tf = torch.mm(anchor_augmented_prob_s, prototypes.t())
        #     # labels = im2cluster[index].repeat(4)
        #     # mask = (labels!=-1)
            
        #     anchors_prob_neighbood = clustering_results[index, :]
            
        #     anchors_prob_neighbood = anchors_prob_neighbood.repeat(4,1)
        #     # anchors_prob_neighbood_s = F.normalize(anchors_prob_neighbood, dim = 1)

        #     # if torch.sum(mask.int()) != 0:
        #     #   kmeans_loss += self.ce(logits_global_kmeans[mask,:], labels[mask])
        #     similarity = torch.bmm(samples.view(4 * b, 1, n), anchors_prob_neighbood.view(4 * b, n, 1)).squeeze()

        #     # similarity = torch.mm(samples, anchors_prob_neighbood.t())
        #     # similarity = torch.einsum('nc,nc->n', [samples, anchors_prob_neighbood_s]).unsqueeze(-1)

        #     # ones = torch.ones_like(similarity)
        #     # similarity = torch.mm(anchors_prob_s, positives_prob_s.t())
        #     # similarity = torch.einsum('nc,nc->n', [anchors_prob_s, positives_prob_s]).unsqueeze(-1)
        #     ones = torch.ones_like(similarity)
        #     # eyes = torch.eye(b).repeat(4, 4).cuda()
        #     # zeros = torch.zeros_like(similarity).squeeze().long()
        #     # zeros = torch.zeros(similarity.shape[0], dtype=torch.long).cuda()

        #     samples_c = F.normalize(samples, dim = 0)
        #     anchors_prob_neighbood_c = F.normalize(anchors_prob_neighbood, dim = 0)

        #     similarity_c = torch.mm(samples_c.t(), anchors_prob_neighbood_c)
        #     # similarity = torch.bmm(anchors_prob_c.t().view( n, 1, 2 * b), positives_prob_c.t().view( n, 2 * b, 1))
        #     # similarity = torch.einsum('cn,nc->c', [anchors_prob_c.t(), positives_prob_c])


        #     labels_c = torch.tensor(list(range(n))).cuda()
        #     # labels = torch.ones_like(similarity)

        #     kmeans_loss += self.bce(similarity, ones) + self.ce(similarity_c, labels_c)





        # margin_loss = self.margin_loss(anchors_prob, positives_prob)
        
        anchors_prob_c = F.normalize(anchors_prob, dim = 0)
        # anchor_augmented_prob_c = F.normalize(anchor_augmented_prob, dim = 0)
        anchor_augmented_prob_c = F.normalize(anchor_augmented_prob, dim = 0)

        # under_anchors_prob_c = F.normalize(under_anchors_prob, dim = 0)
        # # anchor_augmented_prob_c = F.normalize(anchor_augmented_prob, dim = 0)
        # under_anchor_augmented_prob_c = F.normalize(under_neighbors_prob, dim = 0)


        # over_anchors_prob_c = F.normalize(over_anchors_prob, dim = 0)
        # # anchor_augmented_prob_c = F.normalize(anchor_augmented_prob, dim = 0)
        # over_anchor_augmented_prob_c = F.normalize(over_neighbors_prob, dim = 0)
        
        # similarity = torch.bmm(anchors_prob.view( b, 1, n), anchor_augmented_prob.view( b, n, 1))
        similarity = torch.mm(anchors_prob_c.t(), anchor_augmented_prob_c)
        # similarity_under = torch.mm(under_anchors_prob_c.t(), under_anchor_augmented_prob_c)
        # similarity_over = torch.mm(over_anchors_prob_c.t(), over_anchor_augmented_prob_c)
        # similarity = torch.bmm(anchors_prob_c.t().view( n, 1, 2 * b), positives_prob_c.t().view( n, 2 * b, 1))
        # similarity = torch.einsum('cn,nc->c', [anchors_prob_c.t(), positives_prob_c])
        # ones = torch.ones_like(similarity)

        labels = torch.tensor(list(range(n))).cuda()
        # labels_under = torch.tensor(list(range(n1))).cuda()
        # labels_over = torch.tensor(list(range(n2))).cuda()
        # labels = torch.ones_like(similarity)

        # ce_loss =  self.bce(similarity, ones)
        ce_loss =  self.ce(similarity, labels) \
            # + self.ce(similarity_over, labels_over) 
        
        
        centroid_norm  = F.normalize(model.module.cluster_head[0].weight)
        centroid_norm_mul = torch.exp(torch.mm(centroid_norm, centroid_norm.t()) )
        centroid_norm_mul_sum = torch.log(torch.sum(centroid_norm_mul, dim = -1))
        centroid_dissimilar_loss = torch.mean(centroid_norm_mul_sum)


        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True) \
            #  + entropy(torch.mean(over_anchors_prob, 0), input_as_probabilities = True)  
        
        # confidence_ce_loss  = self.confidence_ce(anchors_prob, positives_prob)
        confidence_ce_loss  = self.confidence_ce(anchors, anchor_augmented)

        # Total loss
        total_loss = consistency_loss + ce_loss  - self.entropy_weight * entropy_loss + confidence_ce_loss
        
        return total_loss, consistency_loss, ce_loss, entropy_loss
    
def margin_loss(x, x_tf, neighbood = 10):

    value_x, _ = torch.topk(x.t(), neighbood)

    positive_x = value_x.mean(-1)
    value_x, _ = torch.topk(x.t(), int(x.shape[0]/2), largest=False)

    negative_x = value_x.mean(-1)

    value_x, _ = torch.topk(x_tf.t(), neighbood)

    positive_x_t = value_x.mean(-1)
    value_x, _ = torch.topk(x_tf.t(), int(x.shape[0]/2), largest=False)

    negative_x_t = value_x.mean(-1)


    loss = F.softplus(negative_x_t - positive_x_t) + F.softplus( negative_x-positive_x )

    return loss.mean()


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss
