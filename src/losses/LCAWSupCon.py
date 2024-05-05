import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

class LCAWSupConLoss(nn.Module):
    def __init__(self, sim = 'cosine', tau = 1.0):
        super().__init__()
        self.tau = tau
        self.sim = sim 

    def forward(self, features, labels):
        B, _ = features.shape
        # calculate pair wise similarity 
        sim_mat = self.calculate_sim_matrix(features)
        # division by temperature
        sim_mat = F.log_softmax(sim_mat / self.tau, dim = -1) 
        sim_mat = sim_mat.clone().fill_diagonal_(torch.tensor(0.0))
        # calculating pair wise equal labels for pos pairs
        labels = labels.unsqueeze(1)
        labels_mask = (labels == labels.T).type(torch.float32)
        labels_mask.fill_diagonal_(torch.tensor(0.0))
        # calculating num of positive pairs for each sample
        num_pos = torch.sum(labels_mask, dim = -1)
        # masking out the negative pair log_softmax value
        pos_sim_mat = sim_mat * labels_mask 
        # summing log_softmax value over all positive pairs
        pos_pair_sum = torch.sum(pos_sim_mat, dim = -1)
        # averaging out the log_softmax value, epsilon = 1e-5 is to avoid division by zero
        pos_pairs_avg = torch.div(pos_pair_sum, num_pos + 1e-5)
        # final loss over all features in batch
        loss = -pos_pairs_avg.sum() / B
        return loss

    def calculate_sim_matrix(self, features):
        sim_mat = None
        if self.sim == "mse":
            sim_mat = -torch.cdist(features, features)
        else:
            features = F.normalize(features, dim = -1, p = 2)
            sim_mat = F.cosine_similarity(features[None, :, :], features[:, None, :], dim = -1)
        # filling diagonal with -torch.inf as it will be cancel out while doing softmax
        sim_mat.fill_diagonal_(-torch.tensor(torch.inf))
        return sim_mat 
    
class LCAConClsLoss(nn.Module):
    def __init__(self, sim = 'cosine', tau = 1.0):
        super().__init__()
        self.tau = tau
        self.sim = sim 
        self.ce = nn.CrossEntropyLoss()
        self.lcasupcon = LCAWSupConLoss(sim, tau)
    
    def forward(self, features, scores, labels):
        return self.lcasupcon(features, labels) + self.ce(scores, labels)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

if __name__ == "__main__":
    x1 = torch.rand(4,5)
    x2 = torch.rand(4,5)
    labels = torch.tensor([1,2,3,1])
    lcswsupcon = LCAWSupConLoss(sim = 'cosine')
    lcssupcon_mse = LCAWSupConLoss(sim = 'mse')
    supcon = SupConLoss(temperature = 1.0, base_temperature=1.0)
    print(lcswsupcon(x1, labels))
    print(lcssupcon_mse(x1, labels))
    print(supcon(x1.unsqueeze(1), labels))