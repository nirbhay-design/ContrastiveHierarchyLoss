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
        # calculate pair wise similarity 
        sim_mat = self.calculate_sim_matrix(features)
        # division by temperature
        sim_mat = F.log_softmax(sim_mat / self.tau, dim = -1)
        sim_mat.fill_diagonal_(torch.tensor(0.0))
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
        loss = -pos_pairs_avg.sum()
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

if __name__ == "__main__":
    x1 = torch.rand(4,5)
    x2 = torch.rand(4,5)
    labels = torch.tensor([1,2,3,1])
    lcswsupcon = LCAWSupConLoss(sim = 'cosine')
    lcssupcon_mse = LCAWSupConLoss(sim = 'mse')
    print(lcswsupcon(x1, labels))
    print(lcssupcon_mse(x1, labels))