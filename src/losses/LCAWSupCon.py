import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
from src.dataset.tree import load_distances
import pickle

class SupConLoss(nn.Module):
    def __init__(self, 
                 sim = 'cosine', 
                 tau = 1.0):
        super().__init__()
        self.tau = tau
        self.sim = sim 

    def forward(self, features, labels):
        features = F.normalize(features, dim = -1, p = 2)
        loss = 0
        for idx, i in enumerate(features):
            total_sim = 0
            pos_samples_wt = []
            num_pos = 0
            for jdx, j in enumerate(features):
                if idx != jdx:
                    sim_ij = torch.exp(torch.sum(i*j) / self.tau)
                    total_sim += sim_ij 
                    if labels[idx] == labels[jdx]: # positive_sample
                        pos_samples_wt.append(sim_ij)
                        num_pos += 1

            log_softmax = -torch.log(torch.tensor(pos_samples_wt) / total_sim).sum() / (num_pos + 1e-5)
            loss += log_softmax
        return loss / features.shape[0]            

class LCAWSupConLoss(nn.Module):
    def __init__(self, 
                 hierarchy_dist_path, 
                 idx_to_cls_path, 
                 dataset_name = "tiered-imagenet-224", 
                 sim = 'cosine', 
                 tau = 1.0):
        super().__init__()
        self.tau = tau
        self.sim = sim 

        self.h_dist = load_distances(dataset_name, "ilsvrc", hierarchy_dist_path)
        with open(idx_to_cls_path, "rb") as f:
            self.idx_to_class = pickle.load(f)

    def forward(self, features, labels):
        B, _ = features.shape
        # calculate pair wise similarity 
        sim_mat = self.calculate_sim_matrix(features)
        # calculating lca weighted mask
        # lca_wt_mask = self.calculate_lca_weight_mask(labels)
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
    
    def calculate_lca_weight_mask(self, labels):
        B = labels.shape[0]
        lca_wt_mask = torch.ones((B,B), dtype=torch.float32, device=labels.device)
        for idx, i in enumerate(labels):
            for jdx, j in enumerate(labels):
                if idx != jdx:
                    lca_wt_mask[idx,jdx] = self.h_dist[(self.idx_to_class[i.item()], self.idx_to_class[j.item()])]
        return lca_wt_mask
        
class LCAConClsLoss(nn.Module):
    def __init__(self, sim = 'cosine', tau = 1.0, **kwargs):
        super().__init__()
        self.tau = tau
        self.sim = sim 
        self.ce = nn.CrossEntropyLoss()
        self.lcasupcon = LCAWSupConLoss(sim = sim, tau = tau, **kwargs)
    
    def forward(self, features, scores, labels):
        return self.lcasupcon(features, labels) + self.ce(scores, labels)

if __name__ == "__main__":
    x1 = torch.rand(4,5)
    x2 = torch.rand(4,5)
    labels = torch.tensor([1,2,3,1])
    lcswsupcon = LCAWSupConLoss(
        "src/dataset/hierarchy_pkl", 
        "src/dataset/hierarchy_pkl/tieredimg_idx_to_cls.pkl", 
        sim = 'cosine')
    lcssupcon_mse = LCAWSupConLoss(
        "src/dataset/hierarchy_pkl", 
        "src/dataset/hierarchy_pkl/tieredimg_idx_to_cls.pkl",
        sim = 'mse')
    print(lcswsupcon(x1, labels))
    print(lcssupcon_mse(x1, labels))