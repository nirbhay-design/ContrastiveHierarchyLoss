import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

def mse_sim(x1, x2):
    return -F.mse_loss(x1, x2)

def cosine_sim(x1, x2):
    x1 = F.normalize(x1, dim = -1, p = 2)
    x2 = F.normalize(x2, dim = -1, p = 2)
    return F.cosine_similarity(x1, x2)

class LCAWSupConLoss(nn.Module):
    def __init__(self, simfun):
        super().__init__()
        # simfun -> similarity function
        self.simfun = simfun 

    def forward(self, features, labels):
        pass 

    def compute_loss(self, anchor, positive, negative):
        pass

    def getpositives(self, features, labels, label):
        return features[labels == label]
    
    def getnegatives(self, features, labels, label):
        return features[labels != label]
        

if __name__ == "__main__":
    x1 = torch.rand(4,5)
    x2 = torch.rand(4,5)
    print(mse_sim(x1,x2))
    print(cosine_sim(x1, x2))
    label = torch.tensor([1,2,3,1])
    print(x1)
    print(x1[label == 1])