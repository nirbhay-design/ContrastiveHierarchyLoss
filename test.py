from src.losses.LCAWSupCon import LCAWSupConLoss, SupConLoss
import torch 
from src.dataset.data import TieredImagenet
import torchvision.transforms as transforms 

# ti = TieredImagenet("/workspace/DATASETS/tiered-imagenet/tiered_imagenet", transforms.ToTensor(), False)

x1 = torch.rand(10,5)
# x2 = torch.rand(10,5)
labels = torch.tensor([1,2,3,1,4,5,23,465,200,200])
lcswsupcon = LCAWSupConLoss(
    "src/dataset/hierarchy_pkl", 
    "src/dataset/hierarchy_pkl/tieredimg_idx_to_cls.pkl", 
    sim = 'cosine')
# lcssupcon_mse = LCAWSupConLoss(
#     "src/dataset/hierarchy_pkl", 
#     "src/dataset/hierarchy_pkl/tieredimg_idx_to_cls.pkl",
#     sim = 'mse')
# print(lcswsupcon(x1, labels))
# print(lcssupcon_mse(x1, labels))

supcon = SupConLoss()
print(supcon(x1,labels))
print(lcswsupcon(x1, labels))