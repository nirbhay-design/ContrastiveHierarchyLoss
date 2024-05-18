from src.losses.LCAWSupCon import LCAWSupConLoss, SupConLoss
import torch 
from src.dataset.data import TieredImagenet
import torchvision.transforms as transforms 

train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.6),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])


data = TieredImagenet(
    data_path = '/workspace/DATASETS/imagenet', 
    split_path="datasets_splits/splits_tieredImageNet-H",
    idx_to_cls_path = 'src/dataset/hierarchy_pkl/tieredimg_idx_to_cls.pkl',
    transformations = train_transforms,
    train = True)

print(len(data))

train_dl = torch.utils.data.DataLoader(
        data,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers = 0,
    )

for (img, label) in train_dl:
    print(img.shape, label.shape)