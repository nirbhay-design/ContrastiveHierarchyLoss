import torch 
import torchvision 
import torchvision.transforms as transforms
import os 
from src.dataset.tree import load_distances
from PIL import Image
import pickle 
from torch.utils.data.distributed import DistributedSampler

class TieredImagenet():
    def __init__(self, data_path, transformations, train=True):
        # https://www.kaggle.com/datasets/arjun2000ashok/tieredimagenet
        # data is split given in three directories train -> 351 classes, val -> 97, test -> 160
        classes_list = []
        self.class_map_images = {}
        self.images = []
        sub_paths = ["train", "val", "test"]
        for path in sub_paths:
            cur_path = os.path.join(data_path, path)
            classes_ = os.listdir(cur_path)
            classes_list.extend(classes_)
            classes_path = list(map(lambda x: os.path.join(cur_path, x), classes_))
            for cur_class, class_path in zip(classes_, classes_path):
                images_ = os.listdir(class_path)
                images_path = list(map(lambda x: os.path.join(class_path, x), images_))
                self.class_map_images[cur_class] = images_path

        self.cls_to_idx = dict(zip(classes_list, range(len(classes_list))))
        self.idx_to_cls = {j:i for i,j in self.cls_to_idx.items()}

        idx_to_cls_path = "src/dataset/hierarchy_pkl/tieredimg_idx_to_cls.pkl"
        if not os.path.exists(idx_to_cls_path):
            with open(idx_to_cls_path, "wb") as f:
                pickle.dump(self.idx_to_cls, f)
        else:
            print(f"Path Already Exists: {idx_to_cls_path}")

        self.per_class_train_test = {}
        for class_id, images_path_list in self.class_map_images.items():
            len_images = len(images_path_list)
            num_test = int(0.2 * len_images)
            num_train = len_images - num_test 
            self.per_class_train_test[class_id] = (num_train, num_test)

        for class_id, images_path in self.class_map_images.items():
            if train:
                images_path = images_path[0:self.per_class_train_test[class_id][0]]
            else:
                images_path = images_path[-self.per_class_train_test[class_id][1]:]
            self.images.extend([(ip, class_id) for ip in images_path])

        self.transformations = transformations

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path, str_class = self.images[idx]
        img = Image.open(image_path).convert("RGB")
        img = self.transformations(img)
        int_class = self.cls_to_idx[str_class]
        return img, int_class
    
def TieredImagenetDataLoader(data_dir, image_size, **kwargs):
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.6),
        transforms.AugMix(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_dataset = TieredImagenet(
        data_dir, 
        train_transforms,
        train=True)
    
    test_dataset = TieredImagenet(
        data_dir, 
        test_transforms,
        train=False
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset) if distributed else None 
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers= num_workers
    )

    return train_dl, test_dl, train_dataset, test_dataset

def Cifar100DataLoader(data_dir, image_size, **kwargs):
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.6),
        transforms.AugMix(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_dataset = torchvision.datasets.CIFAR100(
        data_dir, 
        transform = train_transforms,
        train=True,
        download = True)
    
    test_dataset = torchvision.datasets.CIFAR100(
        data_dir, 
        transform=test_transforms,
        train=False,
        download=True
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset) if distributed else None 
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers= num_workers
    )

    return train_dl, test_dl, train_dataset, test_dataset

if __name__ == "__main__":
    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(0.6),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    ti = TieredImagenet("/workspace/DATASETS/tiered-imagenet/tiered_imagenet", transformations)
    img, label = ti[0]
    print(img.shape, label)