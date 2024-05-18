import torch 
import torchvision 
import torchvision.transforms as transforms
import os 
from src.dataset.tree import load_distances
from PIL import Image
import pickle 
from torch.utils.data.distributed import DistributedSampler

class TieredImagenetKaggle():
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
    
class TieredImagenet():
    def __init__(self, data_path, split_path, idx_to_cls_path, transformations, train=True):
        with open(idx_to_cls_path, 'rb') as f:
            self.idx_to_cls = pickle.load(f)
        self.cls_to_idx = {i:j for j,i in self.idx_to_cls.items()}
            
        split_path = os.path.join(split_path, 'train' if train else 'test')
        data_path = os.path.join(data_path, 'train' if train else 'val')


        txt_paths = os.listdir(split_path)
        split_txt_paths = list(map(
            lambda x: os.path.join(split_path, x), 
            txt_paths
        ))

        self.images = []
        
        for txt_file, split_txt_path in zip(txt_paths, split_txt_paths):
            class_id = txt_file.split('.')[0]
            with open(split_txt_path, 'r') as f:
                images_list = f.readlines()
                images_path = list(map(
                    lambda x: os.path.join(data_path, class_id if train else '', x.replace('\n', '')),
                    images_list
                ))
                images_with_class = list(zip(images_path, [class_id for _ in range(len(images_path))]))
                self.images.extend(images_with_class)

        self.transformations = transformations

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path, str_class = self.images[idx]
        img = Image.open(image_path).convert("RGB")
        img = self.transformations(img)
        int_class = self.cls_to_idx[str_class]
        return img, int_class
    
def TieredImagenetDataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']
    split_path = kwargs['split_path']
    idx_to_cls_path = kwargs['idx_to_cls_path']

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(0.6),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_dataset = TieredImagenet(
        data_dir, 
        split_path, 
        idx_to_cls_path,
        train_transforms,
        train=True)
    
    test_dataset = TieredImagenet(
        data_dir, 
        split_path, 
        idx_to_cls_path,
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

def Cifar100DataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']

    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.6),
        transforms.AugMix(),
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
    ti = TieredImagenet(
        data_path = '/workspace/DATASETS/imagenet', 
        split_path="datasets_splits/splits_tieredImageNet-H",
        idx_to_cls_path = 'src/dataset/hierarchy_pkl/tieredimg_idx_to_cls.pkl',
        transformations = transformations,
        train = True
    )
    img, label = ti[0]
    print(img.shape, label)