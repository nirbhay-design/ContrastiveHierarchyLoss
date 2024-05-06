import torch 
import torchvision 
import torchvision.transforms as transforms
import os 
from tree import load_distances

class TieredImagenet():
    def __init__(self, data_path, hierarchy_dist_path):
        # https://www.kaggle.com/datasets/arjun2000ashok/tieredimagenet
        # data is split given in three directories train -> 351 classes, val -> 97, test -> 160
        classes_list = []
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
                self.images.extend(list(zip(images_path, [cur_class for _ in range(len(images_path))])))

        self.cls_to_idx = dict(zip(classes_list, range(len(classes_list))))
        self.idx_to_cls = {j:i for i,j in self.cls_to_idx.items()}

        h_dist = load_distances("tiered-imagenet-224", "ilsvrc", hierarchy_dist_path)
        print(h_dist[(classes_list[1], classes_list[0])])

def CIFAR_dataloader(data_dir):
    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(0.6),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    dataset = torchvision.datasets.CIFAR10(
        root = data_dir, 
        train = True, 
        download=True, 
        transform=transformations)
    
    test_dataset = torchvision.datasets.CIFAR10(
        root = data_dir, 
        train = False, 
        download=True, 
        transform=test_transforms)
    

    train_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size = 128,
        shuffle=True,
        pin_memory=True,
        num_workers = 4
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 128,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    return train_dl, test_dl    

if __name__ == "__main__":
    ti = TieredImagenet("/workspace/DATASETS/tiered-imagenet/tiered_imagenet", "src/dataset/hierarchy_pkl")
