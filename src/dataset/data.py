import torch 
import torchvision 
import torchvision.transforms as transforms
import os 
from src.dataset.tree import load_distances
from PIL import Image
import pickle 

class TieredImagenet():
    def __init__(self, data_path, transformations):
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

        with open("src/dataset/hierarchy_pkl/tieredimg_idx_to_cls.pkl", "wb") as f:
            pickle.dump(self.idx_to_cls, f)

        self.transformations = transformations

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path, str_class = self.images[idx]
        img = Image.open(image_path).convert("RGB")
        img = self.transformations(img)
        int_class = self.cls_to_idx[str_class]
        return img, int_class
    
def TieredImagenetDataLoader(data_dir):
    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(0.6),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    dataset = TieredImagenet(
        data_dir, 
        test_transforms)

    len_data = len(dataset)
    len_test_data = int(0.1 * len_data)
    remaining_data = len_data - len_test_data
    print(f"# of Training points: {remaining_data}\n# of Testing points: {len_test_data}")

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [remaining_data, len_test_data])

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 256,
        shuffle=True,
        pin_memory=True,
        num_workers = 0
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 256,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    return train_dl, test_dl

if __name__ == "__main__":
    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(0.6),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    ti = TieredImagenet("/workspace/DATASETS/tiered-imagenet/tiered_imagenet", transformations)
    img, label = ti[0]
    print(img.shape, label)