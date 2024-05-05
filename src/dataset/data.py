import torch 
import torchvision 
import torchvision.transforms as transforms

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

    dataset = torchvision.dataset.CIFAR10(
        root = data_dir, 
        train = True, 
        download=True, 
        transform=transformations)
    
    test_dataset = torchvision.dataset.CIFAR10(
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