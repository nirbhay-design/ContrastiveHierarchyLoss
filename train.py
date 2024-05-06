from src.losses.LCAWSupCon import LCAConClsLoss
from src.network.Network import Network
from src.dataset.data import TieredImagenetDataLoader
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
import yaml, sys, random, numpy as np
from yaml.loader import SafeLoader

def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
    
    return config_data

def progress(current, total, **kwargs):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    data_ = ""
    for meter, data in kwargs.items():
        data_ += f"{meter}: {round(data,2)}|"
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}|{data_}",end='\r')
    if (current == total):
        print()

def evaluate(model, loader, device, return_logs=False):
    model.eval()
    correct = 0;samples =0
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            # model = model.to(config.device)

            _, scores = model(x)
            predict_prob = F.softmax(scores,dim=1)
            _,predictions = predict_prob.max(1)

            correct += (predictions == y).sum()
            samples += predictions.size(0)
        
            if return_logs:
                progress(idx+1,loader_len)
                # print('batches done : ',idx,end='\r')
        accuracy = round(float(correct / samples), 3)
    model.train()
    return accuracy 

def train(model, train_loader, test_loader, lossfunction, optimizer, eval_every, n_epochs, device, return_logs=False): 
    tval = {'trainacc':[],"trainloss":[]}
    model = model.to(device)
    model.train()
    for epochs in range(n_epochs):
        cur_loss = 0
        curacc = 0
        len_train = len(train_loader)
        for idx , (data,target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            feats, scores = model(data)
            
            loss = lossfunction(feats, scores, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_loss += loss.item() / (len_train)
            scores = F.softmax(scores,dim = 1)
            _,predicted = torch.max(scores,dim = 1)
            correct = (predicted == target).sum()
            samples = scores.shape[0]
            curacc += correct / (samples * len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss=loss.item())
        
        if epochs % eval_every == 0:
            cur_test_acc = evaluate(model, test_loader, device, return_logs)
            print(f"Test Accuracy at epoch: {epochs}: {cur_test_acc}")
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_loss))
        
        print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss: {cur_loss:.3f}")
    
    final_test_acc = evaluate(model, test_loader, device, return_logs)
    print(f"Final Test Accuracy: {final_test_acc}")

    return model, tval

if __name__ == "__main__":
    config = yaml_loader(sys.argv[1])
    
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("environment: ")
    print(f"YAML: {sys.argv[1]}")
    for key, value in config.items():
        print(f"==> {key}: {value}")
    
    model = Network(num_classes=config['num_classes'])
    optimizer = optim.Adam(model.parameters(), lr = config['lr'])
    loss = LCAConClsLoss( 
        sim = 'cosine', tau = 1.0,
        hierarchy_dist_path = config['hierarchy_path'], 
        idx_to_cls_path = config["idx_to_cls_path"], 
        dataset_name = config['dataset_name'])
    
    train_dl, test_dl, train_ds, test_ds = TieredImagenetDataLoader(
        data_dir=config['data_dir'],
        image_size = config['image_size'],
        batch_size = config['batch_size'],
        num_workers=config['num_workers'])
    
    print(f"# of Training Images: {len(train_ds)}")
    print(f"# of Testing Images: {len(test_ds)}")


    return_logs = config['return_logs']
    eval_every = config['eval_every']
    n_epochs = config['n_epochs']
    device = torch.device(f"cuda:{config['gpu_id']}")

    train(
        model,
        train_dl,
        test_dl,
        loss,
        optimizer,
        eval_every,
        n_epochs,
        device,
        return_logs
    )