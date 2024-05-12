from src.losses.LCAWSupCon import LCAConClsLoss
from src.network.Network import Network, MLP
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

def evaluate(model, mlp, loader, device, return_logs=False):
    model.eval()
    mlp.eval()
    correct = 0;samples =0
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            # model = model.to(config.device)

            feats, _ = model(x)
            scores = mlp(feats)

            predict_prob = F.softmax(scores,dim=1)
            _,predictions = predict_prob.max(1)

            correct += (predictions == y).sum()
            samples += predictions.size(0)
        
            if return_logs:
                progress(idx+1,loader_len)
                # print('batches done : ',idx,end='\r')
        accuracy = round(float(correct / samples), 3)
    return accuracy 

def train(
        model, mlp, train_loader,
        test_loader, lossfunction, 
        optimizer, mlp_optimizer, opt_lr_schedular, 
        eval_every, n_epochs, device, return_logs=False): 
    
    tval = {'trainacc':[],"trainloss":[]}
    model = model.to(device)
    mlp = mlp.to(device)
    for epochs in range(n_epochs):
        model.train()
        mlp.train()
        cur_loss = 0
        curacc = 0
        cur_mlp_loss = 0
        len_train = len(train_loader)
        for idx , (data,target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            feats, proj_feat = model(data)
            scores = mlp(feats.detach())            
            
            loss_con, loss_sup = lossfunction(proj_feat, scores, target)
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            mlp_optimizer.zero_grad()
            loss_sup.backward()
            mlp_optimizer.step()

            cur_loss += loss_con.item() / (len_train)
            cur_mlp_loss += loss_sup.item() / (len_train)
            scores = F.softmax(scores,dim = 1)
            _,predicted = torch.max(scores,dim = 1)
            correct = (predicted == target).sum()
            samples = scores.shape[0]
            curacc += correct / (samples * len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), loss_sup=loss_sup.item())
        
        opt_lr_schedular.step()

        if epochs % eval_every == 0:
            cur_test_acc = evaluate(model, mlp, test_loader, device, return_logs)
            print(f"Test Accuracy at epoch: {epochs}: {cur_test_acc}")
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_loss))
        
        print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss_con: {cur_loss:.3f} train_loss_sup: {cur_mlp_loss:.3f}")
    
    final_test_acc = evaluate(model, mlp, test_loader, device, return_logs)
    print(f"Final Test Accuracy: {final_test_acc}")

    return model, tval

def loss_function(loss_type = 'lcacon', **kwargs):
    if loss_type == 'lcacon':
        return LCAConClsLoss(**kwargs)
    else:
        print("{loss_type} Loss is Not Supported")
        return None 
    
def model_optimizer(model, opt_name, **opt_params):
    print(f"using optimizer: {opt_name}")
    if opt_name == "SGD":
        return optim.SGD(model.parameters(), **opt_params)
    elif opt_name == "ADAM":
        return optim.Adam(model.parameters(), **opt_params)
    elif opt_name == "AdamW":
        return optim.AdamW(model.parameters(), **opt_params)
    else:
        print("{opt_name} not available")
        return None

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
    
    model = Network(**config['model_params'])
    mlp = MLP(model.classifier_infeatures, config['num_classes'])

    optimizer = model_optimizer(model, config['opt'], **config['opt_params'])
    print(optimizer)
    mlp_optimizer = model_optimizer(mlp, config['mlp_opt'], **config['mlp_opt_params'])

    opt_lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    loss = loss_function(loss_type = config['loss'], **config.get('loss_params', {}))
    
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
        mlp,
        train_dl,
        test_dl,
        loss,
        optimizer,
        mlp_optimizer,
        opt_lr_schedular,
        eval_every,
        n_epochs,
        device,
        return_logs
    )