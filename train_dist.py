import sys, random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.network.Network import Network, MLP
from train_utils import yaml_loader, progress, evaluate, \
                        model_optimizer, \
                        loss_function, \
                        load_dataset

import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group
import os 

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = "6707"
    init_process_group(backend = 'nccl', rank = rank, world_size = world_size)

def train(
        model, mlp, train_loader,
        test_loader, lossfunction, 
        optimizer, mlp_optimizer, opt_lr_schedular, 
        eval_every, n_epochs, device_id, return_logs=False): 
    
    tval = {'trainacc':[],"trainloss":[]}
    device = torch.device(f"cuda:{device_id}")
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
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), loss_sup=loss_sup.item(), gpu_id = device_id)
        
        opt_lr_schedular.step()

        if epochs % eval_every == 0 and device_id == 0:
            cur_test_acc = evaluate(model, mlp, test_loader, device, return_logs)
            print(f"[GPU{device_id}] Test Accuracy at epoch: {epochs}: {cur_test_acc}")
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_loss))
        
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss_con: {cur_loss:.3f} train_loss_sup: {cur_mlp_loss:.3f}")
    
    if device_id == 0:
        final_test_acc = evaluate(model, mlp, test_loader, device, return_logs)
        print(f"[GPU{device_id}] Final Test Accuracy: {final_test_acc}")

    return model, tval

def main(rank, world_size, config):

    ddp_setup(rank, world_size)

    model = Network(**config['model_params'])
    mlp = MLP(model.classifier_infeatures, config['num_classes'])
    print(f"############################# RANK: {rank} #############################")
    model = model.to(rank)
    mlp = mlp.to(rank)

    model = DDP(model, device_ids = [rank])
    mlp = DDP(mlp, device_ids=[rank])

    optimizer = model_optimizer(model, config['opt'], **config['opt_params'])
    mlp_optimizer = model_optimizer(mlp, config['mlp_opt'], **config['mlp_opt_params'])

    opt_lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    loss = loss_function(loss_type = config['loss'], **config.get('loss_params', {}))
    
    train_dl, test_dl, train_ds, test_ds = load_dataset(
        dataset_name=config['data_name'],
        data_dir=config['data_dir'],
        image_size = config['image_size'],
        batch_size = config['batch_size'],
        num_workers=config['num_workers'],
        distributed = config['distributed'])
    
    if rank == 0:
        print(f"# of Training Images: {len(train_ds)}")
        print(f"# of Testing Images: {len(test_ds)}")


    return_logs = config['return_logs']
    eval_every = config['eval_every']
    n_epochs = config['n_epochs']

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
        rank,
        return_logs
    )

    destroy_process_group()

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

    world_size = torch.cuda.device_count()
    
    mp.spawn(main, args=(world_size, config), nprocs=world_size)