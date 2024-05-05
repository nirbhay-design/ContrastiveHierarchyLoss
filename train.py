from src.losses.LCAWSupCon import LCAConClsLoss
from src.network.Network import Network
from src.dataset.data import CIFAR_dataloader
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 

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
        accuracy = round(correct / samples, 3)
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
                progress(idx+1,len(train_loader))
        
        if eval_every % epochs == 0:
            cur_test_acc = evaluate(model, test_loader, device, return_logs)
            print(f"Test Accuracy at {epochs}: {cur_test_acc}")
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_loss))
        
        print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss: {cur_loss:.3f}")
    
    final_test_acc = evaluate(model, test_loader, device, return_logs)
    print(f"Final Test Accuracy: {final_test_acc}")

    return model, tval

if __name__ == "__main__":
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model = Network(num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9)
    loss = LCAConClsLoss(sim = 'cosine', tau = 1.0)
    train_dl, test_dl = CIFAR_dataloader()

    return_logs = True
    eval_every = 5
    n_epochs = 1
    device = torch.device("cuda:0")

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