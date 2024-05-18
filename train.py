import sys, random
import torch
import torch.optim as optim
import numpy as np
from src.network.Network import Network, MLP
from train_utils import yaml_loader, train, \
                        model_optimizer, \
                        loss_function, \
                        load_dataset

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

    opt_lr_schedular = optim.lr_scheduler.StepLR(optimizer, **config['schedular_params'])

    loss = loss_function(loss_type = config['loss'], **config.get('loss_params', {}))
    
    train_dl, test_dl, train_ds, test_ds = load_dataset(
        dataset_name = config['data_name'],
        data_dir=config['data_dir'],
        image_size = config['image_size'],
        batch_size = config['batch_size'],
        num_workers=config['num_workers'],
        distributed = config['distributed'])
    
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