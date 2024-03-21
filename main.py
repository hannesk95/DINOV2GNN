from dataset import NoduleMNIST3D, GraphNoduleMNIST3D, GrapNSCLC3D, GrapRadcure3D
from param_configurator import ParamConfigurator
import mlflow 
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils import set_seed, save_conda_env
from model import DINOv2, GNN
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from datetime import datetime
from torch_geometric.loader import DataLoader as GeometricDataLoader


def train(config) -> None:

    save_conda_env(config)
    mlflow.log_params(config.__dict__)    

    match config.model_name:
        case "GNN":
            # train_dataset = GraphNoduleMNIST3D(config=config, mode='train')
            # val_dataset = GraphNoduleMNIST3D(config=config, mode='val')
            # test_dataset = GraphNoduleMNIST3D(config=config, mode='test')
            # train_dataset = GrapNSCLC3D(config=config, mode='train')
            # val_dataset = GrapNSCLC3D(config=config, mode='val')
            # test_dataset = GrapNSCLC3D(config=config, mode='test')
            train_dataset = GrapRadcure3D(config=config, mode='train')
            val_dataset = GrapRadcure3D(config=config, mode='val')
            test_dataset = GrapRadcure3D(config=config, mode='test')
            train_loader = GeometricDataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = GeometricDataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)
            test_loader = GeometricDataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)
            model = GNN(config=config).to(config.device)
        case _:
            train_dataset = NoduleMNIST3D(config=config, mode='train')
            val_dataset = NoduleMNIST3D(config=config, mode='val')
            test_dataset = NoduleMNIST3D(config=config, mode='test')
            train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)    
            val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)   
            test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
            model = DINOv2(config=config).to(config.device)  
    
    criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.class_weights).to(config.device)
    
    match config.optimizer:
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, 
                                        nesterov=config.nesterov, weight_decay=config.weight_decay)
        case _:
            raise NotImplementedError(f"Optimizer not implemented!")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)   
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    mlflow.log_param("num_parameters", total_params)

    #######################
    # Training ############
    #######################
    best_val_loss = float('inf')
    best_val_acc = float('-inf')
    best_val_auc = float('-inf')    

    for epoch in range(1, config.epochs+1):
        train_loss_list = []
        train_true_list = []
        train_pred_list = []
        train_score_list = []
        lr = scheduler.get_last_lr()[0]

        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"[Training]    Epoch {epoch:03d} | {config.epochs}")
                inputs, labels = data
                
                match config.model_name:
                    case "GNN":
                        inputs = inputs.to(config.device)
                        labels = labels.view(-1).to(config.device)
                        
                    case _:
                        inputs = inputs.view(-1, 3, 224, 224)
                        inputs = inputs.float().to(config.device)
                        labels = labels.view(-1)
                        labels = labels.long().to(config.device)                                     

                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                pred = torch.max(out, dim=1).indices
                score = torch.softmax(out, dim=1)[:, 1]

                train_loss_list.append(loss.detach().cpu().item())
                train_true_list.extend(labels.detach().cpu().numpy())
                train_pred_list.extend(pred.detach().cpu().numpy())
                train_score_list.extend(score.detach().cpu().numpy())
        
        train_loss = np.mean(train_loss_list)
        train_acc = balanced_accuracy_score(train_true_list, train_pred_list)
        train_auc = roc_auc_score(train_true_list, train_score_list)

        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("train_auc", train_auc, step=epoch)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("learning_rate", lr, step=epoch)
        scheduler.step()

        #######################
        # Validation ##########
        #######################
        eval_loss, eval_acc, eval_auc = evaluate(config, model, val_loader, criterion)
        mlflow.log_metric("val_acc", eval_acc, step=epoch)
        mlflow.log_metric("val_auc", eval_auc, step=epoch)
        mlflow.log_metric("val_loss", eval_loss, step=epoch)
    
    #################
    # Test ##########
    #################    
    eval_loss, eval_acc, eval_auc = evaluate(config, model, test_loader, criterion)
    mlflow.log_metric("test_acc", eval_acc, step=epoch)
    mlflow.log_metric("test_auc", eval_auc, step=epoch)
    mlflow.log_metric("test_loss", eval_loss, step=epoch)


def evaluate(config, model, dataloader, criterion) -> None:
    eval_loss_list = []
    eval_true_list = []
    eval_pred_list = []
    eval_score_list = []
    
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"[Validation]")
                inputs, labels = data
                
                match config.model_name:
                    case "GNN":
                        inputs = inputs.to(config.device)
                        labels = labels.view(-1).to(config.device)
                        
                    case _:
                        inputs = inputs.view(-1, 3, 224, 224)
                        inputs = inputs.float().to(config.device)
                        labels = labels.view(-1)
                        labels = labels.long().to(config.device)
                
                out = model(inputs)

                loss = criterion(out, labels)
                pred = torch.max(out, dim=1).indices
                score = torch.softmax(out, dim=1)[:, 1]

                eval_loss_list.append(loss.detach().cpu().item())
                eval_true_list.extend(labels.detach().cpu().numpy())
                eval_pred_list.extend(pred.detach().cpu().numpy())
                eval_score_list.extend(score.detach().cpu().numpy())

    eval_loss = np.mean(eval_loss_list)
    eval_acc = balanced_accuracy_score(eval_true_list, eval_pred_list)
    eval_auc = roc_auc_score(eval_true_list, eval_score_list)

    return eval_loss, eval_acc, eval_auc


if __name__ == "__main__":
    config = ParamConfigurator()
    set_seed(config.seed)

    mlflow.set_experiment(f'{config.model_name}')
    date = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    with mlflow.start_run(run_name=date):
        train(config)
