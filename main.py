# from dataset_custom import GraphNoduleMNIST3D, GraphSynapseMNIST3D, GraphAdrenalMNIST3D, GraphVesselMNIST3D, GraphOrganMNIST3D, GraphFractureMNIST3D, GraphNSCLC3D, GraphRadcure3D, NSCLC3D
# from dataset_custom import NoduleMNIST3D, SynapseMNIST3D, AdrenalMNIST3D, VesselMNIST3D, OrganMNIST3D, FractureMNIST3D
from dataset_mednist import MedMNIST3D
from dataset_custom import GraphNSCLC3D, GraphRadcure3D
from param_configurator import ParamConfigurator
import mlflow 
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils import set_seed, save_conda_env
from model import DINOv2, GNN, Classifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from datetime import datetime
from torch_geometric.loader import DataLoader as GeometricDataLoader


def train(config) -> None:

    # save_conda_env(config)
    mlflow.log_params(config.__dict__)    

    match config.dataset:
        # MedMNIST Datasets
        case "nodule" | "synapse" | "adrenal" | "vessel" | "organ" | "fracture":
            train_dataset = MedMNIST3D(config=config, mode='train')
            val_dataset = MedMNIST3D(config=config, mode='val')
            test_dataset = MedMNIST3D(config=config, mode='test')
        # case "synapse":
        #     train_dataset = GraphSynapseMNIST3D(config=config, mode='train')
        #     val_dataset = GraphSynapseMNIST3D(config=config, mode='val')
        #     test_dataset = GraphSynapseMNIST3D(config=config, mode='test')
        # case "adrenal":
        #     train_dataset = GraphAdrenalMNIST3D(config=config, mode='train')
        #     val_dataset = GraphAdrenalMNIST3D(config=config, mode='val')
        #     test_dataset = GraphAdrenalMNIST3D(config=config, mode='test')
        # case "vessel":
        #     train_dataset = GraphVesselMNIST3D(config=config, mode='train')
        #     val_dataset = GraphVesselMNIST3D(config=config, mode='val')
        #     test_dataset = GraphVesselMNIST3D(config=config, mode='test')
        # case "organ":
        #     train_dataset = GraphOrganMNIST3D(config=config, mode='train')
        #     val_dataset = GraphOrganMNIST3D(config=config, mode='val')
        #     test_dataset = GraphOrganMNIST3D(config=config, mode='test')
        # case "fracture":
        #     train_dataset = GraphFractureMNIST3D(config=config, mode='train')
        #     val_dataset = GraphFractureMNIST3D(config=config, mode='val')
        #     test_dataset = GraphFractureMNIST3D(config=config, mode='test')

        # Custom Datasets
        case "nsclc":
            train_dataset = GraphNSCLC3D(config=config, mode='train')
            val_dataset = GraphNSCLC3D(config=config, mode='val')
            test_dataset = GraphNSCLC3D(config=config, mode='test')
        case "radcure":
            train_dataset = GraphRadcure3D(config=config, mode='train')
            val_dataset = GraphRadcure3D(config=config, mode='val')
            test_dataset = GraphRadcure3D(config=config, mode='test')
        case "sts":
            raise NotImplementedError("Dataset not yet implemented!")
        case _:
            raise NotImplementedError(f"Given dataset '{config.dataset}' not implemented!")
        
    match config.model_name:
        case "GNN":
            model = GNN(config=config).to(config.device)
            train_loader = GeometricDataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = GeometricDataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)
            test_loader = GeometricDataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)
            
        case "MLP":
            model = Classifier(config=config).to(config.device)
            train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)    
            val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)   
            test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
                
            
            
        
        # case "MLP":
        #     model = Classifier(config=config).to(config.device)

        #     match config.dataset:
        #         # MedMNIST Datasets
        #         case "nodule":
        #             train_dataset = NoduleMNIST3D(config=config, mode='train')
        #             val_dataset = NoduleMNIST3D(config=config, mode='val')
        #             test_dataset = NoduleMNIST3D(config=config, mode='test')
        #         case "synapse":
        #             train_dataset = SynapseMNIST3D(config=config, mode='train')
        #             val_dataset = SynapseMNIST3D(config=config, mode='val')
        #             test_dataset = SynapseMNIST3D(config=config, mode='test')
        #         case "adrenal":
        #             train_dataset = AdrenalMNIST3D(config=config, mode='train')
        #             val_dataset = AdrenalMNIST3D(config=config, mode='val')
        #             test_dataset = AdrenalMNIST3D(config=config, mode='test')
        #         case "vessel":
        #             train_dataset = VesselMNIST3D(config=config, mode='train')
        #             val_dataset = VesselMNIST3D(config=config, mode='val')
        #             test_dataset = VesselMNIST3D(config=config, mode='test')
        #         case "organ":
        #             train_dataset = OrganMNIST3D(config=config, mode='train')
        #             val_dataset = OrganMNIST3D(config=config, mode='val')
        #             test_dataset = OrganMNIST3D(config=config, mode='test')
        #         case "fracture":
        #             train_dataset = FractureMNIST3D(config=config, mode='train')
        #             val_dataset = FractureMNIST3D(config=config, mode='val')
        #             test_dataset = FractureMNIST3D(config=config, mode='test')                
        #         case "nsclc":
        #             train_dataset = NSCLC3D(config=config, mode='train')
        #             val_dataset = NSCLC3D(config=config, mode='val')
        #             test_dataset = NSCLC3D(config=config, mode='test')
        #         case "radcure":
        #             raise NotImplementedError("Dataset not yet implemented!")
        #         case "sts":
        #             raise NotImplementedError("Dataset not yet implemented!")
        #         case _:
        #             raise NotImplementedError(f"Given dataset '{config.dataset}' not implemented!")

        #     train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)    
        #     val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)   
        #     test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
            
            
        # case _:
        #     train_dataset = NoduleMNIST3D(config=config, mode='train')
        #     val_dataset = NoduleMNIST3D(config=config, mode='val')
        #     test_dataset = NoduleMNIST3D(config=config, mode='test')
        #     train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)    
        #     val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)   
        #     test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
        #     model = DINOv2(config=config).to(config.device)  
    
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

                    case "MLP":
                        inputs = inputs.view(-1, inputs.shape[-1])
                        inputs = inputs.to(torch.float32).to(config.device)
                        labels = labels.view(-1)
                        labels = labels.to(torch.long).to(config.device)
                        
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

                if config.n_classes > 2:
                    score = torch.softmax(out, dim=1)
                else:
                    score = torch.softmax(out, dim=1)[:, -1]

                train_loss_list.append(loss.detach().cpu().item())
                train_true_list.extend(labels.detach().cpu().numpy())
                train_pred_list.extend(pred.detach().cpu().numpy())
                train_score_list.extend(score.detach().cpu().numpy())
        
        train_loss = np.mean(train_loss_list)
        train_bacc = balanced_accuracy_score(train_true_list, train_pred_list)
        train_auc = roc_auc_score(train_true_list, train_score_list, multi_class='ovr')
        train_acc = accuracy_score(train_true_list, train_pred_list)

        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("train_bacc", train_bacc, step=epoch)
        mlflow.log_metric("train_auc", train_auc, step=epoch)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("learning_rate", lr, step=epoch)
        scheduler.step()

        #######################
        # Validation ##########
        #######################
        eval_loss, eval_bacc, eval_auc, eval_acc = evaluate(config, model, val_loader, criterion)
        mlflow.log_metric("val_bacc", eval_bacc, step=epoch)
        mlflow.log_metric("val_acc", eval_acc, step=epoch)
        mlflow.log_metric("val_auc", eval_auc, step=epoch)
        mlflow.log_metric("val_loss", eval_loss, step=epoch)
    
    #################
    # Test ##########
    #################    
    eval_loss, eval_bacc, eval_auc, eval_acc = evaluate(config, model, test_loader, criterion)
    mlflow.log_metric("test_bacc", eval_bacc, step=epoch)
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

                    case "MLP":
                        inputs = inputs.view(-1, inputs.shape[-1])
                        inputs = inputs.to(torch.float32).to(config.device)
                        labels = labels.view(-1)
                        labels = labels.to(torch.long).to(config.device)
                        
                    case _:
                        inputs = inputs.view(-1, 3, 224, 224)
                        inputs = inputs.float().to(config.device)
                        labels = labels.view(-1)
                        labels = labels.long().to(config.device)
                
                out = model(inputs)

                loss = criterion(out, labels)
                pred = torch.max(out, dim=1).indices
                if config.n_classes > 2:
                    score = torch.softmax(out, dim=1)
                else:
                    score = torch.softmax(out, dim=1)[:, -1]

                eval_loss_list.append(loss.detach().cpu().item())
                eval_true_list.extend(labels.detach().cpu().numpy())
                eval_pred_list.extend(pred.detach().cpu().numpy())
                eval_score_list.extend(score.detach().cpu().numpy())

    eval_loss = np.mean(eval_loss_list)
    eval_bacc = balanced_accuracy_score(eval_true_list, eval_pred_list)
    eval_auc = roc_auc_score(eval_true_list, eval_score_list, multi_class='ovr')
    eval_acc = accuracy_score(eval_true_list, eval_pred_list)

    return eval_loss, eval_bacc, eval_auc, eval_acc


if __name__ == "__main__":
    config = ParamConfigurator()
    set_seed(config.seed)

    mlflow.set_experiment(f'{config.model_name}_{config.dataset}')
    date = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    with mlflow.start_run(run_name=date):
        train(config)
