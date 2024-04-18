import torch
from torch.utils.data import Dataset
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from torchvision import transforms
from transformers import AutoImageProcessor
from PIL import Image
from sklearn.model_selection import train_test_split

class GraphRadcure3D(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.config = config

        match mode:
            case 'train':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/RADCURE_preprocessed/graph_data/graphs.pt")
                self.data = [data for data in self.data if data.split == "training"]
                self.labels = np.array([graph.label for graph in self.data])
                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
            case 'val':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/RADCURE_preprocessed/graph_data/graphs.pt")
                self.data = [data for data in self.data if data.split == "test"]
                self.labels = np.array([graph.label for graph in self.data])
                self.data, _, _, _ = train_test_split(self.data, self.labels, test_size=0.5, stratify=self.labels, random_state=self.config.seed)                
            case 'test':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/RADCURE_preprocessed/graph_data/graphs.pt")
                self.data = [data for data in self.data if data.split == "test"]
                self.labels = np.array([graph.label for graph in self.data])
                _, self.data, _, _ = train_test_split(self.data, self.labels, test_size=0.5, stratify=self.labels, random_state=self.config.seed)                
            case _:
                raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")

        # match mode:
        #     case 'train':
        #         self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/radcure_train_graphs.pt")
        #         self.labels = np.array([graph.label for graph in self.data])
        #         self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
        #         self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
        #     case 'val':
        #         self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/radcure_val_graphs.pt")                
        #     case 'test':
        #         self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/radcure_test_graphs.pt")                
        #     case _:
        #         raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        graph = self.data[index]
        graph.x = graph.x.to(torch.float32)
        graph.edge_index = graph.edge_index.to(torch.long)
        graph.edge_attr = graph.edge_attr.to(torch.float32)
        # graph.edge_attr = (torch.ones(56)*1000).to(torch.float32)
        # graph.edge_attr = None

        label = torch.tensor(graph.label).long()
        
        return graph, label

class NSCLC3D(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.config = config

        match mode:
            case 'train':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/NSCLC_preprocessed/graph_data/graphs.pt")
                self.labels = np.array([graph.label for graph in self.data])
                self.data, _, self.labels, _ = train_test_split(self.data, self.labels, train_size=0.8, random_state=config.seed, stratify=self.labels)
                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
            case 'val':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/NSCLC_preprocessed/graph_data/graphs.pt")
                self.labels = np.array([graph.label for graph in self.data])
                _, val_test_data, _, val_test_labels = train_test_split(self.data, self.labels, train_size=0.8, random_state=config.seed, stratify=self.labels)
                self.data, _, self.labels, _ = train_test_split(val_test_data, val_test_labels, test_size=0.5, random_state=config.seed, stratify=val_test_labels)
                
            case 'test':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/NSCLC_preprocessed/graph_data/graphs.pt")
                self.labels = np.array([graph.label for graph in self.data])
                _, val_test_data, _, val_test_labels = train_test_split(self.data, self.labels, train_size=0.8, random_state=config.seed, stratify=self.labels)
                _, self.data, _, self.labels = train_test_split(val_test_data, val_test_labels, test_size=0.5, random_state=config.seed, stratify=val_test_labels)
            case _:
                raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        data = self.data[index].x
        label = torch.Tensor([self.labels[index]]).repeat(data.shape[0])

        return data, label

class GraphNSCLC3D(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.config = config

        match mode:
            case 'train':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/NSCLC_preprocessed/graph_data/graphs.pt")
                self.labels = np.array([graph.label for graph in self.data])
                self.data, _, self.labels, _ = train_test_split(self.data, self.labels, train_size=0.8, random_state=config.seed, stratify=self.labels)
                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
            case 'val':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/NSCLC_preprocessed/graph_data/graphs.pt")
                self.labels = np.array([graph.label for graph in self.data])
                _, val_test_data, _, val_test_labels = train_test_split(self.data, self.labels, train_size=0.8, random_state=config.seed, stratify=self.labels)
                self.data, _, self.labels, _ = train_test_split(val_test_data, val_test_labels, test_size=0.5, random_state=config.seed, stratify=val_test_labels)
                
            case 'test':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/NSCLC_preprocessed/graph_data/graphs.pt")
                self.labels = np.array([graph.label for graph in self.data])
                _, val_test_data, _, val_test_labels = train_test_split(self.data, self.labels, train_size=0.8, random_state=config.seed, stratify=self.labels)
                _, self.data, _, self.labels = train_test_split(val_test_data, val_test_labels, test_size=0.5, random_state=config.seed, stratify=val_test_labels)
            case _:
                raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")

        # match mode:
        #     case 'train':
        #         data = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_graphs.pt")
        #         labels = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_labels.pt")
        #         self.data, _, _, _ = train_test_split(data, labels, train_size=0.8, random_state=config.seed)
        #         self.labels = np.array([graph.label for graph in self.data])
        #         self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
        #         self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
        #     case 'val':
        #         data = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_graphs.pt")
        #         labels = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_labels.pt")
        #         _, val_test_data, _, val_test_labels = train_test_split(data, labels, train_size=0.8, random_state=config.seed)
        #         self.data, _, _, _ = train_test_split(val_test_data, val_test_labels, test_size=0.5, random_state=config.seed)
                
        #     case 'test':
        #         data = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_graphs.pt")
        #         labels = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_labels.pt")
        #         _, val_test_data, _, val_test_labels = train_test_split(data, labels, train_size=0.8, random_state=config.seed)
        #         _, self.data, _, _ = train_test_split(val_test_data, val_test_labels, test_size=0.5, random_state=config.seed)
        #     case _:
        #         raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        graph = self.data[index]
        graph.x = graph.x.to(torch.float32)
        graph.edge_index = graph.edge_index.to(torch.long)
        graph.edge_attr = graph.edge_attr.to(torch.float32)
        # graph.edge_attr = (torch.ones(56)*1000).to(torch.float32)
        # graph.edge_attr = None

        label = torch.tensor(graph.label).long()
        
        return graph, label

class Sarcoma(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.config = config

        match mode:
            case 'train':
                self.data = torch.load(f"/home/johannes/Code/DINOV2GNN/data/STS_preprocessed{config.num_slices}/graph_data/graphs.pt")
                self.data = [data for data in self.data if "TUM" in data.patient]
                self.labels = np.array([graph.label for graph in self.data])

                self.data, _, _, _ = train_test_split(self.data, self.labels, train_size=0.8, random_state=config.seed)

                self.labels = np.array([graph.label for graph in self.data])
                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
            case 'val':
                self.data = torch.load(f"/home/johannes/Code/DINOV2GNN/data/STS_preprocessed{config.num_slices}/graph_data/graphs.pt")
                self.data = [data for data in self.data if "TUM" in data.patient] 
                self.labels = np.array([graph.label for graph in self.data])
                _, self.data, _, _ = train_test_split(self.data, self.labels, train_size=0.8, random_state=config.seed)             
            case 'test':
                self.data = torch.load(f"/home/johannes/Code/DINOV2GNN/data/STS_preprocessed{config.num_slices}/graph_data/graphs.pt")
                self.data = [data for data in self.data if "UWS" in data.patient]
            
            case _:
                raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        match self.config.model_name:
            case "GNN":
                data = self.data[index]
                data.x = data.x.to(torch.float32)
                data.edge_index = data.edge_index.to(torch.long)
                data.edge_attr = data.edge_attr.to(torch.float32)                
                label = torch.Tensor([data.label]).to(torch.long)                

            case "MLP":
                data = self.data[index].x
                label = torch.Tensor([self.data[index].label.item()]).to(torch.long).view(-1).repeat(data.shape[0])
            case _:
                raise NotImplementedError(f"Given model architecture '{self.config.model_name} is not implemented!")        

        return data, label