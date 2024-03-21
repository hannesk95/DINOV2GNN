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

class GrapRadcure3D(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.config = config

        match mode:
            case 'train':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/radcure_train_graphs.pt")
                self.labels = np.array([graph.label for graph in self.data])
                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
            case 'val':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/radcure_val_graphs.pt")                
            case 'test':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/radcure_test_graphs.pt")                
            case _:
                raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")
            
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


class GrapNSCLC3D(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.config = config

        match mode:
            case 'train':
                data = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_graphs.pt")
                labels = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_labels.pt")
                self.data, _, _, _ = train_test_split(data, labels, train_size=0.8, random_state=config.seed)
                self.labels = np.array([graph.label for graph in self.data])
                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
            case 'val':
                data = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_graphs.pt")
                labels = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_labels.pt")
                _, val_test_data, _, val_test_labels = train_test_split(data, labels, train_size=0.8, random_state=config.seed)
                self.data, _, _, _ = train_test_split(val_test_data, val_test_labels, test_size=0.5, random_state=config.seed)
                
            case 'test':
                data = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_graphs.pt")
                labels = torch.load("/home/johannes/Code/DINOV2GNN/data/nsclc_labels.pt")
                _, val_test_data, _, val_test_labels = train_test_split(data, labels, train_size=0.8, random_state=config.seed)
                _, self.data, _, _ = train_test_split(val_test_data, val_test_labels, test_size=0.5, random_state=config.seed)
            case _:
                raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")
            
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


class GraphNoduleMNIST3D(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.config = config

        match mode:
            case 'train':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/nodule_train_graphs.pt")
                self.labels = np.array([graph.label.item() for graph in self.data])
                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
            case 'val':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/nodule_val_graphs.pt")
            case 'test':
                self.data = torch.load("/home/johannes/Code/DINOV2GNN/data/nodule_test_graphs.pt")
            case _:
                raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")
            
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
    

class NoduleMNIST3D(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)

        match mode:
            case 'train':
                self.data = np.load("/home/johannes/Code/DINOV2GNN/data/nodulemnist3d_64/train_images.npy")
                self.labels = np.load("/home/johannes/Code/DINOV2GNN/data/nodulemnist3d_64/train_labels.npy")
                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
            case 'val':
                self.data = np.load("/home/johannes/Code/DINOV2GNN/data/nodulemnist3d_64/val_images.npy")
                self.labels = np.load("/home/johannes/Code/DINOV2GNN/data/nodulemnist3d_64/val_labels.npy")
            case 'test':
                self.data = np.load("/home/johannes/Code/DINOV2GNN/data/nodulemnist3d_64/test_images.npy")
                self.labels = np.load("/home/johannes/Code/DINOV2GNN/data/nodulemnist3d_64/test_labels.npy")
            case _:
                raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        volume = self.data[index]

        slices = []
        for i in range(24, 40):
            img_slice = Image.fromarray(volume[:, :, i])
            image = self.processor(images=img_slice, return_tensors="pt")
            image = image["pixel_values"]
            slices.append(image)
        
        images = torch.concatenate(slices, dim=0)
        
        return images, np.repeat(self.labels[index], 16, axis=0)

        
    