import os
import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


class MedMNIST3D(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.config = config

        match mode:
            case 'train':
                self.data = torch.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{self.config.dataset}mnist3d_64_preprocessed_ACS{self.config.num_slices}/graph_data/train_graphs.pt")
                self.labels = np.array([graph.label.item() for graph in self.data])
                self.data, _, self.labels, _ = train_test_split(self.data, self.labels, train_size=self.config.fraction, random_state=self.config.seed, stratify=self.labels)

                # num_samples = int(len(self.data) * self.config.fraction)
                # self.data = self.data[:num_samples]

                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to(self.config.device)
            case 'val':
                self.data = torch.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{self.config.dataset}mnist3d_64_preprocessed_ACS{self.config.num_slices}/graph_data/val_graphs.pt")
            case 'test':
                self.data = torch.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{self.config.dataset}mnist3d_64_preprocessed_ACS{self.config.num_slices}/graph_data/test_graphs.pt")
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
                label = torch.Tensor(data.label).to(torch.long)
            case "MLP":
                data = self.data[index].x
                if self.config.mlp_conditional:
                    data = torch.concat([data, torch.arange(0, self.config.num_slices).view(-1, 1).repeat(1, 1)], dim=1)
                label = torch.Tensor([self.data[index].label.item()]).to(torch.long).view(-1).repeat(data.shape[0])
            case "LSTM":
                data = self.data[index].x
                label = torch.Tensor([self.data[index].label.item()]).to(torch.long)
            case _:
                raise NotImplementedError(f"Given model architecture '{self.config.model_name} is not implemented!")        

        return data, label





    