import os
import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import torch_geometric.data
from torch_geometric.utils import to_undirected

import torch
import torch_geometric
from scipy.stats import pearsonr, spearmanr, entropy
from scipy.spatial.distance import cosine
from torch_geometric.data import Data
from torchmetrics.functional import pairwise_cosine_similarity


class MedMNIST3D(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.config = config

        match mode:
            case 'train':
                self.data = torch.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{self.config.dataset}mnist3d_64_preprocessed_ACS{self.config.num_slices}/graph_data/train_graphs.pt")
                self.labels = np.array([graph.label.item() for graph in self.data])
                
                # if self.config.fraction != 1.0:
                #     self.data, _, self.labels, _ = train_test_split(self.data, self.labels, train_size=self.config.fraction, random_state=28, stratify=self.labels)

                # num_samples = int(len(self.data) * self.config.fraction)
                # self.data = self.data[:num_samples]

                # if self.config.topology != "custom":
                self.data = [self.get_topology(self.data[i]) for i in range(len(self.data))]
                    # self.data = [self.get_topology(self.data[i]) for i in range(len(self.data))]

                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels.flatten())
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to(self.config.device)
            case 'val':
                self.data = torch.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{self.config.dataset}mnist3d_64_preprocessed_ACS{self.config.num_slices}/graph_data/val_graphs.pt")
                # if self.config.topology != "custom":
                self.data = [self.get_topology(self.data[i]) for i in range(len(self.data))]
                    # self.data = [self.get_topology(self.data[i]) for i in range(len(self.data))]
            case 'test':
                self.data = torch.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{self.config.dataset}mnist3d_64_preprocessed_ACS{self.config.num_slices}/graph_data/test_graphs.pt")
                # if self.config.topology != "custom":
                self.data = [self.get_topology(self.data[i]) for i in range(len(self.data))]
                    # self.data = [self.get_topology(self.data[i]) for i in range(len(self.data))]
            case _:
                raise NotImplementedError(f"Given mode '{self.mode}' is not implemented!")
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        match self.config.model_name:
            case "GNN":
                data = self.data[index]
                data.x = data.x.to(torch.float32)

                # if self.config.topology != "custom":
                #     data = self.get_topology(data)
                # else:
                data.edge_index = data.edge_index.to(torch.long)
                data.edge_attr = data.edge_attr.to(torch.float32)

                label = torch.Tensor(data.label).to(torch.long)
            case "MLP":
                data = self.data[index].x
                if self.config.mlp_conditional:
                    data = torch.concat([data, torch.arange(0, self.config.num_slices).view(-1, 1).repeat(1, 1)], dim=1)
                # label = torch.Tensor([self.data[index].label.item()]).to(torch.long).view(-1).repeat(data.shape[0])
                label = torch.Tensor([self.data[index].label.item()]).to(torch.long)
            case "LSTM":
                data = self.data[index].x
                label = torch.Tensor([self.data[index].label.item()]).to(torch.long)
            case _:
                raise NotImplementedError(f"Given model architecture '{self.config.model_name} is not implemented!")        

        return data, label
    
    # def get_topology(self, data) -> torch.tensor:

    #     match self.config.topology:
    #         case "line":
    #             a = torch.arange(0, self.config.num_slices-1).tolist()
    #             b = torch.arange(1, self.config.num_slices).tolist()
    #             edge_index = torch.tensor([a, b], dtype=torch.long)
    #             edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
    #             edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
    #             data.edge_index = edge_index
    #             data.edge_attr = edge_attr               

    #             return data

    #         case "random":
    #             torch.manual_seed(0)
    #             num_edges = int(0.25*(self.config.num_slices*self.config.num_slices))
    #             a = torch.randint(low=0, high=self.config.num_slices, size=(num_edges,)).tolist()
    #             b = torch.randint(low=0, high=self.config.num_slices, size=(num_edges,)).tolist()
    #             edge_index = torch.tensor([a, b], dtype=torch.long)
    #             edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
    #             edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)             
    #             data.edge_index = edge_index
    #             data.edge_attr = edge_attr  

    #             return data
            
    #         case "fully":
    #             edge_index = []
    #             for i in range(self.config.num_slices):
    #                 for j in range(i + 1, self.config.num_slices):
    #                     edge_index.append([i, j])

    #             edge_index = torch.transpose(torch.tensor(edge_index), 0, 1)
    #             edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
    #             edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr) 
    #             data.edge_index = edge_index
    #             data.edge_attr = edge_attr

    #             return data

    #         case "star":
    #             edge_index = []
    #             center_node = int(self.config.num_slices/2)
    #             for i in range(self.config.num_slices):
    #                 if i != center_node:
    #                     edge_index.append([center_node, i])
                
    #             edge_index = torch.transpose(torch.tensor(edge_index), 0, 1)
    #             edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
    #             edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr) 
    #             data.edge_index = edge_index
    #             data.edge_attr = edge_attr

    #             return data
            
    #         case "prism":
    #             a = torch.arange(0*self.config.num_slices, 1*self.config.num_slices-1).tolist()
    #             b = torch.arange(0*self.config.num_slices+1, 1*self.config.num_slices).tolist()

    #             c = torch.arange(1*self.config.num_slices, 2*self.config.num_slices-1).tolist()
    #             d = torch.arange(1*self.config.num_slices+1, 2*self.config.num_slices).tolist()

    #             e = torch.arange(2*self.config.num_slices, 3*self.config.num_slices-1).tolist()
    #             f = torch.arange(2*self.config.num_slices+1, 3*self.config.num_slices).tolist()

    #             edge_index = torch.tensor([a+c+e, b+d+f], dtype=torch.long)

    #             a = torch.arange(self.config.num_slices)
    #             b = a + self.config.num_slices
    #             c = b + self.config.num_slices

    #             a = a.tolist()
    #             b = b.tolist()
    #             c = c.tolist()

    #             edge_index = torch.concat([edge_index, torch.tensor([a, b])], dim=1)
    #             edge_index = torch.concat([edge_index, torch.tensor([a, c])], dim=1)
    #             edge_index = torch.concat([edge_index, torch.tensor([b, c])], dim=1)
    #             edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
    #             edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr) 


    #             if data.x.shape[-1] == 1152:
    #                 x = [data.x[:, i*384:(i+1)*384] for i in range(3)]
    #                 x = torch.concat(x, dim=0)
    #                 data.x = x
                    
    #             data.edge_index = edge_index
    #             data.edge_attr = edge_attr

    #             return data     
    
    def get_topology(self, data) -> torch_geometric.data.Data:

        
        k = self.config.k 
        features = data.x
        num_nodes = self.config.num_slices

        match self.config.topology:

            #####################################################################################
            # volume derived topologies #########################################################
            #####################################################################################

            case "manhattan":
                scores = torch.cdist(features, features, p=1.0)
                scores.fill_diagonal_(torch.nan)
                scores[~torch.isnan(scores)].reshape(scores.shape[0], scores.shape[1] - 1)

                a = torch.repeat_interleave(torch.arange(0, num_nodes), k).view(1, -1)
                b = torch.topk(scores, k=k, dim=1, largest=False).indices.flatten().view(1, -1)
                edge_index = torch.concatenate([a, b], dim=0)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
                data.edge_index = edge_index
                data.edge_attr = edge_attr
            
            case "euclidean":
                # features = features.flatten()
                # features = features.view(-1, 384)
                scores = torch.cdist(features, features, p=2.0)
                scores.fill_diagonal_(torch.nan)
                scores[~torch.isnan(scores)].reshape(scores.shape[0], scores.shape[1] - 1)

                a = torch.repeat_interleave(torch.arange(0, features.shape[0]), k).view(1, -1)
                b = torch.topk(scores, k=k, dim=1, largest=False).indices.flatten().view(1, -1)
                edge_index = torch.concatenate([a, b], dim=0)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                # edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
                data.edge_index = edge_index
                data.edge_attr = edge_attr
            
            case "chebyshev":
                scores = torch.cdist(features, features, p=torch.inf)
                scores.fill_diagonal_(torch.nan)
                scores[~torch.isnan(scores)].reshape(scores.shape[0], scores.shape[1] - 1)

                a = torch.repeat_interleave(torch.arange(0, num_nodes), k).view(1, -1)
                b = torch.topk(scores, k=k, dim=1, largest=False).indices.flatten().view(1, -1)
                edge_index = torch.concatenate([a, b], dim=0)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
                data.edge_index = edge_index
                data.edge_attr = edge_attr
            
            case "cosine":  
                # scores = torch.zeros((num_nodes, num_nodes))
                # features = features.numpy()

                # for i in range(num_nodes):
                #     for j in range(num_nodes):
                #         scores[i, j] = float(cosine(features[i], features[j]))
                scores = pairwise_cosine_similarity(features, zero_diagonal=False)
                scores.fill_diagonal_(torch.nan)
                scores[~torch.isnan(scores)].reshape(scores.shape[0], scores.shape[1] - 1)

                a = torch.repeat_interleave(torch.arange(0, num_nodes), k).view(1, -1)
                b = torch.topk(scores, k=k, dim=1, largest=True).indices.flatten().view(1, -1)
                edge_index = torch.concatenate([a, b], dim=0)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
                data.edge_index = edge_index
                data.edge_attr = edge_attr
            
            case "pearson":
                # scores = torch.zeros((num_nodes, num_nodes))
                # features = features.numpy()

                # for i in range(num_nodes):
                #     for j in range(num_nodes):
                #         scores[i, j] = float(pearsonr(features[i], features[j])[0])

                scores = torch.corrcoef(features)
                scores.fill_diagonal_(torch.nan)
                scores[~torch.isnan(scores)].reshape(scores.shape[0], scores.shape[1] - 1)

                a = torch.repeat_interleave(torch.arange(0, num_nodes), k).view(1, -1)
                b = torch.topk(scores, k=k, dim=1, largest=True).indices.flatten().view(1, -1)
                edge_index = torch.concatenate([a, b], dim=0)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
                data.edge_index = edge_index
                data.edge_attr = edge_attr                
            
            case "spearman":
                scores = torch.zeros((num_nodes, num_nodes))
                features = features.numpy()

                for i in range(num_nodes):
                    for j in range(num_nodes):
                        scores[i, j] = float(spearmanr(features[i], features[j])[0])
                        
                a = torch.repeat_interleave(torch.arange(0, num_nodes), k).view(1, -1)
                b = torch.topk(scores, k=k, dim=1, largest=True).indices.flatten().view(1, -1)
                edge_index = torch.concatenate([a, b], dim=0)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
                data.edge_index = edge_index
                data.edge_attr = edge_attr
            
            case "entropy":
                scores = torch.zeros((num_nodes, num_nodes))
                features = features.numpy()

                for i in range(num_nodes):
                    for j in range(num_nodes):
                        scores[i, j] = float(entropy(features[i], features[j]))
                        
                a = torch.repeat_interleave(torch.arange(0, num_nodes), k).view(1, -1)
                b = torch.topk(scores, k=k, dim=1, largest=False).indices.flatten().view(1, -1)
                edge_index = torch.concatenate([a, b], dim=0)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
                data.edge_index = edge_index
                data.edge_attr = edge_attr
            
            #####################################################################################
            # volume derived topologies #########################################################
            #####################################################################################

            case "line":
                a = torch.arange(0, self.config.num_slices-1).tolist()
                b = torch.arange(1, self.config.num_slices).tolist()
                edge_index = torch.tensor([a, b], dtype=torch.long)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
                data.edge_index = edge_index
                data.edge_attr = edge_attr               

                # return data

            case "random":
                torch.manual_seed(0)
                num_edges = int(0.25*(self.config.num_slices*self.config.num_slices))
                a = torch.randint(low=0, high=self.config.num_slices, size=(num_edges,)).tolist()
                b = torch.randint(low=0, high=self.config.num_slices, size=(num_edges,)).tolist()
                edge_index = torch.tensor([a, b], dtype=torch.long)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)             
                data.edge_index = edge_index
                data.edge_attr = edge_attr  

                # return data
            
            case "fully":
                edge_index = []
                for i in range(self.config.num_slices):
                    for j in range(i + 1, self.config.num_slices):
                        edge_index.append([i, j])

                edge_index = torch.transpose(torch.tensor(edge_index), 0, 1)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr) 
                data.edge_index = edge_index
                data.edge_attr = edge_attr

                # return data

            case "star":
                edge_index = []
                center_node = int(self.config.num_slices/2)
                for i in range(self.config.num_slices):
                    if i != center_node:
                        edge_index.append([center_node, i])
                
                edge_index = torch.transpose(torch.tensor(edge_index), 0, 1)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr) 
                data.edge_index = edge_index
                data.edge_attr = edge_attr

                # return data
            
            case "custom": # line + star
                edge_index = data.edge_index
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                data.edge_index = edge_index
                data.edge_attr = edge_attr
                
            case _:
                raise NotImplementedError(f"Given metric '{self.config.topology}' is not implemented.")

        # edge_index = torch.concatenate([a, b], dim=0)
        # edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
        # edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
        # data.edge_index = edge_index
        # data.edge_attr = edge_attr

        return data
        