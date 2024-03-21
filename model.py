import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import AutoModel, ViTModel
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, GATConv
from torch_geometric.nn import global_mean_pool
import torch


class DINOv2(nn.Module):

     def __init__(self, config):
        super(DINOv2, self).__init__()        

        match config.model_name:
            case 'facebook/dinov2-small':
                self.encoder = AutoModel.from_pretrained(config.model_name)
                self.classifier = nn.Sequential(nn.Linear(384, 128), nn.ELU(), nn.Linear(128, 2))
            case 'facebook/dinov2-base':
                self.encoder = AutoModel.from_pretrained(config.model_name)
                self.classifier = nn.Sequential(nn.Linear(768, 128), nn.ELU(), nn.Linear(128, 2))                
            case 'facebook/dinov2-large':
                self.encoder = AutoModel.from_pretrained(config.model_name)
                self.classifier = nn.Sequential(nn.Linear(768, 128), nn.ELU(), nn.Linear(128, 2))
            case 'facebook/dinov2-giant':
                self.encoder = AutoModel.from_pretrained(config.model_name)
                self.classifier = nn.Sequential(nn.Linear(1536, 128), nn.ELU(), nn.Linear(128, 2))
            case _: 
                raise NotImplementedError(f"Model name '{config.model_name}' not implemented!")

        for param in self.encoder.parameters():
            param.requires_grad = False
     
     def forward(self, x):
                
        x = self.encoder(x)
        # encoder_embedding = x.pooler_output

        x = self.encoder.layernorm(x.pooler_output)
        out = self.classifier(x)
         
        #  embedding = self.embedding(encoder_embedding)
        #  embedding = self.bn(embedding)
        # outputs = self.cls(encoder_embedding)

        return out
     

class GNN(torch.nn.Module):
    def __init__(self, config):
        super(GNN, self).__init__()
        torch.manual_seed(config.seed)

        self.config = config

        match config.gnn_type:
            case "GCNConv":
                self.conv1 = GCNConv(384, self.config.hidden_channels)
                self.conv2 = GCNConv(self.config.hidden_channels, self.config.hidden_channels)
                self.conv3 = GCNConv(self.config.hidden_channels, self.config.hidden_channels)
            
            case "ChebConv":
                self.conv1 = ChebConv(384, self.config.hidden_channels, K=1)
                self.conv2 = ChebConv(self.config.hidden_channels, self.config.hidden_channels, K=1)
                self.conv3 = ChebConv(self.config.hidden_channels, self.config.hidden_channels, K=1)

            case "GraphConv":
                self.conv1 = GraphConv(384, self.config.hidden_channels)
                self.conv2 = GraphConv(self.config.hidden_channels, self.config.hidden_channels)
                self.conv3 = GraphConv(self.config.hidden_channels, self.config.hidden_channels)

            case "GATConv":
                self.conv1 = GATConv(384, self.config.hidden_channels)
                self.conv2 = GATConv(self.config.hidden_channels, self.config.hidden_channels)
                self.conv3 = GATConv(self.config.hidden_channels, self.config.hidden_channels)


            case _:
                raise NotImplementedError("GNN type is not implemented!")

        self.lin = Linear(self.config.hidden_channels, 2)

    def forward(self, data):
        # 1. Obtain node embeddings
        match self.config.gnn_type:
            case "GATConv":
                x = self.conv1(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
                x = x.relu()
                x = self.conv2(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)
                x = x.relu()
                x = self.conv3(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            case _:
                x = self.conv1(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)
                x = x.relu()
                x = self.conv2(x=x, edge_index=data.edge_index, edge_weight=data.edge_attr)
                x = x.relu()
                x = self.conv3(x=x, edge_index=data.edge_index, edge_weight=data.edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
