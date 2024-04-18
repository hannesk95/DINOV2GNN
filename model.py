import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import AutoModel, ViTModel
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, global_sort_pool
from torch_geometric.nn.pool import SAGPooling
import torch
from torch.autograd import Variable


# class Classifier(nn.Module):

#     def __init__(self, config):
#         super(Classifier, self).__init__()

#         self.config = config

#         if config.mlp_conditional == True:
#             input_dim = 2*384 #385
#         else:
#             input_dim = 384

#         self.classifier = nn.Sequential(nn.Linear(input_dim, 288), nn.ReLU(), 
#                                         nn.Linear(288, 192), nn.ReLU(),
#                                         nn.Linear(192, config.n_classes))

#     def forward(self, x):
#         return self.classifier(x)


# class DINOv2(nn.Module):

#      def __init__(self, config):
#         super(DINOv2, self).__init__()        

#         match config.model_name:
#             case 'facebook/dinov2-small':
#                 self.encoder = AutoModel.from_pretrained(config.model_name)
#                 self.classifier = nn.Sequential(nn.Linear(384, 128), nn.ELU(), nn.Linear(128, 2))
#             case 'facebook/dinov2-base':
#                 self.encoder = AutoModel.from_pretrained(config.model_name)
#                 self.classifier = nn.Sequential(nn.Linear(768, 128), nn.ELU(), nn.Linear(128, 2))                
#             case 'facebook/dinov2-large':
#                 self.encoder = AutoModel.from_pretrained(config.model_name)
#                 self.classifier = nn.Sequential(nn.Linear(768, 128), nn.ELU(), nn.Linear(128, 2))
#             case 'facebook/dinov2-giant':
#                 self.encoder = AutoModel.from_pretrained(config.model_name)
#                 self.classifier = nn.Sequential(nn.Linear(1536, 128), nn.ELU(), nn.Linear(128, 2))
#             case _: 
#                 raise NotImplementedError(f"Model name '{config.model_name}' not implemented!")

#         for param in self.encoder.parameters():
#             param.requires_grad = False
     
#      def forward(self, x):
                
#         x = self.encoder(x)
#         # encoder_embedding = x.pooler_output

#         x = self.encoder.layernorm(x.pooler_output)
#         out = self.classifier(x)
         
#         #  embedding = self.embedding(encoder_embedding)
#         #  embedding = self.bn(embedding)
#         # outputs = self.cls(encoder_embedding)

#         return out

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        self.config = config

        if config.mlp_conditional == True:
            input_dim = 1153
        else:
            input_dim = 1152

        self.linear1 = nn.Linear(input_dim, 224)
        self.linear2 = nn.Linear(224, 192)
        self.linear3 = nn.Linear(192, config.n_classes)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x  

class LSTM(torch.nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        torch.manual_seed(config.seed)

        self.config = config
        self.hidden_size = 54

        self.lstm = nn.LSTM(input_size=3*384, hidden_size=self.hidden_size, num_layers=3, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.config.n_classes)

        self.relu = nn.ReLU()

    def forward(self, x):

        h_0 = Variable(torch.zeros(3, x.size(0), self.hidden_size)).to(self.config.device) #hidden state
        c_0 = Variable(torch.zeros(3, x.size(0), self.hidden_size)).to(self.config.device) #internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        # out = self.relu(out)
        out = self.fc(out[:, -1, :])
        # hn = hn.view(-1, 192) #reshaping the data for Dense layer next
        # out = self.relu(hn)
        # out = self.fc(out) #first Dense

        return out
     

class GNN(torch.nn.Module):
    def __init__(self, config):
        super(GNN, self).__init__()
        torch.manual_seed(config.seed)

        self.config = config

        match config.gnn_type:
            case "GCNConv":
                self.conv1 = GCNConv(3*384, self.config.hidden_channels)
                self.conv2 = GCNConv(self.config.hidden_channels, self.config.hidden_channels)
                self.conv3 = GCNConv(self.config.hidden_channels, self.config.hidden_channels)
            
            case "ChebConv":
                self.conv1 = ChebConv(3*384, self.config.hidden_channels, K=1)
                self.conv2 = ChebConv(self.config.hidden_channels, self.config.hidden_channels, K=1)
                self.conv3 = ChebConv(self.config.hidden_channels, self.config.hidden_channels, K=1)                

            case "GraphConv":
                self.conv1 = GraphConv(3*384, self.config.hidden_channels)
                self.conv2 = GraphConv(self.config.hidden_channels, self.config.hidden_channels)
                self.conv3 = GraphConv(self.config.hidden_channels, self.config.hidden_channels)

            case "GATConv":
                self.conv1 = GATConv(3*384, self.config.hidden_channels)
                self.conv2 = GATConv(self.config.hidden_channels, self.config.hidden_channels)
                self.conv3 = GATConv(self.config.hidden_channels, self.config.hidden_channels)

            case "SAGEConv":
                self.conv1 = SAGEConv(3*384, self.config.hidden_channels)
                self.conv2 = SAGEConv(self.config.hidden_channels, self.config.hidden_channels)
                self.conv3 = SAGEConv(self.config.hidden_channels, self.config.hidden_channels)

            case _:
                raise NotImplementedError("GNN type is not implemented!")            
        
        self.relu = nn.ReLU()
        self.pool = SAGPooling(self.config.hidden_channels)
        self.lin = Linear(self.config.hidden_channels, self.config.n_classes)

    def forward(self, data):
        # 1. Obtain node embeddings
        match self.config.gnn_type:
            case "SAGEConv":
                x = self.conv1(x=data.x, edge_index=data.edge_index, )
                x = self.relu(x)
                x = self.conv2(x=x, edge_index=data.edge_index)
                x = self.relu(x)
                x = self.conv3(x=x, edge_index=data.edge_index)
                
            case "GATConv":
                x = self.conv1(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
                x = self.relu(x)
                x = self.conv2(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)
                x = self.relu(x)
                x = self.conv3(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            
            case _:
                x = self.conv1(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)
                x = self.relu(x)               
                x = self.conv2(x=x, edge_index=data.edge_index, edge_weight=data.edge_attr)
                x = self.relu(x)
                x = self.conv3(x=x, edge_index=data.edge_index, edge_weight=data.edge_attr)

        # 2. Readout layer
        match self.config.gnn_readout:
            case "center":
                middle_slice = int(self.config.num_slices/2)
                x = x[middle_slice::self.config.num_slices]
            case "mean":
                x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]
            case "max":
                x = global_max_pool(x, data.batch)
            case "sum":
                x = global_add_pool(x, data.batch)
            case "sort":
                x = global_sort_pool(x, data.batch, k=8)
            case "attention":
                x = self.pool(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)[0]
            case _:
                raise NotImplementedError("Readout function provided is not implemented!")

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
