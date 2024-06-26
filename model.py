import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import AutoModel, ViTModel
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, GATConv, SAGEConv, GINConv, DynamicEdgeConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, global_sort_pool
# from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn.models import GIN
import torch
from torch.autograd import Variable
from torch_geometric.nn.norm import BatchNorm, GraphNorm
from torch_geometric.nn.aggr import AttentionalAggregation, SoftmaxAggregation, PowerMeanAggregation


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

        # self.linear1 = nn.Linear(input_dim, 144)
        # self.linear2 = nn.Linear(144, config.hidden_channels)
        self.linear1 = nn.Linear(input_dim, config.hidden_channels)
        self.linear3 = nn.Linear(config.hidden_channels, config.n_classes)
        # self.linear1 = nn.Linear(input_dim, 224)
        # self.linear2 = nn.Linear(224, 192)
        # self.linear3 = nn.Linear(192, config.n_classes)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        # x = self.linear2(x)
        # x = self.relu(x)

        x = torch.mean(x, dim=1)

        # x = F.dropout(x, p=0.5)#, training=self.training)
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

        # h_0 = Variable(torch.zeros(3, x.size(0), self.hidden_size)).to(self.config.device) #hidden state
        # c_0 = Variable(torch.zeros(3, x.size(0), self.hidden_size)).to(self.config.device) #internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x) #lstm with input, hidden, and internal state
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
                self.relu = nn.ReLU()
                self.lin = Linear(self.config.hidden_channels, self.config.n_classes)

            case "GATConv":
                self.conv1 = GATConv(3*384, self.config.hidden_channels)
                self.relu = nn.ReLU()
                self.lin = Linear(self.config.hidden_channels, self.config.n_classes)

            case "SAGEConv":                
                self.conv1 = SAGEConv(3*384, self.config.hidden_channels//2)
                self.relu = nn.ReLU()
                self.lin = Linear(self.config.hidden_channels//2, self.config.n_classes)
            
            # case "ChebConv":
            #     self.conv1 = ChebConv(3*384, self.config.hidden_channels, K=1, aggr=SoftmaxAggregation(learn=True))
            #     # self.bn1 = GraphNorm(self.config.hidden_channels)
            #     # self.conv2 = ChebConv(self.config.hidden_channels, self.config.hidden_channels, K=1, aggr=SoftmaxAggregation(learn=True))
            #     # self.bn2 = GraphNorm(self.config.hidden_channels)
            #     # self.conv3 = ChebConv(self.config.hidden_channels, self.config.hidden_channels, K=1, aggr=SoftmaxAggregation(learn=True))
            #     # self.bn3 = GraphNorm(self.config.hidden_channels)                

            # case "GraphConv":
            #     self.config.hidden_channels = int(self.config.hidden_channels / 2)
            #     self.conv1 = GraphConv(3*384, self.config.hidden_channels)
            #     self.conv2 = GraphConv(self.config.hidden_channels, self.config.hidden_channels)
            #     self.conv3 = GraphConv(self.config.hidden_channels, self.config.hidden_channels)            
            
            # case "GINConv":
            #     self.conv = GIN(3*384, hidden_channels=self.config.hidden_channels, num_layers=3, norm='BatchNorm')

            # case "DynamicConv":
            #     self.conv1 = DynamicEdgeConv(nn=torch.nn.Sequential(nn.Linear(2*3*384, self.config.hidden_channels), nn.ReLU()), k=6)
            #     self.conv2 = DynamicEdgeConv(nn=torch.nn.Sequential(nn.Linear(2*self.config.hidden_channels, self.config.hidden_channels), nn.ReLU()), k=6)
            #     self.conv3 = DynamicEdgeConv(nn=torch.nn.Sequential(nn.Linear(2*self.config.hidden_channels, self.config.hidden_channels), nn.ReLU()), k=6)

            case _:
                raise NotImplementedError("GNN type is not implemented!")            
        
        

    def forward(self, data):
        # 1. Obtain node embeddings
        match self.config.gnn_type:
            case "SAGEConv" | "GCNConv" | "GATConv":
                x = self.conv1(x=data.x, edge_index=data.edge_index)
                x = self.relu(x)
                # x = self.conv2(x=x, edge_index=data.edge_index)
                # x = self.relu(x)
                # x = self.conv3(x=x, edge_index=data.edge_index)
                
            # case "GATConv":
            #     x = self.conv1(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            #     x = self.relu(x)
            #     x = self.conv2(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            #     x = self.relu(x)
            #     x = self.conv3(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)

            # case "GINConv":
            #     x = self.conv(x=data.x, edge_index=data.edge_index)
            
            # case "GCNConv":
            #     x = self.conv1(x=data.x, edge_index=data.edge_index)
            #     x = self.relu(x)

            # case "DynamicConv":
            #     x = self.conv1(x=data.x, batch=data.batch)
            #     x = self.conv2(x=x, batch=data.batch)
            #     x = self.conv3(x=x, batch=data.batch)
            
            # case "ChebConv":
            #     x = self.conv1(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)
            #     x = self.bn1(x)
            #     x = self.relu(x)               
            #     x = self.conv2(x=x, edge_index=data.edge_index, edge_weight=data.edge_attr)
            #     x = self.bn2(x)
            #     x = self.relu(x)
            #     x = self.conv3(x=x, edge_index=data.edge_index, edge_weight=data.edge_attr)
            #     x = self.bn3(x)

            # case _:
            #     x = self.conv1(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)
            #     x = self.relu(x)               
            #     # x = self.conv2(x=x, edge_index=data.edge_index, edge_weight=data.edge_attr)
            #     # x = self.relu(x)
            #     # x = self.conv3(x=x, edge_index=data.edge_index, edge_weight=data.edge_attr)
            #     out = x



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
            # case "attention":
            #     x = self.pool(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)[0]
            case _:
                raise NotImplementedError("Readout function provided is not implemented!")

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.3)#, training=self.training)
        x = self.lin(x)
        
        return x
