import torchio as tio
import numpy as np
import torch
from model import MLP, GNN
from param_configurator import ParamConfigurator
from torch_geometric.transforms import RandomJitter
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.seed import seed_everything
from transformers import AutoImageProcessor
from transformers import AutoModel
from PIL import Image
from tqdm import tqdm
import torch_geometric
from torch_geometric.data import Data
from torchmetrics.functional import pairwise_cosine_similarity
from torch_geometric.utils import to_undirected
from scipy.stats import pearsonr, spearmanr, entropy
from utils import set_seed
import pandas as pd

gnn_checkpoints = {"nodule":
       [           
        "/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/a96035def30b4821b1ecd71ad7bce5dd/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/58f6d6b3447448fbabe9e74ea7afadd0/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/25511860051041aa84a8294530cf3042/artifacts/best_model.pt"
        ],

        "adrenal":
        [
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/bc25bc4b96d946c29ea7b5ff5571ffbc/artifacts/best_model.pt',
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/c92b722025334b11b5f839d2e497441b/artifacts/best_model.pt',
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/f3e9842ca648416ca7c94662c4ecf94a/artifacts/best_model.pt'
        ],

        "synapse":
        [    
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/0e0a93dbdbc7416e9b737473b8d4dc2c/artifacts/best_model.pt',
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/5614b6a94d464a50bb7458b2a3fb3774/artifacts/best_model.pt',
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/9a8f07092e4146cab801f89b44c76e9d/artifacts/best_model.pt'
        ],

        "fracture":
        [    
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/f2c6e5cb50df4cc5a8c28a44da46adf4/artifacts/best_model.pt',
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/78eb9bbee92243658a8f954764e14e48/artifacts/best_model.pt',
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/86f81c75268b492ba92d856254487e04/artifacts/best_model.pt'
        ],

        "vessel":
        [
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/bf73600714fb490c9872c394190b987a/artifacts/best_model.pt',
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/616f532cc5f340018ccfb5bb231e7d0a/artifacts/best_model.pt',
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/a132210c731140b3b86bed6fea8be0c5/artifacts/best_model.pt'
        ],

        "organ":
        [
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/1c7317519a704e61b64f1cefd9225e5b/artifacts/best_model.pt',
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/b5900624af3b4a6b88d69c67333b2d93/artifacts/best_model.pt',
        '/home/johannes/Code/DINOV2GNN/mlruns/675964103216560380/f0d2bdff04b14b21a2293c502f41f251/artifacts/best_model.pt'
        ]
}

gnn_config = {"nodule": ["SAGEConv", 3, "custom"],
              "adrenal": ["GATConv", 7, "chebyshev"],
              "synapse": ["SAGEConv", 3, "custom"],
              "fracture": ["GATConv", 5, "chebyshev"],
              "vessel": ["GATConv", 7, "manhattan"],
              "organ": ["SAGEConv", 7, "manhattan"]}

mlp_checkpoints = {"nodule":
       [
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/a81191c8a3f74d198d2be7c3f73afd30/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/81d684fa6fde47f786259a09147b831b/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/dfe770e19d844592975ee94f31af1b8c/artifacts/best_model.pt"
        ], 
        
        "adrenal":
        [
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/11583639bb8045e88155f39b2d5f9836/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/9247b149623d41c7994ec2c4af9a7e8f/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/9edc382789dd490c82a1d5dab9dd1e44/artifacts/best_model.pt"
        ],

        "synapse":
        [
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/3c5cf8f88d434e6da07d0c295979cf2e/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/6674402310ff44109b97d917a07c91b1/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/be59842fe89444e1beff5fda1b37a053/artifacts/best_model.pt"
        ],

        "fracture":
        [
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/c5a1db53502042ae80ed5195e935942e/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/23ae663f426a499eb616a517ca9ed7c8/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/7bf07db60dbf43faa219b7612b8b3b59/artifacts/best_model.pt"
        ],

        "vessel":
        [
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/81cd70138e5343b9886768597cbaa262/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/ed0965efe9bf4a37991f1b8183677a20/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/8eba648efb2340e5aabe35845bfaaa6f/artifacts/best_model.pt"
        ],

        "organ":
        [
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/56f88bb296184b5b94439f1920a9347b/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/81a34ea6269042d2bfaf7f4b9bb2a5fe/artifacts/best_model.pt",
        "/home/johannes/Code/DINOV2GNN/mlruns/832304414204132017/4a0c7fe294e344bc8fc44750f11adc7e/artifacts/best_model.pt"
        ]
}

def get_augmented_data(dataset, augmentation, std) -> list:

    # load data
    X = np.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{dataset}mnist3d_64/test_images.npy")
    y = np.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{dataset}mnist3d_64/test_labels.npy")
    
    match augmentation:
        case "noise":   
            augmentator = tio.transforms.RandomNoise(std=(0, std))                       
        
        case "blur":            
            augmentator = tio.transforms.RandomBlur()    

        case "bias_field":
            augmentator = tio.transforms.RandomBiasField(coefficients=std)        
        
    X = [augmentator(np.expand_dims(sample, axis=0)) for sample in X]
    X = np.concatenate(X, axis=0)  

    # define preprocessing and model
    dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dinov2_encoder = AutoModel.from_pretrained("facebook/dinov2-small").cuda()

    # encode data
    data = []
    for sample in tqdm(X):
        encodings = []            
        for dim in range(3):
            slices = []
            for i in range(64):
                match dim:
                    case 0:
                        img_slice = Image.fromarray(sample[i, :, :])
                    case 1:
                        img_slice = Image.fromarray(sample[:, i, :])
                    case 2:
                        img_slice = Image.fromarray(sample[:, :, i])
                
                image = dinov2_processor(images=img_slice, return_tensors="pt")
                image = image["pixel_values"]
                slices.append(image)
            
            images = torch.concatenate(slices, dim=0).cuda()
            x = dinov2_encoder(images)
            encodings.append(dinov2_encoder.layernorm(x.pooler_output).detach().cpu())
        
        data.append(torch.concatenate(encodings, dim=1).unsqueeze(dim=0))
    
    data = torch.concatenate(data, dim=0)
    return [Data(x=features) for features in data]

def get_topology(data, topology, k) -> torch_geometric.data.Data:       
        
        features = data.x
        num_nodes = features.shape[0]

        match topology:

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
                a = torch.arange(0, num_nodes-1).tolist()
                b = torch.arange(1, num_nodes).tolist()
                edge_index = torch.tensor([a, b], dtype=torch.long)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
                data.edge_index = edge_index
                data.edge_attr = edge_attr               

                # return data

            case "random":
                torch.manual_seed(0)
                num_edges = int(0.25*(num_nodes*num_nodes))
                a = torch.randint(low=0, high=num_nodes, size=(num_edges,)).tolist()
                b = torch.randint(low=0, high=num_nodes, size=(num_edges,)).tolist()
                edge_index = torch.tensor([a, b], dtype=torch.long)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)             
                data.edge_index = edge_index
                data.edge_attr = edge_attr  

                # return data
            
            case "fully":
                edge_index = []
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        edge_index.append([i, j])

                edge_index = torch.transpose(torch.tensor(edge_index), 0, 1)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr) 
                data.edge_index = edge_index
                data.edge_attr = edge_attr

                # return data

            case "star":
                edge_index = []
                center_node = int(num_nodes/2)
                for i in range(num_nodes):
                    if i != center_node:
                        edge_index.append([center_node, i])
                
                edge_index = torch.transpose(torch.tensor(edge_index), 0, 1)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr) 
                data.edge_index = edge_index
                data.edge_attr = edge_attr

                # return data
            
            case "custom": # line + star
                num_slices = num_nodes
                middle_slice_idx = int(num_slices/2)
                a_center = list(np.repeat(middle_slice_idx, num_slices))
                b_center = list(np.arange(start=0, stop=num_slices, step=1))

                a_top = list(np.arange(start=0, stop=middle_slice_idx-1))
                b_top = list(np.arange(start=1, stop=middle_slice_idx))

                a_bottom = list(np.arange(start=middle_slice_idx+1, stop=num_slices-1))
                b_bottom = list(np.arange(start=middle_slice_idx+2, stop=num_slices))

                a = a_center+a_top+a_bottom
                b = b_center+b_top+b_bottom

                a.pop(middle_slice_idx)
                b.pop(middle_slice_idx)

                edge_index = torch.concatenate([torch.tensor(a).view(1, -1), torch.tensor(b).view(1, -1)], dim=0)
                edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
                edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr) 
                data.edge_index = edge_index
                data.edge_attr = edge_attr
                
            case _:
                raise NotImplementedError(f"Given metric '{topology}' is not implemented.")

        # edge_index = torch.concatenate([a, b], dim=0)
        # edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.float32)
        # edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)
        # data.edge_index = edge_index
        # data.edge_attr = edge_attr

        return data


if __name__ == "__main__":       

    augmentation = "noise"

    for dataset in ["nodule", "adrenal", "synapse", "vessel", "fracture", "organ"]:

        std_list = []
        run_list = []
        acc_mlp_list = []
        acc_gnn_list = []
        auc_mlp_list = []
        auc_gnn_list = []

        for std in [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0]:  
        # for std in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]:  

            gnn_type = gnn_config[dataset][0]    
            k = gnn_config[dataset][1]
            topology = gnn_config[dataset][2]              

            match augmentation:
                case "jitter":
                    data = torch.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{dataset}mnist3d_64_preprocessed_ACS64/graph_data/test_graphs.pt")
                    y = np.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{dataset}mnist3d_64/test_labels.npy").reshape(-1)
                    augmentator = RandomJitter(translate=1.0)
                    samples = []
                    for sample in data:
                        sample.pos = sample.x
                        samples.append(sample)
                    data = [augmentator(samples[i]) for i in range(len(samples))]
                
                case "noise" | "blur" | "bias_field":
                    data = get_augmented_data(dataset=dataset, augmentation=augmentation, std=std)
                    data = [get_topology(data=sample, topology=topology, k=k) for sample in data]
                    y = np.load(f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{dataset}mnist3d_64/test_labels.npy").reshape(-1)

                case _:
                    raise NotImplementedError("Given augmentation is not implemented!")  

            for run in [0, 1, 2]:

                #############################################                        
                mlp_checkpoint = mlp_checkpoints[dataset][run]
                gnn_checkpoint = gnn_checkpoints[dataset][run]            
                #############################################

                config = ParamConfigurator()   
                config.gnn_type = gnn_type
                config.topology = topology
                config.k = k
                seed_everything(config.seed)
                set_seed(config.seed)

                match dataset:
                    case "fracture":
                        config.n_classes = 3 
                    case "organ":
                        config.n_classes = 11
                    case _:
                        config.n_classes = 2
                
                model_mlp = MLP(config=config).cuda()
                model_mlp.load_state_dict(torch.load(mlp_checkpoint))
                model_mlp.eval()
        
                model_gnn = GNN(config=config).cuda()
                model_gnn.load_state_dict(torch.load(gnn_checkpoint))
                model_gnn.eval()

                mlp_preds = []
                gnn_preds = []
                mlp_scores = []
                gnn_scores = []
                for sample in data:               
                    
                    input_mlp = torch.concat([sample.x.cuda(), torch.arange(0, config.num_slices).view(-1, 1).repeat(1, 1).cuda()], dim=1)                       
                    out_mlp = model_mlp(input_mlp.unsqueeze(dim=0).cuda())                    
                    out_gnn = model_gnn(sample.cuda())

                    mlp_preds.append(torch.max(out_mlp, dim=1).indices.item())
                    gnn_preds.append(torch.max(out_gnn, dim=1).indices.item())

                    if config.n_classes > 2:
                        score_mlp = torch.softmax(out_mlp, dim=1)
                        score_gnn = torch.softmax(out_gnn, dim=1)
                    else:
                        score_mlp = torch.softmax(out_mlp, dim=1)[:, -1]
                        score_gnn = torch.softmax(out_gnn, dim=1)[:, -1]

                    mlp_scores.extend(score_mlp.detach().cpu().numpy())
                    gnn_scores.extend(score_gnn.detach().cpu().numpy())

                    # mlp_scores.append(out_mlp[:, -1].detach().cpu().numpy().item())
                    # gnn_scores.append(out_gnn[:, -1].detach().cpu().numpy().item())
                    
                acc_mlp = accuracy_score(y_true=y, y_pred=mlp_preds)
                acc_gnn = accuracy_score(y_true=y, y_pred=gnn_preds)
                auc_mlp = roc_auc_score(y_true=y, y_score=mlp_scores, multi_class='ovo')
                auc_gnn = roc_auc_score(y_true=y, y_score=gnn_scores, multi_class='ovo')

                print(f"MLP ACC: {acc_mlp}")
                print(f"MLP AUC: {auc_mlp}\n")

                print(f"GNN ACC: {acc_gnn}")
                print(f"GNN AUC: {auc_gnn}\n")

                std_list.append(std)
                run_list.append(run)
                acc_mlp_list.append(acc_mlp)
                acc_gnn_list.append(acc_gnn)
                auc_mlp_list.append(auc_mlp)
                auc_gnn_list.append(auc_gnn)

        df = pd.DataFrame()
        df["std"] = std_list
        df["run"] = run_list
        df["acc_mlp"] = acc_mlp_list
        df["acc_gnn"] = acc_gnn_list
        df["auc_mlp"] = auc_mlp_list
        df["auc_gnn"] = auc_gnn_list

        df.to_csv(f"{dataset}_{augmentation}_robustness_results.csv")

        print(f"{dataset} finished!")
