import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import os
import sys
import numpy as np
import yaml
import argparse
import torchio as tio
import shutil
import torch
from transformers import AutoImageProcessor
from transformers import AutoModel
from PIL import Image
from argparse import Namespace
from torch_geometric.utils import to_undirected
import torch_geometric
from skimage.metrics import structural_similarity


class Preprocessor:
    def __init__(self, config: Namespace) -> None:
        self.config = config
        self.dinov2_processor = AutoImageProcessor.from_pretrained(config.dinov2_model)
        self.dinov2_encoder = AutoModel.from_pretrained(config.dinov2_model).cuda()

        self.config.target_dir = self.config.target_dir + str(self.config.num_slices)

        if os.path.exists(self.config.target_dir):            
            sys.exit(f"\n[INFO] Target dir '{self.config.target_dir}' already exists!\n")
        else:
            os.makedirs(self.config.target_dir)
        

    def get_target_spacing(self) -> tuple:

        spacings = []
        for image_path in tqdm(glob(os.path.join(self.config.data_dir, "*image*.nii.gz")), desc=f"Get target spacing '{self.config.spacing_aggregation}' "):
            img = sitk.ReadImage(image_path, sitk.sitkFloat32)
            spacing = img.GetSpacing()
            spacings.append(spacing)
        
        match self.config.spacing_aggregation:
            case "mean":
                target_spacing = np.mean(spacings, axis=0)
            case "median":
                target_spacing = np.median(spacings, axis=0)
            case _:
                raise NotImplementedError(f"Target spacing aggregation '{self.config.spacing_aggregation}' not implemented!")
            
        return tuple(target_spacing)
    
    
    def resample_volumes(self, target_spacing: tuple) -> None:

        resample = tio.transforms.Resample(target=target_spacing)
        reorient = tio.transforms.ToCanonical()

        for image_path in tqdm(glob(os.path.join(self.config.data_dir, "*image*.nii.gz")), desc=f"Resample volumes "):
            label_path = image_path.replace("image", "label")

            image = tio.ScalarImage(image_path)
            label = tio.LabelMap(label_path)
            label = tio.transforms.Resample(image)(label)

            # image = reorient(image)
            # label = reorient(label)
            image = resample(image)
            label = resample(label)

            image_name = image_path.split("/")[-1]
            label_name = label_path.split("/")[-1]

            sitk.WriteImage(image.as_sitk(), os.path.join(self.config.target_dir, image_name))
            sitk.WriteImage(label.as_sitk(), os.path.join(self.config.target_dir, label_name))
    

    def get_target_subvolume(self) -> tuple:

        dimensions = []
        for image_path in tqdm(glob(os.path.join(self.config.target_dir, "*image*.nii.gz")), desc=f"Get target tumor subvolume '{self.config.subvolume_aggregation}' "):
            label_path = image_path.replace("image", "label")

            image_arr = np.squeeze(tio.ScalarImage(image_path).numpy())
            label_arr = np.squeeze(tio.LabelMap(label_path).numpy())
            label_arr = (label_arr / np.max(label_arr)).astype(int)

            indices = np.where(label_arr == 1)
            
            x_min = np.min(indices[0])            
            y_min = np.min(indices[1])
            z_min = np.min(indices[2])
            
            x_max = np.max(indices[0])
            y_max = np.max(indices[1])
            z_max = np.max(indices[2])

            x_dim = x_max - x_min
            y_dim = y_max - y_min
            z_dim = z_max - z_min

            dimensions.append((x_dim, y_dim, z_dim))
        
        match self.config.subvolume_aggregation:
            case "mean":
                target_subvolume = np.mean(dimensions, axis=0)
            case "median":
                target_subvolume = np.median(dimensions, axis=0)
            case "max":
                target_subvolume = np.max(dimensions, axis=0)
            case _:
                raise NotImplementedError(f"Target subvolume aggregation '{self.config.subvolume_aggregation}' not implemented!")

        return tuple(target_subvolume)
    

    def center_crop_volumes(self, target_subvolume: tuple) -> None:

        crop = tio.transforms.CropOrPad(target_shape=target_subvolume, mask_name='label')

        for image_path in tqdm(glob(os.path.join(self.config.target_dir, "*image*.nii.gz")), desc=f"Crop volumes "):
            label_path = image_path.replace("image", "label")

            subject = tio.Subject(
                image = tio.ScalarImage(image_path),
                label = tio.LabelMap(label_path)
            )

            subject = crop(subject)

            image_name = image_path.split("/")[-1]
            label_name = label_path.split("/")[-1]

            sitk.WriteImage(subject.image.as_sitk(), os.path.join(self.config.target_dir, image_name))
            sitk.WriteImage(subject.label.as_sitk(), os.path.join(self.config.target_dir, label_name))

    
    def clip_hounsfield_units(self, volume: torch.Tensor) -> torch.Tensor:

        volume = torch.clamp(volume, min=self.config.min_HU, max=self.config.max_HU)

        return volume
    
    
    def min_max_scale(self, volume: torch.Tensor) -> torch.Tensor:

        volume = (volume - torch.min(volume)) / (torch.max(volume)- torch.min(volume)) * 255

        return volume

    # def subselect_volume(self, dim: int, volume: torch.Tensor) -> np.ndarray:

    #     middle_slice_idx = int(volume.shape[-1]/2)
    #     lower_idx = middle_slice_idx - int(self.config.num_slices/2)
    #     upper_idx = middle_slice_idx + int(self.config.num_slices/2)

    #     return volume[:, :, lower_idx:upper_idx].numpy().astype(np.uint8)
    
    
    def subselect_volume(self, volume: torch.Tensor, dim: int) -> np.ndarray:

        middle_slice_idx = int(volume.shape[-1]/2)
        lower_idx = middle_slice_idx - int(self.config.num_slices/2)
        upper_idx = middle_slice_idx + int(self.config.num_slices/2)

        match dim:
            case 0:
                return volume[lower_idx:upper_idx, :, :].numpy().astype(np.uint8)
            case 1:
                return volume[:, lower_idx:upper_idx, :].numpy().astype(np.uint8)
            case 2:
                return volume[:, :, lower_idx:upper_idx].numpy().astype(np.uint8)        


    # def get_dinov2_encodings(self, volume: np.ndarray) -> torch.Tensor:

    #     slices = []
    #     for i in range(volume.shape[-1]):
    #         img_slice = Image.fromarray(volume[:, :, i])
    #         image = self.dinov2_processor(images=img_slice, return_tensors="pt")
    #         image = image["pixel_values"]
    #         slices.append(image)
        
    #     images = torch.concatenate(slices, dim=0).cuda()

    #     x = self.dinov2_encoder(images)
    #     x = self.dinov2_encoder.layernorm(x.pooler_output)
        
    #     return x.detach()
    
    def get_dinov2_encodings(self, volume: np.ndarray, dim: int) -> torch.Tensor:

        slices = []
        for i in range(volume.shape[dim]):
            match dim:
                case 0:
                    img_slice = Image.fromarray(volume[i, :, :])
                case 1:
                    img_slice = Image.fromarray(volume[:, i, :])
                case 2:
                    img_slice = Image.fromarray(volume[:, :, i])
            
            image = self.dinov2_processor(images=img_slice, return_tensors="pt")
            image = image["pixel_values"]
            slices.append(image)
        
        images = torch.concatenate(slices, dim=0).cuda()

        x = self.dinov2_encoder(images)
        x = self.dinov2_encoder.layernorm(x.pooler_output)
        
        return x.detach()
    
    def get_graph_structure(self, volume: np.ndarray, encodings: torch.Tensor, label: int, split: str | None) -> torch_geometric.data.Data:

        num_slices = volume.shape[-1]
        middle_slice_idx = int(volume.shape[-1]/2)
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
        edge_weight = torch.tensor([structural_similarity(volume[:, :, edge_index[0, i]], volume[:, :, edge_index[1, i]]) for i in range(edge_index.shape[-1])])

        edge_index, edge_weight = to_undirected(edge_index=edge_index, edge_attr=edge_weight)

        data = torch_geometric.data.Data(x=encodings.cpu(), edge_index=edge_index.cpu(), edge_attr=edge_weight.cpu(), label=label, patient=split)

        return data

def main_medmnist(config: Namespace):

    preprocessor = Preprocessor(config=config)

    train_data = np.load(os.path.join(config.data_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(config.data_dir, "train_labels.npy"))
    train_samples = train_data.shape[0]

    val_data = np.load(os.path.join(config.data_dir, "val_images.npy"))
    val_labels = np.load(os.path.join(config.data_dir, "val_labels.npy"))
    val_samples = val_data.shape[0]

    test_data = np.load(os.path.join(config.data_dir, "test_images.npy"))
    test_labels = np.load(os.path.join(config.data_dir, "test_labels.npy"))
    test_samples = test_data.shape[0]

    data = np.concatenate([train_data, val_data, test_data], axis=0)
    labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)

    graphs = []
    for idx, img_arr in tqdm(enumerate(data), desc="Final preprocessing "): 

        label = labels[idx]
        encodings_list = []
        for dim in range(3):
            # Subselect volume
            volume = preprocessor.subselect_volume(volume=torch.tensor(img_arr), dim=dim)

            # Get DINOV2 encodings
            encodings = preprocessor.get_dinov2_encodings(volume=volume, dim=dim)
            encodings_list.append(encodings)
        encodings = torch.concat(encodings_list, dim=1)

        # Create graph structure
        data = preprocessor.get_graph_structure(volume=volume, encodings=encodings, label=label, split=None)
        graphs.append(data)       

    train_graphs = graphs[:train_samples]
    val_graphs = graphs[train_samples:(train_samples+val_samples)]  
    test_graphs = graphs[(train_samples+val_samples):]

    # Save graphs into file
    if os.path.exists(os.path.join(config.target_dir, "graph_data")):
        shutil.rmtree(os.path.join(config.target_dir, "graph_data"))
        os.makedirs(os.path.join(config.target_dir, "graph_data"))
    else:
        os.makedirs(os.path.join(config.target_dir, "graph_data"))

    torch.save(train_graphs, os.path.join(config.target_dir, "graph_data", "train_graphs.pt"))
    torch.save(val_graphs, os.path.join(config.target_dir, "graph_data", "val_graphs.pt"))
    torch.save(test_graphs, os.path.join(config.target_dir, "graph_data", "test_graphs.pt"))


def main_custom(config: Namespace):

    preprocessor = Preprocessor(config=config)

    # # 1. Get target spacing
    target_spacing = preprocessor.get_target_spacing()

    # # 2. Resample data
    preprocessor.resample_volumes(target_spacing=target_spacing)

    # # 3. Get biggest tumor subvolume
    target_subvolume = preprocessor.get_target_subvolume()

    # # 4. Center crop volumes
    preprocessor.center_crop_volumes(target_subvolume=target_subvolume)

    graphs = []
    for img_path in tqdm(glob(os.path.join(config.target_dir, "*image*.nii.gz")), desc="Final preprocessing "):        
        volume = tio.ScalarImage(img_path).tensor.squeeze()
        label = int(config.label[img_path.split("/")[-1].split("_")[1]])
        
        if "RADCURE" in config.target_dir:
            split = img_path.split("/")[-1].split("_")[2]
        elif "STS" in config.target_dir:
            split = img_path.split("/")[-1][:4]

        # 5. Clip CT-Hounsfield units
        volume = preprocessor.clip_hounsfield_units(volume=volume)

        # 6. Min-Max scale data
        volume = preprocessor.min_max_scale(volume=volume)

        # 7. Subselect volume
        volume = preprocessor.subselect_volume(volume=volume)
        # img_name = img_path.split("/")[-1].replace(".nii.gz", ".npy")
        # np.save(f"/home/johannes/Code/DINOV2GNN/data/temp/{img_name}", volume)        

        # # 8. Get DINOV2 encodings
        encodings = preprocessor.get_dinov2_encodings(volume=volume)

        # # 9. Create graph structure
        data = preprocessor.get_graph_structure(volume=volume, encodings=encodings, label=label, split=split)
        graphs.append(data)

    # 10. Save graphs into file
    if os.path.exists(os.path.join(config.target_dir, "graph_data")):
        shutil.rmtree(os.path.join(config.target_dir, "graph_data"))
        os.makedirs(os.path.join(config.target_dir, "graph_data"))
    else:
        os.makedirs(os.path.join(config.target_dir, "graph_data"))

    torch.save(graphs, os.path.join(config.target_dir, "graph_data", "graphs.pt"))


if __name__ == "__main__":


    for dataset in ['nodule', 'synapse', 'adrenal', 'vessel', 'organ', 'fracture']:
        for num_slices in [8, 16, 24, 32, 48, 64]:


            parser = argparse.ArgumentParser()
            parser.add_argument('-dataset', "--dataset", type=str, help="Name of the dataset", 
                                choices=['nodule', 'synapse', 'adrenal', 'vessel', 'organ', 'fracture', 'nsclc', 'radcure', 'sts'], 
                                default='synapse')
            args = parser.parse_args()

            args.dataset = dataset

            match args.dataset:
                # MedMNIST Datasets
                case "nodule":
                    config_file = "/home/johannes/Code/DINOV2GNN/data/MedMNIST/nodule_config.yaml"
                case "synapse":
                    config_file = "/home/johannes/Code/DINOV2GNN/data/MedMNIST/synapse_config.yaml"
                case "adrenal":
                    config_file = "/home/johannes/Code/DINOV2GNN/data/MedMNIST/adrenal_config.yaml"
                case "vessel":
                    config_file = "/home/johannes/Code/DINOV2GNN/data/MedMNIST/vessel_config.yaml"
                case "organ":
                    config_file = "/home/johannes/Code/DINOV2GNN/data/MedMNIST/organ_config.yaml"
                case "fracture":
                    config_file = "/home/johannes/Code/DINOV2GNN/data/MedMNIST/fracture_config.yaml"
                
                # Custom Datatsets
                case "nsclc":
                    config_file = "/home/johannes/Code/DINOV2GNN/data/Custom/nsclc_config.yaml"
                case "radcure":
                    config_file = "/home/johannes/Code/DINOV2GNN/data/Custom/radcure_config.yaml"
                case "sts":
                    config_file = "/home/johannes/Code/DINOV2GNN/data/Custom/sts_config.yaml"
                case _:
                    raise NotImplementedError(f"Given dataset '{args.dataset} is not implemented!")

            with open(config_file, 'r') as file:
                config = Namespace(**yaml.safe_load(file))
            
            config.num_slices = num_slices
            config.target_dir = f"/home/johannes/Code/DINOV2GNN/data/MedMNIST/{dataset}mnist3d_64_preprocessed_ACS"

            match args.dataset:
                case "nodule" | "synapse" | "adrenal" | "vessel" | "organ" | "fracture":
                    main_medmnist(config=config)        
                case _:
                    main_custom(config=config)
