import random
import numpy as np
import torch
import os
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import mlflow
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch.nn as nn


def set_seed(seed: int) -> None:
    """TODO: Docstring"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def save_conda_env(config) -> None:
    """TODO: Docstring"""

    conda_env = os.environ['CONDA_DEFAULT_ENV']
    command = f"conda env export -n {conda_env} > {config.run_dir}/environment.yml"
    subprocess.call(command, shell=True)
    mlflow.log_artifact(f"{config.run_dir}/environment.yml")


def create_confusion_matrix(y_true, y_pred):
    """TODO: Docstring"""

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    image_path = "./temp.png"
    plt.savefig(image_path)

    image = Image.open(image_path)
    transform = transforms.ToTensor()
    image_tensor = transform(image)

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    os.remove(image_path)

    return image_tensor, sensitivity, specificity

class FocalLoss(nn.Module):

    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'none'):
        """
         Initialize the module. This is the entry point for the module. You can override this if you want to do something other than setting the weights and / or gamma.
         
         Args:
         	 weight: The weight to apply to the layer. If None the layer weights are set to 1.
         	 gamma: The gamma parameter for the layer. Defaults to 2.
         	 reduction: The reduction method to apply. Possible values are'mean'or'std '
        """
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
         Computes NLL loss for each element of input_tensor. This is equivalent to : math : ` L_ { t } ` where L is the log - softmax of the input tensor
         
         Args:
         	 input_tensor: Tensor of shape ( batch_size num_input_features )
         	 target_tensor: Tensor of shape ( batch_size num_target_features )
         
         Returns: 
         	 A tensor of shape ( batch_size num_output_features ) - > loss ( float )
        """
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)

        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction = self.reduction
        )
    

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
     Computes F1 loss for classification. It is used to compute the F1 loss for each class and its predicted values
     
     Args:
     	 y_true: ( torch. Tensor ) Ground truth labels
     	 y_pred: ( torch. Tensor ) Predicted labels
     
     Returns: 
     	 ( torch. Tensor ) Corresponding F1 loss ( tp tn fn fn p r r )
    """
    tp = torch.sum((y_true * y_pred).float(), dim=0)
    tn = torch.sum(((1 - y_true) * (1 - y_pred)).float(), dim=0)
    fp = torch.sum(((1 - y_true) * y_pred).float(), dim=0)
    fn = torch.sum((y_true * (1 - y_pred)).float(), dim=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)
