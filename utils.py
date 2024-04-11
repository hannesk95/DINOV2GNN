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
