import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

BARCH_SIZE = 32

def load_dataset():
    # Define a transform for the data
    transform = transforms.Compose([
        # Convert the PIL images to tensor
        transforms.ToTensor(),

        # Normalize the data to increase accuracy
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Get the dataset, if not exist locally, download them
    # CIFAR10 are RGB images of size 3*32*32, labelled 0-9
    trainset = CIFAR10(root = "./dataset", train=True, download=True, transform=transform)
