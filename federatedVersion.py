from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

import flwr as fl
from flwr.common import Metrics

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

# CIFAR10 labels
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truch"
]

NUM_CLIENTS = 8 # number of clients that attend federated learning
BATCH_SIZE = 32 # how many group of data is loaded each batch

# Each trainloader/valloader pair contains 
# 4500 training examples and 500 validation examples.
def load_dataset():
    """Download and transform CIFAR-10 (train and test)"""
    # Define a transform for the data
    transform = transforms.Compose([
        # Convert the PIL images to tensor
        transforms.ToTensor(),

        # Normalize the data to increase accuracy
        # Hard to calculated the mean and standard derivation
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Get the dataset, if not exist locally, download them
    # CIFAR10 are RGB images of size 3*32*32, labelled int 0-9
    trainset = CIFAR10(
        root = "./dataset", 
        train = True, 
        download = True, 
        transform = transform
    )
    testset = CIFAR10(
        root = "./dataset", 
        train = False, 
        download = True, 
        transform = transform
    )

    # Split training set into NUM_CLIENTS partitions to simulate the individual dataset
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    train_datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # devide each partition of the training set to train and validation and create data loader
    trainloaders = []
    valloaders = []
    for ds in train_datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader # trainloaders and valloaders are list of dataloader


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # Layers

        # 3 * 6 kernels of size 5 * 5 
        # 3 input channels and 6 output channels 
        # for each input channel, 6 kernels
        # 18 kernels of size 5 * 5
        # inputs is 4-dimension tensor: BATCH_SIZE * 3 * width * width
        self.conv1 = nn.Conv2d(3, 6, 5)

        # Max Pool, kewrnel_size: 2 * 2, stride: (2,2)
        self.pool = nn.MaxPool2d(2, 2)

        # 6 * 16 kernels of size 5 * 5 
        # 6 input channels and 16 output channels 
        # for each input channel, 16 kernels
        # 96 kernels of size 5 * 5
        # inputs is 4-dimension tensor: BATCH_SIZE * 6 * width * width
        self.conv2 = nn.Conv2d(6, 16, 5)

        # fc: full connected
        # input and outputs are all 2-dimension tensor
        # in_features: 400, input is BATCH_SIZE * 400
        # out_features: 120, output is of shape BATCH_SZIE * 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # ReLU(x)=(x)^+ = max(0, x)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 64 * 3 * 32 * 32
        x = self.conv1(x)
        # 64 * 6 * 30 * 30
        x = self.relu(x)
        # 64 * 6 * 30 * 30
        x = self.pool(x)
        # 64 * 6 * 15 * 15
        x = self.conv2(x)
        # 64 * 16 * 11 * 11
        x = self.relu(x)
        # 64 * 16 * 11 * 11
        x = self.pool(x)
        # 64 * 16 * 5 * 5
        x = x.view(-1, 16 * 5 * 5) # resize, -1 implies this dimension is defined by other dimensions
        # 64 * 400
        x = self.fc1(x)
        # 64 * 120
        x = self.relu(x)
        # 64 * 120
        x = self.fc2(x)
        # 64 * 84
        x = self.relu(x)
        # 64 * 84
        x = self.fc3(x)
        # 64 * 10
        return x


def train(net, trainloader, epochs: int = 5, verbose=False):
    """Train the network on the training set."""
    # Loss Function: CrossEntropyLoss
    # output a scalar if the parameter [reduction] is not "None"
    criterion = nn.CrossEntropyLoss()

    # SGD, Adam are both optimizers of different optimizing algorithms
    optimizer = torch.optim.Adam(net.parameters())

    net.train() # Let the model enter train mode
    for epoch in range(epochs): # iterations
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            # images: BATCH_SIZE * 3 * 32 * 32
            # labels: BATCH_SIZE
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad() # Set the gradients of all optimized torch.Tensors to zero
            
            # each 3 * 32 * 32 outputs a label list, the outputs is BATCH_SIZE * 10
            outputs = net(images)

            # calculated the loss with the defined loss function
            # return a tensor of size 1, that is, only a float
            loss = criterion(outputs, labels)

            # backpropagate the loss
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss # Note the loiss is calculated from the loss function
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total # accuracy before the optimizer.step()
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# two helper functions 
# one for clients to update the local model of 
# with parameters received from the server
# another for the server to get the updated model parameters
# from client local models

# return a list of parameters of the net, in type list of numpy arrays
def get_parameters(net) -> List[np.ndarray]:
    paramList = []
    for param_name in net.state_dict():
        paramList.append(net.state_dict()[param_name].cpu().numpy())
    return paramList

# Given parameters in list of numpy arrays, set the parameters of the net
def set_parameters(net, parameters: List[np.ndarray]):
    # list of 2-tuples: (param_name: str, param_value: np.ndarray)
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict = True)
    

# client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
    
    '''Overriding the get_parameters function of flwr.client.NumPyClient'''
    def get_parameters(self, config): # return the local parameters
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)
    
    '''
    Overriding the fit function of flwr.client.NumPyClient
    Receive model parameters from the server, 
    train the model parameters on the local data, 
    and return the (updated) model parameters to the server
    '''
    def fit(self, parameters, config): # local train
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=local_epochs)
        return get_parameters(self.net), len(self.trainloader), {}
    
    '''
    Overriding the evaluate function of flwr.client.NumPyClient
    Receive model parameters from the server, 
    evaluate the model parameters on the local data, 
    and return the evaluation result to the server
    '''
    def evaluate(self, parameters, config): # local test
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# for the server to create client
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    # Load model
    net = MyNet().to(DEVICE)

    # Load data(CIFAR-10)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a single flower client
    return FlowerClient(cid, net, trainloader, valloader)
    

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Server-side parameter evaluation
# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = MyNet().to(DEVICE)
    valloader = valloaders[0] # we simulate with the sever has data of the first client
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, valloader)
    print(f"Round {server_round}: Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

# Server-side config transition
def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,  #
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="federated", type=str, help='mode of learning')
    parser.add_argument('--clientNum', default=NUM_CLIENTS, type=int, help='number of clients')
    args = parser.parse_args()

    print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )

    NUM_CLIENTS = args.clientNum

    if (args.mode == "federated"):
        trainloaders, valloaders, testloader = load_dataset()
        params = get_parameters(MyNet())

        # Create FedAvg strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1,  # Sample 100% of available clients for training
            fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=NUM_CLIENTS // 2,  # Never sample less than 10 clients for training
            min_evaluate_clients=NUM_CLIENTS // 2,  # Never sample less than 5 clients for evaluation
            min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=fl.common.ndarrays_to_parameters(params),
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config, # Pass the fit_config function
        )

        # Specify client resources if you need GPU
        client_resources = None
        if DEVICE.type == "cuda":
            client_resources = {"num_gpus": 1}

        # Start simulation
        print("Start Federal Simulation")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=5),
            strategy=strategy,
            client_resources=client_resources,
        )
    else:
        # simulate a centralized training of the first client
        trainloaders, valloaders, testloader = load_dataset()
        trainloader = trainloaders[0]
        valloader = valloaders[0]
        net = MyNet().to(DEVICE)
        train(net, trainloader, 5, True)
        loss, accuracy = test(net, testloader)
        print(f"Test loss: {loss}, accuracy {accuracy}")