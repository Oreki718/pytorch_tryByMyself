# Python 3.7.9
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

BATCH_SIZE = 32
DEVICE = torch.device("cpu")

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

def load_dataset():
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

    # Create data loaders
    # Dataset should be capusuled in dataloaders to be used
    test_dataloader = DataLoader(testset, batch_size = BATCH_SIZE)

    # for X, y in test_dataloader:
    #     print("Shape of X [N, C, H, W]: ", X.shape)
    #     print("Shape of y: ", y.shape, y.dtype)
    #     break

    # devide the training set to train and validation and create data loader
    len_validate = len(trainset) // 10  # 10% validation
    len_train = len(trainset) - len_validate
    lengths = [len_train, len_validate]
    ds_train, ds_val = random_split(trainset, lengths, torch.Generator().manual_seed(42)) # 莱万汀！
    train_dataloader = DataLoader(ds_train, batch_size=BATCH_SIZE)
    validate_dataloader = DataLoader(ds_val, batch_size=BATCH_SIZE)

    return train_dataloader, validate_dataloader, test_dataloader


# By inherit from Module class, redefine __init__ and forward method
# Defined our own model
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


# verbose: Displaying logs or not 
def train(net, trainLoader, epochs: int = 5, verbose=False):
    """Train the network on the training set."""
    # Loss Function: CrossEntropyLoss
    # output a scalar if the parameter [reduction] is not "None"
    criterion = nn.CrossEntropyLoss()

    # SGD, Adam are optimizers of different optimizing algorithms
    optimizer = torch.optim.Adam(net.parameters())

    # Let the model enter train mode
    net.train()
    for epoch in range(epochs):
        # batch by batch, each time BATCH_SIZE data will be loaded
        for images, labels in trainLoader:
            # images: BATCH_SIZE * 3 * 32 * 32
            # labels: BATCH_SIZE * 1
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # each 3 * 32 * 32 outputs a label int, the output should be BATCH_SIZE * 1
            outputs = net(images)

            # calculated the loss with the defined loss function
            # return a tensor of size 1, that is, only a float
            loss = criterion(outputs, labels)

            # backpropagate the loss
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"Epoch {epoch+1}: train loss , accuracy ")


    # Set the gradients of all optimized torch.Tensors to zero
    optimizer.zero_grad()



if __name__ == '__main__':
    trainLoader, valLoader, testLoader = load_dataset()
    net = MyNet()
    train(net, trainLoader, 5, True)
    
