# pytorch_tryByMyself
Records my leaning process of Pytorch
This respository implements a process of learning CIFAR10 with a neuro network, in both centralized version (<code>originVersion.py</code>) and federated version(<code>federatedVersion.py</code>) with flower.

## Pytorch Basics

CIFAR10 are RGB images of size 3*32*32, labelled int 0-9

net.state_dict(): &lt; class 'collections.OrderedDict' &gt; is orderd dictionary of the parameters, the keys are of type string, the name of parameters, and the values are of type torch.Tensor, the value of parameters. For example:

    key              value
    conv1.weight     torch.Size([6, 3, 5, 5])
    conv1.bias       torch.Size([6])
    conv2.weight     torch.Size([16, 6, 5, 5])
    conv2.bias       torch.Size([16])
    fc1.weight       torch.Size([120, 400])
    fc1.bias         torch.Size([120])
    fc2.weight       torch.Size([84, 120])
    fc2.bias         torch.Size([84])
    fc3.weight       torch.Size([10, 84])
    fc3.bias         torch.Size([10])

net.load_state_dict(state_dict)

loss.backward():
loss is calculated from the train data and the parameters of the net. <code>train</code> automatically track the source parameter tensors and calculation and calculate their gradients with the loss

optimizer.step()
update the parameter tensors based on their gradients. The net should be in train mode to let the parameters be modified. The algorithms are decided by the optimizer. 

zip(): 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

    >>> a = [1,2,3]
    >>> b = [4,5,6]
    >>> zipped = zip(a,b)     # 打包为元组的列表
    [(1, 4), (2, 5), (3, 6)]

torch.Tensor.item(): 返回一个数，只能用于只包含一个元素的张量


## Federated Learning with Pytorch
<code>collections</code>: a library that spends the types of python, for example, the OrderedDict type allows dict type with order

<code>typing</code>: used for checking the parameter and return type of 
<code>matplotlib.pyplot</code>: Pyplot 是 Matplotlib 的子库，提供了和 MATLAB 类似的绘图 API。

In Flower, we create clients by implementing subclasses of <code>flwr.client.Client</code> or <code>flwr.client.NumPyClient</code>

To enable the Flower framework to create clients when necessary, we need to implement a function called client_fn that creates a FlowerClient instance on demand. Clients are identified by a client ID, or short cid. The cid can be used, for example, to load different local data partitions for different clients.

The function <code>start_simulation</code> accepts a number of arguments, amongst them the <code>client_fn</code> used to create FlowerClient instances, the number of clients to simulate (num_clients), the number of federated learning rounds (num_rounds), and the strategy. 

The **strategy** encapsulates the federated learning approach/algorithm, for example, Federated Averaging (FedAvg).

As users, we need to tell the framework how to handle/aggregate these custom metrics, and we do so by passing **metric aggregation** functions to the strategy. The two possible functions are fit_metrics_aggregation_fn and evaluate_metrics_aggregation_fn

Flower provides a way to send **configuration** values from the server to the clients using a dictionary. through the <code>config</code> parameter in fit



