# Federated Variational Autoencoder with PyTorch and Flower

O objetivo desse documento é descrever os códigos que realizam um treinamento federado de um Variational Autoencoder (VAE), modelo generativo que conta com um encoder e um decoder. O encoder é responsável por representar um certo dado de entrada em um espaço latente, através de uma média e um desvio padrão de uma distribuição como a gaussiana. Já, o decoder tem a função de reconstruir amostras geradas pela distribuição representada pelo encoder, visando obter o formato original do dado de entrada. 

![Estrutura VAE](https://github.com/gustavoguaragna/FedGen/blob/main/pytorch-federated-variational-autoencoder/images/VAE.png "VAE")

## Estrutura do projeto
- Arquivo de configuração: pyproject.toml
- Arquivo de funções auxiliares: task.py
- Arquivo do cliente: client_app.py
- Arquivo do servidor: server_app.py


## Arquivo de configuração 
O arquivo de configuração tem como objetivo facilitar a reprodutibilidade do projeto, armazenando metadados do projeto como nome, versão, descrição licença, dependências, autores e configurações padrões, mas que podem ser alteradas, conforme desejado.
```
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fedvaeexample"
version = "1.0.0"
description = "Federated Variational Autoencoder Example with PyTorch and Flower"
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.12.0",
    "flwr-datasets[vision]>=0.3.0",
    "torch==2.2.1",
    "torchvision==0.17.1",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "flwrlabs"

[tool.flwr.app.components]
serverapp = "fedvaeexample.server_app:app"
clientapp = "fedvaeexample.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 3
local-epochs = 1
learning-rate = 0.001

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 2
```

## Funções auxiliares

##### Importações
```python
"""fedvae: A Flower app for Federated Variational Autoencoder."""

from collections import OrderedDict
import torch
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
```
##### Classes para o modelo. 
Flatten será usada no encoder para vetorizar a entrada em uma dimensão, enquando UnFlatten será usada no decoder para reformatar conforme o tamanho desejado. O encoder é composto por duas redes convolucionais seguidas de ativações ReLU e pela camada Flatten ao fim. Já o decoder é composto pela camada UnFlatten e por duas redes convolucionais transpostas seguidas das ativações ReLU e tangente hiberbólica. A cada passagem de foward propagation a imagem de entrada passa pelo encoder, que retorna a média, o desvio padrão de uma distribuição no espaço latente, assim como uma amostra dessa distribuição. Essa amostra passa pelo decoder que retorna a imagem sintética no formato original da imagem de entrada.

```python
class Flatten(nn.Module):
    """Flattens input by reshaping it into a one-dimensional tensor."""

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """Unflattens a tensor converting it to a desired shape."""

    def forward(self, input):
        return input.view(-1, 16, 6, 6)


class Net(nn.Module):
    def __init__(self, h_dim=576, z_dim=10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=4, stride=2
            ),  # [batch, 6, 15, 15]
            nn.ReLU(),
            nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, stride=2
            ),  # [batch, 16, 6, 6]
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2),
            nn.Tanh(),
        )

    def reparametrize(self, h):
        """Reparametrization layer of VAE."""
        mu, logvar = self.fc1(h), self.fc2(h)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def encode(self, x):
        """Encoder of the VAE."""
        h = self.encoder(x)
        z, mu, logvar = self.reparametrize(h)
        return z, mu, logvar

    def decode(self, z):
        """Decoder of the VAE."""
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z_decode = self.decode(z)
        return z_decode, mu, logvar

```
A função _load_data_ é responsável por carregar os dados do CIFAR10, dividí-los entre os clientes e pré-processá-los.

```python
fds = None  # Cache FederatedDataset

def load_data(partition_id, num_partitions):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader
```
###### Função de treino do modelo
A função de perda do VAE é uma composição de duas perdas:
- Perda de reconstrução: MSE(imagem entrada, imagem reconstruída).
- Divergência de Kullback-Leibler (KLD): Mede quanto a distribuição de probabilidade predita pelo modelo diverge da distribuição de porbabilidade esperada.

```python
def train(net, trainloader, epochs, learning_rate, device):
    """Train the network on the training set."""
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    for _ in range(epochs):
        # for images, _ in trainloader:
        for batch in trainloader:
            images = batch["img"]
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.05 * kld_loss
            loss.backward()
            optimizer.step()
```
##### Função de teste do modelo 
```python
def test(net, testloader, device):
    """Validate the network on the entire test set."""
    total, loss = 0, 0.0
    with torch.no_grad():
        # for data in testloader:
        for batch in testloader:
            images = batch["img"].to(device)
            # images = data[0].to(DEVICE)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
            total += len(images)
    return loss / total
```
A função _generate_ pode ser utilizada para gerar uma imagem sintética a partir de uma imagem real e da VAE treinada. A função _get_weights_ prepara os pesos do modelo para serem transmitidos entre clientes e servidor em forma de lista de arrays numpy. A função _set_weights_ atualiza os parâmetros do modelo com os novos pesos recebidos. 

```python
def generate(net, image):
    """Reproduce the input with trained VAE."""
    with torch.no_grad():
        return net.forward(image)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

```

## Arquivo do cliente

```python
"""fedvaeexample: A Flower / PyTorch app for Federated Variational Autoencoder."""

import torch
from fedvaeexample.task import Net, get_weights, load_data, set_weights, test, train

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


class CifarClient(NumPyClient):
    def __init__(self, trainloader, testloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            epochs=self.local_epochs,
            learning_rate=self.lr,
            device=self.device,
        )
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss = test(self.net, self.testloader, self.device)
        return float(loss), len(self.testloader), {}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read the run_config to fetch hyperparameters relevant to this run
    trainloader, testloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    return CifarClient(trainloader, testloader, local_epochs, learning_rate).to_client()


app = ClientApp(client_fn=client_fn)

```

## Servidor




[GitHub Original](https://github.com/adap/flower/tree/main/examples/pytorch-federated-variational-autoencoder)
