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
dataset = "mnist"  # Alterar para "cifar10" conforme necessário

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

    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, input):
        return input.view(*self.target_shape)


class Net(nn.Module):
    def __init__(self, dataset="mnist", z_dim=10) -> None:
        super().__init__()
        if dataset == "mnist":
            in_channels = 1
            out_channels = 1
            self.encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=1, out_channels=6, kernel_size=4, stride=2
                ),  # [batch,6,13,13]
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=6, out_channels=16, kernel_size=5, stride=2
                ),  # [batch,16,5,5]
                nn.ReLU(),
                Flatten(),
            )
            h_dim = 16 * 5 * 5  # 400
            self.fc1 = nn.Linear(h_dim, z_dim)
            self.fc2 = nn.Linear(h_dim, z_dim)
            self.fc3 = nn.Linear(z_dim, h_dim)
            self.decoder = nn.Sequential(
                UnFlatten((-1, 16, 5, 5)),  # [batch,16,5,5]
                nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=2),  # [batch,6,15,15]
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=4, stride=2),  # [batch,1,28,28]
                nn.Tanh(),
            )
        elif dataset == "cifar10":
            in_channels = 3
            out_channels = 3
            self.encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=3, out_channels=6, kernel_size=4, stride=2
                ),  # [batch,6,15,15]
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=6, out_channels=16, kernel_size=5, stride=2
                ),  # [batch,16,6,6]
                nn.ReLU(),
                Flatten(),
            )
            h_dim = 16 * 6 * 6  # 576
            self.fc1 = nn.Linear(h_dim, z_dim)
            self.fc2 = nn.Linear(h_dim, z_dim)
            self.fc3 = nn.Linear(z_dim, h_dim)
            self.decoder = nn.Sequential(
                UnFlatten((-1, 16, 6, 6)),  # [batch,16,6,6]
                nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=2),  # [batch,6,15,15]
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2),  # [batch,3,32,32]
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Dataset {dataset} not supported")

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
A função _load_data_ é responsável por carregar os dados do CIFAR10 ou MNIST, dividí-los entre os clientes e pré-processá-los.
```python
fds = None  # Cache FederatedDataset

def load_data(partition_id, num_partitions, dataset="mnist"):
    """Load partition dataset (MNIST or CIFAR10)."""
    # Only initialize FederatedDataset once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        if dataset == "mnist":
            fds = FederatedDataset(
                dataset="mnist",
                partitioners={"train": partitioner},
            )
        elif dataset == "cifar10":
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        else:
            raise ValueError(f"Dataset {dataset} not supported")
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    if dataset == "mnist":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5,), (0.5,))]  # MNIST has 1 channel
        )
    elif dataset == "cifar10":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # CIFAR-10 has 3 channels
        )

    def apply_transforms(batch, dataset="mnist"):
        if dataset == "mnist":
          imagem = "image"
        elif dataset == "cifar10":
          imagem = "img"
        """Apply transforms to the partition from FederatedDataset."""
        batch[imagem] = [pytorch_transforms(img) for img in batch[imagem]]
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
def train(net, trainloader, epochs, learning_rate, device, dataset="mnist"):
    """Train the network on the training set."""
    if dataset == "mnist":
      imagem = "image"
    elif dataset == "cifar10":
      imagem = "img"
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[imagem]
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
def test(net, testloader, device, dataset="mnist"):
    """Validate the network on the entire test set."""
    if dataset == "mnist":
      imagem = "image"
    elif dataset == "cifar10":
      imagem = "img"
    total, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[imagem].to(device)
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
##### Importações

```python
"""fedvaeexample: A Flower / PyTorch app for Federated Variational Autoencoder."""
import torch
from fedvaeexample.task import Net, get_weights, load_data, set_weights, test, train
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
```
##### Classe do cliente
O cliente herda a classe _NumpyClient_ do _Flower_ e define seus atributos modelo, _trainloader_, _testloader_, número de épocas locais de treino, _learning rate_ e _device_. Define o método _fit_ que é responsável pelo treinamento do modelo, usando as funções definidas no arquivo _task.py_ _set_weights_, _train_ e _get_weights_, retornando os parâmentros atualizados localmente e o número de amostras para cálculo de agregação do _FedAvg_. O método _evaluate_ utiliza também a função _test_ do arquivo _task.py_ para calcular a função de perda de teste no modelo treinado em um respectivo cliente.
```python
class FedVaeClient(NumPyClient):
    def __init__(self, trainloader, testloader, local_epochs, learning_rate, dataset):
        self.net = Net(dataset=dataset)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            epochs=self.local_epochs,
            learning_rate=self.lr,
            device=self.device,
            dataset=self.dataset
        )
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss = test(self.net, self.testloader, self.device, dataset=self.dataset)
        return float(loss), len(self.testloader), {}
```
A função _client_fn_ é responsável por construir as instâncias de clientes que irão rodar em uma aplicação de cliente. As informações como número de épocas locais e _learning-rate_ são obtidas do arquivo _pyproject.toml_ através do _run_config_.

```python
def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read the run_config to fetch hyperparameters relevant to this run
    dataset = context.run_config["dataset"]  # Novo parâmetro
    trainloader, testloader = load_data(partition_id, num_partitions, dataset=dataset)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    return FedVaeClient(trainloader, testloader, local_epochs, learning_rate, dataset).to_client()


app = ClientApp(client_fn=client_fn)

```

## Servidor
##### Importações
```python
"""fedvaeexample: A Flower / PyTorch app for Federated Variational Autoencoder."""

from fedvaeexample.task import Net, get_weights
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import os  # Importar para verificar a existência de arquivos
```
Classe que define uma estratégia tal como o _FedAvg_, mas que salva os pesos do modelo e o valor da função de perda a cada rodada.
```python
class FedAvg_Save(FedAvg):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def aggregate_fit(self, server_round, results, failures):
        # Agrega os resultados da rodada
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Salva o modelo após a agregação
            self.save_model(aggregated_parameters, server_round)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # Agrega os resultados da avaliação
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Salva a perda após a avaliação
        self.save_loss(aggregated_loss, server_round)

        return aggregated_loss, aggregated_metrics

    def save_model(self, parameters, server_round):
        # Converte os parâmetros para ndarrays
        ndarrays = parameters_to_ndarrays(parameters)
        # Cria uma instância do modelo
        model = Net(dataset=self.dataset)
        # Define os pesos do modelo
        set_weights(model, ndarrays)
        # Salva o modelo no disco com o nome específico do dataset
        model_path = f"model_round_{server_round}_{self.dataset}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em {model_path}")

    def save_loss(self, loss, server_round):
        # Salva a perda em um arquivo de texto específico do dataset
        loss_file = f"losses_{self.dataset}.txt"
        with open(loss_file, "a") as f:
            f.write(f"Rodada {server_round}, Perda: {loss}\n")
        print(f"Perda da rodada {server_round} salva em {loss_file}")
```

Função para definir configurações para a execução do servidor como número de rodadas e estratégia de agregação. Os parâmetros iniciais do modelo também são definidos.

```python
def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(initial_parameters=parameters)
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
```
## Resultados
Para rodar o código nesse esquema, basta inicialmente instalar as dependências com:
```bash
pip install -e ./pytorch-federated-variational-autoencoder/
```
Então, para rodar:
```bash
flwr run ./pytorch-federated-variational-autoencoder/
```
Após completar o treinamento, é esperado o seguinte output, contento os valores da função de perda por rodada:

![FL Treino](https://github.com/gustavoguaragna/FedGen/blob/main/pytorch-federated-variational-autoencoder/images/FL_trained.png "Fim de Treinamento")

Esses valores podem ser melhor analisados a partir da um gráfico. Nossa estratégia utilizada no treinamento salvou os valores da função de perda por rodada em um arquivo txt, de modo que esses valores podem ser obtidos a partir do código abaixo:

```python
import re
rounds = []
losses = []

# Define the regex pattern to extract numbers
pattern = r"Rodada\s+(\d+),\s+Perda:\s+([0-9.]+)"

with open(f"losses_{dataset}.txt", 'r') as file:
    for line in file:
        line = line.strip()
        if line:  # Ensure the line is not empty
            match = re.match(pattern, line)
            if match:
                round_num = int(match.group(1))
                loss_val = float(match.group(2))
                rounds.append(round_num)
                losses.append(loss_val)
            else:
                print(f"Ignored line (unexpected format): {line}")
```
E o gráfico é plotado com o código:
```python
plt.figure(figsize=(10, 6))
plt.plot(rounds, losses, marker='o', linestyle='-', color='b', label='Loss')
plt.xlabel('Round', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True)
plt.xticks(rounds)  # Set x-ticks to be the round numbers
plt.legend()

plt.savefig("losses.png")
plt.show()
```
![Loss por Rodada](https://github.com/gustavoguaragna/FedGen/blob/main/pytorch-federated-variational-autoencoder/images/losses.png "Loss por Rodada")

Também podemos analisar visualmente imagens sintéticas geradas pelo nosso VAE a partir de imagens reais.
Primeiramente, vamos baixar as imagens do banco de dados e normalizar para serem entrada do nosso modelo treinado.
```python
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

# Defina as mesmas transformações usadas durante o treinamento
if dataset == "mnist":
    transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    dataset_obj = MNIST(root='./data', train=False, download=True, transform=transform)
elif dataset == "cifar10":
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_obj = CIFAR10(root='./data', train=False, download=True, transform=transform)
else:
    raise ValueError(f"Dataset {dataset} not supported")
```
Agora, vamos buscar imagens do banco de dados, até obter uma imagem de cada classe para fins didáticos. Em seguida, geramos imagens sintéticas a partir das imagens reais selecionadas.
```python
num_classes = 10
images_dict = {}  # Dicionário para armazenar imagens por dígito

for i, (img, label) in enumerate(dataset_obj):
    if label not in images_dict:
        images_dict[label] = img
    if len(images_dict) == num_classes:
        break

# Verifique se todos os dígitos foram encontrados
if len(images_dict) < num_classes:
    raise ValueError("Não foi possível encontrar uma imagem para cada classe.")

# Ordenar as imagens de 0 a 9
images = [images_dict[classe] for classe in range(num_classes)]
# Certifique-se de que as imagens estão no dispositivo correto
images = [img.to(device) for img in images]

# Empilhe as imagens em um único tensor
input_batch = torch.stack(images)

# Gere as reconstruções usando a função generate
with torch.no_grad():
    reconstructed_images, _, _ = fedvae.generate(model, input_batch)
```
Então, desnormalizamos as imagens.
```python
import matplotlib.pyplot as plt
import numpy as np

# Desnormalize as imagens (inverta a normalização aplicada)
def denormalize(imgs):
    imgs = imgs * 0.5 + 0.5  # Escala de volta para [0,1]
    imgs = torch.clamp(imgs, 0, 1)
    return imgs

input_batch = denormalize(input_batch.cpu())
reconstructed_images = denormalize(reconstructed_images.cpu())

# Converter para numpy
input_batch = input_batch.numpy()
reconstructed_images = reconstructed_images.numpy()
```
Finalmente, podemos gerar uma vizualização, comparando as imagens sintéticas geradas com as reais.
```python
# Criar a figura composta
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16, 20))
fig.suptitle(f"Comparação de Imagens Originais e Reconstruídas ({dataset.upper()})", fontsize=20)

for i in range(5):
    # Dígitos 0-4 na primeira e segunda colunas
    orig_idx = i  # Dígito 0-4
    recon_idx = i

    if dataset == "mnist":
        orig_img = np.squeeze(input_batch[orig_idx])  # Remover dimensão de canal
        recon_img = np.squeeze(reconstructed_images[recon_idx])
        cmap = 'gray'
    elif dataset == "cifar10":
        orig_img = np.transpose(input_batch[orig_idx], (1, 2, 0))  # (C, H, W) -> (H, W, C)
        recon_img = np.transpose(reconstructed_images[recon_idx], (1, 2, 0))
        cmap = None  # RGB

    # Coluna 1: Original (0-4)
    ax_orig = axes[i, 0]
    ax_orig.imshow(orig_img, cmap=cmap)
    digit_label = f"Dígito {orig_idx}" if dataset == "mnist" else f"Classe {orig_idx}"
    ax_orig.set_title(f"Original {digit_label}", fontsize=14)
    ax_orig.axis('off')

    # Coluna 2: Reconstruída (0-4)
    ax_recon = axes[i, 1]
    ax_recon.imshow(recon_img, cmap=cmap)
    ax_recon.set_title(f"Reconstruída {digit_label}", fontsize=14)
    ax_recon.axis('off')

    # Dígitos 5-9 na terceira e quarta colunas
    orig_idx_2 = i + 5  # Dígito 5-9
    recon_idx_2 = i + 5

    if dataset == "mnist":
        orig_img_2 = np.squeeze(input_batch[orig_idx_2])
        recon_img_2 = np.squeeze(reconstructed_images[recon_idx_2])
        cmap_2 = 'gray'
    elif dataset == "cifar10":
        orig_img_2 = np.transpose(input_batch[orig_idx_2], (1, 2, 0))
        recon_img_2 = np.transpose(reconstructed_images[recon_idx_2], (1, 2, 0))
        cmap_2 = None

    # Coluna 3: Original (5-9)
    ax_orig_2 = axes[i, 2]
    ax_orig_2.imshow(orig_img_2, cmap=cmap_2)
    digit_label_2 = f"Dígito {orig_idx_2}" if dataset == "mnist" else f"Classe {orig_idx_2}"
    ax_orig_2.set_title(f"Original {digit_label_2}", fontsize=14)
    ax_orig_2.axis('off')

    # Coluna 4: Reconstruída (5-9)
    ax_recon_2 = axes[i, 3]
    ax_recon_2.imshow(recon_img_2, cmap=cmap_2)
    ax_recon_2.set_title(f"Reconstruída {digit_label_2}", fontsize=14)
    ax_recon_2.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajusta o layout para não sobrepor o título
# Salvar a figura
output_path = f"comparison_{dataset}.png"
plt.savefig(output_path)
plt.close()

print(f"Figura comparativa salva em {output_path}")
```
![Imagens Geradas](https://github.com/gustavoguaragna/FedGen/blob/main/pytorch-federated-variational-autoencoder/images/comparison_mnist.png "Imagens Sintéticas por Classe")

Aqui está o [GitHub Original](https://github.com/adap/flower/tree/main/examples/pytorch-federated-variational-autoencoder) configurado somente para o CIFAR10 e que não salva os modelos e as perdas por rodada.
