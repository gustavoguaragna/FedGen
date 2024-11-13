# Federated Variational Autoencoder with PyTorch and Flower

O objetivo desse documento é descrever os códigos que realizam um treinamento federado de um Variational Autoencoder (VAE), modelo generativo que conta com um encoder e um decoder. O encoder é responsável por representar um certo dado de entrada em um espaço latente, através de uma média e um desvio padrão de uma distribuição como a gaussiana. Já, o decoder tem a função de reconstruir amostras geradas pela distribuição representada pelo encoder, visando obter o formato original do dado de entrada. 

![Estrutura VAE](https://github.com/gustavoguaragna/FedGen/blob/main/pytorch-federated-variational-autoencoder/images/VAE.png "VAE")
