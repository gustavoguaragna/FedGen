"""fedvaeexample: A Flower / PyTorch app for Federated Variational Autoencoder."""

from fedvaeexample.task import Net, get_weights, set_weights
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import torch

class FedAvg_Save(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        # Chama o método original para obter os parâmetros agregados
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Salva o modelo após a agregação
            self.save_model(aggregated_parameters, server_round)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # Chama o método original para obter a perda agregada
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Salva a perda após a avaliação
        self.save_loss(aggregated_loss, server_round)

        return aggregated_loss, aggregated_metrics

    def save_model(self, parameters, server_round):
        # Converte os parâmetros para ndarrays
        ndarrays = parameters_to_ndarrays(parameters)
        # Cria uma instância do modelo
        model = Net()
        # Define os pesos do modelo
        set_weights(model, ndarrays)
        # Salva o modelo no disco
        model_path = f"model_round_{server_round}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em {model_path}")

    def save_loss(self, loss, server_round):
        # Salva a perda em um arquivo de texto
        with open("losses.txt", "a") as f:
            f.write(f"Rodada {server_round}, Perda: {loss}\n")
        print(f"Perda da rodada {server_round} salva: {loss}")



def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg_Save(initial_parameters=parameters)
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
