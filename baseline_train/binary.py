import sys
import os

# Adicionar o diretório pai ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import prepare_data, train_models

if __name__ == "__main__":
    # Imprime o diretório de trabalho atual
    print(os.getcwd())

    # Define o caminho dos dados
    data_path = os.path.join("..", "tcc-results", "data-results", "data_prepared_enem_2022.parquet")

    # Prepara os dados
    X, y = prepare_data(data_path, "binary", "TP_ESCOLA")

    # Executa o treinamento utilizando o método baseline
    train_models(
        X, y,
        reduction_methods=["baseline"],
        n_components=[None],  # Nenhum componente necessário para o método baseline
        average_type="binary",
        experiment="baseline_experiment"
    )
