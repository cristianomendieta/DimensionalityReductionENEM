import sys
import os

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from utils import prepare_data, train_models

if __name__ == "__main__":
    print(os.getcwd())
    
    n_components_list = [2, 3, 5, 10, 20]
    linear_methods = ['autoencoder']
    
    data_path = os.path.join(project_root, "tcc-results", "data-results", "data_prepared_enem_2022.parquet")
    
    X, y = prepare_data(data_path, "binary", "TP_ESCOLA")
    train_models(X, y, linear_methods, n_components_list, "binary", "autoencoder_binary")