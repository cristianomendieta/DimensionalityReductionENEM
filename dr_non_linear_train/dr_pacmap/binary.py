import sys
import os

# Adicionar o diretório pai ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

if __name__ == "__main__":
    print(os.getcwd())
    n_components_list = [2, 3, 5, 10, 20]
    linear_methods = ['pacmap']
    
    X, y = prepare_data("../tcc-results/data-results/data_prepared_enem_2022.parquet", "binary", "TP_ESCOLA")
    train_models(X, y, linear_methods, n_components_list, "binary", "pacmap_binary")
    
    