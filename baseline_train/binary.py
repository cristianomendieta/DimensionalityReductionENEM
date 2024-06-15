import sys
import os

# Adicionar o diretório pai ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

if __name__ == "__main__":
    print(os.getcwd())
    X, y = prepare_data("../tcc-results/data-results/data_prepared_enem_2022.parquet", "binary", "TP_ESCOLA")
    train_models(X, y, "binary:logistic", 2, ["baseline"], [53], "binary")
    
    