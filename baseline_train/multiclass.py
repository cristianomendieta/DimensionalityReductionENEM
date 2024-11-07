import sys
import os

# Adicionar o diretório pai ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

if __name__ == "__main__":
    X, y = prepare_data("../tcc-results/data-results/data_prepared_enem_2022.parquet", "multiclass", "faixa_renda_familiar")
    train_models(X, y, ["baseline"], [53], "weight", "baseline_multiclass", "multi:softmax", 5)
    
    