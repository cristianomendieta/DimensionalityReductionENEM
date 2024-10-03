import sys
import os

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from rfe import prepare_data, train_models

if __name__ == "__main__":
    n_features_list = [2, 3, 5, 10, 20]
    
    data_path = os.path.join(project_root, "tcc-results", "data-results", "data_prepared_enem_2022.parquet")
    
    # Preparar os dados para multiclasse
    X, y = prepare_data(data_path, "multiclass", "faixa_renda_familiar")
    
    # Definir o treinamento de modelos usando o método RFE para multiclasse
    results_df = train_models(X, y, n_features_list, "weighted", "rfe_feature_selection_multiclass", "multi:softmax", num_classes=5)
    
    print("\nResults Summary:")
    print(results_df)
    
    # Encontrar o modelo com melhor desempenho
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    print("\nBest Performing Model:")
    print(f"Number of Features: {best_model['Number of Features']}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"F1 Score: {best_model['F1 Score']:.4f}")
    print(f"Training Time: {best_model['Training Time']:.2f} seconds")
