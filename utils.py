import pandas as pd
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.preprocessing import TargetEncoder, LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA, TruncatedSVD, FastICA


NUMERIC_COLS = [
    "TP_FAIXA_ETARIA",
    "TP_ESTADO_CIVIL",
    "TP_COR_RACA",
    "TP_NACIONALIDADE",
    "TP_ST_CONCLUSAO",
    "TP_ANO_CONCLUIU",
    "IN_TREINEIRO",
    "CO_MUNICIPIO_PROVA",
    "CO_UF_PROVA",
    "TP_PRESENCA_CN",
    "TP_PRESENCA_CH",
    "TP_PRESENCA_LC",
    "TP_PRESENCA_MT",
    "NU_NOTA_COMP2",
    "NU_NOTA_COMP3",
    "NU_NOTA_COMP4",
    "NU_NOTA_COMP5",
    "TP_ESCOLA"
]

CATEGORICAL_COLS = [
    "TP_SEXO",
    "Q001",
    "Q002",
    "Q003",
    "Q004",
    "Q007",
    "Q008",
    "Q009",
    "Q010",
    "Q011",
    "Q012",
    "Q013",
    "Q014",
    "Q015",
    "Q016",
    "Q017",
    "Q018",
    "Q019",
    "Q020",
    "Q021",
    "Q022",
    "Q023",
    "Q024",
    "Q025",
    "NO_MUNICIPIO_PROVA",
    "TP_STATUS_REDACAO",
    "SG_UF_PROVA",
    "faixa_renda_familiar"
]


def generate_pipeline(method, n_components=None):
    numeric_transformer = Pipeline(steps=[("scaler", MinMaxScaler())])

    categorical_transformer = Pipeline(steps=[("encoder", OrdinalEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    if method == "baseline":
        return Pipeline(steps=[("preprocessor", preprocessor)])

    elif method == "pca":
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("reduction_method", PCA(n_components=n_components)),
            ]
        )

    elif method == "svd":
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("reduction_method", TruncatedSVD(n_components=n_components)),
            ]
        )

    elif method == "ica":
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("reduction_method", FastICA(n_components=n_components)),
            ]
        )
    
def encode_target(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder
    
# Função para fazer o mapeamento inverso
def map_values(value, mapping):
    for key, values in mapping.items():
        if value in values:
            return key
    return value
        
def prepare_data(data_path: str, target_type: str, target_col: str):
    df_microdados = pd.read_parquet(data_path)
    if target_col in CATEGORICAL_COLS:
        CATEGORICAL_COLS.remove(target_col)
    if target_col in NUMERIC_COLS:
        NUMERIC_COLS.remove(target_col)
    if target_type == "multiclass":
        map_renda = {
            "A": ["P", "Q"], # 15 A 25 SALÁRIOS
            "B": ["N", "O"], # 10 A 15 SALÁRIOS
            "C": ["J", "K", "L", "M"], # 5 A 10 SALÁRIOS
            "D": ["E", "F", "G", "H", "I"], # 2 A 5 SALÁRIOS
            "E":  ["A", "B", "C"], # ATÉ 2 SALÁRIOS
        }
        df_microdados[target_col] = df_microdados[target_col].apply(map_values, mapping=map_renda)
        
        X = df_microdados.drop('faixa_renda_familiar', axis=1)
        y = df_microdados['faixa_renda_familiar']
        y, encoder = encode_target(y)
        
    elif target_type == "binary":
        df_microdados = df_microdados.loc[df_microdados["TP_ESCOLA"] != 1]

        X = df_microdados.drop(target_col, axis=1)
        y = df_microdados[target_col]

        # replace target values [2, 3] to [0, 1]
        y = y.replace({2: 0, 3: 1})
        
    return X, y


def train_models(X, y, objective, num_classes, reduction_methods, n_components, average_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=400, n_jobs=-1)

    k_folds = 5
    
    df_results = pd.DataFrame()
    results_list = []
    
    for method in reduction_methods:
        for component in n_components:
            results = {}
            print(f"Reduction method: {method} - n_components: {component}")
            
            pipe = generate_pipeline(method, component)
                        
            X_train_method = pipe.fit_transform(X_train)
            X_test_transformed = pipe.transform(X_test)

            kf = KFold(n_splits=k_folds, shuffle=True)
            cv_results = cross_val_score(model, X_train_method, y_train, cv=kf)

            # get training time
            start_time = time.time()
            model.fit(X_train_method, y_train)
            end_time = time.time()

            # save model
            with open(f'./results/models/{method}_{component}.pkl', 'wb') as file:
                pickle.dump(model, file)

            training_time = end_time - start_time
            print("Tempo de treinamento:", training_time)

            y_predicted = model.predict(X_test_transformed)

            # get classification metrics
            test_accuracy = accuracy_score(y_test, y_predicted)
            test_f1_score = f1_score(y_test, y_predicted, average=average_type)
            test_precision = precision_score(y_test, y_predicted, average=average_type)
            test_recall = recall_score(y_test, y_predicted, average=average_type)

            print("Accuracy médio na validação cruzada:", np.mean(cv_results))
            print("Accuracy no conjunto de teste:", test_accuracy)

            print("F1 Score no conjunto de teste:", test_f1_score)
            print("Precision no conjunto de teste:", test_precision)
            
            print("Accuracy médio na validação cruzada:", np.mean(cv_results))
            print("Accuracy no conjunto de teste:", test_accuracy)
            
            # Adicionando métricas ao dicionário de resultados
            results['Reduction Method'] = method
            results['n_components'] = component
            results['CV Accuracy'] = np.mean(cv_results)
            results['Test Accuracy'] = test_accuracy
            results['Test F1 Score'] = test_f1_score
            results['Test Precision'] = test_precision
            results['Test Recall'] = test_recall
            results['Training Time'] = training_time
            
            results_list.append(results)
        

            cm = confusion_matrix(y_test, y_predicted)

            # Criar o display da matriz de confusão
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)

            # Salvar a figura em um arquivo
            plt.savefig(f'./metrics/mc/matriz_de_confusao_{method}_{component}.png')

            # Fechar a figura para liberar memória
            plt.close()