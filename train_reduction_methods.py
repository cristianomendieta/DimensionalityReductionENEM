from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import TargetEncoder

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import time
import pickle
from xgboost import XGBClassifier

import pandas as pd
import time
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import pacmap
from sklearn.model_selection import train_test_split



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
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "TP_LINGUA",
    "NU_NOTA_REDACAO",
    "NU_NOTA_COMP1",
    "NU_NOTA_COMP2",
    "NU_NOTA_COMP3",
    "NU_NOTA_COMP4",
    "NU_NOTA_COMP5"
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

    categorical_transformer = Pipeline(steps=[("encoder", TargetEncoder(smooth="auto"))])

    preprocessor = ColumnTransformer(
        transformers=[
            #("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    if method == "nan":
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

    elif method == "tsne":
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("pre_tsne", PCA(n_components=5)),
                ("reduction_method", TSNE(n_components=n_components)),
            ]
        )
    elif method == 'pacmap':
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('reduction_method', pacmap.PaCMAP(n_components=n_components, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0))
        ])


def train_models(reduction_method_list, n_components_list, src_path, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=400)

    k_folds = 5

    df_linear_results = pd.DataFrame()
    results_list = []

    for reduction_method in reduction_method_list:
        for n in n_components_list:
            results = {}
            print(f"Reduction method: {reduction_method} - n_components: {n}")
            pipe = generate_pipeline(reduction_method, n)

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            X_train_method = pipe.fit_transform(X_train, y_train)
            X_test_transformed = pipe.transform(X_test)
            print("saiu")
            kf = KFold(n_splits=k_folds, shuffle=True)
            cv_results = cross_val_score(model, X_train_method, y_train, cv=kf)
            
            # get training time
            start_time = time.time()
            model.fit(X_train_method, y_train)
            end_time = time.time()
            
            # save model
            with open(f'{src_path}/model_{reduction_method}_{n}.pkl', 'wb') as file:
                pickle.dump(model, file)
            
            training_time = end_time - start_time
            
            y_predicted = model.predict(X_test_transformed)
            
            # get classification metrics
            test_accuracy = accuracy_score(y_test, y_predicted)
            test_f1_score = f1_score(y_test, y_predicted)
            test_precision = precision_score(y_test, y_predicted)
            test_recall = recall_score(y_test, y_predicted)

            print("Accuracy médio na validação cruzada:", np.mean(cv_results))
            print("Accuracy no conjunto de teste:", test_accuracy)
            
            # Adicionando métricas ao dicionário de resultados
            results['Reduction Method'] = reduction_method
            results['n_components'] = n
            results['CV Accuracy'] = np.mean(cv_results)
            results['Test Accuracy'] = test_accuracy
            results['Test F1 Score'] = test_f1_score
            results['Test Precision'] = test_precision
            results['Test Recall'] = test_recall
            results['Training Time'] = training_time
            
            results_list.append(results)
            
    # Criando DataFrame a partir da lista de resultados
    df_linear_results = pd.DataFrame(results_list)

    # save results
    df_linear_results.to_parquet(f'{src_path}/no-linear.parquet')
    
    
if __name__ == '__main__':
    reduction_method_list = ['pacmap']
    n_components_list = [2, 3]
    src_path = "../tcc-results/data-results/no-linear-binaria/"
    
    data_enem_binary_classifier = pd.read_parquet("../tcc-results/data-results/data_prepared_enem_2022.parquet")
    data_enem_binary_classifier = data_enem_binary_classifier.loc[data_enem_binary_classifier["TP_ESCOLA"] != 1]
    print(data_enem_binary_classifier.shape)

    X = data_enem_binary_classifier.drop('TP_ESCOLA', axis=1)
    y = data_enem_binary_classifier['TP_ESCOLA']

    # replace target values [2, 3] to [0, 1]
    y = y.replace({2: 0, 3: 1})
        
    train_models(reduction_method_list, n_components_list, src_path, X, y)
