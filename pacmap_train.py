import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

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

if __name__ == '__main__':
    data_enem_binary_classifier = pd.read_parquet("../tcc-results/data-results/data_prepared_enem_2022.parquet")
    data_enem_binary_classifier = data_enem_binary_classifier.loc[data_enem_binary_classifier["TP_ESCOLA"] != 1]
    print(data_enem_binary_classifier.shape)

    X = data_enem_binary_classifier.drop('TP_ESCOLA', axis=1)
    y = data_enem_binary_classifier['TP_ESCOLA']

    # replace target values [2, 3] to [0, 1]
    y = y.replace({2: 0, 3: 1})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=400)
    
    results_list = []
    results = {}
    
    categorical_transformer = Pipeline(steps=[("encoder", TargetEncoder(smooth="auto"))])
    numeric_transformer = Pipeline(steps=[("scaler", MinMaxScaler())])
    
    preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_COLS),
                ("cat", categorical_transformer, CATEGORICAL_COLS),
            ]
    )
    
    X_train_method_preprocessed = preprocessor.fit_transform(X_train, y_train)
    
    embedding = pacmap.PaCMAP(n_components=2) 

    # fit the data (The index of transformed data corresponds to the index of the original data)
    X_transformed = embedding.fit_transform(X_train_method_preprocessed)
    
    X_test_preprocessed = preprocessor.transform(X_test)
    X_test_transformed = embedding.transform(X_test_preprocessed)
    
    k_folds = 5
    
    
    print("saiu")
    kf = KFold(n_splits=k_folds, shuffle=True)
    cv_results = cross_val_score(model, X_transformed, y_train, cv=kf)
    
    # get training time
    start_time = time.time()
    model.fit(X_transformed, y_train)
    end_time = time.time()
    
    # save model
    with open(f'../tcc-results/model_pacmap_2.pkl', 'wb') as file:
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
    results['Reduction Method'] = "pacmap"
    results['n_components'] = 2
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
    df_linear_results.to_parquet(f'../tcc-results/no-linear_pacmap2.parquet')
    


