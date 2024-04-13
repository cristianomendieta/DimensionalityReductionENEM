
import pandas as pd
import gc
import time
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, KernelPCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error
import pacmap


NUMERIC_COLS = [
    "TP_FAIXA_ETARIA",
    "TP_ESTADO_CIVIL",
    "TP_COR_RACA",
    "TP_NACIONALIDADE",
    "TP_ST_CONCLUSAO",
    "TP_ANO_CONCLUIU",
    "TP_ESCOLA",
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
    "NU_NOTA_COMP5",
    "Q005",
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

    ### não será utilizado pois não é eficiente para um conjunto de dados com muitas linhas como é o nosso caso, no algoritmo do kpca a dimensionalidade é aumentada
    ### e nesse caso como temos mais de 3 milhões de linhas, o algoritmo não é eficiente
    elif method == "kernel_pca":
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "reduction_method",
                    KernelPCA(
                        n_components=n_components, kernel="rbf", gamma=10, alpha=0.1
                    ),
                ),
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

    elif method == "umap":
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("reduction_method", UMAP(n_components=n_components)),
            ]
        )
    elif method == 'pacmap':
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('reduction_method', pacmap.PaCMAP(n_components=n_components, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0))
        ])


def get_preserved_variance_ratio(X, X_transformed):
    # Calcula a variância total dos dados originais
    total_variance = np.var(X, axis=0).sum()

    # Calcula a variância após a redução de dimensionalidade
    reduced_variance = np.var(X_transformed, axis=0).sum()

    # Calcula a proporção da variância preservada
    preserved_variance_ratio = reduced_variance / total_variance

    return preserved_variance_ratio


def get_dimension_reduction_results(list_methods, n_components_list, X_train):
    results = []

    for method in list_methods:
        for n in n_components_list:
            print(f"Method: {method} - n_components: {n}")
            # dr is short for dimensionality reduction
            pipeline_dr = generate_pipeline(method, n)

            start_time = time.time()
            
            X_train_transformed_dr = pipeline_dr.fit_transform(X_train)

            end_time = time.time()
            execution_time = end_time - start_time

            print(f"Execution time: {execution_time} seconds")

            pipeline_without_dr = generate_pipeline('nan')
            X_train_transformed = pipeline_without_dr.fit_transform(X_train)

            reduction_method_mse = None
            X_restored = None
            #if method != 'tsne':
                #X_restored = pipeline_dr.named_steps['reduction_method'].inverse_transform(X_train_transformed_dr)

                #reduction_method_mse = mean_squared_error(X_train_transformed, X_restored)

            #preserved_variance_ratio = get_preserved_variance_ratio(X_train_transformed, X_train_transformed_dr)
            preserved_variance_ratio = None
        
            # save X_train_transformed_dr as parquet
            X_train_transformed_dr_df = pd.DataFrame(X_train_transformed_dr)
            X_train_transformed_dr_df.to_parquet(f'./results/X_train_transformed_dr_{method}_{n}.parquet')

            results.append({
                'Method': method,
                'n_components': n,
                'Preserved Variance Ratio': preserved_variance_ratio,
                'MSE': reduction_method_mse,
                'time': execution_time
            })

            del pipeline_dr
            del pipeline_without_dr
            del X_train_transformed_dr_df
            del X_train_transformed_dr
            del X_train_transformed
            del X_restored
            gc.collect()

    # Convertendo a lista de dicionários em um DataFrame
    results_df = pd.DataFrame(results)

    return results_df
  
if __name__ == '__main__':
    X_train = pd.read_parquet('./use_data/X_train.parquet')
    list_linear_methods = ['pacmap']
    n_components_list = [3]
    results_df = get_dimension_reduction_results(list_linear_methods, n_components_list, X_train)
    results_df.to_parquet('./results/pacmap_metrics_3.parquet')

