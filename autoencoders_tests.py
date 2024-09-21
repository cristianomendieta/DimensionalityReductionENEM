import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import pacmap
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Constants
NUMERIC_COLS = [
    "TP_FAIXA_ETARIA", "TP_ESTADO_CIVIL", "TP_COR_RACA", "TP_NACIONALIDADE",
    "TP_ST_CONCLUSAO", "TP_ANO_CONCLUIU", "IN_TREINEIRO", "CO_MUNICIPIO_PROVA",
    "CO_UF_PROVA", "TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC",
    "TP_PRESENCA_MT", "NU_NOTA_COMP2", "NU_NOTA_COMP3", "NU_NOTA_COMP4",
    "NU_NOTA_COMP5", "TP_ESCOLA"
]

CATEGORICAL_COLS = [
    "TP_SEXO", "Q001", "Q002", "Q003", "Q004", "Q007", "Q008", "Q009", "Q010",
    "Q011", "Q012", "Q013", "Q014", "Q015", "Q016", "Q017", "Q018", "Q019",
    "Q020", "Q021", "Q022", "Q023", "Q024", "Q025", "NO_MUNICIPIO_PROVA",
    "TP_STATUS_REDACAO", "SG_UF_PROVA", "faixa_renda_familiar"
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
        return Pipeline(steps=[("preprocessor", preprocessor), ("reduction", PCA(n_components=n_components))])
    elif method == "svd":
        return Pipeline(steps=[("preprocessor", preprocessor), ("reduction", TruncatedSVD(n_components=n_components))])
    elif method == "tsne":
        return Pipeline(steps=[("preprocessor", preprocessor), ("reduction", TSNE(n_components=n_components))])
    else:
        return Pipeline(steps=[("preprocessor", preprocessor)])

def create_autoencoder(input_dim, encoding_dim, learning_rate=0.001):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)
    
    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, autoencoder.get_layer('bottleneck').output)
    
    # Compile
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return autoencoder, encoder

def train_autoencoder(X_train, X_test, encoding_dim, epochs=200, batch_size=32, learning_rate=0.001):
    input_dim = X_train.shape[1]
    autoencoder, encoder = create_autoencoder(input_dim, encoding_dim, learning_rate)
    
    history = autoencoder.fit(X_train, X_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(X_test, X_test),
                              verbose=1)
    
    return autoencoder, encoder, history

def encode_target(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder

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
            "A": ["P", "Q"],
            "B": ["N", "O"],
            "C": ["J", "K", "L", "M"],
            "D": ["E", "F", "G", "H", "I"],
            "E": ["A", "B", "C"],
        }
        df_microdados[target_col] = df_microdados[target_col].apply(map_values, mapping=map_renda)
        
        X = df_microdados.drop(target_col, axis=1)
        y = df_microdados[target_col]
        y, encoder = encode_target(y)
        
    elif target_type == "binary":
        df_microdados = df_microdados.loc[df_microdados["TP_ESCOLA"] != 1]
        X = df_microdados.drop(target_col, axis=1)
        y = df_microdados[target_col]
        y = y.replace({2: 0, 3: 1})
    
    return X, y

def train_models(X, y, reduction_methods, n_components, average_type, experiment, objective=None, num_classes=None):
    os.makedirs('./results/models', exist_ok=True)
    os.makedirs('./results/metrics/mc', exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if objective and num_classes:
        model = XGBClassifier(n_estimators=400, objective=objective, num_class=num_classes, eval_metric='logloss', n_jobs=-1)
    else:
        model = XGBClassifier(n_estimators=400, eval_metric='logloss', n_jobs=-1)

    k_folds = 5
    results_list = []
    
    for method in reduction_methods:
        for component in n_components:
            print(f"Reduction method: {method} - n_components: {component}")
            
            if method == "pacmap":
                pipe = generate_pipeline("baseline")
                X_train_preprocessed = pipe.fit_transform(X_train)
                X_test_preprocessed = pipe.transform(X_test)
                
                embedding = pacmap.PaCMAP(n_components=component)
                X_train_reduced = embedding.fit_transform(X_train_preprocessed)
                X_test_reduced = embedding.transform(X_test_preprocessed)
            
            elif method == "autoencoder":
                pipe = generate_pipeline("baseline")
                X_train_preprocessed = pipe.fit_transform(X_train)
                X_test_preprocessed = pipe.transform(X_test)
                
                autoencoder, encoder, history = train_autoencoder(
                    X_train_preprocessed, 
                    X_test_preprocessed, 
                    encoding_dim=component,
                    epochs=200,
                    batch_size=32,
                    learning_rate=0.001
                )
                
                X_train_reduced = encoder.predict(X_train_preprocessed)
                X_test_reduced = encoder.predict(X_test_preprocessed)
                
                # Plot training history
                plt.figure(figsize=(10, 5))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title(f'Autoencoder Training History (Encoding Dim: {component})')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(f'./results/metrics/autoencoder_history_{component}.png')
                plt.close()
            
            else:
                pipe = generate_pipeline(method, component)
                X_train_reduced = pipe.fit_transform(X_train)
                X_test_reduced = pipe.transform(X_test)

            # Cross-validation
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            cv_results = cross_val_score(model, X_train_reduced, y_train, cv=kf)

            # Training
            start_time = time.time()
            model.fit(X_train_reduced, y_train)
            training_time = time.time() - start_time

            # Save model
            with open(f'./results/models/{method}_{component}.pkl', 'wb') as file:
                pickle.dump(model, file)

            # Predictions and metrics
            y_predicted = model.predict(X_test_reduced)
            test_accuracy = accuracy_score(y_test, y_predicted)
            test_f1_score = f1_score(y_test, y_predicted, average=average_type)
            test_precision = precision_score(y_test, y_predicted, average=average_type)
            test_recall = recall_score(y_test, y_predicted, average=average_type)

            print(f"CV Accuracy: {np.mean(cv_results):.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test F1 Score: {test_f1_score:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Training Time: {training_time:.2f} seconds")

            # Save results
            results = {
                'Reduction Method': method,
                'n_components': component,
                'CV Accuracy': np.mean(cv_results),
                'Test Accuracy': test_accuracy,
                'Test F1 Score': test_f1_score,
                'Test Precision': test_precision,
                'Test Recall': test_recall,
                'Training Time': training_time
            }
            results_list.append(results)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_predicted)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.savefig(f'./results/metrics/mc/confusion_matrix_{method}_{component}.png')
            plt.close()

    # Save all results
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(f'./results/metrics/{experiment}.csv', index=False)
