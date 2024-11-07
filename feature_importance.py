import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import os
import time
import pickle
from utils import map_values, CATEGORICAL_COLS, NUMERIC_COLS, generate_pipeline

def encode_target(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder

def plot_feature_importance(feature_names, importance, title):
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(importance)), importance)
    plt.xticks(range(len(importance)), feature_names, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'./results/metrics/{title.replace(" ", "_").lower()}.png')
    plt.close()

def train_models(X, y, n_features_list, average_type, experiment, objective=None, num_classes=None):
    os.makedirs('./results/models', exist_ok=True)
    os.makedirs('./results/metrics/mc', exist_ok=True)
    
    # # Encode target if necessary
    # if y.dtype == object:
    y, encoder = encode_target(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = generate_pipeline("baseline", 0)
    X_train_transformed = pipe.fit_transform(X_train)
    X_test_transformed = pipe.transform(X_test)
    
    if objective and num_classes:
        base_model = XGBClassifier(n_estimators=400, objective=objective, num_class=num_classes, eval_metric='logloss', n_jobs=-1)
    else:
        base_model = XGBClassifier(n_estimators=400, eval_metric='logloss', n_jobs=-1)
    
    results_list = []
    
    # Train and evaluate model with all features
    start_time = time.time()
    model_all = XGBClassifier(random_state=42, n_estimators=400, eval_metric='logloss', n_jobs=-1)
    model_all.fit(X_train_transformed, y_train)
    all_features_time = time.time() - start_time
    
    y_pred_all = model_all.predict(X_test_transformed)
    accuracy_all = accuracy_score(y_test, y_pred_all)
    f1_all = f1_score(y_test, y_pred_all, average=average_type)
    precision_all = precision_score(y_test, y_pred_all, average=average_type)
    recall_all = recall_score(y_test, y_pred_all, average=average_type)
    
    results_list.append({
        'Number of Features': X.shape[1],
        'Accuracy': accuracy_all,
        'F1 Score': f1_all,
        'Precision': precision_all,
        'Recall': recall_all,
        'Training Time': all_features_time
    })
    
    # Plot feature importance for all features
    feature_importance = model_all.feature_importances_
    
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=pipe.get_feature_names_out(input_features=X.columns))
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=pipe.get_feature_names_out(input_features=X.columns))
    plot_feature_importance(X_train_transformed.columns, feature_importance, "Feature Importance (All Features)")
    
    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = X_train_transformed.columns[sorted_indices]
    
    # Train models with selected features based on importance
    for n_features in n_features_list:
        print(f"Number of features: {n_features}")
        
        selected_features = sorted_features[:n_features]
        
        start_time = time.time()
        if objective and num_classes:
            model_selected = XGBClassifier(n_estimators=400, objective=objective, num_class=num_classes, eval_metric='logloss', n_jobs=-1)
        else:
            model_selected = XGBClassifier(n_estimators=400, eval_metric='logloss', n_jobs=-1)
        model_selected.fit(X_train_transformed[selected_features], y_train)
        selected_features_time = time.time() - start_time
        
        y_pred_selected = model_selected.predict(X_test_transformed[selected_features])
        accuracy_selected = accuracy_score(y_test, y_pred_selected)
        f1_selected = f1_score(y_test, y_pred_selected, average=average_type)
        precision_selected = precision_score(y_test, y_pred_selected, average=average_type)
        recall_selected = recall_score(y_test, y_pred_selected, average=average_type)
        
        results_list.append({
            'Number of Features': n_features,
            'Accuracy': accuracy_selected,
            'F1 Score': f1_selected,
            'Precision': precision_selected,
            'Recall': recall_selected,
            'Training Time': selected_features_time
        })
        
        print(f"Accuracy with {n_features} features: {accuracy_selected:.4f}")
        print(f"F1 Score with {n_features} features: {f1_selected:.4f}")
        print(f"Training time with {n_features} features: {selected_features_time:.2f} seconds")
        
        # Save model
        with open(f'./results/models/top_{n_features}_features.pkl', 'wb') as file:
            pickle.dump(model_selected, file)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_selected)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f'./results/metrics/mc/confusion_matrix_top_{n_features}_features.png')
        plt.close()
        
        # Plot feature importance for selected features
        feature_importance_selected = model_selected.feature_importances_
        plot_feature_importance(selected_features, feature_importance_selected, f"Feature Importance (Top {n_features} Features)")
    
    # Save all results
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(f'./results/metrics/{experiment}.csv', index=False)
    
    return df_results

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
        
        X = df_microdados.drop(target_col, axis=1)
        y = df_microdados[target_col]
        y, encoder = encode_target(y)
        
    elif target_type == "binary":
        df_microdados = df_microdados.loc[df_microdados["TP_ESCOLA"] != 1]
        X = df_microdados.drop(target_col, axis=1)
        y = df_microdados[target_col]
        y = y.replace({2: 0, 3: 1})
    
    return X, y
