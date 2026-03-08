"""
Script para entrenar un modelo de clasificación utilizando la técnica con mejor desempeño
que fuera seleccionada durante la experimentación.
"""
# Importaciones generales
import pandas as pd
import mlflow
import mlflow.sklearn
# Importaciones para el preprocesamiento y modelado
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample
# Importaciones para la evaluación / experimentación
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeClassifier


def load_data(path):
    """Función para cargar los datos desde un archivo CSV."""
    df = pd.read_csv(path)
    X = df.drop("y", axis=1)
    y = df["y"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def create_preprocessor(X_train):
    numerical_columns = X_train.select_dtypes(exclude="object").columns
    categorical_columns = X_train.select_dtypes(include="object").columns

    X_train = X_train.copy()
    int_columns = X_train.select_dtypes(include="int").columns
    for col in int_columns:
        X_train[col] = X_train[col].astype("float")

    # Actualizar numerical_cols
    numerical_columns = X_train.select_dtypes(exclude="object").columns

    # Pipeline para valores numéricos
    num_pipeline = Pipeline(steps=[
        ("RobustScaler", RobustScaler())
    ])

    # Pipeline para valores categóricos
    cat_pipeline = Pipeline(steps=[
        ("OneHotEncoder", OneHotEncoder(drop="first", sparse_output=False))
    ])

    # Se configuran los preprocesadores
    preprocessor_full = ColumnTransformer([
        ("num_pipeline", num_pipeline, numerical_columns),
        ("cat_pipeline", cat_pipeline, categorical_columns)
    ]).set_output(transform="pandas")

    return preprocessor_full, X_train

#TODO: continuar desde aqui con balanceo, modela y guardado con mlflow