import pandas as pd
from pandera.pandas import DataFrameSchema, Column
import pytest


@pytest.fixture
def datos_banco():
    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")
    return df


def test_esquema(datos_banco):
    esquema = DataFrameSchema({
        "age": Column(int, nullable=False),
        "job": Column(str, nullable=False),
        "marital": Column(str, nullable=False),
        "education": Column(str, nullable=False),
        "default": Column(str, nullable=True),
        "housing": Column(str, nullable=False),
        "loan": Column(str, nullable=False),
        "contact": Column(str, nullable=False),
        "month": Column(str, nullable=False),
        "day_of_week": Column(str, nullable=False),
        "duration": Column(int, nullable=False),
        "campaign": Column(int, nullable=False),
        "pdays": Column(int, nullable=False),
        "previous": Column(int, nullable=False),
        "poutcome": Column(str, nullable=False),
        "emp.var.rate": Column(float, nullable=False),
        "cons.price.idx": Column(float, nullable=False),
        "cons.conf.idx": Column(float, nullable=False),
        "euribor3m": Column(float, nullable=False),
        "nr.employed": Column(float, nullable=False),
        "y": Column(str, nullable=False),
    })

    esquema.validate(datos_banco)


def test_basico(datos_banco):
    df = datos_banco

    assert not df.empty, "El DataFrame está vacío."
    assert df.isnull().sum().sum() == 0, "El DataFrame contiene valores nulos."
    # assert df.duplicated().sum() == 0, "El DataFrame contiene filas duplicadas."
    assert df.shape[1] == 21, (
        f"El DataFrame debería tener 21 columnas, pero tiene {df.shape[1]}."
    )


def test_rango_edad(datos_banco):
    df = datos_banco
    assert df["age"].between(0, 120).all(), (
        "La edad debe estar entre 0 y 120."
    )
    
def test_campaign_valida(datos_banco):
    df = datos_banco
    assert (df["campaign"] >= 1).all(), (
        "La cantidad de contactos en campaña debe ser al menos 1."
    )