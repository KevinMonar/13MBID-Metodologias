# Importación de librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def visualizar_datos(
    fuente: str = "data/raw/bank-additional-full.csv",
    salida: str = "docs/figures/",
):
    """
    Genera gráficos del dataset y los exporta como imágenes.
    """

    # Crear el directorio de salida si no existe
    Path(salida).mkdir(parents=True, exist_ok=True)

    # Leer los datos
    df = pd.read_csv(fuente, sep=";")

    # (solo para confirmar ejecución)
    print("OK: dataset cargado. Filas:", len(df))
    print("OK: guardando gráficos en:", str(Path(salida).resolve()))

    # Gráfico 1: Distribución de la variable objetivo
    plt.figure(figsize=(6, 4))
    sns.countplot(x="y", data=df)
    plt.title("Distribución de la variable objetivo (suscripción al depósito)")
    plt.xlabel("¿Suscribió un depósito a plazo?")
    plt.ylabel("Cantidad de clientes")
    plt.tight_layout()
    plt.savefig(f"{salida}/distribucion_target.png")
    plt.close()

    # Gráfico 2: Distribución del nivel educativo
    plt.figure(figsize=(6, 4))
    col = "education"
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title("Distribución del nivel educativo")
    plt.xlabel("Cantidad")
    plt.ylabel("Nivel educativo")
    plt.tight_layout()
    plt.savefig(f"{salida}/distribucion_educacion.png")
    plt.close()

    # Gráfico 3: Distribución del tipo de trabajo
    plt.figure(figsize=(8, 4))
    col = "job"
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title("Distribución del tipo de trabajo")
    plt.xlabel("Cantidad")
    plt.ylabel("Tipo de trabajo")
    plt.tight_layout()
    plt.savefig(f"{salida}/distribucion_trabajo.png")
    plt.close()

    # Gráfico 4: Distribución por día de la semana
    plt.figure(figsize=(8, 4))
    col = "day_of_week"
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title("Distribución de contactos por día de la semana")
    plt.xlabel("Cantidad")
    plt.ylabel("Día de la semana")
    plt.tight_layout()
    plt.savefig(f"{salida}/distribucion_dia_semana.png")
    plt.close()

    # --- NUEVO: asegurar que son numéricos (por si hay NaN o texto) ---
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    # Gráfico 5: Distribución de la edad de los clientes
    plt.figure(figsize=(6, 4))
    sns.histplot(df["age"].dropna(), bins=30, kde=True)
    plt.title("Distribución de la edad de los clientes")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(f"{salida}/distribucion_edad.png")
    plt.close()

    # Gráfico 6: Distribución de la duración del contacto
    plt.figure(figsize=(6, 4))
    sns.histplot(df["duration"].dropna(), bins=30, kde=True)
    plt.title("Distribución de la duración del contacto")
    plt.xlabel("Duración (segundos)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(f"{salida}/distribucion_duracion.png")
    plt.close()

    print("OK: gráficos generados (incluye edad y duración).")


if __name__ == "__main__":
    visualizar_datos()