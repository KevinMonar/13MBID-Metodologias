from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Dict

app = FastAPI(
    title="Modelo de Clasificación de Clientes Bancarios",
    description="API para predecir si un cliente bancario va a realizar un depósito a plazo fijo.",
    version="1.0.1",
)

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Edad del cliente")
    job: str = Field(..., description="Tipo de trabajo")
    marital: str = Field(..., description="Estado civil")
    education: str = Field(..., description="Nivel educativo")
    housing: str = Field(..., description="Tiene crédito hipotecario")
    loan: str = Field(..., description="Tiene préstamo personal")
    contact: str = Field(..., description="Tipo de comunicación")
    month: str = Field(..., description="Mes del último contacto")
    day_of_week: str = Field(..., description="Día de la semana del último contacto")
    duration: int = Field(..., ge=0, description="Duración del último contacto en segundos")
    campaign: int = Field(..., ge=1, description="Número de contactos realizados durante esta campaña")
    previous: int = Field(..., ge=0, description="Número de contactos realizados antes de esta campaña")
    pdays: int = Field(..., ge=0, description="Número de días desde el último contacto (999 si nunca fue contactado)")
    poutcome: str = Field(..., description="Resultado de la campaña anterior")
    emp_var_rate: float = Field(..., description="Tasa de variación de empleo")
    cons_price_idx: float = Field(..., description="Índice de precios al consumidor")
    cons_conf_idx: float = Field(..., description="Índice de confianza del consumidor")
    euribor3m: float = Field(..., description="Tasa Euribor a 3 meses")
    nr_employed: float = Field(..., description="Número de empleados")
    contacted_before: str = Field(..., description="Si fue contactado anteriormente: 'yes' o 'no'")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "job": "technician",
                "marital": "married",
                "education": "university.degree",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "duration": 200,
                "campaign": 2,
                "previous": 0,
                "pdays": 999,
                "poutcome": "nonexistent",
                "emp_var_rate": 1.1,
                "cons_price_idx": 93.994,
                "cons_conf_idx": -36.4,
                "euribor3m": 4.857,
                "nr_employed": 5191.0,
                "contacted_before": "no"
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    probability: Dict[str, float]
    model_info: Dict[str, str]


# Rutas de modelo y preprocesador
MODEL_PATH = "models/decision_tree_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"Modelo cargado correctamente: {type(model).__name__}")
    print(f"Preprocesador cargado correctamente: {type(preprocessor).__name__}")
except FileNotFoundError as e:
    model = None
    preprocessor = None
    raise RuntimeError(f"No se pudo cargar el modelo o preprocesador: {e}")
except Exception as e:
    model = None
    preprocessor = None
    raise RuntimeError(f"Error al cargar el modelo: {e}")


@app.get("/")
def root():
    return {
        "message": "API del modelo de predicción para clientes bancarios.",
        "version": "1.0.1",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_type": type(model).__name__ if model else None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="El modelo o preprocesador no están cargados.")

    try:
        # Convertir request a DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Convertir enteros a float como en entrenamiento
        int_columns = input_data.select_dtypes(include='int').columns
        for col in int_columns:
            input_data[col] = input_data[col].astype('float')

        # Aplicar preprocesamiento
        input_processed = preprocessor.transform(input_data)

        # Predicción
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0]

        class_labels = model.classes_
        probability_dict = {
            str(class_labels[i]): float(probability[i])
            for i in range(len(class_labels))
        }

        model_info = {
            "model_type": type(model).__name__,
            "preprocessor_type": type(preprocessor).__name__
        }

        return PredictionResponse(
            prediction=str(prediction),
            probability=probability_dict,
            model_info=model_info
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la solicitud: {str(e)}")