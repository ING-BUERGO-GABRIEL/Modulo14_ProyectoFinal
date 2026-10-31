import os
import socket
from typing import List, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import mlflow.pyfunc

# Nombre del Pod en Kubernetes (inyectado por Downward API) o hostname del contenedor
POD_NAME = os.getenv("POD_NAME", os.getenv("HOSTNAME", socket.gethostname()))
MODEL_NAME = "modelo-inmobiliario-prod"
MODEL_VERSION = "5@champion"

app = FastAPI(
    title="California Housing Inference Service",
    description="Servicio de inferencia empaquetado para Kubernetes con modelo MLflow registrado",
    version="1.0.0"
)

# Carga del modelo champion empaquetado
MODEL_PATH = os.getenv("MODEL_PATH", "model_champion")
print(f"Cargando modelo MLflow desde: {MODEL_PATH} ...")
try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    model = None

class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Ingreso mediano del bloque (en decenas de miles)")
    HouseAge: float = Field(..., description="Antigüedad mediana de las viviendas")
    AveRooms: float = Field(..., description="Número promedio de habitaciones")
    AveBedrms: float = Field(..., description="Número promedio de dormitorios")
    Population: float = Field(..., description="Población del bloque")
    AveOccup: float = Field(..., description="Ocupación promedio por vivienda")
    Latitude: float = Field(..., description="Latitud")
    Longitude: float = Field(..., description="Longitud")

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 3.5,
                "HouseAge": 25.0,
                "AveRooms": 5.2,
                "AveBedrms": 1.05,
                "Population": 1200.0,
                "AveOccup": 3.0,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
    }

class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    pod_name: str
    status: str

@app.get("/")
def read_root():
    return {
        "service": "California Housing ML Ingestion & Inference API",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "pod_name": POD_NAME,
        "status": "ready" if model is not None else "model_not_loaded"
    }

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {
        "status": "healthy",
        "pod_name": POD_NAME,
        "model_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Convertir a DataFrame con los nombres exactos de columnas
        data_dict = features.model_dump()
        df = pd.DataFrame([data_dict])
        
        preds = model.predict(df)
        pred_value = float(preds[0])
        
        return PredictionResponse(
            prediction=round(pred_value, 4),
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            pod_name=POD_NAME,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")

@app.post("/predict_batch")
def predict_batch(items: List[HousingFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        df = pd.DataFrame([item.model_dump() for item in items])
        preds = model.predict(df)
        return {
            "predictions": [round(float(p), 4) for p in preds],
            "count": len(preds),
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "pod_name": POD_NAME,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia batch: {str(e)}")

