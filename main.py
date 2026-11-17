from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predictor import ejecutar_warmup, generar_predicciones


@asynccontextmanager
async def lifespan(_: FastAPI):
    ejecutar_warmup()
    yield


app = FastAPI(
    title="Heladería Ciocolatto — Servicio de Predicción IA",
    version="1.0.0",
    lifespan=lifespan,
)


class RegistroHistorialInput(BaseModel):
    fecha: str
    cantidad: int = Field(ge=0)


class ProductoHistorialInput(BaseModel):
    producto_id: int
    nombre: Optional[str] = None
    producto_temporada: bool = False
    historial: List[RegistroHistorialInput]


class PredictRequest(BaseModel):
    dias: int = Field(default=7, ge=1, le=30)
    fecha_corte: Optional[str] = None
    productos: List[ProductoHistorialInput]


@app.get("/health")
def health():
    return {"status": "ok", "motor": "StatsForecast AutoETS"}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        productos_dict = [p.model_dump() for p in payload.productos]
        resultados = generar_predicciones(
            productos=productos_dict,
            dias=payload.dias,
            fecha_corte_str=payload.fecha_corte,
        )
        return {
            "motor": "StatsForecast AutoETS",
            "resultados": resultados,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(exc)}")
