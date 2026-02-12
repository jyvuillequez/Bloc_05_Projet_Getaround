from typing import Any, List
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

description = """
Getaround Pricing API - Estimation du prix de location

Cette API permet d’estimer le prix de location journalier d’un véhicule à partir de ses caractéristiques.

- GET /
Endpoint d’accueil : vérifie que l’API fonctionne et affiche les informations utiles pour démarrer (statut, modèle chargé, etc.).

- POST /predict
Endpoint de prédiction : envoie les caractéristiques du véhicule au format JSON et récupère en réponse une estimation du prix par jour.

Pour obtenir une prédiction, utilise /predict avec une requête POST contenant les données du véhicule au format attendu.

Format attendu pour /predict :
`{"input": [[...], [...]]}` où chaque ligne respecte l’ordre des colonnes.
"""

tags_metadata = [
  {"name": "Prediction", "description": "Prédiction du prix de location."}
]

app = FastAPI(
    title="Getaround Pricing API",
    version="1.0.0",
    description=description,
    openapi_tags=tags_metadata,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"

FEATURE_NAMES = [
    "model_key",
    "mileage",
    "engine_power",
    "fuel",
    "paint_color",
    "car_type",
    "private_parking_available",
    "has_gps",
    "has_air_conditioning",
    "automatic_car",
    "has_getaround_connect",
    "has_speed_regulator",
    "winter_tires",
]

model = None


class PredictRequest(BaseModel):
    input: List[List[Any]]  # batch: {"input": [[...], [...]]}


@app.on_event("startup")
def startup():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Getaround Pricing API",
        "model_loaded": model is not None,
        "docs": "/docs",
        "feature_order": FEATURE_NAMES,
        "example": {
            "input": [[
                "Peugeot", 13131, 110, "diesel", "grey", "convertible",
                True, True, True, True, True, True, True
            ]]
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded (model.joblib missing).")

    if not req.input:
        raise HTTPException(400, "Field 'input' must be a non-empty list.")

    n_expected = len(FEATURE_NAMES)
    if any(len(row) != n_expected for row in req.input):
        raise HTTPException(
            400,
            f"Each row must have {n_expected} values in this order: {FEATURE_NAMES}"
        )

    X = pd.DataFrame(req.input, columns=FEATURE_NAMES)

    try:
        y = model.predict(X)
        return {"prediction": [round(float(v), 2) for v in y]}
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")
