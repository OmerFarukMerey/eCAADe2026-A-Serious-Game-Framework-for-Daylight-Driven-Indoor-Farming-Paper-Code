"""FastAPI server for GHR/DLI prediction using XGBoost ONNX model."""

import math
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal

# --- Constants ---
DLI_RATIO = 0.00726460
MODEL_PATH = "models/unity_export/daylight_model.onnx"

# --- Load ONNX model once at startup ---
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# --- Enums ---
class Orientation(str, Enum):
    N = "N"
    NE = "NE"
    NW = "NW"
    E = "E"
    S = "S"
    SE = "SE"
    W = "W"

class Level(str, Enum):
    G = "G"
    M = "M"
    U = "U"

# --- Request / Response ---
class PredictRequest(BaseModel):
    day_of_year: int = Field(..., ge=1, le=365)
    orientation: Orientation
    level: Level
    wwr: float = Field(..., ge=0.0, le=1.0)
    window_transmittance: Literal[0.1, 0.6, 0.9]
    tree_width_m: Literal[3, 6]
    tree_present: bool

class PredictResponse(BaseModel):
    GHR: float
    DLI: float
    season: str

# --- Feature encoding ---
def encode_features(r: PredictRequest) -> np.ndarray:
    day = r.day_of_year
    tree_int = int(r.tree_present)

    # Cyclical
    day_sin = math.sin(2 * math.pi * day / 365)
    day_cos = math.cos(2 * math.pi * day / 365)

    # Season
    if day <= 59 or day >= 335:
        season_num = 0
    elif day <= 151:
        season_num = 1
    elif day <= 243:
        season_num = 2
    else:
        season_num = 3

    season_sin = math.sin(2 * math.pi * season_num / 4)
    season_cos = math.cos(2 * math.pi * season_num / 4)

    # Interactions
    effective_trans = r.window_transmittance * (1 - tree_int * 0.9)
    wwr_x_trans = r.wwr * r.window_transmittance
    tree_effect = r.tree_width_m * tree_int

    # One-hot (no Building — model generalizes via building properties)
    ori = [r.orientation == o for o in ["E", "N", "NE", "NW", "S", "SE", "W"]]
    lvl = [r.level == l for l in ["G", "M", "U"]]

    features = [
        day, tree_int, r.window_transmittance, r.wwr, r.tree_width_m,
        day_sin, day_cos, season_sin, season_cos,
        *ori, *lvl,
        effective_trans, wwr_x_trans, tree_effect,
    ]
    return np.array([features], dtype=np.float32)


SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}

def get_season(day: int) -> str:
    if day <= 59 or day >= 335:
        return "Winter"
    elif day <= 151:
        return "Spring"
    elif day <= 243:
        return "Summer"
    return "Autumn"

# --- App ---
app = FastAPI(title="P.L.A.N.T.S. Daylight Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    features = encode_features(req)
    ghr = float(session.run(None, {input_name: features})[0].flatten()[0])
    return PredictResponse(GHR=round(ghr, 2), DLI=round(ghr * DLI_RATIO, 4), season=get_season(req.day_of_year))

@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "features": 22}
