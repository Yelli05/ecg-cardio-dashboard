import math
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from model import ECGCNN

MODEL_PATH = Path(__file__).parent / "ecg_model.pth"
N_STEPS = 187
CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular ectopic (S)",
    "Ventricular ectopic (V)",
    "Fusion (F)",
    "Unknown / paced (Q)",
]

ml = {}  # holds the loaded model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: load the model a single time, not per request
    model = ECGCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    ml["model"] = model
    yield
    ml.clear()


app = FastAPI(title="ECG Arrhythmia Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ECGRequest(BaseModel):
    values: list[float] = Field(..., min_length=N_STEPS, max_length=N_STEPS)

    @field_validator("values")
    @classmethod
    def must_be_finite(cls, v):
        if not all(math.isfinite(x) for x in v):
            raise ValueError("values must be finite numbers")
        return v


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml}


@app.post("/predict")
def predict(req: ECGRequest):
    x = torch.tensor(req.values, dtype=torch.float32).reshape(1, 1, N_STEPS)
    with torch.no_grad():
        probs = torch.softmax(ml["model"](x), dim=1)[0]
    idx = int(probs.argmax())
    return {
        "predicted_class": idx,
        "label": CLASS_NAMES[idx],
        "is_abnormal": idx != 0,
        "confidence": float(probs[idx]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }