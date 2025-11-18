from pydantic import BaseModel, Field
from typing import List, Optional

class ECGSignal(BaseModel):
    signal: List[float] = Field(..., description="ECG signal values")
    description: Optional[str] = Field(default="ECG signal")

class PredictionResponse(BaseModel):
    Normal: float
    Supraventricular: float
    Ventricular: float
    Fusion: float
    Unknown: float
    predicted_class: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
