# inference.py
import torch
import numpy as np
import joblib
import os
from pathlib import Path
import logging

# Importujemy jedną, spójną definicję modelu
from model_definition import ECG_CNN_LSTM_Deep

# Konfiguracja ścieżek i urządzenia
MODEL_PATH = os.getenv('MODEL_PATH', './models/best_model.pth')
SCALER_PATH = os.getenv('SCALER_PATH', './models/scaler.pkl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Globalne zmienne
model = None
scaler = None
logger = logging.getLogger(__name__)

def load_model():
    """Wczytuje model i scaler z zapisanych plików."""
    global model, scaler
    try:
        model = ECG_CNN_LSTM_Deep(num_classes=5, dropout=0.45)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        scaler = joblib.load(SCALER_PATH)

        logger.info(f"✓ Model wczytany z {MODEL_PATH}")
        logger.info(f"✓ Scaler wczytany z {SCALER_PATH}")
        return True
    except Exception as e:
        logger.error(f"✗ Błąd wczytywania modela/scalera: {e}")
        import traceback
        traceback.print_exc()
        return False

def is_model_loaded():
    """Sprawdza, czy model i scaler są wczytane."""
    return model is not None and scaler is not None

def predict_ecg(signal: list[float], uncertainty_threshold: float = 0.6):
    """
    Klasyfikuje sygnał EKG.
    Ta funkcja jest sercem logiki predykcji i jest używana przez wszystkie endpointy.
    """
    if model is None or scaler is None:
        logger.error("Próba predykcji przy nie wczytanym modelu.")
        return {"error": "Model not loaded"}

    try:
        # 1. Konwersja i wyrównanie długości do 187 próbek
        signal_array = np.array(signal, dtype=np.float32)
        if len(signal_array) < 187:
            signal_array = np.pad(signal_array, (0, 187 - len(signal_array)), mode='constant')
        elif len(signal_array) > 187:
            signal_array = signal_array[:187]

        # 2. Normalizacja
        signal_array = scaler.transform(signal_array.reshape(1, -1))[0]

        # 3. Przygotowanie tensora o poprawnym kształcie (1, 1, 187)
        signal_tensor = torch.from_numpy(signal_array).float().to(device)
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(1)  # Kształt: (1, 1, 187)

        # 4. Predykcja
        with torch.no_grad():
            outputs = model(signal_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        # 5. Formatowanie wyniku
        class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        is_uncertain = confidence < uncertainty_threshold

        return {
            "Normal": float(probabilities[0]),
            "Supraventricular": float(probabilities[1]),
            "Ventricular": float(probabilities[2]),
            "Fusion": float(probabilities[3]),
            "Unknown": float(probabilities[4]),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "threshold": uncertainty_threshold
        }

    except Exception as e:
        logger.error(f"BŁĄD w predict_ecg: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Wczytaj model przy starcie aplikacji
load_model()