import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path
import os

MODEL_PATH = os.getenv('MODEL_PATH', './models/best_model.pth')
SCALER_PATH = os.getenv('SCALER_PATH', './models/scaler.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None
scaler = None

class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)

        self.lstm = nn.LSTM(input_size=512, hidden_size=192, num_layers=2,
                            batch_first=True, dropout=0.45, bidirectional=True)

        self.attention = nn.Linear(384, 1)

        self.dropout1 = nn.Dropout(0.45)
        self.fc1 = nn.Linear(384, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)

        self.dropout2 = nn.Dropout(0.45)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.dropout3 = nn.Dropout(0.45)
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)

        self.dropout4 = nn.Dropout(0.45)
        self.fc4 = nn.Linear(64, 5)

        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch, samples, channels) = (1, 187, 1)
        x = x.transpose(1, 2)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4(self.conv4(x)))

        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)

        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        x = self.dropout1(context)
        x = self.relu(self.bn_fc1(self.fc1(x)))

        x = self.dropout2(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))

        x = self.dropout3(x)
        x = self.relu(self.bn_fc3(self.fc3(x)))

        x = self.dropout4(x)
        x = self.fc4(x)

        return x


def load_model():
    """Load model and scaler from saved files"""
    global model, scaler
    try:
        model = ECGModel()
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        scaler = joblib.load(SCALER_PATH)

        print(f"✓ Model loaded from {MODEL_PATH}")
        print(f"✓ Scaler loaded from {SCALER_PATH}")
        return True
    except Exception as e:
        print(f"✗ Error loading model/scaler: {e}")
        import traceback
        traceback.print_exc()
        return False


def is_model_loaded():
    return model is not None and scaler is not None


def predict_ecg(signal, uncertainty_threshold=0.6):
    """Predict ECG class from signal"""
    if model is None or scaler is None:
        return {"error": "Model not loaded"}

    try:
        signal_array = np.array(signal, dtype=np.float32)

        if len(signal_array) < 187:
            signal_array = np.pad(signal_array, (0, 187 - len(signal_array)), mode='constant')
        elif len(signal_array) > 187:
            signal_array = signal_array[:187]

        signal_array = scaler.transform(signal_array.reshape(1, -1))[0]

        # (1, 187, 1)
        signal_tensor = torch.from_numpy(signal_array).float().to(device)
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(-1)

        print(f"DEBUG: Signal tensor shape: {signal_tensor.shape}")

        with torch.no_grad():
            outputs = model(signal_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        is_uncertain = confidence < uncertainty_threshold

        print(f"DEBUG: Prediction: {predicted_class}, Confidence: {confidence:.4f}")

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
        print(f"ERROR in predict_ecg: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Load model on startup
load_model()