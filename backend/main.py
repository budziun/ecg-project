"""
FastAPI application for ECG Arrhythmia Classification
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import ECGSignal, PredictionResponse, HealthResponse
from inference import load_model, predict_ecg, is_model_loaded, scaler
import logging
import pandas as pd
import numpy as np
import io
import random
from pathlib import Path
import os

DATA_PATH = Path(os.getenv('DATA_PATH', './data'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ECG Arrhythmia Classifier",
    description="API dla klasyfikacji zaburzeń rytmu serca z sygnału EKG (z normalizacją)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== STARTUP ==================

@app.on_event("startup")
async def startup_event():
    """Load model and scaler when app starts"""
    logger.info("🚀 Starting up ECG Classifier API...")
    logger.info("Loading model and scaler...")
    success = load_model()
    if success:
        logger.info("✅ API ready for predictions (with normalization)")
    else:
        logger.error("❌ Failed to load model/scaler - check paths")

# ================== HEALTH CHECK ==================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": is_model_loaded()
    }

@app.get("/", tags=["Info"])
async def root():
    return {
        "message": "ECG Arrhythmia Classifier API",
        "version": "1.0.0",
        "docs": "http://localhost:8000/docs",
        "model_loaded": is_model_loaded(),
        "note": "All signals are normalized using StandardScaler before prediction"
    }

# ================== SINGLE PREDICTION ==================

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(ecg_data: ECGSignal):
    """
    Classify single ECG signal
    - Signal will be automatically padded/truncated to 187 samples
    - Signal will be normalized using the trained scaler
    """
    try:
        if not is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please try again later."
            )

        if len(ecg_data.signal) == 0:
            raise HTTPException(
                status_code=400,
                detail="Signal cannot be empty"
            )

        result = predict_ecg(ecg_data.signal)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        logger.info(f"✅ Prediction: {result['predicted_class']} (confidence: {result['confidence']:.2%})")
        return result

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"❌ Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ================== FILE UPLOAD - CSV ==================
# still under coding - not working 5.11.20252
@app.post("/upload-csv", tags=["File Upload"])
async def upload_csv_file(file: UploadFile = File(...)):
    try:
        if not is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")

        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents), header=None)
        except:
            raise HTTPException(status_code=400, detail="Invalid CSV format")

        if df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        signal_values = df.iloc[0].values.astype(np.float32)
        signal_values = signal_values[~np.isnan(signal_values)]

        if len(signal_values) == 0:
            raise HTTPException(status_code=400, detail="No valid signal data found")

        logger.info(f"📥 CSV Upload - Raw signal length: {len(signal_values)}")
        logger.info(f"   Raw: Min={signal_values.min():.4f}, Max={signal_values.max():.4f}")

        trailing_zeros = 0
        for i in range(len(signal_values) - 1, -1, -1):
            if signal_values[i] == 0:
                trailing_zeros += 1
            else:
                break

        if trailing_zeros > 50:
            signal_for_plot = signal_values[:-trailing_zeros]
        else:
            signal_for_plot = signal_values.copy()

        signal_values = signal_values.copy()
        logger.info(f"   After cleanup: length={len(signal_values)}, Min={signal_values.min():.4f}, Max={signal_values.max():.4f}")

        # ===== LENGTH ADJUSTMENT =====
        target_length = 187
        if len(signal_values) < target_length:
            padding_length = target_length - len(signal_values)
            signal_values = np.pad(signal_values, (0, padding_length), mode='constant', constant_values=0)
            logger.info(f"   ✅ Padded with {padding_length} zeros for model")
        elif len(signal_values) > target_length:
            signal_values = signal_values[:target_length]
            logger.info(f"   ✅ Truncated to {target_length}")

        # ===== NORMALIZATION =====
        logger.info(f"   Before scaler: Min={signal_values.min():.4f}, Max={signal_values.max():.4f}, Mean={signal_values.mean():.4f}")

        signal_array = signal_values.reshape(1, -1)
        signal_normalized = scaler.transform(signal_array)[0]

        logger.info(f"   After scaler: Min={signal_normalized.min():.4f}, Max={signal_normalized.max():.4f}, Mean={signal_normalized.mean():.4f}, Std={signal_normalized.std():.4f}")

        if np.any(np.isnan(signal_normalized)):
            raise HTTPException(status_code=400, detail="Normalization failed")

        signal_reshaped = signal_normalized.reshape(187, 1)

        # ===== PREDICTION =====
        result = predict_ecg(signal_reshaped)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        result['signal_for_plot'] = signal_for_plot.tolist()
        result['signal_normalized'] = signal_normalized.tolist()

        logger.info(f"✅ Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
        logger.info(f"   Plot signal: {len(signal_for_plot)} samples (bez padding)")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# ================== TEST DATA SAMPLES ==================

@app.get("/test-samples", tags=["Test Data"])
async def get_test_samples(count: int = 5):
    """
    Get random samples from test dataset (X_test.npy, y_test.npy) with predictions
    Test data will be normalized using the trained scaler
    """
    try:
        if not is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Load test data
        X_test_path = DATA_PATH / 'X_test.npy'
        y_test_path = DATA_PATH / 'y_test.npy'

        if not X_test_path.exists() or not y_test_path.exists():
            raise FileNotFoundError("Test data files not found in ../data/")

        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

        # Get random samples
        max_count = min(count, len(X_test))
        indices = random.sample(range(len(X_test)), max_count)

        samples = []
        class_map = {
            0: 'Normal',
            1: 'Supraventricular',
            2: 'Ventricular',
            3: 'Fusion',
            4: 'Unknown'
        }

        for idx in indices:
            signal = X_test[idx].tolist()
            label = int(y_test[idx])

            # Predict (with normalization)
            result = predict_ecg(signal)
            result['true_label'] = class_map.get(label, 'Unknown')
            result['true_label_id'] = label
            result['index'] = int(idx)
            result['is_correct'] = result['predicted_class'] == result['true_label']
            samples.append(result)

        logger.info(f"✅ Generated {len(samples)} test samples")
        return {"samples": samples, "count": len(samples)}

    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Test samples error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================== BATCH PREDICTIONS - CSV ==================
@app.post("/batch-predict-csv", tags=["Batch"])
async def batch_predict_csv(file: UploadFile = File(...)):
    try:
        if not is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")

        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), header=None)

        if df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        results = []
        correct_count = 0

        class_map = {
            0: 'Normal',
            1: 'Supraventricular',
            2: 'Ventricular',
            3: 'Fusion',
            4: 'Unknown'
        }

        for idx in range(len(df)):
            try:
                signal = df.iloc[idx, :-1].values.astype(np.float32)
                label = int(df.iloc[idx, -1])

                result = predict_ecg(signal)
                result['true_label'] = class_map.get(label, 'Unknown')
                result['true_label_id'] = label
                result['index'] = int(idx)

                is_correct = result['predicted_class'] == result['true_label']
                result['is_correct'] = is_correct

                if is_correct:
                    correct_count += 1

                results.append(result)
            except Exception as e:
                logger.warning(f"⚠️  Sample {idx} failed: {e}")
                continue

        accuracy = correct_count / len(results) if len(results) > 0 else 0
        logger.info(f"✅ Batch prediction: {len(results)} samples, accuracy: {accuracy:.2%}")

        return {
            "predictions": results,
            "count": len(results),
            "accuracy": accuracy,
            "correct": correct_count
        }

    except Exception as e:
        logger.error(f"❌ Batch CSV error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ================== EXCEPTION HANDLER ==================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# ================== RUN ==================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
