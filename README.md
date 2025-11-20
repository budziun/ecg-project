# ECG Arrhythmia Classifier

An AI-powered web application for automatic detection and classification of cardiac arrhythmias from ECG signals using deep learning models trained on the MIT-BIH Arrhythmia Database.

##  Project Overview

This project implements a hybrid **CNN-LSTM deep neural network** architecture to identify five different types of heartbeat patterns from ECG signals:

- **0 - Normal** - Standard normal heartbeat
- **1 - Supraventricular** - Premature atrial/nodal contractions
- **2 - Ventricular** - Premature ventricular contractions
- **3 - Fusion** - Fusion of ventricular and normal beats
- **4 - Unknown** - Unclassifiable or noise beats

##  Project Structure

Project Structure:
- ecg-project/
    - frontend/
        - src/
            - components/
                - AboutModal.tsx        # Project modal
                - UploadECG.tsx         # CSV upload component
                - ResultDisplay.tsx     # Model predictions
                - TestSamples.tsx       # Test samples gallery
                - ...
            - services/
                - api.ts                # Backend API integration
            - public/
                - qm.png                # Quality metrics visualization
    - backend/
        - main.py                  # API routes
        - inference.py             # Model inference
        - model_definition.py      # CNN-LSTM architecture
        - train_knn.py             # Training script
        - data_get.py              # Dataset prep
        - ltsm_cnn.py              # Training pipeline
        - requirements.txt         # Dependencies
        - models/
            - best_model.pth         # Trained model
            - scaler.pkl             # Scaler for normalization
    - results/
        - confusion_matrix.png
        - training_curves.png
    - test_samples_csv/ # 100 raw csv file to test upload function
    - docker-compose.yml
    - README.md
    - SETUP.md
    - data_get.py - file to download and split dataset
    - ltsm_cnn.py - file with ltsm cnn training 
    - model_definition.py
    - requirements.txt
    - test_imports.py # test requirments install 
    - train_knn.py # simple knn model baseline

##  Machine Learning Architecture

### Dataset: MIT-BIH Arrhythmia Database

- **Source:** [Kaggle MIT-BIH Heartbeat Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- **Size:** 48 half-hour recordings from 47 subjects
- **Samples:** 110,000+ annotated heartbeats
- **Signal Format:** 2-channel ambulatory ECG recordings
- **Preprocessing:** 187-point ECG window per heartbeat

### Model Architecture: CNN-LSTM Deep Network

The system employs a sophisticated hybrid deep learning architecture:

Input (batch, 1, 187 time-steps)

↓

[Conv1d: 64 filters, kernel=7] → BatchNorm → MaxPool

↓

[Conv1d: 128 filters, kernel=5] → BatchNorm → MaxPool

↓

[Conv1d: 256 filters, kernel=3] → BatchNorm → MaxPool

↓

[Conv1d: 512 filters, kernel=3] → BatchNorm

↓

Bidirectional LSTM (192 hidden units, 2 layers)

↓

Attention Mechanism (weighted context aggregation)

↓

Dense Layers (256 → 128 → 64) with BatchNorm & Dropout

↓

Output Classification (5 classes)


**Key Features:**

- **Convolutional Layers** - Extract spatial and temporal features from raw ECG signals
- **LSTM Layers** - Capture long-term dependencies and temporal patterns
- **Batch Normalization** - Stabilize training and accelerate convergence
- **Attention Mechanism** - Focus on important time steps in the sequence
- **Dropout Regularization** - Prevent overfitting (p=0.45)
- **Data Augmentation** - Handle class imbalance with aggressive augmentation techniques:
    - Time warping
    - Jittering
    - Scaling
    - Rolling shifts

### Training Pipeline (`ltsm_cnn.py`)

1. **Data Preparation** (`data_get.py`):
    - Load MIT-BIH CSV files
    - Split into train (85%) / validation (15%) / test
    - Reshape to (batch, 1, 187) for CNN input
    - Stratified split to maintain class distribution

2. **Normalization**:
    - StandardScaler applied per training set
    - Fit parameters saved to `models/scaler.pkl` for inference
    - Prevents data leakage between splits

3. **Imbalanced Learning Handling**:
    - Weighted random sampling based on class frequency
    - Focal Loss (γ=2.5) to down-weight easy negatives
    - Class weights computed as: `weight[i] = (1 / count[i]) ^ 0.45`

4. **Training**:
    - Optimizer: AdamW (lr=0.001, weight_decay=2e-5)
    - Learning rate scheduler: ReduceLROnPlateau
    - Early stopping: patience=10 epochs
    - Loss: Focal Loss with class weighting
    - Gradient clipping: max_norm=1.0

5. **Evaluation Metrics**:
    - Accuracy
    - F1-Score (macro & weighted) - handles class imbalance
    - Confusion Matrix
    - Per-class precision, recall, F1

### Results

Training produces:
- Best model checkpoint: `models/best_model.pth`
- Confusion matrix visualization
- Training curves (loss, accuracy, F1 score)
- Per-class classification metrics

#### Sample Output Visualizations

**Confusion Matrix**
![Confusion Matrix](https://github.com/budziun/ecg-project/blob/main/results/confusion_matrix.png?raw=true)

**Training Curves**
![Training Curves](https://github.com/budziun/ecg-project/blob/main/results/training_curves.png?raw=true)

**Quality Metrics - Normal vs Abnormal**
![Quality Metrics](https://github.com/budziun/ecg-project/blob/main/frontend/public/qm.png?raw=true)

##  Quick Start

### Prerequisites

- Docker & Docker Compose
- Git

### Installation & Running

Clone repository
git clone https://github.com/budziun/ecg-project.git
cd ecg-project

Start with Docker
docker-compose up --build

Access applications
Frontend: http://localhost:3000
API Docs: http://localhost:8000/docs
text

For detailed setup instructions, see [SETUP.md](SETUP.md).

##  Technology Stack

**Machine Learning:**
- PyTorch - Deep learning framework
- NumPy, Pandas - Data manipulation
- scikit-learn - Preprocessing & metrics

**Backend:**
- FastAPI - High-performance API
- Swagger - Interactive API documentation
- Docker - Containerization

**Frontend:**
- React + TypeScript - Web application
- Tailwind CSS - Styling
- Recharts - Data visualization

##  Features

### Web Application

✅ **CSV Upload** - Upload ECG signal CSV files for classification

✅ **Visualization** - View ECG signals and prediction results

✅ **Test Samples** - Try with pre-loaded test ECG samples

##  Team

- **Maciej Świder** - [GitHub](https://github.com/MacSwider) | Project Manager, Data Scientist
- **Jakub Budzich** - [GitHub](https://github.com/budziun) | ML/Web Engineer, Tech Lead
- **Adam Czaplicki** - [GitHub](https://github.com/AdamCzp) | UX Designer, QA Specialist

##  References

1. **Dataset:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
2. **Kaggle:** [Heartbeat Classification Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
3. **Architecture:** CNN-LSTM hybrid for time-series medical signal classification
4. **Techniques:** Focal Loss, Attention Mechanisms, Data Augmentation, Weighted Sampling

##  Documentation

- **Setup Guide:** [SETUP.md](SETUP.md) - Installation and Docker instructions
- **ML Notebook:** [See Jupyter Notebook](ecg.ipynb) for complete ML workflow, data exploration, training details, and results analysis
- **API Docs:** Available at http://localhost:8000/docs when running



**University of Warmia and Mazury in Olsztyn • Computer Science • 2025**

