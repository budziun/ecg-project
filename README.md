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
    - ecg.ipynb # Complete ML training notebook
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

### Training the Model

#### Option 1: Use Jupyter Notebook (Recommended) 

The complete end-to-end ML pipeline is available in **`ecg.ipynb`**:

Install Jupyter
pip install jupyter

Launch notebook
jupyter notebook ecg.ipynb

**What's included in the notebook:**
- ✅ Data download from Kaggle API
- ✅ Preprocessing & normalization
- ✅ Data augmentation for class imbalance
- ✅ KNN baseline model training
- ✅ CNN-LSTM model training
- ✅ Comprehensive evaluation on train/val/test sets
- ✅ Visualizations (ROC curves, confusion matrices, PR curves)
- ✅ Detailed performance metrics and analysis

**Simply run all cells** to reproduce the complete training pipeline from scratch!

#### Option 2: Use Python Scripts

**1. Data Preparation** (`data_get.py`):
python data_get.py

- Downloads MIT-BIH dataset from Kaggle
- Splits into train (85%) / validation (15%) / test
- Reshapes to (batch, 1, 187) for CNN input
- Saves to `data/` directory

**2. Train KNN Baseline** (`train_knn.py`):
python train_knn.py

- Trains simple K-Nearest Neighbors classifier
- Provides baseline metrics for comparison

**3. Train CNN-LSTM Model** (`ltsm_cnn.py`):
python ltsm_cnn.py

- Trains deep CNN-LSTM model
- Saves best model to `models/best_model.pth`
- Generates visualizations in `results/`

### Training Configuration

- **Normalization:** StandardScaler fitted on training data
- **Scaler:** Saved to `models/scaler.pkl` for inference
- **Loss Function:** Focal Loss (γ=2.5) with class weighting
- **Optimizer:** AdamW (lr=0.001, weight_decay=2e-5)
- **Scheduler:** ReduceLROnPlateau (patience=5)
- **Early Stopping:** Patience=10 epochs
- **Regularization:** Dropout (0.45), Gradient clipping (max_norm=1.0)

### Results

**Final Model Performance:**

| Dataset | Accuracy | F1 (macro) | F1 (weighted) |
|---------|----------|------------|---------------|
| Train | 99.65% | 99.50% | 99.65% |
| Validation | 98.80% | 93.21% | 98.81% |
| Test | **98.67%** | **92.74%** | **98.67%** |

**KNN Baseline Performance:**

| Dataset | Accuracy | F1 (macro) | F1 (weighted) |
|---------|----------|------------|---------------|
| Test | 97.43% | 86.72% | 97.31% |

** CNN-LSTM achieves 1.24% accuracy improvement and 6.02% macro F1 improvement over KNN baseline!**

#### Sample Output Visualizations

**Confusion Matrix**
![Confusion Matrix](https://github.com/budziun/ecg-project/blob/main/results/confusion_matrix.png?raw=true)

**Training Curves**
![Training Curves](https://github.com/budziun/ecg-project/blob/main/results/training_curves.png?raw=true)

**Precision Recall Curves**
![Precision Recall Curves](https://github.com/budziun/ecg-project/blob/main/results/recal.png?raw=true)

**ROC Curves**
![ROC Curves](https://github.com/budziun/ecg-project/blob/main/results/roc.png?raw=true)

**Quality Metrics - Normal vs Abnormal**
![Quality Metrics](https://github.com/budziun/ecg-project/blob/main/frontend/public/qm.png?raw=true)

## 📸 Application Screenshots

### Main Interface - upload ecg signal via csv or mit-bh test record
![Main Interface](https://github.com/budziun/ecg-project/blob/main/results/ecg_main.png?raw=true)

### Classification Results

**Normal Heartbeat Detection**
![Normal Classification](https://github.com/budziun/ecg-project/blob/main/results/ecg_normal.png?raw=true)

**Ventricular Arrhythmia Detection**
![Ventricular Classification](https://github.com/budziun/ecg-project/blob/main/results/ecg_ventricular.png?raw=true)

##  Quick Start of WEB app

### Prerequisites

- Docker & Docker Compose
- Git
- (Optional) Python 3.8+ for training from scratch

### Installation & Running

#### Run Web Application (Using Pre-trained Model)

Clone repository
git clone https://github.com/budziun/ecg-project.git
cd ecg-project

Start with Docker
docker-compose up --build

Access applications
Frontend: http://localhost:3000
API Docs: http://localhost:8000/docs

#### Train Model from Scratch

Option 1: Use Jupyter Notebook (Recommended)
pip install jupyter
jupyter notebook ecg.ipynb

Option 2: Use Python scripts
pip install -r requirements.txt
python data_get.py # Download & prepare data
python train_knn.py # Train KNN baseline
python ltsm_cnn.py # Train CNN-LSTM model

For detailed setup instructions, see [SETUP.md](SETUP.md).

##  Technology Stack

**Machine Learning:**
- PyTorch - Deep learning framework
- NumPy, Pandas - Data manipulation
- scikit-learn - Preprocessing & metrics
- Jupyter - Interactive development

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

### ML Development

✅ **Complete Pipeline** - End-to-end training in Jupyter Notebook

✅ **Baseline Comparison** - KNN vs CNN-LSTM performance

✅ **Comprehensive Metrics** - ROC, PR curves, confusion matrices

✅ **Production Ready** - Saved models + scaler for deployment

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
- **ML Notebook:** [ecg.ipynb](ecg.ipynb) - Complete ML workflow, training, and analysis
- **API Docs:** Available at http://localhost:8000/docs when running

---

**University of Warmia and Mazury in Olsztyn • Computer Science • 2025**



