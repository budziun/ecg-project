# 🫀 ECG Project - Setup Guide

A quick guide to get your ECG Project running smoothly.  
Follow the steps below to set up your environment and verify that everything works.

---

## ⚙️ Prerequisites
- **Python**: 3.11 or 3.12 (recommended)
- **Git**: optional, but useful for cloning the repository

---

## 🚀 Setup Instructions

### 1️⃣ Clone or Download the Project
```bash
git clone <repo-url>
cd ECG_Project
```

If you don't use Git, you can download the ZIP from the repository page and extract it manually.

---

### 2️⃣ Create a Virtual Environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### 4️⃣ Verify Installation
To confirm that all dependencies are correctly installed:
```bash
python test_imports.py
```

If no errors appear, you’re good to go.

---

## 🧩 Troubleshooting

### ❗ Pandas installation fails on Windows + Python 3.13
**Solution:** Use Python **3.11** or **3.12** instead.

---

### ❗ PyTorch not installing
**Solution:** Install the CPU version manually:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### ❗ Import errors
**Solution:** Make sure your virtual environment is activated.  
You should see `(venv)` at the beginning of your terminal line.

---

## 🗂️ Project Structure
```
ECG_Project/
├── venv/            # Virtual environment (don't commit)
├── data/            # Dataset storage
├── models/          # Saved models
├── notebooks/       # Jupyter notebooks
├── requirements.txt # Dependencies
└── *.py             # Python scripts
```

---

## 👥 Team
- **Maciej** — Project Manager  
- **Adam** — QA / UX  
- **Jakub** — AI Engineer
