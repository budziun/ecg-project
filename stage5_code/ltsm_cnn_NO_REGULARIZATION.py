# ================== 0. IMPORTS ==================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# ================== 1. LOAD DATA ==================
print("Loading data...")
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')
X_val = np.load('../data/X_val.npy')
y_val = np.load('../data/y_val.npy')
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

print(f"Original Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Original class distribution (train): {np.bincount(y_train)}")

# ================== 1.2 LOAD PRE-FITTED SCALER ==================
print("\n" + "="*60)
print("LOADING SHARED SCALER (fitted on original data)")
print("="*60)

# ZAŁADUJ ISTNIEJĄCY SCALER:
scaler = joblib.load('../models/scaler.pkl')

# ✅ PRAWIDŁOWY RESHAPE - normalizuj i reshape prawidłowo
X_train_normalized = scaler.transform(X_train.reshape(-1, 187)).reshape(-1, 1, 187)
X_val_normalized = scaler.transform(X_val.reshape(-1, 187)).reshape(-1, 1, 187)
X_test_normalized = scaler.transform(X_test.reshape(-1, 187)).reshape(-1, 1, 187)

print(f"✓ Scaler loaded from '../models/scaler.pkl'")
print(f"\nScaler statistics (shared across all experiments):")
print(f"  Mean: {scaler.mean_[0]:.4f}")
print(f"  Std: {scaler.scale_[0]:.4f}")

print(f"\nData shapes after normalization:")
print(f"  X_train: {X_train_normalized.shape}")
print(f"  X_val: {X_val_normalized.shape}")
print(f"  X_test: {X_test_normalized.shape}")

# ================== 1.5 NO DATA AUGMENTATION ==================
print("\n" + "=" * 60)
print("⚠️ NO DATA AUGMENTATION - Using IMBALANCED original data")
print("=" * 60)

print(f"\nTraining set class distribution (IMBALANCED):")
class_dist = np.bincount(y_train)
for idx, count in enumerate(class_dist):
    print(f"  Class {idx}: {count} samples ({100 * count / len(y_train):.1f}%)")

max_count = np.max(class_dist)
min_count = np.min(class_dist)
print(f"Class imbalance ratio: {max_count / min_count:.1f}x")

# ================== 2. PREPARE DATA ==================
X_train_tensor = torch.FloatTensor(X_train_normalized)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val_normalized)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test_normalized)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# ❌ NO WEIGHTED SAMPLING - pure random
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"\nDataLoader created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")


# ================== 3. MODEL WITHOUT REGULARIZATION ==================
class ECG_CNN_LSTM_NO_REG(nn.Module):
    def __init__(self, num_classes=5):
        super(ECG_CNN_LSTM_NO_REG, self).__init__()

        # ❌ NO DROPOUT, NO BATCHNORM

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)

        self.lstm = nn.LSTM(input_size=512, hidden_size=192, num_layers=2,
                            batch_first=True, bidirectional=True)

        self.attention = nn.Linear(384, 1)

        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.relu(self.conv4(x))

        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)

        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        x = self.relu(self.fc1(context))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x


# ================== 4. SETUP ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

torch.manual_seed(42)
np.random.seed(42)

model = ECG_CNN_LSTM_NO_REG(num_classes=5)
model = model.to(device)

# ❌ NO FOCAL LOSS, NO CLASS WEIGHTS - use standard CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# ❌ NO WEIGHT DECAY (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)


# ================== 5. TRAINING FUNCTIONS ==================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # ❌ NO GRADIENT CLIPPING

        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1_macro


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted, all_labels, all_preds


# ================== 6. TRAINING LOOP ==================
print("\n" + "=" * 80)
print("⚠️ Training Model WITHOUT REGULARIZATION (NO AUGMENTATION - IMBALANCED)")
print("=" * 80 + "\n")

num_epochs = 50
best_f1 = 0.0

train_losses, train_accs, train_f1s = [], [], []
val_losses, val_accs, val_f1s = [], [], []

start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate(model, val_loader, criterion, device)

    if val_f1_macro > best_f1:
        best_f1 = val_f1_macro
        torch.save(model.state_dict(), 'models/no_reg_no_aug/best_model.pth')

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_f1s.append(train_f1)

    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1_macro)

    # ✅ PRINT EVERY EPOCH
    print(f"Epoch [{epoch + 1:02d}/{num_epochs}]")
    print(f"  Train: Loss={train_loss:.4f} | Acc={train_acc:.4f} | F1={train_f1:.4f}")
    print(
        f"  Val:   Loss={val_loss:.4f} | Acc={val_acc:.4f} | F1={val_f1_macro:.4f} | F1_weighted={val_f1_weighted:.4f}")
    print()

training_time = time.time() - start_time
print(f"Total training time: {training_time:.2f}s ({training_time / 60:.1f} minutes)")

# ================== 7. FINAL EVALUATION ==================
print("\n" + "=" * 80)
print("Final Evaluation on Test Set (WITHOUT REGULARIZATION - NO AUGMENTATION)")
print("=" * 80 + "\n")

model.load_state_dict(torch.load('models/no_reg_no_aug/best_model.pth'))
test_loss, test_acc, test_f1_macro, test_f1_weighted, y_true, y_pred = evaluate(
    model, test_loader, criterion, device)

print(f"Test Results:")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  F1 (macro): {test_f1_macro:.4f}")
print(f"  F1 (weighted): {test_f1_weighted:.4f}")

print("\nPer-class Classification Report:")
print(classification_report(y_true, y_pred,
                            target_names=['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown'],
                            zero_division=0))

# ================== 8. VISUALIZATIONS ==================
os.makedirs('results/no_reg_no_aug', exist_ok=True)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Normal', 'Supraventr', 'Ventricular', 'Fusion', 'Unknown'],
            yticklabels=['Normal', 'Supraventr', 'Ventricular', 'Fusion', 'Unknown'])
plt.title('Confusion Matrix - NO REGULARIZATION (NO AUGMENTATION)', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('results/no_reg_no_aug/confusion_matrix.png', dpi=300)
print("\n✓ Confusion matrix saved")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, label='Train', linewidth=2, color='blue')
axes[0].plot(val_losses, label='Validation', linewidth=2, color='red')
axes[0].set_title('Loss (NO REG - NO AUG)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Train', linewidth=2, color='blue')
axes[1].plot(val_accs, label='Validation', linewidth=2, color='red')
axes[1].set_title('Accuracy (NO REG - NO AUG)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

axes[2].plot(train_f1s, label='Train', linewidth=2, color='blue')
axes[2].plot(val_f1s, label='Validation', linewidth=2, color='red')
axes[2].set_title('F1 Score (NO REG - NO AUG)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('F1 Score', fontsize=12)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/no_reg_no_aug/training_curves.png', dpi=300)
print("✓ Training curves saved")

# ================== 9. SAVE METRICS ==================
import json

metrics = {
    'train_acc': [float(x) for x in train_accs],
    'val_acc': [float(x) for x in val_accs],
    'train_f1': [float(x) for x in train_f1s],
    'val_f1': [float(x) for x in val_f1s],
    'test_acc': float(test_acc),
    'test_f1': float(test_f1_macro),
    'best_val_f1': float(best_f1)
}

with open('results/no_reg_no_aug/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✓ Metrics saved to 'results/no_reg_no_aug/metrics.json'")
print("\n✅ Training WITHOUT REGULARIZATION (NO AUGMENTATION) completed!")
