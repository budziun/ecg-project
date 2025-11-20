# ================== 0. IMPORTS ==================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from model_definition import ECG_CNN_LSTM_Deep

# ================== 1. LOAD DATA ==================
print("Loading data...")
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

print(f"Original Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Original class distribution: {np.bincount(y_train)}")

# ================== 1.2 ADD NORMALIZATION ==================
print("\n" + "=" * 60)
print("NORMALIZING DATA")
print("=" * 60)

# Fit scaler on training data
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, 187)
X_train_normalized = scaler.fit_transform(X_train_flat).reshape(-1, 1, 187)

# Transform val and test using training statistics
X_val_normalized = scaler.transform(X_val.reshape(-1, 187)).reshape(-1, 1, 187)
X_test_normalized = scaler.transform(X_test.reshape(-1, 187)).reshape(-1, 1, 187)

# Save scaler for inference
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

print(f"✓ Scaler saved to 'models/scaler.pkl'")
print(f"\nNormalization statistics:")
print(f"  Mean: {scaler.mean_[0]:.4f}")
print(f"  Std: {scaler.scale_[0]:.4f}")

print(f"\nBefore normalization - Min: {X_train.min():.4f}, Max: {X_train.max():.4f}")
print(f"After normalization  - Min: {X_train_normalized.min():.4f}, Max: {X_train_normalized.max():.4f}")

X_train = X_train_normalized
X_val = X_val_normalized
X_test = X_test_normalized


# ================== 1.5 AGGRESSIVE DATA AUGMENTATION ==================
def augment_ecg_aggressive(X, y, target_class, n_augment):
    """More aggressive augmentation with multiple techniques"""
    class_mask = (y == target_class)
    class_samples = X[class_mask]
    augmented_X = []
    augmented_y = []

    for _ in range(n_augment):
        for sample in class_samples:
            sample_flat = sample.squeeze()
            aug_type = np.random.randint(0, 4)

            if aug_type == 0:
                alpha = np.random.uniform(0.85, 1.15)
                new_length = int(187 * alpha)
                warped = np.interp(np.linspace(0, 187, new_length), np.arange(187), sample_flat)
                if len(warped) < 187:
                    warped = np.pad(warped, (0, 187 - len(warped)), mode='edge')
                else:
                    warped = warped[:187]
                augmented = warped

            elif aug_type == 1:
                start = np.random.randint(0, 10)
                sliced = np.roll(sample_flat, -start)
                noise = np.random.normal(0, 0.04, 187)
                augmented = sliced + noise

            elif aug_type == 2:
                smooth = np.random.uniform(0.5, 2.0, 187)
                smooth = np.convolve(smooth, np.ones(10) / 10, mode='same')
                augmented = sample_flat * smooth

            else:
                alpha = np.random.uniform(0.9, 1.1)
                new_length = int(187 * alpha)
                warped = np.interp(np.linspace(0, 187, new_length), np.arange(187), sample_flat)
                if len(warped) < 187:
                    warped = np.pad(warped, (0, 187 - len(warped)), mode='edge')
                else:
                    warped = warped[:187]
                noise = np.random.normal(0, 0.03, 187)
                scale = np.random.uniform(0.9, 1.1)
                augmented = (warped + noise) * scale

            augmented_X.append(augmented.reshape(1, 187))
            augmented_y.append(target_class)

    return np.array(augmented_X), np.array(augmented_y)


print("\nPerforming data augmentation...")
aug_X_list, aug_y_list = [X_train], [y_train]

aug_X, aug_y = augment_ecg_aggressive(X_train, y_train, target_class=1, n_augment=6)
aug_X_list.append(aug_X)
aug_y_list.append(aug_y)
print(f" Supraventricular: +{len(aug_X)} samples")

aug_X, aug_y = augment_ecg_aggressive(X_train, y_train, target_class=3, n_augment=15)
aug_X_list.append(aug_X)
aug_y_list.append(aug_y)
print(f" Fusion: +{len(aug_X)} samples")

aug_X, aug_y = augment_ecg_aggressive(X_train, y_train, target_class=2, n_augment=3)
aug_X_list.append(aug_X)
aug_y_list.append(aug_y)
print(f" Ventricular: +{len(aug_X)} samples")

aug_X, aug_y = augment_ecg_aggressive(X_train, y_train, target_class=4, n_augment=1)
aug_X_list.append(aug_X)
aug_y_list.append(aug_y)
print(f" Unknown: +{len(aug_X)} samples")

X_train = np.vstack(aug_X_list)
y_train = np.hstack(aug_y_list)

print(f"\nAugmented training set: {X_train.shape}")
print(f"New class distribution: {np.bincount(y_train)}")

# ================== 2. PREPARE DATA ==================
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

class_counts = np.bincount(y_train)
class_weights = np.power(1.0 / class_counts, 0.45)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights_tensor = torch.FloatTensor(class_weights)

print(f"Class weights: {class_weights_tensor}")

sample_weights = [class_weights[int(label)] for label in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# ================== 2.5 FOCAL LOSS ==================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


# ================== 3. DEEP CNN-LSTM MODEL ==================
class ECG_CNN_LSTM_Deep(nn.Module):
    def __init__(self, num_classes=5, dropout=0.45):
        super(ECG_CNN_LSTM_Deep, self).__init__()

        # Input shape: (batch, 1, 187)
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
                            batch_first=True, dropout=dropout, bidirectional=True)

        self.attention = nn.Linear(384, 1)

        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(384, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)

        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.dropout3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)

        self.dropout4 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Input: (batch, 1, 187)
        # Conv1d preserves channel dimension

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4(self.conv4(x)))

        # Transpose for LSTM: (batch, 1, 512) -> (batch, 512, 1)? NO!
        # LSTM expects: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, length, 512)

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


# ================== 4. SETUP ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

torch.manual_seed(42)
np.random.seed(42)

model = ECG_CNN_LSTM_Deep(num_classes=5, dropout=0.45)
model = model.to(device)

class_weights_tensor = class_weights_tensor.to(device)
criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.5)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')

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
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1_macro, f1_weighted, all_labels, all_preds


# ================== 6. TRAINING LOOP ==================
print("\nTraining CNN-LSTM Deep Model WITH NORMALIZATION")
print("=" * 80)

num_epochs = 50
best_f1 = 0.0
patience_counter = 0
patience = 10

train_losses, train_accs, train_f1s = [], [], []
val_losses, val_accs, val_f1s = [], [], []

start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate(model, val_loader, criterion, device)

    scheduler.step(val_f1_macro)

    if val_f1_macro > best_f1:
        best_f1 = val_f1_macro
        torch.save(model.state_dict(), 'models/best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_f1s.append(train_f1)

    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1_macro)

    print(f"Epoch [{epoch + 1:02d}/{num_epochs}]")
    print(f" Train: Loss={train_loss:.4f} | Acc={train_acc:.4f} | F1={train_f1:.4f}")
    print(
        f" Val:   Loss={val_loss:.4f} | Acc={val_acc:.4f} | F1={val_f1_macro:.4f} | F1_weighted={val_f1_weighted:.4f}")
    print()

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

training_time = time.time() - start_time
print(f"Total training time: {training_time:.2f}s ({training_time / 60:.1f} minutes)")

# ================== 6_1. FINAL EVALUATION ==================
print("\n" + "=" * 80)
print("Final Evaluation on Test Set")
print("=" * 80 + "\n")

model.load_state_dict(torch.load('models/best_model.pth'))
test_loss, test_acc, test_f1_macro, test_f1_weighted, y_true, y_pred = evaluate(
    model, test_loader, criterion, device)

print(f"Test Results:")
print(f" Accuracy: {test_acc:.4f}")
print(f" F1 (macro): {test_f1_macro:.4f}")
print(f" F1 (weighted): {test_f1_weighted:.4f}")

print("\nPer-class Classification Report:")
print(classification_report(y_true, y_pred,
                            target_names=['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']))

# ================== 8. VISUALIZATIONS ==================
os.makedirs('results', exist_ok=True)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Supraventr', 'Ventricular', 'Fusion', 'Unknown'],
            yticklabels=['Normal', 'Supraventr', 'Ventricular', 'Fusion', 'Unknown'])
plt.title('Confusion Matrix - CNN-LSTM Deep Model (FIXED)', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300)
print("\nConfusion matrix saved as 'confusion_matrix.png'")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, label='Train', linewidth=2, color='blue')
axes[0].plot(val_losses, label='Validation', linewidth=2, color='red')
axes[0].set_title('Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Train', linewidth=2, color='blue')
axes[1].plot(val_accs, label='Validation', linewidth=2, color='red')
axes[1].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

axes[2].plot(train_f1s, label='Train', linewidth=2, color='blue')
axes[2].plot(val_f1s, label='Validation', linewidth=2, color='red')
axes[2].set_title('F1 Score (Macro)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('F1 Score', fontsize=12)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=300)
print("Training curves saved as 'training_curves.png'")

print("\nTraining completed.")
print(f"Best validation F1 (macro): {best_f1:.4f}")
print("Model saved as 'models/best_model.pth'")
print("Scaler saved as 'models/scaler.pkl'")
