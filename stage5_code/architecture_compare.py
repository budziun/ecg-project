# ================== ARCHITECTURE EXPERIMENTS ==================
# Testuje różne konfiguracje architektury i generuje PNG tabelkę
# Użycie: python architecture_experiments.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
import os
import json
import matplotlib.pyplot as plt


# ==================== FOCAL LOSS ====================
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


# ==================== ARCHITECTURE VARIANTS ====================

class ECG_CNN_LSTM_3CNN(nn.Module):
    """Config A: 3 CNN layers + 2 LSTM + Attention"""

    def __init__(self, num_classes=5, dropout=0.45):
        super(ECG_CNN_LSTM_3CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2, 2)

        # SKIP conv4!

        self.lstm = nn.LSTM(input_size=256, hidden_size=192, num_layers=2,
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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

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


class ECG_CNN_LSTM_1LSTM(nn.Module):
    """Config B: 4 CNN layers + 1 LSTM + Attention"""

    def __init__(self, num_classes=5, dropout=0.45):
        super(ECG_CNN_LSTM_1LSTM, self).__init__()
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

        # Only 1 LSTM layer!
        self.lstm = nn.LSTM(input_size=512, hidden_size=192, num_layers=1,
                            batch_first=True, dropout=0, bidirectional=True)

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


class ECG_CNN_LSTM_NoAttention(nn.Module):
    """Config C: 4 CNN + 2 LSTM (NO Attention)"""

    def __init__(self, num_classes=5, dropout=0.45):
        super(ECG_CNN_LSTM_NoAttention, self).__init__()
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

        # NO attention - use mean pooling

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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.relu(self.bn4(self.conv4(x)))

        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)

        # Mean pooling (NO attention)
        context = torch.mean(lstm_out, dim=1)

        x = self.dropout1(context)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout3(x)
        x = self.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout4(x)
        x = self.fc4(x)
        return x


class ECG_CNN_LSTM_Full(nn.Module):
    """Config D: Full model (4CNN + 2LSTM + Attention) - BASELINE"""

    def __init__(self, num_classes=5, dropout=0.45):
        super(ECG_CNN_LSTM_Full, self).__init__()
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


# ==================== TRAINING FUNCTIONS ====================
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

    return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds,
                                                                                     average='macro')


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

    return (total_loss / len(loader), accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average='macro'),
            f1_score(all_labels, all_preds, average='weighted'))


# ==================== MAIN ====================
print("=" * 80)
print("ARCHITECTURE EXPERIMENTS")
print("=" * 80)

# Load data
print("\nLoading data...")
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# Normalize
import joblib

scaler = joblib.load('models/scaler.pkl')
X_train = scaler.transform(X_train.reshape(-1, 187)).reshape(-1, 1, 187)
X_val = scaler.transform(X_val.reshape(-1, 187)).reshape(-1, 1, 187)
X_test = scaler.transform(X_test.reshape(-1, 187)).reshape(-1, 1, 187)


# Augment
def augment_ecg_aggressive(X, y, target_class, n_augment):
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
                augmented = warped[:187] if len(warped) >= 187 else np.pad(warped, (0, 187 - len(warped)), mode='edge')
            elif aug_type == 1:
                augmented = np.roll(sample_flat, -np.random.randint(0, 10)) + np.random.normal(0, 0.04, 187)
            elif aug_type == 2:
                smooth = np.random.uniform(0.5, 2.0, 187)
                smooth = np.convolve(smooth, np.ones(10) / 10, mode='same')
                augmented = sample_flat * smooth
            else:
                warped = np.interp(np.linspace(0, 187, int(187 * np.random.uniform(0.9, 1.1))), np.arange(187),
                                   sample_flat)
                augmented = (warped[:187] if len(warped) >= 187 else np.pad(warped, (0, 187 - len(warped)),
                                                                            mode='edge')) * np.random.uniform(0.9,
                                                                                                              1.1) + np.random.normal(
                    0, 0.03, 187)
            augmented_X.append(augmented.reshape(1, 187))
            augmented_y.append(target_class)
    return np.array(augmented_X), np.array(augmented_y)


aug_X_list, aug_y_list = [X_train], [y_train]
for target, n_aug in [(1, 6), (3, 15), (2, 3), (4, 1)]:
    aug_X, aug_y = augment_ecg_aggressive(X_train, y_train, target, n_aug)
    aug_X_list.append(aug_X)
    aug_y_list.append(aug_y)

X_train = np.vstack(aug_X_list)
y_train = np.hstack(aug_y_list)

# Dataloaders
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

class_counts = np.bincount(y_train)
class_weights = np.power(1.0 / class_counts, 0.45)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights_tensor = torch.FloatTensor(class_weights)

sample_weights = [class_weights[int(label)] for label in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=128, sampler=sampler)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=128, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=128,
                         shuffle=False)

# Configs
configs = {
    "Config_A_3CNN": ECG_CNN_LSTM_3CNN,
    "Config_B_1LSTM": ECG_CNN_LSTM_1LSTM,
    "Config_C_NoAttn": ECG_CNN_LSTM_NoAttention,
    "Config_D_Full": ECG_CNN_LSTM_Full
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = {}

# Train each config
for config_name, model_class in configs.items():
    print(f"\n{'=' * 80}")
    print(f"Training: {config_name}")
    print(f"{'=' * 80}")

    torch.manual_seed(42)
    np.random.seed(42)

    model = model_class(num_classes=5, dropout=0.45).to(device)
    criterion = FocalLoss(alpha=class_weights_tensor.to(device), gamma=2.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_f1 = 0.0
    for epoch in range(30):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_f1_w = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/20: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")

    test_loss, test_acc, test_f1, test_f1_w = evaluate(model, test_loader, criterion, device)
    results[config_name] = {
        "Architecture": config_name.replace("Config_", "").replace("_", " "),
        "Test Accuracy": float(test_acc),
        "Test F1 (macro)": float(test_f1),
        "Best Val F1": float(best_f1)
    }

    print(f"\n✅ {config_name}:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test F1 (macro): {test_f1:.4f}")

# Save results
os.makedirs('results/architecture_experiments', exist_ok=True)

df_results = pd.DataFrame(results).T
df_results = df_results.sort_values('Test F1 (macro)', ascending=False)

print("\n" + "=" * 80)
print("ARCHITECTURE EXPERIMENTS - RESULTS")
print("=" * 80)
print(df_results.to_string())
print("=" * 80)

df_results.to_csv('results/architecture_experiments/comparison.csv')

# Generate PNG table
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

table_data = [[col for col in df_results.columns]]
for idx, row in df_results.iterrows():
    table_data.append([f"{val:.4f}" if isinstance(val, float) else str(val) for val in row.values])

table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(len(df_results.columns)):
    table[(0, i)].set_facecolor('#1f77b4')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(df_results) + 1):
    if i == 1:
        for j in range(len(df_results.columns)):
            table[(i, j)].set_facecolor('#90EE90')
    elif i % 2 == 0:
        for j in range(len(df_results.columns)):
            table[(i, j)].set_facecolor('#F0F0F0')

plt.figtext(0.5, 0.97, 'Architecture Experiments: Ablation Study', ha='center', fontsize=16, fontweight='bold')
plt.figtext(0.5, 0.02, 'Comparing different architectural configurations', ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)
plt.savefig('results/architecture_experiments/comparison_table.png', dpi=300, bbox_inches='tight')

print("\n✅ Results saved:")
print("   - results/architecture_experiments/comparison.csv")
print("   - results/architecture_experiments/comparison_table.png")
