# ================== HYPERPARAMETER TUNING ==================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

# ================== LOAD DATA ==================
print("Loading data...")
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')
X_val = np.load('../data/X_val.npy')
y_val = np.load('../data/y_val.npy')
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

# NORMALIZE
scaler = joblib.load('../models/scaler.pkl')  # Use shared scaler!
X_train = scaler.transform(X_train.reshape(-1, 187)).reshape(-1, 1, 187)
X_val = scaler.transform(X_val.reshape(-1, 187)).reshape(-1, 1, 187)
X_test = scaler.transform(X_test.reshape(-1, 187)).reshape(-1, 1, 187)


# ================== MODEL ==================
class ECG_CNN_LSTM(nn.Module):
    def __init__(self, num_classes=5, dropout=0.45):
        super(ECG_CNN_LSTM, self).__init__()

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


# ================== FOCAL LOSS ==================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


# ================== TRAINING & EVALUATION ==================
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
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted, all_labels, all_preds


# ================== HYPERPARAMETER CONFIGS ==================
configs = {
    "Config_A_Best": {
        "dropout": 0.45,
        "learning_rate": 0.0008,
        "batch_size": 128,
        "weight_decay": 2e-5,
        "gamma": 2.5,
        "epochs": 30
    },
    "Config_B_HighDropout": {
        "dropout": 0.5,
        "learning_rate": 0.0008,
        "batch_size": 128,
        "weight_decay": 2e-5,
        "gamma": 2.5,
        "epochs": 30
    },
    "Config_C_LowDropout": {
        "dropout": 0.3,
        "learning_rate": 0.0008,
        "batch_size": 128,
        "weight_decay": 2e-5,
        "gamma": 2.5,
        "epochs": 30
    },
    "Config_D_HighLR": {
        "dropout": 0.45,
        "learning_rate": 0.001,
        "batch_size": 128,
        "weight_decay": 2e-5,
        "gamma": 2.5,
        "epochs": 30
    },
    "Config_E_LowLR": {
        "dropout": 0.45,
        "learning_rate": 0.0005,
        "batch_size": 128,
        "weight_decay": 2e-5,
        "gamma": 2.5,
        "epochs": 30
    }
}

# ================== RUN TUNING ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = {}

for config_name, params in configs.items():
    print(f"\n{'=' * 80}")
    print(f"Testing: {config_name}")
    print(f"Params: dropout={params['dropout']}, lr={params['learning_rate']}, batch_size={params['batch_size']}")
    print(f"{'=' * 80}\n")

    # Prepare data
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Class weighting
    class_counts = np.bincount(y_train)
    class_weights = np.power(1.0 / class_counts, 0.45)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    sample_weights = [class_weights[int(label)] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler)

    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model setup
    torch.manual_seed(42)
    np.random.seed(42)

    model = ECG_CNN_LSTM(num_classes=5, dropout=params['dropout'])
    model = model.to(device)

    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=params['gamma'])

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params['learning_rate'],
                                  weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=3)

    # Training
    best_f1 = 0.0
    val_f1s = []

    for epoch in range(params['epochs']):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_f1_w, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1

        val_f1s.append(val_f1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{params['epochs']}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")

    # Test evaluation
    test_loss, test_acc, test_f1_macro, test_f1_weighted, y_true, y_pred = evaluate(
        model, test_loader, criterion, device)

    results[config_name] = {
        "dropout": params['dropout'],
        "learning_rate": params['learning_rate'],
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_weighted": float(test_f1_weighted),
        "best_val_f1": float(best_f1)
    }

    print(f"\n✅ {config_name} Results:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test F1 (macro): {test_f1_macro:.4f}")
    print(f"   Best Val F1: {best_f1:.4f}\n")

# ================== SAVE & DISPLAY RESULTS ==================
os.makedirs('../results/hyperparameter_tuning', exist_ok=True)

# Save JSON
with open('../results/hyperparameter_tuning/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Create comparison table
df_results = pd.DataFrame(results).T
df_results = df_results.sort_values('test_f1_macro', ascending=False)

print("\n" + "=" * 100)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 100)
print(df_results.to_string())
print("=" * 100)

# Save CSV
df_results.to_csv('results/hyperparameter_tuning/comparison.csv')
print("\n✅ Results saved to 'results/hyperparameter_tuning/'")
