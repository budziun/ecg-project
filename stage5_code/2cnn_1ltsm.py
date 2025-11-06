import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
import json


# ================== BASELINE MODEL (SIMPLER) ==================
class ECG_CNN_LSTM_Baseline(nn.Module):
    """Simplified baseline for comparison"""

    def __init__(self, num_classes=5, dropout=0.3):
        super(ECG_CNN_LSTM_Baseline, self).__init__()

        # Only 2 CNN layers (vs 4 in main model)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2, 2)

        # Single LSTM layer (vs 2 in main model)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1,
                            batch_first=True, dropout=0, bidirectional=True)

        # No attention mechanism

        # Simpler FC layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)

        # Simple mean pooling instead of attention
        x = torch.mean(lstm_out, dim=1)

        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ================== TRAINING SCRIPT ==================
def train_baseline_model():
    """Train baseline model for comparison"""

    print("=" * 80)
    print("EXPERIMENT 1: BASELINE MODEL")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # NO DATA AUGMENTATION for baseline

    # Prepare tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Simple class weights
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)

    print(f"Class weights: {class_weights_tensor}")

    # DataLoaders - no weighted sampler
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Smaller batch
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    model = ECG_CNN_LSTM_Baseline(num_classes=5, dropout=0.3)
    model = model.to(device)

    # Simple CrossEntropy loss (no Focal Loss)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

    # Standard Adam optimizer (not AdamW)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # No scheduler

    # Training functions
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

    # Training loop
    print("Training Baseline Model")
    print("-" * 80)

    num_epochs = 30
    best_f1 = 0.0
    results = {
        'model_name': 'Baseline CNN-LSTM',
        'architecture': '2 CNN + 1 BiLSTM',
        'params': {
            'dropout': 0.3,
            'batch_size': 64,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'loss': 'CrossEntropy',
            'augmentation': False,
            'attention': False
        },
        'train_history': [],
        'val_history': []
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate(model, val_loader, criterion, device)

        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            torch.save(model.state_dict(), 'models/baseline_model.pth')

        results['train_history'].append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'accuracy': train_acc,
            'f1_macro': train_f1
        })
        results['val_history'].append({
            'epoch': epoch + 1,
            'loss': val_loss,
            'accuracy': val_acc,
            'f1_macro': val_f1_macro,
            'f1_weighted': val_f1_weighted
        })

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1:02d}/{num_epochs}] | "
                  f"Train: Loss={train_loss:.4f}, F1={train_f1:.4f} | "
                  f"Val: Loss={val_loss:.4f}, F1={val_f1_macro:.4f}")

    training_time = time.time() - start_time

    # Final evaluation
    print("\n" + "-" * 80)
    print("Final Evaluation on Test Set")
    print("-" * 80)

    model.load_state_dict(torch.load('models/baseline_model.pth'))
    test_loss, test_acc, test_f1_macro, test_f1_weighted, y_true, y_pred = evaluate(
        model, test_loader, criterion, device)

    results['test_results'] = {
        'accuracy': test_acc,
        'f1_macro': test_f1_macro,
        'f1_weighted': test_f1_weighted,
        'best_val_f1': best_f1
    }
    results['training_time'] = training_time

    print(f"\nTest Accuracy:     {test_acc:.4f}")
    print(f"Test F1 (macro):   {test_f1_macro:.4f}")
    print(f"Test F1 (weighted): {test_f1_weighted:.4f}")
    print(f"Best Val F1:       {best_f1:.4f}")
    print(f"Training time:     {training_time:.2f}s")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Normal', 'Supraventr', 'Ventricular', 'Fusion', 'Unknown']))

    # Save results
    with open('results/experiment1_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Baseline model training completed!")
    print(f"Results saved to: results/experiment1_baseline.json")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = train_baseline_model()