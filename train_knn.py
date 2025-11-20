import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

# Load data_old
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Normalize
scaler = RobustScaler() # StandardScaler() , MinMaxScaler() , RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
print("🤖 Training KNN...")
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan') # euclidean , manhattan , cosine , minkowski
knn.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

y_pred = knn.predict(X_test_scaled)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"\n✅ RESULTS:")
print(f"   Training time: {train_time:.2f}s")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   F1-Score (weighted): {f1_weighted:.4f}")
print(f"   F1-Score (macro): {f1_macro:.4f}")

print("\n📋 Per-class report:")
print(classification_report(y_test, y_pred,
      target_names=['Normal', 'Supraventr', 'Ventricular', 'Fusion', 'Unknown']))
