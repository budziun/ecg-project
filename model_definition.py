import torch
import torch.nn as nn

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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = x.transpose(1, 2)  # Transpose for LSTM
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