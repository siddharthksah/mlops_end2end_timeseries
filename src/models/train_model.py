import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


ticker = "^GSPC"  # S&P 500 ticker
# start_date = "2015-01-01"
# Calculate the date 35 years ago
start_date = (datetime.now() - timedelta(days=365*35)).strftime("%Y-%m-%d")
end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

data = yf.download(ticker, start=start_date, end=end_date)

# Keep only the 'Close' column
data = data[['Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Create sequences of data for the LSTM model
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input for the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension
X_test = torch.tensor(X_test, dtype=torch.float32)

# Create DataLoader for training data
train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False)


# Define LSTM model
# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
#         super(LSTMModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.dropout = nn.Dropout(0.2)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
#         out = self.dropout(out[:, -1, :])
#         out = self.fc(out)
#         return out

class ComplexLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ComplexLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm1(x, (h0.detach(), c0.detach()))
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm2(out, (h0.detach(), c0.detach()))
        
        out = self.batch_norm(out[:, -1, :])
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

input_dim = 1
hidden_dim = 100
num_layers = 2
output_dim = 1
model = ComplexLSTMModel(input_dim, hidden_dim, num_layers, output_dim)


# Initialize the model, loss function, and optimizer
# model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
model = ComplexLSTMModel(input_dim, hidden_dim, num_layers, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs = inputs.requires_grad_()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Make predictions on the test data
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.detach().numpy()
    y_pred = scaler.inverse_transform(y_pred)  # De-normalize the predictions

# Calculate the actual test prices
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Make a prediction for today
last_sequence = np.reshape(scaled_data[-sequence_length:], (1, sequence_length, 1))
last_sequence = torch.tensor(last_sequence, dtype=torch.float32)
with torch.no_grad():
    today_pred = model(last_sequence)
    today_pred = today_pred.detach().numpy()
    today_pred = scaler.inverse_transform(today_pred)

print(f"Today's predicted stock price: {today_pred[0][0]}")

model_path = "stock_prediction_model.pt"
torch.save(model.state_dict(), model_path)
print("Model saved to", model_path)


