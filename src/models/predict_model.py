import torch
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F

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

# Load the trained model
model_path = "stock_prediction_model.pt"
model = ComplexLSTMModel(1, 100, 2, 1)
model.load_state_dict(torch.load(model_path))
model.eval()

def predict_stock_price(ticker):
    start_date = (datetime.now() - timedelta(days=365*35)).strftime("%Y-%m-%d")
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    sequence_length = 60
    last_sequence = np.reshape(scaled_data[-sequence_length:], (1, sequence_length, 1))
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32)

    with torch.no_grad():
        today_pred = model(last_sequence)
        today_pred = today_pred.detach().numpy()
        today_pred = scaler.inverse_transform(today_pred)

    return today_pred[0][0]

if __name__ == "__main__":
    ticker = "^GSPC"
    predicted_price = predict_stock_price(ticker)
    print(f"Today's predicted stock price for {ticker}: {predicted_price}")
