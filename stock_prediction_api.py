from fastapi import FastAPI
from pydantic import BaseModel
import torch
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.models.complex_lstm_model import ComplexLSTMModel


app = FastAPI()

# Load the trained model
model_path = "stock_prediction_model.pt"
model = ComplexLSTMModel(1, 100, 2, 1)
model.load_state_dict(torch.load(model_path))
model.eval()

class TickerInput(BaseModel):
    ticker: str

@app.post("/predict")
def predict(ticker: str = "^GSPC"):
        
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

    response = {"predicted_price": today_pred[0][0].item()}
    return response
