import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
from complex_lstm_model import ComplexLSTMModel  # Import the ComplexLSTMModel class


def predict_stock_price(ticker, model_path, sequence_length=60):
    # Download historical stock data
    start_date = (datetime.now() - timedelta(days=365*35)).strftime("%Y-%m-%d")
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Keep only the 'Close' column
    data = data[['Close']]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    # Prepare the sequence for prediction
    last_sequence = np.reshape(scaled_data[-sequence_length:], (1, sequence_length, 1))
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32)

    # Load the model
    input_dim = 1
    hidden_dim = 100
    num_layers = 2
    output_dim = 1
    model = ComplexLSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Make a prediction
    with torch.no_grad():
        today_pred = model(last_sequence)
        today_pred = today_pred.detach().numpy()
        today_pred = scaler.inverse_transform(today_pred)

    return today_pred[0][0]


if __name__ == "__main__":
    ticker = "^GSPC"  # S&P 500 ticker
    model_path = "stock_prediction_model.pt"
    predicted_price = predict_stock_price(ticker, model_path)
    print(f"Today's predicted stock price: {predicted_price}")
