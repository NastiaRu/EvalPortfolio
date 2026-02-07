#Made by Chat GPT
# =======================
# STOCK TRADING LSTM SIM
# =======================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# 1️⃣ LOAD DATA
# -----------------------
df = pd.read_csv("stocks_macro_2025.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Stocks to trade
STOCKS = ["AAPL", "AMZN", "MSFT", "NVDA"]

# Use only stock prices for LSTM input
prices = df[STOCKS].values

# -----------------------
# 2️⃣ CREATE SEQUENCES
# -----------------------
SEQ_LEN = 10  # number of past days to use

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(prices, SEQ_LEN)

# Convert to tensors
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)


# -----------------------
# 3️⃣ DEFINE LSTM MODEL
# -----------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out

model = StockLSTM(input_size=len(STOCKS))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# 4️⃣ TRAIN MODEL
# -----------------------
EPOCHS = 50000
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

variance_y=torch.var(y_tensor, unbiased=False).item()
print(variance_y)

#5 optimal purchases