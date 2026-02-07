#Made by Chat GPT
# =======================
# STOCK TRADING LSTM SIM
# =======================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt
import stock_agent
from stock_agent import TradingEnv, PolicyNetwork


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

#print(X_tensor)
# -----------------------
# 3️⃣ DEFINE LSTM MODEL
# -----------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.Dropout=nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.Dropout(out)
        out = self.fc(out[:, -1, :])  # last time step
        return out

model = StockLSTM(input_size=len(STOCKS))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# 4️⃣ TRAIN MODEL
# -----------------------
variance_y=torch.var(y_tensor, unbiased=False).item()
EPOCHS = 10
loss_history=[]
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | MSE-Variance Ratio: {loss.item()/variance_y:.4f}")

#plt.figure(figsize=(8,5))
#plt.plot(loss_history)
#plt.xlabel("Epoch")
#plt.ylabel("MSE Loss")
#plt.title("Training MSE Loss vs Epochs")
#plt.grid(True)
#plt.show()

torch.save(model.state_dict(), 'model_dict.pt')

#RL

env_prices=[]
curr_prices=X_tensor

for i in range(len(prices) - SEQ_LEN-1):
    new_prices=model(curr_prices[i:i+1])
    new_prices_np=new_prices.cpu().detach().numpy()
    env_prices.append(new_prices_np)
print(env_prices)
env_prices=np.array(env_prices).squeeze()
env_prices=torch.tensor(env_prices)

#Add some noise to encourage exploration
returns = torch.diff(env_prices, dim=0, prepend=env_prices[:1])
env_prices = env_prices * (1 + 0.05 * returns)



#print(env_prices)

# Drop stock columns
macro_df = df.drop(columns=STOCKS).copy()

# If you have a Date column, drop or convert it
if 'Date' in macro_df.columns:
    macro_df = macro_df.drop(columns=['Date'])

# Convert all remaining columns to float
macro_numeric = macro_df.astype(np.float32).values

env = TradingEnv(prices=env_prices, macro=macro_numeric, initial_cash=10000)
policy = PolicyNetwork(state_dim=14,n_stocks=4)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)
stock_agent.train(env, policy, optimizer, episodes=1000)


#5 optimal purchases