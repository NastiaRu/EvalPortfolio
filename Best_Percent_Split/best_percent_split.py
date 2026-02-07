#Made by Chat GPT
# =======================
# STOCK TRADING LSTM SIM
# =======================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
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


env_prices=y_tensor
#print(env_prices.shape)

#for i in range(len(prices) - SEQ_LEN-1):
    #new_prices=curr_prices[i:i+1]
    #new_prices_np=new_prices.detach().numpy()
    #env_prices.append(new_prices_np)
#env_prices=np.array(env_prices).squeeze()
#env_prices=torch.tensor(env_prices)
#print(env_prices.shape)

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

initial_shares = np.array([4, 1, 2, 5])
env = TradingEnv(prices=y_tensor, macro=macro_numeric, initial_cash=10000, initial_shares=initial_shares)
policy = PolicyNetwork(state_dim=14,n_stocks=4)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)
episode_rewards=[]
cash_history=[]
portfolio_history=[]
episode_rewards, cash_history, portfolio_history = stock_agent.train(env, policy, optimizer, episodes=100, gamma=0.99)

#Given portfolio- recommend actions and estimate returns with RNN
rand_portfolio=np.random.randint(1,5,size=4)
env.initial_cash=10000
print(rand_portfolio)
# 1️⃣ Get a single state from your environment
env.reset()
state = env._get_state()  # shape: [state_dim]

# 2️⃣ Add batch dimension for PyTorch
state_t = state.unsqueeze(0)  # shape: [1, state_dim]

# 3️⃣ Forward pass through your policy network
logits = policy(state_t)      # shape: [1, n_stocks*3]

# 4️⃣ Reshape to separate each stock
logits = logits.view(env.n_stocks, 3)  # shape: [n_stocks, 3]

# 5️⃣ Sample actions for each stock
actions = []
for i in range(env.n_stocks):
    masked_logits = logits[i].clone()
    
    # mask impossible actions (optional)
    if env.shares[i] <= 0:
        masked_logits[2] = -1e9
    if env.cash < env.prices[env.t][i]:
        masked_logits[1] = -1e9

    dist = Categorical(logits=masked_logits)
    action = dist.sample()
    actions.append(action.item())

#actions = np.array(actions)
print("Final shares", env.shares)

# SMOOTH REWARDS
window = 20
smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(cash_history, label="Cash")
plt.plot(portfolio_history, label="Portfolio Value")
plt.xlabel("Step")
plt.ylabel("USD")
plt.title("Cash & Portfolio Over Time")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(episode_rewards, alpha=0.3, label="Raw Reward")
plt.plot(range(window-1, len(episode_rewards)), smoothed_rewards, color='red', label="Smoothed Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Rewards (Sharpe-Based)")
plt.legend()
plt.grid(True)

plt.show()


#5 optimal purchases

def get_recommended_actions(state_tensor, policy, price_vector, initial_shares=None, initial_cash=10000):
    """
    Returns recommended actions for a single step.
    
    state_tensor: torch.Tensor of shape [state_dim]
    policy: your PolicyNetwork
    price_vector: np.array or torch.Tensor with current stock prices
    initial_shares: np.array of current shares for each stock
    initial_cash: float
    """
    n_stocks = len(price_vector)
    
    # Convert prices to numpy float32
    if torch.is_tensor(price_vector):
        price_vector = price_vector.cpu().numpy()
    price_vector = np.array(price_vector, dtype=np.float32)
    
    # Initialize shares
    shares = np.zeros(n_stocks, dtype=np.float32) if initial_shares is None else np.array(initial_shares, dtype=np.float32)
    cash = float(initial_cash)

    # Forward pass
    state_t = state_tensor.unsqueeze(0)  # add batch dim
    logits = policy(state_t)
    logits = logits.view(n_stocks, 3)  # 3 actions per stock

    actions = []

    for i in range(n_stocks):
        masked_logits = logits[i].clone()

        # Mask illegal actions
        if shares[i] <= 0:
            masked_logits[2] = -1e9  # can't sell
        if cash < price_vector[i]:
            masked_logits[1] = -1e9  # can't buy

        dist = torch.distributions.Categorical(logits=masked_logits)
        a = dist.sample()
        actions.append(int(a.item()))

        # Update shares and cash to simulate step
        if a.item() == 1:  # BUY
            if cash >= price_vector[i]:
                cash -= price_vector[i]
                shares[i] += 1
        elif a.item() == 2:  # SELL
            if shares[i] >= 1:
                cash += price_vector[i]
                shares[i] -= 1

    return np.array(actions), shares, cash