import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
#import matplotlib.pyplot as plt

episode_rewards = []

class TradingEnv:
    def __init__(self, prices, macro, initial_cash=10000):
        """
        prices: np.array [T, N_stocks] 
        macro: np.array [T, N_features]
        """
        self.prices = prices
        self.macro = macro
        self.n_stocks = prices.shape[1]
        self.initial_cash = initial_cash
        self.reset()
        
    def reset(self):
        self.t = 0
        self.cash = self.initial_cash
        self.shares = np.zeros(self.n_stocks)
        self.done = False
        self.portfolio_value = self.cash
        return self._get_state()
    
    def _get_state(self):
      # Flatten prices and ensure float32
      prices_t = np.array(self.prices[self.t], dtype=np.float32).ravel()
    
      # Flatten macro indicators and ensure float32
      macro_t = np.array(self.macro[self.t], dtype=np.float32).ravel()
    
      # Cash as 1D array
      cash_t = np.array([self.cash / self.initial_cash], dtype=np.float32)
    
      # Shares normalized, flattened
      shares_t = np.array(self.shares / (self.initial_cash / np.mean(prices_t)), dtype=np.float32).ravel()
    
      # Concatenate everything into 1D numeric array
      state = np.concatenate([prices_t, macro_t, cash_t, shares_t])
    
      # Convert to PyTorch tensor
      return torch.tensor(state, dtype=torch.float32)

    
    def step(self, action):
        next_state=torch.tensor([], dtype=torch.float32)
        """
        action: np.array [N_stocks] with discrete actions:
        0 = hold, 1 = buy 1 unit, 2 = sell 1 unit
        """
        if self.done:
            raise ValueError("Episode is done, call reset()")

        price = self.prices[self.t]
        for i, a in enumerate(action):
            if a == 1:  # BUY
                if self.cash >= price[i]:
                    self.cash -= price[i]
                    self.shares[i] += 1
            elif a == 2:  # SELL
                if self.shares[i] > 0:
                    self.cash += price[i]
                    self.shares[i] -= 1

        # Update portfolio
        price = price.detach().cpu().numpy()
        self.portfolio_value = self.cash + np.sum(self.shares * price)

        # Compute reward: change in portfolio value
        if self.t == 0:
            reward = 0
        else:
            #Add sharpe reward?
            reward = self.portfolio_value - self.prev_value

        self.prev_value = self.portfolio_value

        # Move time forward
        self.t += 1
        if self.t >= len(self.prices):
            self.done = True
        if not self.done:
            next_state = self._get_state()
        else:
            next_state = None
        return next_state, reward, self.done, {}
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)
        
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
    return discounted_rewards

def train(env, policy, optimizer, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state)
            m = Categorical(probs)
            action = m.sample()
            state, reward, done, _ = env.step(action)

            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            # Inside the train function, after an episode ends:

            if done:
                episode_rewards.append(sum(rewards))
                discounted_rewards = compute_discounted_rewards(rewards)
                policy_loss = []
                for log_prob, Gt in zip(log_probs, discounted_rewards):
                    policy_loss.append(-log_prob * Gt)
                optimizer.zero_grad()
                policy_loss = torch.cat(policy_loss).sum()
                policy_loss.backward()
                optimizer.step()

                if episode % 50 == 0:
                    print(f"Episode {episode}, Total Reward: {sum(rewards)}")
                break
