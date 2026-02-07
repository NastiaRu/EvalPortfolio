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
            reward = 100*np.log(self.portfolio_value / self.prev_value)
            reward=np.clip(reward,-1.0,1.0)
            #print(reward)

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
    def __init__(self, state_dim, n_stocks):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_stocks * 3),  # 3 actions per stock
            #nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)
        
def compute_discounted_rewards(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, dtype=torch.float32)
    std = returns.std()

    if std < 1e-4:
        return returns  # DO NOT NORMALIZE

    return (returns - returns.mean()) / (std + 1e-5)

def train(env, policy, optimizer, episodes=1000, gamma=0.99):
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        entropies=[]

        while not done:
            state_t = state.unsqueeze(0)  # Add batch dimension
            # ---- FORWARD PASS ----
            logits = policy(state_t)  
            logits = logits.view(env.n_stocks, 3)
            if episode % 50 == 0:
                print("LOGITS START:", logits[0].detach().cpu().numpy())

            actions = []
            step_log_probs = []

            # ---- SAMPLE PER-STOCK ACTIONS ----
            for i in range(env.n_stocks):
                masked_logits = logits[i].clone()

                # ACTION 2 = SELL
                if env.shares[i] <= 0:
                  masked_logits[2] = -1e9

                # ACTION 1 = BUY
                if env.cash < env.prices[env.t][i]:
                  masked_logits[1] = -1e9

                dist = Categorical(logits=masked_logits)

                a = dist.sample()
                actions.append(a.item())
                step_log_probs.append(dist.log_prob(a))
                entropies.append(dist.entropy())

            actions = np.array(actions)

            # ---- ENV STEP ----
            next_state, reward, done, _ = env.step(actions)

            log_probs.append(torch.stack(step_log_probs).sum())
            rewards.append(reward)

            state = next_state

        # ---- EPISODE END: POLICY GRADIENT ----
        discounted_rewards = compute_discounted_rewards(rewards, gamma)

        policy_loss = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * Gt)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        policy_loss=torch.stack(policy_loss).sum()
        entropy_loss=-torch.stack(entropies).sum()
        loss=policy_loss-0.01*entropy_loss
        loss.backward()
        print(policy.fc[0].weight.grad.abs().mean())
        optimizer.step()

        episode_rewards.append(sum(rewards))

        


        if episode % 50 == 0:
            print("LOGITS END:", policy(state_t)[0].detach().cpu().numpy())
            with torch.no_grad():
                probs=torch.softmax(logits, dim=-1)
                print("DEBUG ACTION PROBS (STOCK 0):", probs[0].cpu().numpy())
            print(f"EPISODE {episode} | TOTAL REWARD: {sum(rewards):.2f}")
