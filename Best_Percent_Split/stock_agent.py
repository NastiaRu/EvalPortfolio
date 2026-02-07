import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


class TradingEnv:
    def __init__(self, prices, macro, initial_cash=10000, initial_shares=None, sharpe_window=20):
        self.prices = prices
        self.macro = macro
        self.n_stocks = prices.shape[1]
        self.initial_cash = float(initial_cash)
        self.sharpe_window = sharpe_window

        # Always numpy float32
        self.initial_shares = initial_shares if initial_shares is not None else np.zeros(self.n_stocks, dtype=np.float32)
        self.shares = self.initial_shares.copy()
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = self.initial_cash
        self.shares = self.initial_shares.copy()
        self.done = False
        self.portfolio_value = self.cash
        self.prev_value = self.cash
        self.past_returns = []
        return self._get_state()

    def _get_state(self):
        prices_t = np.array(self.prices[self.t], dtype=np.float32).ravel()
        macro_t = np.array(self.macro[self.t], dtype=np.float32).ravel()
        cash_t = np.array([self.cash / self.initial_cash], dtype=np.float32)
        shares_t = self.shares / max(self.initial_cash / np.mean(prices_t), 1e-6)
        state = np.concatenate([prices_t, macro_t, cash_t, shares_t.astype(np.float32)])
        return torch.tensor(state, dtype=torch.float32)

    def step(self, action):
        if self.done:
            raise ValueError("Episode is done. Call reset()")

        price = np.array(self.prices[self.t], dtype=np.float32)

        # Enforce legal actions
        for i, a in enumerate(action):
            if a == 1:  # BUY
                if self.cash >= price[i]:
                    self.cash -= price[i]
                    self.shares[i] += 1
            elif a == 2:  # SELL
                if self.shares[i] >= 1:
                    self.cash += price[i]
                    self.shares[i] -= 1
            # HOLD = do nothing

        # Update portfolio
        self.portfolio_value = self.cash + np.sum(self.shares * price)
        step_return = (self.portfolio_value - self.prev_value) / self.prev_value
        self.prev_value = self.portfolio_value

        self.past_returns.append(step_return)
        if len(self.past_returns) > self.sharpe_window:
            self.past_returns.pop(0)

        mean_r = np.mean(self.past_returns)
        std_r = np.std(self.past_returns)
        reward = np.clip(mean_r / (std_r + 1e-6), -1.0, 1.0)

        self.t += 1
        if self.t >= len(self.prices):
            self.done = True
            next_state = None
        else:
            next_state = self._get_state()

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
    episode_rewards = []
    cash_history = []
    portfolio_history = []

    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        entropies = []
        done = False
        actions=[]

        while not done:
            state_t = state.unsqueeze(0)
            logits = policy(state_t)
            logits = logits.view(env.n_stocks, 3)

            actions = []
            step_log_probs = []

            for i in range(env.n_stocks):
                masked_logits = logits[i].clone()
                if env.shares[i] <= 0:
                    masked_logits[2] = -1e9
                if env.cash < env.prices[env.t][i]:
                    masked_logits[1] = -1e9

                dist = torch.distributions.Categorical(logits=masked_logits)
                a = dist.sample()
                actions.append(a.item())
                step_log_probs.append(dist.log_prob(a))
                entropies.append(dist.entropy())

            actions = np.array(actions)
            next_state, reward, done, _ = env.step(actions)

            # TRACK LOG_PROBS, REWARDS
            log_probs.append(torch.stack(step_log_probs).sum())
            rewards.append(reward)

            # TRACK CASH AND PORTFOLIO
            cash_history.append(env.cash)
            portfolio_history.append(env.portfolio_value)

            # OPTIONAL: PRINT ACTIONS
            #print(f"t={env.t} | Actions={actions} | Cash={env.cash:.2f} | Portfolio={env.portfolio_value:.2f}")

            state = next_state

        # policy gradient
        discounted_rewards = compute_discounted_rewards(rewards, gamma)
        policy_loss = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * Gt)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        entropy_loss = -torch.stack(entropies).sum()
        total_loss = loss - 0.01 * entropy_loss
        total_loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))

        print(f"EPISODE {episode} | Actions: {actions} | Cash: {env.cash:.2f} | Shares: {env.shares} | TOTAL REWARD: {sum(rewards):.2f}")

    return episode_rewards, cash_history, portfolio_history
