import torch.nn as nn
import torch
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x, mask=None):
        logits = self.fc(x)

        if mask is not None:
            # Prevent -inf in backprop by using masked_fill with float('-1e9')
            logits = logits.masked_fill(mask == 0, -1e9)

        dist = Categorical(logits=logits)
        return dist

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

def train_ppo(policy_net, value_net, buffer, optimizer, clip_eps, gamma, lam):
    obs = torch.stack(buffer.obs).to(device)
    actions = torch.tensor(buffer.actions).to(device)
    rewards = buffer.rewards
    dones = buffer.dones
    old_log_probs = torch.tensor(buffer.log_probs).to(device)
    values = torch.tensor(buffer.values).to(device)

    # Compute returns and advantages using GAE
    returns, advantages = compute_gae(rewards, values, dones, gamma, lam)
    returns = returns.to(device)
    advantages = advantages.to(device)

    for _ in range(ppo_epochs):
        dist = policy_net(obs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        ratio = (new_log_probs - old_log_probs).exp()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_preds = value_net(obs)
        value_loss = F.mse_loss(value_preds, returns)

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
