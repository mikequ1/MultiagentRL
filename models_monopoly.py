import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import collections
import random
import numpy as np


class CentralizedDQNAgent:
    def __init__(self, obs_dim, act_dim, n_agents, lr=1e-3, buf_size=100000, batch_size=128, gamma=0.95, target_update=100):
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.model = nn.Sequential(
            nn.Linear(n_agents * obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_agents * act_dim)
        )
        self.target_model = nn.Sequential(
            nn.Linear(n_agents * obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_agents * act_dim)
        )
        self.target_model.load_state_dict(self.model.state_dict())
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = collections.deque(maxlen=buf_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.steps = 0

    def select(self, obs, eps, masks, acting_agent):
        state = torch.tensor(np.array(np.concatenate(obs)), dtype=torch.float32).unsqueeze(0)

        if random.random() < eps:
            legal_actions = np.where(masks[acting_agent])[0]
            if len(legal_actions) == 0:
                return random.randint(0, self.act_dim)
            return np.random.choice(legal_actions)

        with torch.no_grad():
            q_values = self.model(state).view(self.n_agents, -1)
            agent_q = q_values[acting_agent]

            illegal = (torch.tensor(masks[acting_agent]) == 0)
            agent_q[illegal] = -1e9

            return agent_q.argmax().item()

    def store(self, s, a, r, s2, d):
        self.buffer.append((np.concatenate(s), a, r, np.concatenate(s2), d))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

        # Convert to tensors
        states_b = torch.tensor(np.array(states_b), dtype=torch.float32)
        actions_b = torch.tensor(np.array(actions_b), dtype=torch.int64)
        rewards_b = torch.tensor(np.array(rewards_b), dtype=torch.float32)
        next_states_b = torch.tensor(np.array(next_states_b), dtype=torch.float32)
        dones_b = torch.tensor(np.array(dones_b), dtype=torch.float32)

        # Forward pass
        q_vals = self.model(states_b).view(self.batch_size, self.n_agents, -1)
        next_q_vals = self.target_model(next_states_b).view(self.batch_size, self.n_agents, -1)

        # Mask for active agents (action != -1)
        valid_mask = (actions_b != -1)

        # Replace -1 actions with dummy index (won't matter since we'll mask them out)
        safe_actions = actions_b.clone()
        safe_actions[~valid_mask] = 0

        # Gather predicted Q-values for taken actions
        chosen_q = q_vals.gather(2, safe_actions.unsqueeze(-1)).squeeze(-1)
        max_next_q = next_q_vals.max(dim=2)[0]

        # Compute targets
        targets = rewards_b + self.gamma * max_next_q * (1 - dones_b[:, 0].unsqueeze(1))

        # Apply the valid_mask before computing loss
        masked_pred_q = chosen_q[valid_mask]
        masked_targets = targets.detach()[valid_mask]

        if masked_pred_q.numel() == 0:
            return  # no valid samples in this batch

        loss = F.mse_loss(masked_pred_q, masked_targets)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())


class DQNAgent:
    def __init__(self, obs_dim, propose_dim, accept_dim, lr=1e-3,
                 buf_size=5000, batch_size=64, gamma=0.99, target_update=100):
        self.propose_net = nn.Sequential(nn.Linear(obs_dim,64), nn.ReLU(), nn.Linear(64, propose_dim))
        self.propose_target = nn.Sequential(nn.Linear(obs_dim,64), nn.ReLU(), nn.Linear(64, propose_dim))
        self.propose_target.load_state_dict(self.propose_net.state_dict())
        self.accept_net = nn.Sequential(nn.Linear(obs_dim,64), nn.ReLU(), nn.Linear(64, accept_dim))
        self.accept_target = nn.Sequential(nn.Linear(obs_dim,64), nn.ReLU(), nn.Linear(64, accept_dim))
        self.accept_target.load_state_dict(self.accept_net.state_dict())
        self.opt = optim.Adam(list(self.propose_net.parameters()) + list(self.accept_net.parameters()), lr=lr)
        self.buffer = collections.deque(maxlen=buf_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.steps = 0

    def select(self, obs, eps, is_proposer):
        state_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        if is_proposer:
            if random.random() < eps:
                return random.randrange(self.propose_net[-1].out_features)
            with torch.no_grad():
                return int(self.propose_net(state_v).argmax(1).item())
        else:
            if random.random() < eps:
                return random.randrange(self.accept_net[-1].out_features)
            with torch.no_grad():
                return int(self.accept_net(state_v).argmax(1).item())

    def store(self, s, a, r, s2, d, is_prop):
        self.buffer.append((s, a, r, s2, d, is_prop))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s,a,r,s2,d,is_prop = zip(*batch)
        s_v = torch.from_numpy(np.array(s)).float()
        s2_v= torch.from_numpy(np.array(s2)).float()
        a_v = torch.tensor(a).long()
        r_v = torch.tensor(r, dtype=torch.float32)
        d_v = torch.tensor(d, dtype=torch.float32)
        is_p_v = torch.tensor(is_prop)

        q_prop = self.propose_net(s_v)
        q_acc  = self.accept_net(s_v)

        # Split actions
        a_prop = torch.clamp(a_v, max=q_prop.shape[1]-1)
        a_acc  = torch.clamp(a_v, max=q_acc.shape[1]-1)

        q = torch.where(
            is_p_v,
            q_prop.gather(1, a_prop.unsqueeze(1)).squeeze(1),
            q_acc.gather(1, a_acc.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            q2_prop = self.propose_target(s2_v).max(1)[0]
            q2_acc  = self.accept_target(s2_v).max(1)[0]
            q2 = torch.where(is_p_v, q2_prop, q2_acc)
            tgt = r_v + self.gamma * q2 * (1 - d_v)

        loss = F.mse_loss(q, tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.propose_target.load_state_dict(self.propose_net.state_dict())
            self.accept_target.load_state_dict(self.accept_net.state_dict())



class MADDPGActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, act_dim), nn.Softmax(dim=-1))

    def forward(self, obs):
        return self.net(obs)

class MADDPGCritic(nn.Module):
    def __init__(self, full_obs_dim, full_act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(full_obs_dim + full_act_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, full_obs, full_acts):
        return self.net(torch.cat([full_obs, full_acts], dim=-1))

class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, full_obs_dim, full_act_dim, lr=1e-3):
        self.actor = MADDPGActor(obs_dim, act_dim)
        self.target_actor = MADDPGActor(obs_dim, act_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = MADDPGCritic(full_obs_dim, full_act_dim)
        self.target_critic = MADDPGCritic(full_obs_dim, full_act_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

    def select(self, obs, eps=0.0):
        obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = self.actor(obs_v).squeeze(0)
        if random.random() < eps:
            return random.randrange(len(probs))
        return int(probs.argmax().item())


class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, act_dim), nn.Softmax(dim=-1))

    def forward(self, obs):
        return self.net(obs)

class MAPPOCritic(nn.Module):
    def __init__(self, full_obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(full_obs_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, full_obs):
        return self.net(full_obs)

class MAPPOAgent:
    def __init__(self, obs_dim, act_dim, full_obs_dim, lr=1e-3, clip_param=0.2):
        self.actor = MAPPOActor(obs_dim, act_dim)
        self.critic = MAPPOCritic(full_obs_dim)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.clip_param = clip_param

    def select(self, obs, eps=0.0):
        obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = self.actor(obs_v).squeeze(0)
        if random.random() < eps:
            return random.randrange(len(probs))
        return int(probs.argmax().item())

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, logprob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

def compute_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_update(agent, buffer, batch_size=64, epochs=4, gamma=0.99, lam=0.95):
    next_value = agent.critic(torch.tensor(buffer.obs[-1], dtype=torch.float32).unsqueeze(0)).item()
    advs, rets = compute_gae(buffer.rewards, buffer.dones, buffer.values, next_value, gamma, lam)

    obs_tensor = torch.tensor(buffer.obs, dtype=torch.float32)
    actions_tensor = torch.tensor(buffer.actions)
    logprobs_tensor = torch.tensor(buffer.logprobs)
    advantages_tensor = torch.tensor(advs, dtype=torch.float32)
    returns_tensor = torch.tensor(rets, dtype=torch.float32)

    for _ in range(epochs):
        indices = np.arange(len(buffer.obs))
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            obs_b = obs_tensor[batch_idx]
            act_b = actions_tensor[batch_idx]
            old_logprobs_b = logprobs_tensor[batch_idx]
            adv_b = advantages_tensor[batch_idx]
            ret_b = returns_tensor[batch_idx]

            # Policy loss
            probs = agent.actor(obs_b)
            dist = torch.distributions.Categorical(probs)
            logprobs = dist.log_prob(act_b)
            ratio = (logprobs - old_logprobs_b).exp()
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - agent.clip_param, 1 + agent.clip_param) * adv_b
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = agent.critic(obs_b).squeeze(1)
            critic_loss = F.mse_loss(values, ret_b)

            # Optimize
            agent.actor_opt.zero_grad()
            actor_loss.backward()
            agent.actor_opt.step()

            agent.critic_opt.zero_grad()
            critic_loss.backward()
            agent.critic_opt.step()

    return actor_loss.item(), critic_loss.item()
