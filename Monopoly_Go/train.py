import numpy as np
import torch
import torch.nn.functional as F

from monopoly_go import monopoly_go_v0
from monopoly_go.utils import flattened_length
from dqn import DQN, ReplayBuffer

device = torch.device("mps")

# Hyperparameters
num_episodes = 2000
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
batch_size = 128
lr = 3e-5
target_update_freq = 5

# Environment setup
env = monopoly_go_v0.env(render_mode="human")
env.reset()
sample_obs = env.observe("player_0")
obs_size = flattened_length
action_size = env.action_space("player_0").n

# Model + target
policy_net = DQN(obs_size, action_size).to(device)
target_net = DQN(obs_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
buffer = ReplayBuffer()

agent_to_train = "player_0"
winners = [0, 0, 0]

terminated = [False, False, False]


for ep in range(num_episodes):
    env = monopoly_go_v0.env(render_mode="human")
    env.reset()
    last_obs = {}

    while True:
        agent = env.agent_selection
        obs, reward, term, trunc, info = env.last()

        rewards_gotten = 0

        if term:
            terminated[env.curr_agent_index] = True
            env.step(None)

        if not term and agent == agent_to_train:
            # ε-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space(agent).sample(info["action_mask"])
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(obs_tensor)
                    mask = torch.tensor(info["action_mask"], dtype=torch.bool).unsqueeze(0)  # shape [1, 611]
                    assert mask.sum() > 0, "All actions masked — bug in masking logic"
                    q_values[~mask] = -float('inf')
                    action = torch.argmax(q_values).item()

            last_obs[agent] = obs
            last_action = action
        else:
            action = env.action_space(agent).sample(info["action_mask"])

        env.step(action if not term else None)

        if agent == agent_to_train:
            next_obs, r, done_flag, _, _ = env.last()
            buffer.add((last_obs[agent], last_action, r, next_obs, done_flag))

        if all(terminated):
            print(env._cumulative_rewards)
            terminated = [False] * 3
            break

    if env.winner >= 0:
        winners[env.winner] += 1

    # Training step
    if len(buffer.buffer) >= batch_size:
        obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(batch_size)
        obs_b = obs_b.float().to(device)
        act_b = act_b.long().to(device)
        rew_b = rew_b.float().to(device)
        next_obs_b = next_obs_b.float().to(device)
        done_b = done_b.float().to(device)

        q_vals = policy_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_vals = target_net(next_obs_b).max(1)[0]
            target = rew_b + gamma * next_q_vals * (1 - done_b)

        loss = F.mse_loss(q_vals, target)
        print(f"Loss: {loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network
    if ep % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {ep} done. ε={epsilon:.3f}")

print(winners)