import os
import time
import json
import math
import pickle
import heapq
from collections import deque
from typing import Tuple, Optional, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================
# CRITICAL FIX: Import metrics from map_env
# ============================
from map_env import compute_map_metrics  # ✅ Use environment's correct version

# ============================
# Utilities
# ============================

class RunningMeanStd:
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: float):
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var = max(1e-6, ((self.count - 1) * self.var + delta * delta2) / (self.count))

    def normalize(self, x: float) -> float:
        return (x - self.mean) / math.sqrt(self.var + 1e-8)

_TILES = {0: "land", 1: "water", 2: "sand"}
_TILE_CMAP = ListedColormap(["#2b8c2b", "#4da6ff", "#f2d08b"])

# ============================
# Replay Buffer with Action Masks
# ============================

class ReplayBuffer:
    """Replay buffer storing action masks for hierarchical training."""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        self.buffer.append((state, action, reward, next_state, done, action_mask, next_action_mask))
    
    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones, action_masks, next_action_masks = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones),
                np.array(action_masks), np.array(next_action_masks))
    
    def __len__(self):
        return len(self.buffer)

# ============================
# Dueling Q-Network
# ============================

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size: Tuple[int, int], action_size: int, input_channels: int = 5):  # ✅ CHANGED from 1 to 5
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        conv_out_size = self._get_conv_out(state_size, input_channels)

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, action_size)
    
    def _get_conv_out(self, shape, in_ch):
        o = torch.zeros(1, in_ch, *shape)
        o = self.conv3(self.conv2(self.conv1(o)))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        value = self.value_stream(x)
        adv = self.advantage_stream(x)
        return value + (adv - adv.mean(1, keepdim=True))

# ============================
# RL Agent
# ============================

class RLAgent:
    def __init__(self,
                 state_size: Tuple[int, int],
                 action_size: int,
                 device: Optional[torch.device] = None,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 batch_size: int = 64,
                 eps_start: float = 0.5,  # FIX 3: Start with less exploration (was 1.0)
                 eps_end: float = 0.05,
                 eps_decay_steps: int = 100_000,
                 eps_schedule: str = "linear",
                 reward_scale: float = 1.0,
                 reward_clamp: Optional[Tuple[float, float]] = (-10.0, 10.0),
                 tau: float = 0.005,
                 buffer_capacity: int = 200_000,
                 input_channels: int = 5,  # ✅ CHANGED from 1 to 5
                 min_replay_size: int = 5000,
                 eps_decay_episodes: int = 800,
                 load_path: Optional[str] = None):
        
        self.min_replay_size = min_replay_size
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.state_size = state_size
        self.action_size = action_size
        self.current_episode = 0
        self.max_action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.reward_clamp = reward_clamp
        self.tau = tau

        # Epsilon parameters
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.eps_schedule = eps_schedule
        self.total_steps = 0

        self.reward_stats = RunningMeanStd()

        # Networks
        self.policy_net = DuelingQNetwork(state_size, action_size, input_channels).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size, input_channels).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)
        self.learn_steps = 0
        self.last_loss = None
        
        # Target update tracking
        self.steps = 0
        self.target_update_every = 10

        if load_path and os.path.exists(load_path):
            self.load_checkpoint(load_path, load_buffer=False)

    def current_epsilon(self):
        """Compute epsilon based on step or episode count."""
        frac = min(self.total_steps / max(1, self.eps_decay_steps), 1.0)
        if self.eps_schedule == "linear":
            return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)
        elif self.eps_schedule == "exponential":
            decay_rate = math.log(self.eps_end / max(1e-8, self.eps_start)) / max(1, self.eps_decay_steps)
            return max(self.eps_end, self.eps_start * math.exp(decay_rate * self.total_steps))
        elif self.eps_schedule == "cosine":
            return self.eps_end + 0.5 * (self.eps_start - self.eps_end) * (1 + math.cos(math.pi * frac))
        return self.eps_end

    def act(self, state: np.ndarray, action_mask: np.ndarray, deterministic: bool = False) -> int:
        """
        Epsilon-greedy action selection with masking.
        Handles variable action space sizes in hierarchical mode.
        """
        epsilon = 0.0 if deterministic else self.current_epsilon()
        
        # Get valid action indices
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0  # Fallback
        
        # Random exploration
        if not deterministic and np.random.random() < epsilon:
            return np.random.choice(valid_actions)
        
        # Greedy: select best Q-value among valid actions
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).cpu().numpy()[0]
            
            # Only evaluate valid actions to avoid indexing error
            valid_q_values = q_values[valid_actions]
            best_valid_idx = np.argmax(valid_q_values)
            return valid_actions[best_valid_idx]

    def remember(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        """Store transition with action masks."""
        self.memory.push(state, action, reward, next_state, done, action_mask, next_action_mask)

    def replay(self) -> Optional[float]:
        """Train with masked Double DQN targets and replay warmup."""
        # Check MIN_REPLAY_SIZE before training
        if len(self.memory) < self.min_replay_size or len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones, _, next_masks = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_masks = torch.BoolTensor(next_masks).to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Masked Double DQN target computation
        with torch.no_grad():
            # Online network selects action
            next_q_online = self.policy_net(next_states)
            masked_next_q = torch.full_like(next_q_online, float('-inf'))
            masked_next_q[next_masks] = next_q_online[next_masks]
            next_actions = masked_next_q.argmax(dim=1)
            
            # Target network evaluates action
            next_q_target = self.target_net(next_states)
            next_q_value = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_value
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update step counters
        self.learn_steps += 1
        self.total_steps += 1
        self.steps += 1
        
        # Soft update target network
        if self.steps % self.target_update_every == 0:
            self.soft_update()
        
        self.last_loss = loss.item()
        return loss.item()
    
    def soft_update(self, tau: float = None):
        """Soft update target network with policy network weights."""
        if tau is None:
            tau = self.tau
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save_checkpoint(self, path: str, save_buffer: bool = False, keep_meta: bool = True):
        """Save agent checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": int(self.total_steps),
            "learn_steps": int(self.learn_steps),
            "current_episode": int(self.current_episode),
            "eps_start": float(self.eps_start),
            "eps_end": float(self.eps_end),
            "eps_decay_steps": int(self.eps_decay_steps),
            "eps_schedule": self.eps_schedule,
        }
        torch.save(payload, path)
        if save_buffer:
            with open(path + ".buffer.pkl", "wb") as f:
                pickle.dump(list(self.memory.buffer), f)
        if keep_meta:
            meta = {"saved_at": time.time(), "episode": int(self.current_episode)}
            with open(path + ".meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    def load_checkpoint(self, path: str, load_buffer: bool = False):
        """Load agent checkpoint."""
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["policy_state"])
        self.target_net.load_state_dict(data["target_state"])
        try:
            self.optimizer.load_state_dict(data["optimizer"])
        except:
            print("[Agent] Optimizer state not fully restored.")
        self.total_steps = int(data.get("total_steps", 0))
        self.learn_steps = int(data.get("learn_steps", 0))
        self.current_episode = int(data.get("current_episode", 0))
        self.eps_start = float(data.get("eps_start", self.eps_start))
        self.eps_end = float(data.get("eps_end", self.eps_end))
        self.eps_decay_steps = int(data.get("eps_decay_steps", self.eps_decay_steps))
        self.eps_schedule = data.get("eps_schedule", self.eps_schedule)
        if load_buffer and os.path.exists(path + ".buffer.pkl"):
            with open(path + ".buffer.pkl", "rb") as f:
                arr = pickle.load(f)
            self.memory.buffer = deque(arr, maxlen=self.memory.buffer.maxlen)

    def evaluate(self, env_factory: Callable[[], object], episodes: int = 5, deterministic: bool = True):
        """Evaluate agent on multiple episodes."""
        results = []
        worlds = []
        for ep in range(episodes):
            env = env_factory()
            state = env.reset()
            done = False
            while not done:
                mask = env.get_valid_action_mask()
                a = self.act(state, mask, deterministic=deterministic)
                state, r, done = env.step(a)
            world = env.world
            worlds.append(world)
            m = compute_map_metrics(world)  # ✅ Now uses correct metrics
            results.append(m)
        return results, worlds

    def score_world(self, world: np.ndarray, total_reward: float) -> float:
        """Compute combined score from world and reward."""
        m = compute_map_metrics(world)  # ✅ Now uses correct metrics
        water_balance = 1.0 - abs(m["water_frac"] - 0.45)
        return total_reward + 0.5 * water_balance + 0.3 * m["sand_quality"] + 0.2 * m["largest_land_frac"]

# ============================
# Improved Agent with Episode-based Epsilon
# ============================

class ImprovedRLAgent(RLAgent):
    def __init__(self, *args, **kwargs):
        # Extract episode-based decay parameter
        self.eps_decay_episodes = kwargs.pop('eps_decay_episodes', 2000)
        super().__init__(*args, **kwargs)
        self.current_episode = 0
    
    def current_epsilon(self):
        """Episode-based epsilon decay for hierarchical mode."""
        if hasattr(self, 'eps_decay_episodes'):
            frac = min(self.current_episode / max(1, self.eps_decay_episodes), 1.0)
        else:
            frac = min(self.total_steps / max(1, self.eps_decay_steps), 1.0)
        
        if self.eps_schedule == "cosine":
            return self.eps_end + 0.5 * (self.eps_start - self.eps_end) * (1 + math.cos(math.pi * frac))
        elif self.eps_schedule == "exponential":
            return max(self.eps_end, self.eps_start * (self.eps_end / self.eps_start) ** frac)
        else:  # linear
            return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)
    
    def on_episode_start(self):
        """Call at the start of each episode to update epsilon."""
        self.current_episode += 1

# ============================
# Display helper (metrics now imported from map_env)
# ============================

def show_world(world: np.ndarray, title: str = None):
    plt.figure(figsize=(4,4))
    plt.imshow(world, cmap=_TILE_CMAP, vmin=0, vmax=2)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
# ============================