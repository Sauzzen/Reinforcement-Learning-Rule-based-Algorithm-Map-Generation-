# rl_agent.py
import os
import random
from collections import deque
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------
# Device Utils
# --------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Replay Buffer
# --------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state if next_state is not None else np.zeros_like(state), dtype=np.float32)
        self.buffer.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

# --------------------
# Dueling Q-Network
# --------------------
class DuelingQNetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], output_dim: int, hidden_dim: int = 512):
        super().__init__()
        H, W = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Compute conv output size
        with torch.no_grad():
            sample = torch.zeros(1, 1, H, W)
            conv_out = self.conv(sample)
            conv_out_size = int(np.prod(conv_out.shape[1:]))

        self.fc_common = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
        )

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        if x.dim() == 3:  # add channel dim
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.fc_common(x)
        val = self.value_stream(x)
        adv = self.adv_stream(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# --------------------
# RL Agent
# --------------------
class RLAgent:
    def __init__(
        self,
        state_size: Tuple[int, int],
        action_size: int,
        device: Optional[torch.device] = None,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 100_000,
        reward_scale: float = 1.0,
        reward_clamp: Optional[Tuple[float, float]] = (-10.0, 10.0),
        tau: float = 0.005,
        buffer_capacity: int = 200_000,
        load_path: Optional[str] = None
    ):
        self.device = device or get_device()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.reward_clamp = reward_clamp
        self.tau = tau

        # Networks
        self.policy_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay
        self.memory = ReplayBuffer(buffer_capacity)

        # Epsilon
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = max(1, eps_decay_steps)
        self.total_steps = 0

        # Training bookkeeping
        self.learn_steps = 0
        self.last_loss = None

        # Optional model load
        if load_path and os.path.exists(load_path):
            self.load(load_path)

    # -----------------
    # Epsilon schedule
    # -----------------
    def current_epsilon(self) -> float:
        frac = min(self.total_steps / self.eps_decay_steps, 1.0)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    # -----------------
    # Action selection
    # -----------------
    def act(self, state):
        self.total_steps += 1
        if random.random() < self.current_epsilon():
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.policy_net(state_t)
            return int(q_vals.argmax(dim=1).item())

    # -----------------
    # Store transition
    # -----------------
    def remember(self, state, action, reward, next_state, done):
        r = reward * self.reward_scale
        if self.reward_clamp is not None:
            lo, hi = self.reward_clamp
            r = max(lo, min(hi, r))
        self.memory.push(state, action, r, next_state, done)

    # -----------------
    # Replay / training
    # -----------------
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update()
        self.learn_steps += 1
        self.last_loss = float(loss.item())
        return self.last_loss

    # -----------------
    # Soft update target network
    # -----------------
    def soft_update(self, tau: Optional[float] = None):
        tau = tau if tau is not None else self.tau
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    # -----------------
    # Save / Load
    # -----------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "learn_steps": self.learn_steps
        }, path)
        print(f"💾 Saved checkpoint to {path}")

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["policy_state"])
        self.target_net.load_state_dict(data.get("target_state", data["policy_state"]))
        if "optimizer" in data:
            try:
                self.optimizer.load_state_dict(data["optimizer"])
            except Exception:
                print("⚠️ Could not load optimizer state.")
        self.total_steps = data.get("total_steps", 0)
        self.learn_steps = data.get("learn_steps", 0)
        print(f"🔁 Loaded checkpoint from {path}")

    # -----------------
    # Reset training
    # -----------------
    def reset_training(self, lr: Optional[float] = None):
        self.policy_net.apply(self.policy_net._init_weights)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory.clear()
        self.total_steps = 0
        self.learn_steps = 0
        self.last_loss = None
        if lr:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.optimizer.param_groups[0]["lr"])
        print("🔁 Training reset (networks reinitialized, memory cleared).")

    # -----------------
    # Info
    # -----------------
    def info(self):
        return {
            "device": str(self.device),
            "total_steps": self.total_steps,
            "learn_steps": self.learn_steps,
            "memory_size": len(self.memory),
            "last_loss": self.last_loss,
            "epsilon": self.current_epsilon()
        }
