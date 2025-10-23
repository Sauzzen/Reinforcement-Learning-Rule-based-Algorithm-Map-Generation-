import torch
import numpy as np
from pathlib import Path
from map_env import MapEnvironment
from train_agent import ImprovedRLAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MapEnvironment(size=64, action_mode="hierarchical", max_steps=15)
agent = ImprovedRLAgent(
    state_size=(64, 64),
    action_size=9,
    input_channels=4,
    device=device
)

# Check if best model exists
best_model_path = "checkpoints/best_model.pth"
if not Path(best_model_path).exists():
    print(f"❌ {best_model_path} NOT FOUND!")
    exit(1)

# Load model
data = torch.load(best_model_path, map_location=device)
agent.policy_net.load_state_dict(data["policy_state"])
print(f"✅ Loaded {best_model_path}")

# Test single map generation
Path("worlds").mkdir(exist_ok=True)
state = env.reset(init_mode="perlin", init_seed=99999)
done = False
steps = 0

while not done and steps < 15:
    action_mask = env.get_valid_action_mask()
    if env.should_skip_learning():
        state, _, done = env.step(0)
    else:
        action = agent.act(state, action_mask, deterministic=True)
        state, _, done = env.step(action)
    steps += 1

world = env.world.copy()

# Save map
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

with open("worlds/test_map.json", "w") as f:
    json.dump({
        "width": 64,
        "height": 64,
        "tiles": world.tolist(),
        "metadata": {"seed": 99999, "episode": "test"}
    }, f)

cmap = ListedColormap(["#2b8c2b", "#4da6ff", "#f2d08b"])
plt.imsave("worlds/test_map.png", world, cmap=cmap)

print("✅ Saved worlds/test_map.png and worlds/test_map.json")
print("If you see this, the code works! Check why main.py isn't reaching this section.")
