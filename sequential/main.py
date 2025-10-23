# main.py
import os
import json
import numpy as np
import torch

from map_env import MapEnvironment
from train_agent import RLAgent
from rule_based_polish import RuleBasedPolisher
import visualize_map
from metrics_plot import MetricsLogger   # ✅ Metrics tracking

# -----------------
# Training settings
# -----------------
EPISODES = 1500
WORLD_SIZE = 32
SAVE_INTERVAL = 10
TRAIN_INTERVAL = 4
TARGET_UPDATE = 2000  # steps for hard target update

# -----------------
# Setup directories
# -----------------
os.makedirs("worlds", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)  # ✅ Folder for heatmaps
RESUME_FILE = "checkpoints/resume.json"

# -----------------
# Resume helpers
# -----------------
def load_resume():
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "r") as f:
            data = json.load(f)
        return data.get("step_count", 0), data.get("best_score", -float("inf"))
    return 0, -float("inf")

def save_resume(step_count, best_score):
    data = {
        "step_count": int(step_count),
        "best_score": float(best_score),
    }
    with open(RESUME_FILE, "w") as f:
        json.dump(data, f)

# -----------------
# Main training loop
# -----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    step_count, best_score = load_resume()
    best_world = None

    print(f"🌍 Training world {WORLD_SIZE}x{WORLD_SIZE} (Perlin-noise initialized)")

    # ----------------- initialize environment & agent -----------------
    env = MapEnvironment(size=WORLD_SIZE)
    agent = RLAgent(state_size=env.state_size, action_size=env.action_size, device=device)
    print(f"Agent initialized on {agent.device}")

    metrics = MetricsLogger()

    best_model_path = "checkpoints/best_model.h5"
    if os.path.exists(best_model_path):
        agent.load(best_model_path)
        print("✅ Loaded previous best model.")

    # ----------------- training episodes -----------------
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

            # Train agent
            if step_count % TRAIN_INTERVAL == 0 and len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                if loss is not None and step_count % 100 == 0:
                    print(f"[Step {step_count}] Loss: {loss:.4f}")

            # Update target network
            if step_count % TARGET_UPDATE == 0:
                agent.soft_update(tau=1.0)
                print(f"[Step {step_count}] 🔄 Target network updated.")

        # ----------------- Save maps and heatmaps -----------------
        world = env.get_full_world()
        visualize_map.save_full_map(world, episode, save_dir="worlds")
        visualize_map.save_heatmap(world, episode, save_dir="heatmaps")  # ✅ Heatmap

        # Metrics logging
        metrics.log(episode, world, total_reward)

        avg_step_reward = total_reward / (WORLD_SIZE * WORLD_SIZE)
        print(f"[Episode {episode}] Total={total_reward:.2f}, Avg/step={avg_step_reward:.3f}, eps≈{agent.current_epsilon():.3f}")

        # ----------------- Update best model -----------------
        if total_reward > best_score:
            best_score = total_reward
            best_world = world.copy()
            agent.save(best_model_path)
            visualize_map.save_full_map(best_world, f"best_ep{episode}", save_dir="worlds")
            print(f"[Episode {episode}] 🎉 New best! Total={best_score:.2f}")
        # Save checkpoints & intermediate metrics plot
        if episode % SAVE_INTERVAL == 0:
            checkpoint_path = f"checkpoints/ep{episode}.h5"
            agent.save(checkpoint_path)
            save_resume(step_count, best_score)
            print(f"[Episode {episode}] ✅ Checkpoint saved")
            metrics.plot(show=False, save_path=f"metrics/metrics_ep{episode}.png")
            metrics.save_to_csv(f"metrics/training_log_ep{episode}.csv")


    # ----------------- Final rule-based polish -----------------
    if best_world is not None:
        polisher = RuleBasedPolisher()
        polished_world = polisher.apply(best_world)
        visualize_map.save_full_map(polished_world, "final_best", save_dir="worlds")
        visualize_map.save_heatmap(polished_world, "final_best", save_dir="heatmaps")
        metrics.save_to_csv("metrics/training_log_final.csv")
        print("🏁 Training complete. Polished best world saved.")
    else:
        print("⚠️ No best world generated!")

    # Plot final metrics
    metrics.plot(show=True, save_path="metrics/metrics_final.png")


if __name__ == "__main__":
    main()
