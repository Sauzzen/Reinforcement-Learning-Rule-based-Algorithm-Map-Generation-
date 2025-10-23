# main.py - Updated for hierarchical terrain generation with live learning curves
from __future__ import annotations
import os, json, time, pickle, logging, heapq
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for stability
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import torch
import sys
import traceback
from collections import deque
from train_agent import ImprovedRLAgent
# Import with error handling
try:
    from map_env import MapEnvironment, compute_map_metrics, LAND, WATER, SAND
    from train_agent import RLAgent
    from metrics_plot import MetricsLogger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)

# ---------- Updated Config for Hierarchical Mode ----------
GRID_SIZE = 64
EPISODES = 5000  # Fewer episodes but more focused
SAVE_INTERVAL = 50
VISUALIZE_EVERY = 25
TARGET_UPDATE_STEPS = 100  # More frequent target updates
TRAIN_INTERVAL = 1
BATCH_SIZE = 32  # Smaller batch for faster learning
MAX_EPISODE_STEPS = 15
ACTION_MODE = "hierarchical"
CHECKPOINT_DIR = "checkpoints"
RESUME_PATH = os.path.join(CHECKPOINT_DIR, "resume.json")
EPS_DECAY_EPISODES = 800  # Faster decay for hierarchical (was 1500)
MIN_REPLAY_SIZE = 5000     # Replay warmup threshold
EVAL_EVERY = 100           # Run deterministic eval every N episodes
EVAL_SEEDS = list(range(50000, 50010))

# Create directories
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path("metrics").mkdir(parents=True, exist_ok=True)
Path("worlds").mkdir(parents=True, exist_ok=True)

tile_cmap = ListedColormap(["#2b8c2b", "#4da6ff", "#f2d08b"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Performance helpers
if device.type == "cuda":
    # Enable cuDNN autotuner (good for fixed-size inputs)
    torch.backends.cudnn.benchmark = True
    # Optionally set deterministic=False for best perf (True improves reproducibility)
    torch.backends.cudnn.deterministic = False


# ---------- Safe helper functions ----------
def safe_save_checkpoint(agent, path, save_buffer=False, metadata=None):
    """Safely save checkpoint with error handling"""
    try:
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "policy_state": agent.policy_net.state_dict(),
            "target_state": getattr(agent, 'target_net', agent.policy_net).state_dict(),
            "optimizer": agent.optimizer.state_dict() if hasattr(agent, "optimizer") else None,
            "total_steps": getattr(agent, "total_steps", 0),
            "learn_steps": getattr(agent, "learn_steps", 0),
            "eps_start": getattr(agent, "eps_start", 1.0),
            "eps_end": getattr(agent, "eps_end", 0.05),
            "eps_decay_steps": getattr(agent, "eps_decay_steps", 100000),
            "eps_schedule": getattr(agent, "eps_schedule", "linear")
        }
        
        torch.save(checkpoint_data, path)
        
        meta = metadata or {}
        meta.update({
            "total_steps": checkpoint_data["total_steps"],
            "learn_steps": checkpoint_data["learn_steps"],
            "timestamp": time.time(),
            "action_mode": ACTION_MODE
        })
        
        with open(path + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        if save_buffer and hasattr(agent, "memory") and hasattr(agent.memory, "buffer"):
            try:
                with open(path + ".buffer.pkl", "wb") as f:
                    pickle.dump(list(agent.memory.buffer), f)
            except Exception as e:
                logger.warning(f"Could not save buffer: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint {path}: {e}")
        return False

def safe_load_checkpoint(agent, path, load_buffer=False):
    """Safely load checkpoint with error handling"""
    try:
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return False
            
        data = torch.load(path, map_location=device)
        
        # Load model states
        if "policy_state" in data:
            agent.policy_net.load_state_dict(data["policy_state"])
        if "target_state" in data and hasattr(agent, "target_net"):
            agent.target_net.load_state_dict(data["target_state"])
        
        # Load optimizer
        if "optimizer" in data and data["optimizer"] is not None and hasattr(agent, "optimizer"):
            try:
                agent.optimizer.load_state_dict(data["optimizer"])
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")
        
        # Load training state
        for attr in ["total_steps", "learn_steps", "eps_start", "eps_end", "eps_decay_steps", "eps_schedule"]:
            if attr in data and hasattr(agent, attr):
                setattr(agent, attr, data[attr])
        
        # Load buffer if requested
        if load_buffer:
            buf_path = path + ".buffer.pkl"
            if os.path.exists(buf_path):
                try:
                    with open(buf_path, "rb") as f:
                        arr = pickle.load(f)
                    from collections import deque
                    maxlen = getattr(agent.memory.buffer, "maxlen", None)
                    agent.memory.buffer = deque(arr, maxlen=maxlen)
                    logger.info("Replay buffer restored.")
                except Exception as e:
                    logger.warning(f"Could not load buffer: {e}")
        
        logger.info(f"Checkpoint loaded: {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint {path}: {e}")
        return False

def safe_save_resume_metadata(resume_path, episode, total_steps, best_score, best_model_path):
    """Safely save resume metadata"""
    try:
        meta = {
            "episode": int(episode),
            "total_steps": int(total_steps),
            "best_score": float(best_score),
            "best_model_path": best_model_path,
            "action_mode": ACTION_MODE,
            "updated": time.time()
        }
        with open(resume_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save resume metadata: {e}")


def compute_combined_score(world, total_reward, weights=None):
    """Simplified scoring - just use episode reward + metrics for evaluation"""
    return float(total_reward), compute_map_metrics(world)

def safe_show_map(world, episode, total_reward, metrics, pause_time=0.001):
    """Safely display map with error handling"""
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(world, cmap=tile_cmap, vmin=0, vmax=2, interpolation='nearest')
        
        title = f"Ep {episode}  Reward {total_reward:.1f}  "
        title += f"Land {metrics.get('land_frac', 0):.2f}  "
        title += f"Water {metrics.get('water_frac', 0):.2f}  "
        title += f"Sand {metrics.get('sand_frac', 0):.2f}"
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        # Save instead of show to avoid display issues
        save_path = f"metrics/preview_ep{episode}.png"
        plt.savefig(save_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        logger.warning(f"Could not display map for episode {episode}: {e}")
        if 'fig' in locals():
            plt.close(fig)

def safe_export_map(world, out_json_path, out_png_path, seed=None, episode=None, init_mode="mixed"):
    """Safely export map with error handling"""
    try:
        payload = {
            "width": int(world.shape[1]),
            "height": int(world.shape[0]),
            "tiles": world.astype(int).tolist(),
            "metadata": {
                "seed": seed,
                "episode": episode,
                "init_mode": init_mode,
                "action_mode": ACTION_MODE,
                "timestamp": time.time(),
                "tile_ids": {"LAND": 0, "WATER": 1, "SAND": 2}  # ✅ ADDED
            }
        }
        Path(os.path.dirname(out_json_path) or ".").mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(payload, f)
        plt.imsave(out_png_path, world, cmap=tile_cmap)
        return True
    except Exception as e:
        logger.warning(f"Could not export map: {e}")
        return False


# ---------- Main Setup ----------
logger.info(f"Starting terrain generation training with {ACTION_MODE} mode")

# Initialize environment with hierarchical mode
env = MapEnvironment(size=GRID_SIZE, action_mode=ACTION_MODE, max_steps=MAX_EPISODE_STEPS)

# Initialize agent with correct action size for hierarchical mode
action_size = 9  # This will be 9 for hierarchical mode
logger.info(f"Action size: {action_size} for mode: {ACTION_MODE}")

agent = ImprovedRLAgent(
    state_size=(GRID_SIZE, GRID_SIZE),
    action_size=action_size,
    input_channels=4,
    device=device,
    gamma=0.98,  # Higher discount for multi-step rewards
    lr=5e-4,     # Higher learning rate for faster adaptation
    batch_size=BATCH_SIZE,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay_episodes=1500,  # NEW: Episode-based decay
    eps_schedule="cosine",
    reward_scale=1.0,
    reward_clamp=(-10.0, 10.0),
    tau=0.005,
    buffer_capacity=25000
)

# Initialize live plotting variables
episode_rewards = deque(maxlen=1000)
episode_numbers = deque(maxlen=1000) 
live_fig = None
live_ax = None

# Initialize metrics logger
try:
    metrics_logger = MetricsLogger(save_dir="metrics", live_every=5, save_every=25)
except Exception as e:
    logger.warning(f"Could not initialize metrics logger: {e}")
    metrics_logger = None

# Training state
best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
best_score = -float('inf')
start_episode = 1

# Initialization modes for variety
INIT_MODES = ["perlin"]
init_mode_cycle = 0

# ---------- Training Loop ----------
logger.info("Starting training loop...")

# Add evaluation configuration
EVAL_EVERY = 100
EVAL_SEEDS = list(range(50000, 50010))  # Fixed seeds for deterministic eval
eval_rewards_history = deque(maxlen=100)

try:
    outer_bar = tqdm(range(start_episode, EPISODES + 1), desc="Training Episodes", ncols=100)

    for episode in outer_bar:
        agent.on_episode_start()  # ✅ UPDATE EPSILON
        
        try:
            # Cycle through initialization modes
            init_mode = INIT_MODES[init_mode_cycle % len(INIT_MODES)]
            if episode % 20 == 0:
                init_mode_cycle += 1
            # Deterministic but varied seeds
            init_seed = np.random.randint(0, 1_000_000) + episode
            if episode % 50 == 0:
                init_mode_cycle = np.random.randint(0, len(INIT_MODES))

            # Reset environment
            state = env.reset(init_mode=init_mode, init_seed=init_seed)
            done = False
            episode_steps = 0
            total_reward = 0.0
            losses = []

            # Episode loop with action masking
            while not done and episode_steps < MAX_EPISODE_STEPS:
                # ✅ GET ACTION MASK FOR CURRENT PHASE
                action_mask = env.get_valid_action_mask()
                
                # ✅ SKIP LEARNING DURING SHORELINE AUTO-GENERATION
                if env.should_skip_learning():
                    # Shoreline phase: automatic sand placement, no agent action
                    next_state, reward, done = env.step(0)  # Dummy action
                    state = next_state
                    total_reward += reward
                    episode_steps += 1
                    continue
                
                # ✅ ACT WITH MASK
                action = agent.act(state, action_mask)  # Now expects mask as 2nd arg
                next_state, reward, done = env.step(action)
                
                # ✅ GET NEXT STATE MASK FOR REPLAY BUFFER
                next_action_mask = env.get_valid_action_mask()
                
                # ✅ STORE WITH MASKS
                agent.remember(state, action, reward, next_state, done, action_mask, next_action_mask)

                # Training with replay warmup
                if len(agent.memory) >= BATCH_SIZE:
                    loss = agent.replay()  # Now returns None if < MIN_REPLAY_SIZE
                    if loss is not None:
                        losses.append(loss)

                state = next_state
                total_reward += reward
                episode_steps += 1

            # Episode finished - compute metrics and score
            world = env.world
            combined_score, metrics = compute_combined_score(world, total_reward)
            
            # ✅ GET PER-PHASE REWARD BREAKDOWN
            phase_rewards = env.get_phase_reward_breakdown()
            
            # Track rewards for live plotting
            episode_rewards.append(total_reward)
            episode_numbers.append(episode)

            # ✅ DETERMINISTIC EVALUATION EVERY 100 EPISODES
            if episode % EVAL_EVERY == 0:
                eval_rewards = []
                for eval_seed in EVAL_SEEDS:
                    eval_state = env.reset(init_mode='mixed', init_seed=eval_seed)
                    eval_reward = 0.0
                    eval_done = False
                    eval_steps = 0
                    
                    while not eval_done and eval_steps < MAX_EPISODE_STEPS:
                        eval_mask = env.get_valid_action_mask()
                        
                        if env.should_skip_learning():
                            eval_state, r, eval_done = env.step(0)
                            eval_reward += r
                        else:
                            eval_action = agent.act(eval_state, eval_mask, deterministic=True)
                            eval_state, r, eval_done = env.step(eval_action)
                            eval_reward += r
                        
                        eval_steps += 1
                    
                    eval_rewards.append(eval_reward)
                
                eval_mean = np.mean(eval_rewards)
                eval_std = np.std(eval_rewards)
                eval_rewards_history.append(eval_mean)
                
                logger.info(f"Ep {episode} | Eval: {eval_mean:.2f} ± {eval_std:.2f} | "
                           f"Epsilon: {agent.current_epsilon():.3f}")

            # Update live plot every 10 episodes
            if episode % 10 == 0:
                try:
                    if live_fig is None:
                        live_fig, live_ax = plt.subplots(figsize=(12, 8))
                    
                    live_ax.clear()
                    
                    if len(episode_rewards) >= 2:
                        # Plot raw training rewards
                        live_ax.plot(list(episode_numbers), list(episode_rewards), 
                                    'b-', alpha=0.5, label='Training Rewards', linewidth=1)
                        
                        # Moving average of training rewards
                        if len(episode_rewards) >= 10:
                            ma_window = min(20, len(episode_rewards))
                            moving_avg = []
                            rewards_list = list(episode_rewards)
                            for i in range(len(rewards_list)):
                                start = max(0, i - ma_window + 1)
                                moving_avg.append(np.mean(rewards_list[start:i+1]))
                            
                            live_ax.plot(list(episode_numbers), moving_avg, 
                                        'r-', linewidth=2, label=f'Training MA ({ma_window})')
                        
                        # ✅ PLOT EVAL REWARDS SEPARATELY
                        if eval_rewards_history:
                            eval_episodes = [ep for ep in episode_numbers if ep % EVAL_EVERY == 0]
                            eval_episodes = eval_episodes[-len(eval_rewards_history):]
                            if len(eval_episodes) == len(eval_rewards_history):
                                live_ax.plot(eval_episodes, list(eval_rewards_history), 
                                           'g-', linewidth=3, marker='o', markersize=6,
                                           label='Eval Rewards (Deterministic)')
                    
                    # Mark best models
                    if combined_score > best_score:
                        live_ax.scatter([episode], [total_reward], 
                                    color='gold', s=200, marker='*', 
                                    label='New Best', zorder=5, edgecolors='black', linewidths=2)
                    
                    live_ax.set_xlabel('Episode', fontsize=12)
                    live_ax.set_ylabel('Reward', fontsize=12)
                    live_ax.set_title(
                        f'Learning Curve - Episode {episode} | '
                        f'Current: {total_reward:.2f} | Best: {best_score:.2f} | '
                        f'ε: {agent.current_epsilon():.3f}', 
                        fontsize=14
                    )
                    live_ax.legend(loc='best')
                    live_ax.grid(True, alpha=0.3)
                    
                    # Better y-axis limits
                    if episode_rewards:
                        all_rewards = list(episode_rewards)
                        y_min = max(0, min(all_rewards) - 0.5)
                        y_max = max(6, max(all_rewards) + 0.5)
                        live_ax.set_ylim(y_min, y_max)
                    
                    plt.tight_layout()
                    plt.savefig(f"metrics/live_learning_curve_ep{episode}.png", dpi=100, bbox_inches='tight')
                    plt.close(live_fig)
                    live_fig = None
                    
                except Exception as e:
                    logger.warning(f"Live plotting failed: {e}")
                    import traceback
                    traceback.print_exc()

            # Log metrics with phase breakdown
            if metrics_logger:
                try:
                    # ✅ LOG PER-PHASE REWARDS
                    metrics_logger.log(episode, world, total_reward)
                except Exception as e:
                    logger.warning(f"Metrics logging failed: {e}")

            # Update progress bar
            eps_val = agent.current_epsilon()
            avg_loss = np.mean(losses) if losses else 0.0
            
            # ✅ SHOW REPLAY BUFFER STATUS
            buffer_status = f"{len(agent.memory)}/{agent.min_replay_size}" if len(agent.memory) < agent.min_replay_size else "Ready"
            
            outer_bar.set_postfix({
                "Reward": f"{total_reward:.2f}",
                "Score": f"{combined_score:.2f}",
                "Phase": getattr(env, 'phase', 'N/A'),
                "Water": f"{metrics.get('water_frac', 0):.2f}",
                "Sand": f"{metrics.get('sand_frac', 0):.2f}",
                "Eps": f"{eps_val:.3f}",
                "Loss": f"{avg_loss:.4f}",
                "Buffer": buffer_status
            })

            # Save periodic checkpoint
            if episode % SAVE_INTERVAL == 0:
                path = f"{CHECKPOINT_DIR}/ep{episode}.pth"
                safe_save_checkpoint(agent, path, save_buffer=False, 
                                   metadata={"episode": episode, "score": combined_score})
                
                if metrics_logger:
                    try:
                        metrics_logger.save_plots(save_path=f"metrics/metrics_ep{episode}.png", show=False)
                    except Exception as e:
                        logger.warning(f"Could not save metrics plots: {e}")

            # Save best model
            if combined_score > best_score:
                best_score = combined_score
                if safe_save_checkpoint(agent, best_model_path, save_buffer=False,
                                      metadata={"episode": episode, "score": combined_score}):
                    logger.info(f"New best model saved at ep {episode} (score={combined_score:.3f})")

            # Save resume state
            safe_save_resume_metadata(RESUME_PATH, episode, getattr(agent, "total_steps", 0), 
                                    best_score, best_model_path)

            # Visualize occasionally
            if episode % VISUALIZE_EVERY == 0:
                safe_show_map(world, episode, total_reward, metrics)

        except Exception as e:
            logger.error(f"Error in episode {episode}: {e}")
            logger.error(traceback.format_exc())
            continue

    outer_bar.close()

except KeyboardInterrupt:
    logger.info("Training interrupted by user")
except Exception as e:
    logger.error(f"Fatal error in training: {e}")
    logger.error(traceback.format_exc())
finally:
    if metrics_logger:
        metrics_logger.save_plots(save_path="metrics/final_metrics.png", show=False)
    logger.info("Training completed")

    

# ---------- Post-training exports ----------
logger.info("Generating final maps...")

if os.path.exists(best_model_path):
    if safe_load_checkpoint(agent, best_model_path, load_buffer=False):
        Path("worlds").mkdir(exist_ok=True)
        
        for seed in range(50):  # Generate fewer maps to avoid issues
            try:
                env.reset(init_mode="perlin", init_seed=seed * 1000)
                done = False
                step_count = 0
                
                while not done and step_count < MAX_EPISODE_STEPS:
                    action = agent.act(env.observe(), env.get_current_action_size(), deterministic=True)
                    _, _, done = env.step(action)
                    step_count += 1
                
                world = env.world.copy()
                safe_export_map(world,
                              out_json_path=f"worlds/final_map_seed{seed}.json",
                              out_png_path=f"worlds/final_map_seed{seed}.png",
                              seed=seed, episode="final")
                
                logger.info(f"Generated final map {seed+1}/50")
                
            except Exception as e:
                logger.warning(f"Could not generate final map {seed}: {e}")

logger.info("Training completed!")