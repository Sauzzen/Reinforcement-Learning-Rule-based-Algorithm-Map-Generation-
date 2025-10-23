# main.py - Updated for hierarchical terrain generation with live learning curves
from __future__ import annotations
import os, json, time, pickle, logging, heapq
from pathlib import Path
import numpy as np
import pandas as pd
import csv
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
    from map_env import MapEnvironment, compute_map_metrics, LAND, WATER, SAND, sample_villages_poisson
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
MAX_EPISODE_STEPS = 20
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
                "timestamp": time.time()
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
def to_py(o):
    # Defensive JSON converter for NumPy/scalars/arrays/tuples/sets
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    return o
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
    input_channels=5,  # 4 previous states + current
    device=device,
    gamma=0.98,  # Higher discount for multi-step rewards
    lr=5e-4,     # Higher learning rate for faster adaptation
    batch_size=BATCH_SIZE,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay_episodes=800,  # NEW: Episode-based decay
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
            # FIX 1: Use consistent target for learning stability
            env.desired_water_frac = 0.40  # Fixed target instead of random

            state = env.reset(init_mode=init_mode, init_seed=init_seed)
            done = False
            episode_steps = 0
            total_reward = 0.0
            losses = []

            # Episode loop with action masking
            while not done and episode_steps < MAX_EPISODE_STEPS:
                # ✅ GET ACTION MASK FOR CURRENT PHASE
                action_mask = env.get_valid_action_mask()
                
                # ✅ ACT WITH MASK
                action = agent.act(state, action_mask)  # Now expects mask as 2nd arg
                next_state, reward, done = env.step(action)
                
                # ✅ GET NEXT STATE MASK FOR REPLAY BUFFER
                next_action_mask = env.get_valid_action_mask()
                
                # ✅ STORE WITH MASKS
                agent.remember(state, action, reward, next_state, done, action_mask, next_action_mask)

                # Training with replay warmup
                if len(agent.memory) >= agent.min_replay_size:  # ✅ Use min_replay_size, not BATCH_SIZE
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
                logger.info(f"🔍 Running deterministic eval at episode {episode}")
                
                # Switch to eval mode
                agent.policy_net.eval()
                
                eval_rewards = []
                eval_metrics_list = []
                
                with torch.no_grad():  # Disable gradient computation for eval
                    for eval_seed in EVAL_SEEDS:
                        # FIX: Set fixed target for consistent eval
                        env.desired_water_frac = 0.45
                        
                        eval_state = env.reset(init_mode='mixed', init_seed=eval_seed)
                        eval_reward = 0.0
                        eval_done = False
                        eval_steps = 0
                        
                        while not eval_done and eval_steps < MAX_EPISODE_STEPS:
                            eval_mask = env.get_valid_action_mask()
                            
                            # FIX: No skip logic - all actions are learned now
                            eval_action = agent.act(eval_state, eval_mask, deterministic=True)
                            eval_state, r, eval_done = env.step(eval_action)
                            eval_reward += r
                            eval_steps += 1
                        
                        eval_rewards.append(eval_reward)
                        
                        # FIX: Collect metrics for acceptance criteria
                        final_metrics = compute_map_metrics(env.world)
                        final_metrics['ratio_error'] = abs(final_metrics['water_frac'] - env.desired_water_frac)
                        final_metrics['seed'] = eval_seed
                        eval_metrics_list.append(final_metrics)
                        # FIX: Save PNG and JSON every 100 episode
                        # 
                        #s
                        if episode % 100 == 0:
                            save_path = Path(f"eval_panel/ep{episode}_seed{eval_seed}")
                            save_path.parent.mkdir(exist_ok=True, parents=True)
                            
                            # Save PNG
                            env.save_png(str(save_path) + ".png")
                            
                            # Save JSON with metadata
                            env.export_to_json(str(save_path) + ".json", metadata={
                                'episode': episode,
                                'seed': eval_seed,
                                'desired_water_frac': env.desired_water_frac,
                                'ratio_error': final_metrics['ratio_error'],
                                'sand_quality': final_metrics['sand_quality'],
                                'largest_land_frac': final_metrics['largest_land_frac'],
                                'water_frac': final_metrics['water_frac']
                            })
                
                # Restore training mode
                agent.policy_net.train()
                
                # Compute eval statistics
                eval_mean = np.mean(eval_rewards)
                eval_std = np.std(eval_rewards)
                eval_rewards_history.append(eval_mean)
                
                # FIX: Log acceptance criteria metrics
                mean_ratio_error = np.mean([m['ratio_error'] for m in eval_metrics_list])
                mean_sand_quality = np.mean([m['sand_quality'] for m in eval_metrics_list])
                mean_largest_land = np.mean([m['largest_land_frac'] for m in eval_metrics_list])
                
                # Count how many seeds pass acceptance
                passing_seeds = sum(1 for m in eval_metrics_list 
                                if m['ratio_error'] <= 0.02 
                                and m['sand_quality'] >= 0.75 
                                and 0.20 <= m['largest_land_frac'] <= 0.60)
                
                logger.info(f"📊 Eval Episode {episode}:")
                logger.info(f"  Mean Reward: {eval_mean:.3f} ± {eval_std:.3f}")
                logger.info(f"  Mean Ratio Error: {mean_ratio_error:.4f}")
                logger.info(f"  Mean Sand Quality: {mean_sand_quality:.3f}")
                logger.info(f"  Mean Largest Land: {mean_largest_land:.3f}")
                logger.info(f"  ✅ Passing Seeds: {passing_seeds}/{len(EVAL_SEEDS)}")
                
                # FIX: Save eval metrics to CSV
                eval_df = pd.DataFrame(eval_metrics_list)
                eval_df.to_csv(f"metrics/eval_ep{episode}.csv", index=False)

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
# ---------- Post-training exports ----------
logger.info("Generating final maps...")

if os.path.exists(best_model_path):
    if safe_load_checkpoint(agent, best_model_path, load_buffer=False):
        Path("worlds").mkdir(exist_ok=True)

        # Define colormap for PNG export
        TILE_CMAP = ListedColormap(['#2b8c2b', '#4da6ff', '#f2d08b'])  # LAND, WATER, SAND

        agent.policy_net.eval()
        logger.info("✅ Loaded best model - starting map generation...")

        # ✅ GENERATE RANDOM SEEDS (not fixed seed + 10000)
        base_seed = int(time.time()) % 1000000  # Time-based seed
        generation_seeds = [base_seed + i * 137 for i in range(50)]  # Spread out seeds

        logger.info(f"Using base seed: {base_seed} for map generation")

        for idx, seed in enumerate(generation_seeds):
            print(f"DEBUG: Generating map {idx+1}/50 (seed={seed})...")
            try:
                # Set target water fraction
                env.desired_water_frac = 0.375  # Target: 37.5%

                state = env.reset(init_mode="perlin", init_seed=seed)
                done = False
                step_count = 0

                # Roll out with the trained policy
                with torch.no_grad():
                    while not done and step_count < MAX_EPISODE_STEPS:
                        action_mask = env.get_valid_action_mask()
                        action = agent.act(state, action_mask, deterministic=True)
                        state, _, done = env.step(action)
                        step_count += 1

                # Compute metrics
                final_metrics = compute_map_metrics(env.world)
                final_metrics['ratio_error'] = abs(final_metrics['water_frac'] - env.desired_water_frac)

                # ✅ GENERATE VILLAGES (correct signature)
                villages = sample_villages_poisson(
                    env.world,
                    count=5,
                    d_min_water=3.0,
                    d_min_sand=2.0,
                    r_min=8.0,
                    border_margin=5,
                    seed=seed + 999
                )

                print(
                    f"  Metrics: water={final_metrics['water_frac']:.3f}, "
                    f"sand={final_metrics.get('sand_frac', 0):.3f}, "
                    f"Villages: {len(villages)}"
                )

                # ✅ VISUALIZE WITH VILLAGES
                plt.figure(figsize=(8, 8))
                plt.imshow(env.world, cmap=TILE_CMAP, vmin=0, vmax=2)
                if villages:
                    for r, c in villages:
                        plt.scatter(c, r, c='red', s=100, marker='o',
                                    edgecolors='white', linewidths=2, zorder=10)
                plt.title(
                    f"W:{final_metrics['water_frac']:.1%} "
                    f"L:{final_metrics['land_frac']:.1%} "
                    f"S:{final_metrics['sand_frac']:.1%} "
                    f"V:{len(villages)}",
                    fontsize=10
                )
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(f"worlds/final_map_{idx:03d}.png", dpi=100, bbox_inches='tight')
                plt.close()

                # ✅ SAVE JSON WITH VILLAGES (cast to Python types)
                villages_py = [[int(r), int(c)] for (r, c) in villages]
                json_data = {
                    "map_size": int(env.size),
                    "seed": int(seed),
                    "episode": "final",
                    "init_mode": "perlin",
                    "terrain": env.world.astype(int).tolist(),
                    "villages": villages_py,
                    "metrics": {
                        "water_frac": float(final_metrics["water_frac"]),
                        "land_frac": float(final_metrics["land_frac"]),
                        "sand_frac": float(final_metrics.get("sand_frac", 0)),
                        "largest_land_frac": float(final_metrics["largest_land_frac"]),
                        "sand_quality": float(final_metrics.get("sand_quality", 0)),
                        "ratio_error": float(final_metrics["ratio_error"]),
                    }
                }

                with open(f"worlds/final_map_{idx:03d}.json", 'w') as f:
                    json.dump(json_data, f, indent=2, default=to_py)

                if (idx + 1) % 10 == 0:
                    logger.info(f"Generated {idx+1}/50 maps")

            except Exception as e:
                logger.warning(f"Could not generate map {idx}: {e}")
                traceback.print_exc()

        agent.policy_net.train()
        logger.info("✅ Generated 50 final maps in worlds/ directory")

        # ═══════════════════════════════════════════════════════════
        # SUMMARY STATISTICS
        # ═══════════════════════════════════════════════════════════
        try:
            logger.info("\n" + "="*60)
            logger.info("📊 FINAL MAPS SUMMARY")
            logger.info("="*60)

            water_fracs = []
            sand_fracs = []
            land_fracs = []
            ratio_errors = []
            village_counts = []

            for idx in range(50):
                json_path = f"worlds/final_map_{idx:03d}.json"
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        metrics = data.get('metrics', {})
                        villages_j = data.get('villages', [])

                        w = metrics.get('water_frac', 0.0)
                        s = metrics.get('sand_frac', 0.0)
                        l = metrics.get('land_frac', 0.0)

                        # ✅ ONLY ADD NON-ZERO VALUES
                        if w > 0 or s > 0 or l > 0:
                            water_fracs.append(float(w))
                            sand_fracs.append(float(s))
                            land_fracs.append(float(l))
                            ratio_errors.append(abs(float(w) - 0.375))
                            village_counts.append(int(len(villages_j)))

            if water_fracs:
                logger.info(f"Successfully loaded {len(water_fracs)}/50 maps")
                logger.info(f"\nWater Ratio:")
                logger.info(f"  Mean:   {np.mean(water_fracs)*100:.1f}% (target: 37.5%)")
                logger.info(f"  Median: {np.median(water_fracs)*100:.1f}%")
                logger.info(f"  StdDev: {np.std(water_fracs)*100:.1f}%")
                logger.info(f"  Range:  {np.min(water_fracs)*100:.1f}% - {np.max(water_fracs)*100:.1f}%")

                logger.info(f"\nSand Ratio:")
                logger.info(f"  Mean:   {np.mean(sand_fracs)*100:.1f}% (target: 4-7%)")
                logger.info(f"  Median: {np.median(sand_fracs)*100:.1f}%")
                logger.info(f"  Range:  {np.min(sand_fracs)*100:.1f}% - {np.max(sand_fracs)*100:.1f}%")

                logger.info(f"\nLand Ratio:")
                logger.info(f"  Mean:   {np.mean(land_fracs)*100:.1f}%")
                logger.info(f"  Median: {np.median(land_fracs)*100:.1f}%")

                logger.info(f"\nVillages:")
                logger.info(f"  Mean:   {np.mean(village_counts):.1f} per map")
                logger.info(f"  Median: {np.median(village_counts):.0f}")
                logger.info(f"  Range:  {np.min(village_counts):.0f} - {np.max(village_counts):.0f}")

                logger.info(f"\nAccuracy:")
                logger.info(f"  Avg Error: {np.mean(ratio_errors)*100:.1f}%")
                logger.info(f"  Within ±5%:  {sum(e < 0.05 for e in ratio_errors)}/{len(ratio_errors)} maps")
                logger.info(f"  Within ±10%: {sum(e < 0.10 for e in ratio_errors)}/{len(ratio_errors)} maps")
            else:
                logger.error("❌ No valid map data found in JSON files!")

            logger.info("="*60 + "\n")

        except Exception as e:
            logger.warning(f"Could not generate summary statistics: {e}")
            traceback.print_exc()
