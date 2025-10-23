# generate_map.py
"""
Runtime map generator for trained RL models.

Class: MapGenerator

Requirements:
 - PyTorch
 - NumPy
 - matplotlib
 - (optional) rule_based_polish.RuleBasedPolisher
 - The project's MapEnvironment (map_env.py) should be importable.

Usage:
    gen = MapGenerator("checkpoints/best_model.pth", config_path="checkpoints/config.json")
    grid, metrics = gen.generate_map(seed=123, apply_polish=True)
    gen.export_json(grid, "exports/map_001.json", metadata={"seed":123})
    gen.export_png(grid, "exports/map_001.png")
    gen.generate_batch(3, "exports/", apply_polish=True)
"""

from __future__ import annotations
import os
import json
import time
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Try importing environment and polisher from local modules (assumes they exist)
try:
    from map_env import MapEnvironment, compute_map_metrics  # map_env.py from your project
except Exception:
    MapEnvironment = None
    compute_map_metrics = None

try:
    from rule_based_polish import RuleBasedPolisher
except Exception:
    RuleBasedPolisher = None

# Tile color map (index 0 LAND, 1 WATER, 2 SAND)
_TILE_COLORS = ["#2b8c2b", "#4da6ff", "#f2d08b"]
_TILE_CMAP = ListedColormap(_TILE_COLORS)


# -----------------------------
# DuelingQNetwork (fallback)
# -----------------------------
# We try to import the Dueling network used in training; if not available,
# define a local compatible version. This must match the architecture used when saving the model.
try:
    # try common names used in training code
    from train_agent import DuelingQNetwork  # or rl_agent
except Exception:
    try:
        from train_agent import DuelingQNetwork
    except Exception:
        DuelingQNetwork = None

if DuelingQNetwork is None:
    import torch.nn as nn
    import torch.nn.functional as F

    class DuelingQNetwork(nn.Module):
        """
        Minimal Dueling Q-Network matching many DQN conv encoders:
        - conv1: in_channels -> 32, kernel 8, stride 4
        - conv2: 32 -> 64, kernel 4, stride 2
        - conv3: 64 -> 64, kernel 3, stride 1
        - fc -> hidden (512) -> value & advantage
        """
        def __init__(self, state_size: Tuple[int, int], output_dim: int, input_channels: int = 1, hidden_dim: int = 512):
            super().__init__()
            H, W = state_size
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

            # compute conv output size
            with torch.no_grad():
                sample = torch.zeros(1, input_channels, H, W)
                x = F.relu(self.conv1(sample))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                conv_out_size = int(x.numel() / x.shape[0])

            self.fc_common = nn.Linear(conv_out_size, hidden_dim)
            self.value_stream = nn.Linear(hidden_dim, 1)
            self.adv_stream = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x: (B, C, H, W)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc_common(x))
            val = self.value_stream(x)
            adv = self.adv_stream(x)
            return val + adv - adv.mean(dim=1, keepdim=True)


# -----------------------------
# MapGenerator
# -----------------------------
class MapGenerator:
    """
    MapGenerator loads a trained PyTorch model and generates maps by running the model
    inside MapEnvironment step-by-step (one tile placement per action).

    Parameters
    ----------
    model_path : str
        Path to the .pth checkpoint (can be raw state_dict or dict with "policy_state").
    config_path : Optional[str]
        Optional JSON config with keys:
          {
            "state_size": [H, W],
            "action_size": int,
            "input_channels": int,
            "obs_mode": "channels" or "single",
            "init_mode": "mixed" | "perlin" | "islands"
          }
        If not provided, defaults are used and some values are inferred.
    device : Optional[str or torch.device]
        Device string, e.g. "cuda" or "cpu". If None, auto-detect.
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None, device: Optional[Any] = None):
        assert os.path.exists(model_path), f"Model file not found: {model_path}"
        self.model_path = model_path

        # load config if provided
        self.config = {}
        if config_path:
            assert os.path.exists(config_path), f"Config file not found: {config_path}"
            with open(config_path, "r") as f:
                self.config = json.load(f)

        # determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # set defaults and infer missing config fields
        self.state_size = tuple(self.config.get("state_size", (100, 100)))
        self.action_size = int(self.config.get("action_size", 3))  # 3 tiles: LAND/WATER/SAND
        self.input_channels = int(self.config.get("input_channels", 4))  # default to channels mode (land, water, sand, shore)
        self.obs_mode = self.config.get("obs_mode", "channels")  # 'single' or 'channels'
        self.init_mode_default = self.config.get("init_mode", "mixed")

        # build model architecture
        self.model = DuelingQNetwork(self.state_size, self.action_size, input_channels=self.input_channels)
        self.model.to(self.device)
        self.model.eval()

        # load checkpoint intelligently (support different save styles)
        self._load_weights(model_path)

    # -----------------------------
    # Internal: load weights robustly
    # -----------------------------
    def _load_weights(self, path: str):
        """
        Load PyTorch checkpoint. Supports:
         - saved state_dict directly
         - dict containing "policy_state" (as in trainer.save_checkpoint)
         - dict containing "model_state" or "state_dict"
        """
        ckpt = torch.load(path, map_location=self.device)
        state_dict = None
        if isinstance(ckpt, dict):
            # common save formats
            for key in ("policy_state", "model_state", "state_dict"):
                if key in ckpt:
                    state_dict = ckpt[key]
                    break
            # maybe the dict is directly a state_dict (parameter -> tensor)
            if state_dict is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt
        else:
            # other types unlikely
            state_dict = None

        if state_dict is None:
            raise RuntimeError("Could not locate model state dict in checkpoint. Provide a config or a pure state_dict .pth.")

        # load into model (allow missing/extra keys)
        self.model.load_state_dict(state_dict, strict=False)
        print(f"[MapGenerator] Loaded model weights from {path}")

    # -----------------------------
    # generate_map
    # -----------------------------
    def generate_map(self,
                     seed: Optional[int] = None,
                     apply_polish: bool = True,
                     init_mode: Optional[str] = None,
                     deterministic: bool = True,
                     verbose: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a single map by running the model in MapEnvironment.

        Steps:
          - Create MapEnvironment(size=state_size, seed=seed, obs_mode matched to model).
          - Reset environment with chosen init_mode.
          - Step until done: get observation, pass through model, select action (argmax), env.step(action).
          - Optionally apply rule-based polishing.
          - Compute metrics (uses map_env.compute_map_metrics if available).
          - Return final grid (2D np.ndarray) and metrics dict.
        """
        if MapEnvironment is None:
            raise RuntimeError("MapEnvironment not available. Make sure map_env.py is in PYTHONPATH.")

        # choose init_mode
        init_mode = init_mode or self.init_mode_default

        # create environment with matching observation mode
        env = MapEnvironment(size=self.state_size[0], obs_mode=self.obs_mode, seed=seed)
        obs = env.reset(init_mode=init_mode, init_seed=seed)

        done = False
        step = 0
        with torch.no_grad():
            while not done:
                # obs may be HxW (single) or CxHxW (channels). Convert to tensor [B,C,H,W]
                state_t = self._obs_to_tensor(obs)
                q = self.model(state_t)  # shape (1, action_size)
                if deterministic:
                    action = int(q.argmax(dim=1).item())
                else:
                    # simple epsilon-free stochastic choice using softmax
                    probs = torch.softmax(q.squeeze(0).cpu(), dim=0).numpy()
                    action = int(np.random.choice(len(probs), p=probs))
                obs, reward, done = env.step(action)
                step += 1
                if verbose and step % 5000 == 0:
                    print(f"[MapGenerator] Step {step}...")

        world = env.get_full_world()

        # optional polish
        if apply_polish:
            if RuleBasedPolisher is None:
                if verbose:
                    print("[MapGenerator] rule_based_polish not found; skipping polish.")
            else:
                pol = RuleBasedPolisher()
                world = pol.apply(world)

        # compute metrics if available
        metrics = {}
        if compute_map_metrics is not None:
            metrics = compute_map_metrics(world)
        else:
            # fallback compute basic metrics
            metrics = {
                "land_frac": float(np.mean(world == 0)),
                "water_frac": float(np.mean(world == 1)),
                "sand_frac": float(np.mean(world == 2)),
            }

        # include seed and info
        metrics.update({"seed": seed, "init_mode": init_mode, "steps": step})
        return world, metrics

    # -----------------------------
    # observation -> tensor helper
    # -----------------------------
    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """
        Convert environment observation (HxW or CxHxW numpy) to torch tensor shape (1,C,H,W) on device.
        Assumes obs is float32 or can be cast to float32.
        """
        if isinstance(obs, np.ndarray):
            a = obs.astype(np.float32)
            if a.ndim == 2:
                # single channel: add channel dim
                a = a[None, :, :]
            elif a.ndim == 3:
                # channels already C,H,W
                pass
            else:
                raise ValueError(f"Unexpected obs ndim {a.ndim}")
            # add batch dim
            a = np.expand_dims(a, axis=0)  # 1 x C x H x W
            t = torch.tensor(a, dtype=torch.float32, device=self.device)
            return t
        else:
            raise ValueError("Obs must be a numpy array")

    # -----------------------------
    # export_json
    # -----------------------------
    def export_json(self, grid: np.ndarray, filename: str, 
                    metadata: Optional[Dict[str, Any]] = None,
                    villages: Optional[Dict[str, Any]] = None,
                    distance_layers: Optional[Dict[str, Any]] = None):
        """
        Export grid with optional villages and distance_layers (backward compatible).
        """
        ensure_dir_for_file(filename)
        H, W = grid.shape
        
        payload = {
            'width': int(W),
            'height': int(H),
            'tiles': grid.astype(int).tolist(),
            'metadata': metadata or {}
        }
        
        # Add optional Phase 2 extensions
        if villages is not None:
            payload['villages'] = villages
        if distance_layers is not None:
            payload['distance_layers'] = distance_layers
        
        # Ensure tile_ids in metadata for semantic locking
        if 'tile_ids' not in payload['metadata']:
            payload['metadata']['tile_ids'] = {'LAND': 0, 'WATER': 1, 'SAND': 2}
        
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)
        return filename


    # -----------------------------
    # export_png
    # -----------------------------
    def export_png(self, grid: np.ndarray, filename: str, dpi: int = 150):
        """
        Save a PNG preview of the grid using a tiled colormap. Saves an RGB PNG.
        """
        ensure_dir_for_file(filename)
        # Convert integer grid to RGBA using ListedColormap
        cmap = _TILE_CMAP
        # matplotlib's colormap expects floats 0..1 or discrete indices; we'll map discretely
        # Normalize indices to [0,1] by dividing by max index (2)
        img_rgba = cmap(grid.astype(int) / 2.0)  # returns (H,W,4) float32
        # Save as PNG via plt.imsave
        plt.imsave(filename, img_rgba, dpi=dpi)
        return filename

    # -----------------------------
    # generate_batch
    # -----------------------------
    def generate_batch(self,
                       n_maps: int,
                       output_dir: str,
                       apply_polish: bool = True,
                       seeds: Optional[List[int]] = None,
                       init_mode: Optional[str] = None,
                       prefix: str = "map"):
        """
        Generate multiple maps and save them as JSON + PNG.

        Parameters:
          n_maps: number of maps to generate
          output_dir: where to save files (will be created)
          apply_polish: whether to apply the rule-based polisher
          seeds: optional list of seeds. If None, seeds will be generated using time-based RNG.
          init_mode: optional init_mode override for environment (perlin/islands/mixed)
          prefix: filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        json_dir = os.path.join(output_dir, "json")
        png_dir = os.path.join(output_dir, "png")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        # prepare seeds
        rng = np.random.RandomState(int(time.time()) & 0xFFFFFF)
        if seeds is None:
            seeds = [int(rng.randint(0, 2 ** 31 - 1)) for _ in range(n_maps)]
        assert len(seeds) == n_maps, "Length of seeds must equal n_maps"

        results = []
        for i, seed in enumerate(seeds):
            idx = i + 1
            world, metrics = self.generate_map(seed=seed, apply_polish=apply_polish, init_mode=init_mode)
            ts = int(time.time())
            json_filename = os.path.join(json_dir, f"{prefix}_{idx:03d}_s{seed}_{ts}.json")
            png_filename = os.path.join(png_dir, f"{prefix}_{idx:03d}_s{seed}_{ts}.png")

            metadata = {
                "seed": seed,
                "generated_at": ts,
                "model": os.path.basename(self.model_path),
                "init_mode": init_mode or self.init_mode_default,
            }
            metrics.update(metadata)

            self.export_json(world, json_filename, metadata=metrics)
            self.export_png(world, png_filename)

            results.append({"idx": idx, "seed": seed, "json": json_filename, "png": png_filename, "metrics": metrics})
            print(f"[MapGenerator] Saved map {idx}/{n_maps} (seed={seed}) -> {json_filename}, {png_filename}")
        return results


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example: load model and generate 3 maps
    model_path = "checkpoints/best_model.pth"  # replace with your checkpoint
    config_path = "checkpoints/config.json"    # optional: contains state_size, input_channels, etc.

    # instantiate generator
    gen = MapGenerator(model_path=model_path, config_path=(config_path if os.path.exists(config_path) else None))

    # generate single map
    grid, metrics = gen.generate_map(seed=42, apply_polish=True, verbose=True)
    print("Metrics:", metrics)
    out_json = "exports/map_single.json"
    out_png = "exports/map_single.png"
    gen.export_json(grid, out_json, metadata=metrics)
    gen.export_png(grid, out_png)
    print("Saved:", out_json, out_png)

    # batch generate 3 maps
    results = gen.generate_batch(3, "exports/batch", apply_polish=True)
    print("Batch results:", results)
