# metrics_plot.py
from __future__ import annotations
"""
MetricsLogger for RL tile-map generation (LAND / WATER / SAND).

Features
- Log per-episode metrics: total reward, land/water/sand fractions,
  largest land blob fraction, sand quality.
- Real-time map display (non-blocking) using matplotlib (efficient updates via set_data).
- Periodic saving of historical plots (PNG) and optional animated GIF of map evolution.
- Handles varying grid sizes (32x32 .. 100x100) and provides optional zooming.
- Simple API to integrate into an RL training loop.

Tile convention:
    0 = LAND  (green)
    1 = WATER (blue)
    2 = SAND  (tan)

Usage example (bottom of file).
"""

import matplotlib
matplotlib.use("Agg")  
import os
import math
import time
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# optional dependencies
try:
    from scipy.ndimage import label
except Exception:
    label = None  # fallback if scipy not installed

try:
    import imageio
except Exception:
    imageio = None  # optional GIF support

# color map for tiles: LAND, WATER, SAND
_TILE_COLORS = ["#2b8c2b", "#4da6ff", "#f2d08b"]
_TILE_CMAP = ListedColormap(_TILE_COLORS)


# -------------------------
# Helper metric functions
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def largest_blob_fraction(grid: np.ndarray, tile_val: int) -> float:
    """Return fraction of cells in the largest connected component of tile_val.
    Uses scipy.ndimage.label (8-connectivity) if available; otherwise returns fraction of tile_val.
    """
    if grid.size == 0:
        return 0.0
    if label is None:
        # fallback: return simple fraction (not actually largest connected component)
        return float(np.mean(grid == tile_val))
    mask = (grid == tile_val).astype(np.int32)
    labeled, ncomp = label(mask, structure=np.ones((3, 3)))
    if ncomp == 0:
        return 0.0
    counts = np.bincount(labeled.ravel())
    if counts.size <= 1:
        return 0.0
    largest = counts[1:].max()
    return float(largest) / float(grid.size)


def sand_quality(grid: np.ndarray) -> float:
    """Proportion of sand tiles that are adjacent (3x3) to both land and water."""
    if grid.size == 0:
        return 0.0
    sand_positions = np.argwhere(grid == 2)
    if sand_positions.size == 0:
        return 0.0
    good = 0
    rows, cols = grid.shape
    for r, c in sand_positions:
        r0, r1 = max(0, r - 1), min(rows, r + 2)
        c0, c1 = max(0, c - 1), min(cols, c + 2)
        block = grid[r0:r1, c0:c1]
        if (block == 0).any() and (block == 1).any():
            good += 1
    return good / len(sand_positions)


def compute_metrics(grid: np.ndarray, total_reward: float) -> Dict[str, float]:
    """Compute the standard metrics for a grid and return a dict with reward included."""
    land_frac = float(np.mean(grid == 0))
    water_frac = float(np.mean(grid == 1))
    sand_frac = float(np.mean(grid == 2))
    largest_land_frac = largest_blob_fraction(grid, 0)
    sand_q = sand_quality(grid)
    
    # ✅ NEW: Additional evaluation metrics
    water_dev_low = max(0.0, 0.30 - water_frac)
    water_dev_high = max(0.0, water_frac - 0.50)
    max_frac = max(land_frac, water_frac, sand_frac)
    diversity_flag = 1 if max_frac > 0.65 else 0
    
    return {
        "total_reward": float(total_reward),
        "land_frac": land_frac,
        "water_frac": water_frac,
        "sand_frac": sand_frac,
        "largest_land_frac": largest_land_frac,
        "sand_quality": sand_q,
        "water_dev_low": water_dev_low,      # NEW
        "water_dev_high": water_dev_high,    # NEW
        "diversity_flag": diversity_flag,    # NEW
    }


# -------------------------
# MetricsLogger class
# -------------------------
class MetricsLogger:
    """
    Log metrics and display maps during RL training.

    Public methods:
      - log(episode, grid, total_reward): record episode metrics and preview image
      - display_map(grid=None, pause=0.001): immediate non-blocking update of the shown map
      - maybe_show_live(episode): call every episode to update when interval reached
      - maybe_save(episode): save plots/CSV/GIF periodically
      - save_plots(save_path): save summary plots (PNG)
      - save_csv(csv_path): save metrics to CSV
      - save_gif(gif_path, fps=6): optional animated GIF (requires imageio)
      - clear(): clear internal buffers

    Initialization:
      MetricsLogger(save_dir="metrics", live_every=1, save_every=50, moving_avg_window=10, preview_dpi=80, keep_frames=200, zoom=None)

    Parameters:
      save_dir: root path for outputs
      live_every: show live preview every N episodes (0 disables live)
      save_every: save plots/CSV every N episodes (0 disables periodic saves)
      moving_avg_window: window size for moving-average smoothing of reward plot
      preview_dpi: DPI for preview PNGs
      keep_frames: how many preview frames to keep in memory for GIF
      zoom: optional tuple (r0,r1,c0,c1) to zoom into a subregion for live display
    """

    def __init__(
        self,
        save_dir: str = "metrics",
        live_every: int = 1,
        save_every: int = 50,
        moving_avg_window: int = 10,
        preview_dpi: int = 80,
        keep_frames: int = 500,
        zoom: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.save_dir = save_dir
        self.live_every = max(0, int(live_every))
        self.save_every = max(0, int(save_every))
        self.moving_avg_window = max(1, int(moving_avg_window))
        self.preview_dpi = int(preview_dpi)
        self.keep_frames = int(keep_frames)
        self.zoom = zoom

        # directories
        self.live_dir = os.path.join(self.save_dir, "live")
        self.plots_dir = os.path.join(self.save_dir, "plots")
        self.previews_dir = os.path.join(self.save_dir, "previews")
        ensure_dir(self.save_dir)
        ensure_dir(self.live_dir)
        ensure_dir(self.plots_dir)
        ensure_dir(self.previews_dir)

        # data stores
        self.episodes: List[int] = []
        self.total_rewards: List[float] = []
        self.land_frac: List[float] = []
        self.water_frac: List[float] = []
        self.sand_frac: List[float] = []
        self.largest_land_frac: List[float] = []
        self.sand_quality: List[float] = []

        # preview frames for GIF (RGB arrays or saved PNG-path loaded arrays)
        self._frames: List[np.ndarray] = []

        # live plotting state
        self._fig = None
        self._ax = None
        self._im = None
        self._text = None
        
        self.sand_quality: List[float] = []
    
    # ✅ NEW: Additional metric storage
        self.water_dev_low: List[float] = []
        self.water_dev_high: List[float] = []
        self.diversity_flag: List[int] = []

    # -------------------------
    # Logging
    # -------------------------
    def log(self, episode: int, grid: np.ndarray, total_reward: float):
        """
        Record metrics for one episode and save a preview image.

        grid: 2D numpy array of ints {0,1,2}
        """
        if not isinstance(grid, np.ndarray):
            raise ValueError("grid must be a numpy.ndarray")
        if grid.ndim != 2:
            raise ValueError("grid must be 2D (H x W)")

        metrics = compute_metrics(grid, total_reward)
        
        self.episodes.append(int(episode))
        self.total_rewards.append(metrics["total_reward"])
        self.land_frac.append(metrics["land_frac"])
        self.water_frac.append(metrics["water_frac"])
        self.sand_frac.append(metrics["sand_frac"])
        self.largest_land_frac.append(metrics["largest_land_frac"])
        self.sand_quality.append(metrics["sand_quality"])
        self.water_dev_low.append(metrics["water_dev_low"])
        self.water_dev_high.append(metrics["water_dev_high"])
        self.diversity_flag.append(metrics["diversity_flag"])
        # Save preview (small PNG) and keep frame in memory
        preview_path = os.path.join(self.previews_dir, f"ep_{int(episode):05d}.png")
        self._save_preview(grid, preview_path, episode, metrics)

        # Try to load image as array for GIF; fall back to raw grid if reading fails
        try:
            arr = plt.imread(preview_path)
        except Exception:
            arr = grid.copy()
        self._frames.append(arr)
        if len(self._frames) > self.keep_frames:
            self._frames.pop(0)

        # Optionally display live and/or save periodic plots
        if self.live_every and (episode % self.live_every == 0):
            self.display_map(grid, overlay=metrics)
        if self.save_every and (episode % self.save_every == 0):
            # save PNG plots and CSV
            plot_path = os.path.join(self.plots_dir, f"metrics_ep{episode:05d}.png")
            self.save_plots(plot_path)
            csv_path = os.path.join(self.plots_dir, f"metrics_ep{episode:05d}.csv")
            self.save_csv(csv_path)
        
        # ✅ NEW: Store additional metrics
        self.water_dev_low.append(metrics["water_dev_low"])
        self.water_dev_high.append(metrics["water_dev_high"])
        self.diversity_flag.append(metrics["diversity_flag"])
    # -------------------------
    # Preview image
    # -------------------------
    def _save_preview(self, grid: np.ndarray, path: str, episode: int, metrics: Dict[str, float]):
        """Save a small annotated PNG preview for quick inspection and GIF creation."""
        ensure_dir(os.path.dirname(path) or ".")
        fig = plt.figure(figsize=(4, 4), dpi=self.preview_dpi // 10 or 80)
        ax = fig.add_subplot(111)
        if self.zoom:
            r0, r1, c0, c1 = self.zoom
            img = grid[r0:r1, c0:c1]
        else:
            img = grid
        ax.imshow(img, cmap=_TILE_CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax.axis("off")
        title = f"Ep {episode}  R {metrics['total_reward']:.2f}\n" \
                f"W {metrics['water_frac']:.2f} L {metrics['land_frac']:.2f} S {metrics['sand_frac']:.2f}"
        ax.set_title(title, fontsize=8)
        fig.tight_layout(pad=0)
        fig.savefig(path, dpi=80)
        plt.close(fig)

    # -------------------------
    # Real-time display
    # -------------------------
    def display_map(self, grid: np.ndarray = None, overlay: Optional[Dict[str, float]] = None, pause: float = 0.001):
        """
        Non-blocking display/update of current map.
        If no grid is provided, uses the last preview frame (if available).
        overlay: optional dict returned from compute_metrics for text overlay.
        pause: matplotlib pause time to allow the UI to update.
        """
        # determine image data
        if grid is None:
            if not self._frames:
                return
            frame = self._frames[-1]
            is_rgb = (frame.ndim == 3 and frame.shape[2] in (3, 4))
            img = frame
        else:
            frame = grid
            is_rgb = False
            if self.zoom:
                r0, r1, c0, c1 = self.zoom
                img = frame[r0:r1, c0:c1]
            else:
                img = frame

        # initialize figure if needed
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            if is_rgb:
                self._im = self._ax.imshow(img, interpolation="nearest")
            else:
                self._im = self._ax.imshow(img, cmap=_TILE_CMAP, vmin=0, vmax=2, interpolation="nearest")
            self._ax.axis("off")
            self._text = self._ax.text(0.01, 0.99, "", ha="left", va="top", color="white",
                                       fontsize=9, transform=self._ax.transAxes,
                                       bbox=dict(facecolor='black', alpha=0.4, pad=3))
            self._fig.tight_layout()
        else:
            if is_rgb:
                self._im.set_data(img)
            else:
                self._im.set_data(img)

        # overlay metrics text
        if overlay:
            txt = f"Ep {int(self.episodes[-1]) if self.episodes else '?'}  R {overlay['total_reward']:.2f}\n" \
                  f"W {overlay['water_frac']:.2f}  L {overlay['land_frac']:.2f}  S {overlay['sand_frac']:.2f}"
            self._text.set_text(txt)
        else:
            # try to show last metrics if present
            if self.episodes:
                txt = f"Ep {self.episodes[-1]}  R {self.total_rewards[-1]:.2f}\n" \
                      f"W {self.water_frac[-1]:.2f}  L {self.land_frac[-1]:.2f}  S {self.sand_frac[-1]:.2f}"
                self._text.set_text(txt)

        plt.pause(pause)

    # -------------------------
    # Save summary plots
    # -------------------------
    def save_plots(self, save_path: Optional[str] = None, show: bool = False):
        """
        Save a 2x2 summary plot:
          - Reward (with moving average)
          - Tile fractions (land/water/sand)
          - Largest land fraction
          - Sand quality
        """
        if not self.episodes:
            return
        if save_path is None:
            save_path = os.path.join(self.plots_dir, "metrics_summary.png")
        ensure_dir(os.path.dirname(save_path) or ".")

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        # reward
        axs[0, 0].plot(self.episodes, self.total_rewards, label="Reward")
        if len(self.total_rewards) >= self.moving_avg_window:
            ma = self._moving_average(self.total_rewards, self.moving_avg_window)
            axs[0, 0].plot(self.episodes, ma, label=f"MA({self.moving_avg_window})")
        axs[0, 0].set_title("Total Reward")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # fractions
        axs[0, 1].plot(self.episodes, self.land_frac, label="Land")
        axs[0, 1].plot(self.episodes, self.water_frac, label="Water")
        axs[0, 1].plot(self.episodes, self.sand_frac, label="Sand")
        axs[0, 1].set_title("Tile Fractions")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # largest land fraction
        axs[1, 0].plot(self.episodes, self.largest_land_frac, label="Largest Land Fraction")
        axs[1, 0].set_title("Largest Land Blob Fraction")
        axs[1, 0].grid(True)

        # sand quality
        axs[1, 1].plot(self.episodes, self.sand_quality, label="Sand Quality")
        axs[1, 1].set_title("Sand Quality")
        axs[1, 1].grid(True)

        for ax in axs.flat:
            ax.set_xlabel("Episode")

        fig.tight_layout()
        fig.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    # -------------------------
    # Save CSV
    # -------------------------
    def save_csv(self, csv_path: str):
        """Save tabular metrics as CSV."""
        if not self.episodes:
            return
        ensure_dir(os.path.dirname(csv_path) or ".")
        import csv
        
        # ✅ UPDATED: Extended header
        header = ["episode", "total_reward", "land_frac", "water_frac", "sand_frac", 
                "largest_land_frac", "sand_quality", 
                "water_dev_low", "water_dev_high", "diversity_flag"]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, ep in enumerate(self.episodes):
                writer.writerow([
                    ep,
                    self.total_rewards[i],
                    self.land_frac[i],
                    self.water_frac[i],
                    self.sand_frac[i],
                    self.largest_land_frac[i],
                    self.sand_quality[i],
                    self.water_dev_low[i],
                    self.water_dev_high[i],
                    self.diversity_flag[i],
                ])

    # -------------------------
    # Save animated GIF
    # -------------------------
    def save_gif(self, gif_path: str, fps: int = 6):
        """Save an animated GIF of the collected preview frames. Needs imageio."""
        if imageio is None:
            raise RuntimeError("imageio is required for GIF export (pip install imageio).")
        if not self._frames:
            raise RuntimeError("No frames available to write GIF.")
        ensure_dir(os.path.dirname(gif_path) or ".")
        imageio.mimsave(gif_path, self._frames, fps=fps)
        return gif_path

    # -------------------------
    # Utilities
    # -------------------------
    def _moving_average(self, data: List[float], w: int) -> List[float]:
        if len(data) < w:
            return data[:]
        a = np.array(data, dtype=float)
        c = np.cumsum(np.insert(a, 0, 0.0))
        ma = (c[w:] - c[:-w]) / float(w)
        pad = [ma[0]] * (w - 1)
        return pad + ma.tolist()

    def clear(self):
        """Clear all stored metrics and frames (does not delete files)."""
        self.episodes.clear()
        self.total_rewards.clear()
        self.land_frac.clear()
        self.water_frac.clear()
        self.sand_frac.clear()
        self.largest_land_frac.clear()
        self.sand_quality.clear()
        self._frames.clear()
        if self._fig:
            try:
                plt.close(self._fig)
            except Exception:
                pass
        self._fig = None
        self._ax = None
        self._im = None
        self._text = None
        self.sand_quality.clear()
        
        # ✅ NEW: Clear additional metrics
        self.water_dev_low.clear()
        self.water_dev_high.clear()
        self.diversity_flag.clear()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Simple demo showing usage with a synthetic map
    logger = MetricsLogger(save_dir="metrics_demo", live_every=1, save_every=5, moving_avg_window=5)

    for ep in range(1, 21):
        # create a synthetic grid (100x100) with islands and sand
        size = 100
        rng = np.random.RandomState(ep)
        base = rng.rand(size, size)
        grid = np.where(base > 0.55, 0, 1).astype(np.int32)
        # thin sand band near threshold
        grid[(base > 0.52) & (base <= 0.58)] = 2
        reward = float(100.0 * np.mean(grid == 0) - 80.0 * np.mean(grid == 1) + rng.randn() * 3.0)
        logger.log(ep, grid, reward)
        # the display_map call is handled by logger.log when live_every triggers

    # save outputs
    summary_png = os.path.join("metrics_demo", "plots", "summary.png")
    logger.save_plots(summary_png, show=False)
    logger.save_csv(os.path.join("metrics_demo", "plots", "metrics.csv"))
    if imageio is not None:
        try:
            logger.save_gif(os.path.join("metrics_demo", "live", "evolution.gif"), fps=6)
            print("Saved GIF.")
        except Exception as e:
            print("GIF save failed:", e)
    print("Demo complete. Previews in metrics_demo/previews, plots in metrics_demo/plots.")
