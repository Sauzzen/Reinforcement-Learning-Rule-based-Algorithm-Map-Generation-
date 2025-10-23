# visualize_map.py
"""
Visualization utilities for saving map chunks, full maps, and heatmaps as PNGs.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------
# Tile color map
# -----------------
COLORS = {
    0: (141, 217, 109),   # LAND - green
    1: (51, 102, 255),    # WATER - blue
    2: (238, 214, 175),   # SAND - light beige
    3: (170, 170, 170),   # ROCK - gray
    4: (30, 30, 30),      # ROAD - dark
}

DEFAULT_SCALE = 10


def _tile_to_rgb(tile: int) -> tuple[int, int, int]:
    """Convert a tile integer code into an RGB color."""
    return COLORS.get(int(tile), (0, 0, 0))


# -----------------
# Chunk visualization
# -----------------
def save_chunk_png(chunk_grid: np.ndarray, path_png: str, scale: int = 20) -> None:
    """Save a small chunk (2D grid) as PNG."""
    n = chunk_grid.shape[0]
    img = Image.new("RGB", (n, n))
    px = img.load()

    for i in range(n):
        for j in range(n):
            px[j, i] = _tile_to_rgb(chunk_grid[i, j])

    img = img.resize((n * scale, n * scale), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    img.save(path_png)


# -----------------
# World visualization
# -----------------
def stitch_chunks(chunks_list: list[np.ndarray], chunks_per_row: int = 5) -> np.ndarray | None:
    """Combine multiple chunks into a single world grid."""
    if not chunks_list:
        return None

    chunk_n = chunks_list[0].shape[0]
    rows = []
    nrows = (len(chunks_list) + chunks_per_row - 1) // chunks_per_row

    for r in range(nrows):
        start = r * chunks_per_row
        row_chunks = chunks_list[start:start + chunks_per_row]

        # Pad incomplete row with LAND tiles
        if len(row_chunks) < chunks_per_row:
            pad = [np.full((chunk_n, chunk_n), 0, dtype=int)] * (chunks_per_row - len(row_chunks))
            row_chunks += pad

        rows.append(np.hstack(row_chunks))

    return np.vstack(rows)


def save_world_png(world_grid: np.ndarray, path_png: str, scale: int = DEFAULT_SCALE) -> None:
    """Save a 2D world grid as a PNG image."""
    h, w = world_grid.shape
    img = Image.new("RGB", (w, h))
    px = img.load()

    for i in range(h):
        for j in range(w):
            px[j, i] = _tile_to_rgb(world_grid[i, j])

    img = img.resize((w * scale, h * scale), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    img.save(path_png)


def save_full_map(full_map: np.ndarray, episode: int, save_dir: str = "full_maps", scale: int = DEFAULT_SCALE) -> None:
    """Save the full map with episode-based filename."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"full_map_ep{episode}.png")
    save_world_png(full_map, path, scale=scale)


# -----------------
# Heatmap visualization
# -----------------
def save_heatmap(grid: np.ndarray, episode: int, save_dir: str = "heatmaps") -> None:
    """
    Save a heatmap of the map for terrain visualization.

    Args:
        grid: 2D numpy array of tile codes.
        episode: Episode index for filename.
        save_dir: Directory to save heatmap.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6,6))
    cmap = np.array([
        [141/255, 217/255, 109/255],   # LAND - green
        [51/255, 102/255, 255/255],    # WATER - blue
        [238/255, 214/255, 175/255],   # SAND - beige
        [0.66, 0.66, 0.66],            # ROCK - gray
        [0.12, 0.12, 0.12]             # ROAD - dark
    ])
    # Convert tile codes to RGB image for heatmap
    img_rgb = cmap[grid.astype(int)]
    plt.imshow(img_rgb, interpolation='nearest')
    plt.title(f"Episode {episode}")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"heatmap_ep{episode}.png"), bbox_inches='tight')
    plt.close()
