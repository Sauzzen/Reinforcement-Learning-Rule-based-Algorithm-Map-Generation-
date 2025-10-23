import os
import sys
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

print("CWD:", os.getcwd())

proj = r"F:\Map Generation Theoretical 5\Generation Trainer V2"
if proj not in sys.path:
    sys.path.insert(0, proj)

from map_env import MapEnvironment, compute_map_metrics, LAND, WATER, SAND

outdir = os.path.join(proj, "eval_panel")
os.makedirs(outdir, exist_ok=True)

seeds = range(50000, 50010)
ok = 0

# Correct color mapping: LAND=0 (green), WATER=1 (blue), SAND=2 (tan)
cmap = ListedColormap(["#2b8c2b", "#4da6ff", "#f2d08b"])

for s in seeds:
    try:
        env = MapEnvironment(size=64, seed=s)
        
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        
        # Find the actual terrain grid
        grid = None
        for attr in ['_grid', '_map', 'map', 'terrain', 'state', '_terrain']:
            if hasattr(env, attr):
                candidate = getattr(env, attr)
                if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
                    grid = candidate.copy()
                    break
        
        # If still no grid, reconstruct carefully from observation
        if grid is None:
            if obs.ndim == 3 and obs.shape[0] in [3, 4, 5]:  
                # Multi-channel one-hot encoding
                grid = np.argmax(obs, axis=0).astype(np.uint8)
            elif obs.ndim == 2:
                grid = obs.copy()
            else:
                raise AttributeError(f"Could not extract grid. Obs shape: {obs.shape}")
        
        # Verify the grid makes sense by checking unique values
        unique_vals = np.unique(grid)
        print(f"[DEBUG] seed {s}: unique tile values = {unique_vals}")
        
        metrics = compute_map_metrics(grid)
        
        # Export JSON
        json_path = os.path.join(outdir, f"seed_{s}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "width": int(grid.shape[1]),
                "height": int(grid.shape[0]),
                "tiles": grid.tolist(),
                "seed": int(s),
                "metrics": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v) for k, v in metrics.items()}
            }, f, indent=2)
        
        # Export PNG
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
        ax.axis('off')
        png_path = os.path.join(outdir, f"seed_{s}.png")
        fig.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        
        ok += 1
        print(f"[OK] seed {s} - water: {metrics.get('water_frac', 0):.2f}, land: {metrics.get('land_frac', 0):.2f}, sand: {metrics.get('sand_frac', 0):.2f}")
    except Exception as e:
        print(f"[FAIL] seed {s}: {e}")
        traceback.print_exc()

print(f"Done. Wrote {ok}/{len(list(seeds))} maps to {outdir}")
