# rule_based_polish.py
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from map_env import LAND, WATER, SAND, NUM_TILES

class RuleBasedPolisher:
    def __init__(self, min_island_frac: float = 0.15, max_island_frac: float = 0.6):
        """
        min_island_frac / max_island_frac: fraction of world size for largest island
        """
        self.min_island_frac = min_island_frac
        self.max_island_frac = max_island_frac

    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def count_blobs(grid, val):
        labeled, _ = label(grid == val)
        return np.bincount(labeled.flatten())[1:]  # ignore background

    @staticmethod
    def get_neighbors(r, c, n):
        return [(nr, nc) for nr in (r-1, r, r+1) for nc in (c-1, c, c+1)
                if 0 <= nr < n and 0 <= nc < n and not (nr == r and nc == c)]

    # -----------------------------
    # Polishing steps
    # -----------------------------
    def apply(self, grid: np.ndarray) -> np.ndarray:
        n = grid.shape[0]
        new_grid = grid.copy()

        # Step 1: Ensure largest island is within target fraction
        largest_frac = self._adjust_islands(new_grid, n)

        # Step 2: Apply shoreline rules (LAND-SAND-WATER)
        new_grid = self._shoreline_rules(new_grid, n)

        # Step 3: Fill isolated water pockets
        new_grid = self._fill_water_pockets(new_grid, n)

        return new_grid

    # -----------------------------
    # Internal methods
    # -----------------------------
    def _adjust_islands(self, grid, n):
        # Identify LAND blobs
        sizes = self.count_blobs(grid, LAND)
        if len(sizes) == 0:
            return 0.0

        largest_idx = np.argmax(sizes)
        largest_size = sizes[largest_idx]
        frac = largest_size / (n * n)

        # Remove extra islands if too many
        if frac > self.max_island_frac:
            labeled, _ = label(grid == LAND)
            mask = labeled != (largest_idx + 1)
            grid[mask] = WATER
        elif frac < self.min_island_frac:
            # Expand largest island by converting nearby water to LAND
            labeled, _ = label(grid == LAND)
            island_mask = labeled == (largest_idx + 1)
            self._expand_island(grid, island_mask, n)

        return frac

    def _expand_island(self, grid, island_mask, n):
        # For each border LAND cell, convert adjacent WATER to LAND
        for r in range(n):
            for c in range(n):
                if island_mask[r, c]:
                    for nr, nc in self.get_neighbors(r, c, n):
                        if grid[nr, nc] == WATER:
                            grid[nr, nc] = LAND

    def _shoreline_rules(self, grid, n):
        new_grid = grid.copy()
        for r in range(n):
            for c in range(n):
                if grid[r, c] == LAND:
                    for nr, nc in self.get_neighbors(r, c, n):
                        if grid[nr, nc] == WATER:
                            new_grid[r, c] = LAND
                            new_grid[nr, nc] = SAND
        return new_grid

    def _fill_water_pockets(self, grid, n):
        # Fill any small enclosed water pockets with LAND
        water_mask = grid == WATER
        filled = binary_fill_holes(~water_mask)  # fill holes
        pockets = filled & water_mask
        grid[pockets] = LAND
        return grid
