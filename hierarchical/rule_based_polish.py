# rule_based_polish.py
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from map_env import LAND, WATER, SAND, NUM_TILES

class RuleBasedPolisher:
    def __init__(self, min_island_frac: float = 0.15, max_island_frac: float = 0.6, max_pocket_size: int = 10):
        """
        min_island_frac / max_island_frac: fraction of world size for largest island
        max_pocket_size: maximum size of water pockets to fill (in cells)
        """
        self.min_island_frac = min_island_frac
        self.max_island_frac = max_island_frac
        self.max_pocket_size = max_pocket_size

    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def count_blobs(grid, val):
        labeled, _ = label(grid == val)
        sizes = np.bincount(labeled.flatten())[1:]  # ignore background
        return labeled, sizes

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
        self._adjust_islands(new_grid, n)

        # Step 2: Apply shoreline rules (LAND-SAND-WATER)
        new_grid = self._shoreline_rules(new_grid, n)

        # Step 3: Fill small isolated water pockets only
        new_grid = self._fill_water_pockets(new_grid, n)

        # Step 4: Clean up stray sand tiles
        new_grid = self._clean_sand(new_grid, n)

        return new_grid

    # -----------------------------
    # Internal methods
    # -----------------------------
    def _adjust_islands(self, grid, n):
        # Identify LAND blobs
        labeled, sizes = self.count_blobs(grid, LAND)
        if len(sizes) == 0:
            return

        largest_idx = np.argmax(sizes)
        largest_size = sizes[largest_idx]
        frac = largest_size / (n * n)

        # Remove smaller islands if main island is too big
        if frac > self.max_island_frac:
            # Keep only the largest island
            mask = labeled != (largest_idx + 1)
            grid[mask & (grid == LAND)] = WATER
        elif frac < self.min_island_frac:
            # Expand largest island iteratively
            island_mask = labeled == (largest_idx + 1)
            target_size = int(self.min_island_frac * n * n)
            current_size = largest_size
            
            # Iteratively expand until we reach target or can't expand further
            max_iterations = 20
            for _ in range(max_iterations):
                if current_size >= target_size:
                    break
                expanded = self._expand_island_once(grid, island_mask, n)
                if not expanded:  # No more cells to expand into
                    break
                # Recalculate island mask and size
                island_mask = grid == LAND
                current_size = island_mask.sum()

    def _expand_island_once(self, grid, island_mask, n):
        """Expand island by one layer, return True if any expansion occurred"""
        candidates = []
        
        # Find all water cells adjacent to current island
        for r in range(n):
            for c in range(n):
                if island_mask[r, c]:  # This is part of the island
                    for nr, nc in self.get_neighbors(r, c, n):
                        if grid[nr, nc] == WATER:
                            candidates.append((nr, nc))
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        if not candidates:
            return False
        
        # Convert all candidate cells to land
        for r, c in candidates:
            grid[r, c] = LAND
        
        return True

    def _shoreline_rules(self, grid, n):
        """Create SAND buffer between LAND and WATER"""
        new_grid = grid.copy()
        
        # Find water cells adjacent to land and convert to sand
        for r in range(n):
            for c in range(n):
                if grid[r, c] == WATER:
                    # Check if adjacent to LAND
                    has_land_neighbor = any(
                        grid[nr, nc] == LAND 
                        for nr, nc in self.get_neighbors(r, c, n)
                    )
                    if has_land_neighbor:
                        new_grid[r, c] = SAND
        
        return new_grid

    def _fill_water_pockets(self, grid, n):
        """Fill small enclosed water pockets with LAND"""
        labeled, sizes = self.count_blobs(grid, WATER)
        
        # Fill pockets smaller than max_pocket_size
        for i, size in enumerate(sizes):
            if size <= self.max_pocket_size:
                pocket_mask = labeled == (i + 1)
                grid[pocket_mask] = LAND
        
        return grid

    def _clean_sand(self, grid, n):
        """Remove stray sand tiles that don't form proper shores"""
        new_grid = grid.copy()
        
        for r in range(n):
            for c in range(n):
                if grid[r, c] == SAND:
                    neighbors = [grid[nr, nc] for nr, nc in self.get_neighbors(r, c, n)]
                    
                    # Check if sand is actually between land and water
                    has_land = LAND in neighbors
                    has_water = WATER in neighbors
                    
                    if not (has_land and has_water):
                        # This sand isn't a proper shore, convert to dominant neighbor
                        land_count = neighbors.count(LAND)
                        water_count = neighbors.count(WATER)
                        sand_count = neighbors.count(SAND)
                        
                        if land_count > water_count and land_count > sand_count:
                            new_grid[r, c] = LAND
                        elif water_count > sand_count:
                            new_grid[r, c] = WATER
                        # If sand is dominant or tie, keep as sand
        
        return new_grid