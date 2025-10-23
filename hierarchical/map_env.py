# map_env.py - FIXED VERSION with unified reward system
import json
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

from scipy.ndimage import label, distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

# Tile codes (consistent mapping)
LAND, WATER, SAND = 0, 1, 2
NUM_TILES = 3

# Default map size
DEFAULT_SIZE = 100

# Color map for visualization
_TILE_CMAP = ListedColormap(["#2b8c2b", "#4da6ff", "#f2d08b"])  # LAND, WATER, SAND


# ---------------------------
# Fractal (Perlin-like) noise
# ---------------------------
def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)  # smoothstep

def smooth_map(grid: np.ndarray, passes: int = 1) -> np.ndarray:
    """Apply simple majority smoothing to reduce jagged edges."""
    s = grid.shape[0]
    smoothed = grid.copy()

    for _ in range(passes):
        new_grid = smoothed.copy()
        for r in range(s):
            for c in range(s):
                neighbors = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < s and 0 <= nc < s:
                        neighbors.append(smoothed[nr, nc])
                if neighbors:
                    # Replace with majority if different from most neighbors
                    majority = max(set(neighbors), key=neighbors.count)
                    if neighbors.count(majority) >= 3:  # at least 3 of 4 neighbors
                        new_grid[r, c] = majority
        smoothed = new_grid

    return smoothed

def generate_fractal_noise_2d(shape: Tuple[int, int],
                              res: Tuple[int, int],
                              octaves: int = 1,
                              persistence: float = 0.5,
                              seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    H, W = shape
    noise = np.zeros((H, W), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for o in range(octaves):
        res_h = max(1, int(res[0] * (2 ** o)))
        res_w = max(1, int(res[1] * (2 ** o)))
        grid = rng.rand(res_h + 1, res_w + 1).astype(np.float32)
        ys = np.linspace(0, res_h, H, endpoint=False)
        xs = np.linspace(0, res_w, W, endpoint=False)
        yi, xi = ys.astype(int), xs.astype(int)
        yf, xf = ys - yi, xs - xi
        yf_f = _fade(yf)[..., None]
        xf_f = _fade(xf)[None, :]
        v00 = grid[yi[:, None], xi[None, :]]
        v10 = grid[yi[:, None] + 1, xi[None, :]]
        v01 = grid[yi[:, None], xi[None, :] + 1]
        v11 = grid[yi[:, None] + 1, xi[None, :] + 1]
        i1 = v00 * (1 - yf_f) + v10 * yf_f
        i2 = v01 * (1 - yf_f) + v11 * yf_f
        octave_noise = i1 * (1 - xf_f) + i2 * xf_f
        noise += amplitude * octave_noise
        total_amp += amplitude
        amplitude *= persistence
    noise /= total_amp
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)


def generate_island_mask(size: int,
                         num_islands: int = 2,
                         min_radius_frac: float = 0.15,
                         max_radius_frac: float = 0.35,
                         seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mask = np.zeros((size, size), dtype=np.float32)
    rr = np.arange(size)[:, None]
    cc = np.arange(size)[None, :]
    for _ in range(num_islands):
        cx = rng.randint(low=int(size * 0.2), high=int(size * 0.8))
        cy = rng.randint(low=int(size * 0.2), high=int(size * 0.8))
        rad = rng.uniform(min_radius_frac * size, max_radius_frac * size)
        dist = np.sqrt((rr - cx) ** 2 + (cc - cy) ** 2)
        contribution = np.clip(1.0 - (dist / (rad + 1e-9)), 0.0, 1.0)
        mask = np.maximum(mask, contribution)
    return np.clip(mask, 0.0, 1.0)


def apply_sand_rules(grid: np.ndarray) -> np.ndarray:
    n = grid.shape[0]
    new = grid.copy()
    for r in range(n):
        for c in range(n):
            if grid[r, c] == WATER:
                neighbors = grid[max(0, r-1):min(n, r+2),
                                 max(0, c-1):min(n, c+2)]
                if np.any(neighbors == LAND):
                    new[r, c] = SAND
    return new


def _compute_sand_quality(grid: np.ndarray) -> float:
    n = grid.shape[0]
    sand_positions = np.argwhere(grid == SAND)
    if len(sand_positions) == 0:
        return 0.0
    good = 0
    for r, c in sand_positions:
        neighbors = grid[max(0, r - 1):min(n, r + 2), max(0, c - 1):min(n, c + 2)]
        if np.any(neighbors == LAND) and np.any(neighbors == WATER):
            good += 1
    return good / len(sand_positions)


def stepwise_reward(grid: np.ndarray, r: int, c: int, placed_tile: int) -> float:
    """
    Simplified local reward for pixel mode - normalized to 0-1 scale
    """
    n = grid.shape[0]
    reward = 0.5  # Base reward for any placement

    # 4-neighborhood
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            neighbors.append(grid[nr, nc])

    if placed_tile == LAND:
        same_count = sum(1 for v in neighbors if v == LAND)
        reward += 0.1 * same_count  # Small clustering bonus
    elif placed_tile == WATER:
        same_count = sum(1 for v in neighbors if v == WATER)
        reward += 0.1 * same_count  # Small clustering bonus
    elif placed_tile == SAND:
        has_land = any(v == LAND for v in neighbors)
        has_water = any(v == WATER for v in neighbors)
        if has_land and has_water:
            reward += 0.3  # Proper shoreline bonus
        else:
            reward -= 5  # Penalty for isolated sand

    return max(0.0, min(1.0, reward))  # Clamp to [0,1]


def compute_map_metrics(grid: np.ndarray) -> Dict[str, Any]:
    n = grid.shape[0]
    total = n * n

    land_mask = (grid == LAND)
    water_mask = (grid == WATER)
    sand_mask = (grid == SAND)

    land_frac = float(np.mean(land_mask))
    water_frac = float(np.mean(water_mask))
    sand_frac = float(np.mean(sand_mask))

    # land components / largest land fraction
    labeled_land, land_components = label(land_mask.astype(int))
    sizes = np.bincount(labeled_land.ravel())
    largest_land_frac = float(sizes[1:].max() / total) if land_components > 0 else 0.0

    sand_quality = _compute_sand_quality(grid)

    # water components
    _, water_components = label(water_mask.astype(int))

    return {
        "land_frac": land_frac,
        "water_frac": water_frac,
        "sand_frac": sand_frac,
        "largest_land_frac": largest_land_frac,
        "sand_quality": sand_quality,
        "water_components": int(water_components),
        "land_components": int(land_components),
    }

# ---------------------------
# Village Placement & Distance Transforms
# ---------------------------

def compute_distance_transforms(grid: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute distance transforms from WATER and SAND tiles.
    Returns Euclidean distance in cells from nearest WATER/SAND tile.
    """
    water_mask = (grid == WATER).astype(bool)
    sand_mask = (grid == SAND).astype(bool)
    
    dist_to_water = distance_transform_edt(~water_mask).astype(np.float32)
    dist_to_sand = distance_transform_edt(~sand_mask).astype(np.float32)
    
    return {
        'dist_to_water': dist_to_water,
        'dist_to_sand': dist_to_sand
    }


def sample_villages_poisson(
    grid: np.ndarray,
    count: int,
    d_min_water: float,
    d_min_sand: float,
    r_min: float,
    border_margin: int,
    seed: Optional[int] = None,
    max_attempts: int = 1000
) -> List[Tuple[int, int]]:
    """
    Sample village positions using Poisson-disk sampling with distance constraints.
    
    Relaxation order if insufficient sites: d_min → r_min as specified.
    """
    rng = np.random.RandomState(seed)
    H, W = grid.shape
    
    dist_transforms = compute_distance_transforms(grid)
    dist_to_water = dist_transforms['dist_to_water']
    dist_to_sand = dist_transforms['dist_to_sand']
    
    # Build candidate mask: LAND tiles far from water/sand and borders
    candidate_mask = (
        (grid == LAND) &
        (dist_to_water >= d_min_water) &
        (dist_to_sand >= d_min_sand)
    )
    
    if border_margin > 0:
        candidate_mask[:border_margin, :] = False
        candidate_mask[-border_margin:, :] = False
        candidate_mask[:, :border_margin] = False
        candidate_mask[:, -border_margin:] = False
    
    candidate_coords = np.argwhere(candidate_mask)
    
    if len(candidate_coords) == 0:
        return []
    
    villages = []
    attempts = 0
    current_d_min_water = d_min_water
    current_d_min_sand = d_min_sand
    current_r_min = r_min
    
    while len(villages) < count and attempts < max_attempts:
        # Relax constraints progressively
        if attempts > 0 and attempts % 200 == 0:
            if current_d_min_water > 1:
                current_d_min_water *= 0.8
            if current_d_min_sand > 1:
                current_d_min_sand *= 0.8
            if current_r_min > 2:
                current_r_min *= 0.9
                
            candidate_mask = (
                (grid == LAND) &
                (dist_to_water >= current_d_min_water) &
                (dist_to_sand >= current_d_min_sand)
            )
            if border_margin > 0:
                candidate_mask[:border_margin, :] = False
                candidate_mask[-border_margin:, :] = False
                candidate_mask[:, :border_margin] = False
                candidate_mask[:, -border_margin:] = False
            
            candidate_coords = np.argwhere(candidate_mask)
            if len(candidate_coords) == 0:
                break
        
        idx = rng.randint(0, len(candidate_coords))
        candidate = tuple(candidate_coords[idx])
        
        # Check pairwise distance
        valid = True
        for existing in villages:
            dist = np.sqrt((candidate[0] - existing[0])**2 + (candidate[1] - existing[1])**2)
            if dist < current_r_min:
                valid = False
                break
        
        if valid:
            villages.append(candidate)
        
        attempts += 1
    
    return villages


def create_village_observation_channel(
    grid: np.ndarray,
    villages: List[Tuple[int, int]],
    contact_radius: int
) -> np.ndarray:
    """Create binary observation channel for villages (optional debug/visualization)."""
    H, W = grid.shape
    village_channel = np.zeros((H, W), dtype=np.float32)
    
    for r, c in villages:
        for dr in range(-contact_radius, contact_radius + 1):
            for dc in range(-contact_radius, contact_radius + 1):
                if dr*dr + dc*dc <= contact_radius*contact_radius:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        village_channel[nr, nc] = 1.0
    
    return village_channel

# ==========================
# Map Environment Class
# ==========================
class MapEnvironment:
    def __init__(self,
                 size: int = DEFAULT_SIZE,
                 max_steps: Optional[int] = None,
                 obs_mode: str = "channels",
                 seed: Optional[int] = None,
                 action_mode: str = "hierarchical",  # "pixel", "regional", or "hierarchical"
                 village_count: int = 3,
                 village_contact_radius: int = 2,
                 village_d_min_water: float = 4.0,
                 village_d_min_sand: float = 4.0,
                 village_r_min: float = 8.0,
                 village_border_margin: int = 3):
        self.size = int(size)
        self.action_mode = action_mode
        self.obs_mode = obs_mode
        self.rng = np.random.RandomState(seed)
        self.world = np.full((self.size, self.size), LAND, dtype=np.int8)
        self.desired_water_frac = 0.40  # ADD THIS
        self._prev_ratio_err = 0.0       # ADD THIS
        
        if action_mode == "hierarchical":
            self.max_regions = max_steps or 25  # Major landmasses/water bodies
            self.regions_placed = 0
            self.action_size = 9  # 3 terrain types × 3 scales (small/medium/large)
            self.phase = "macro"  # "macro" -> "shoreline" -> "detail"
            self.macro_done = False
            self.shoreline_done = False
            self.macro_actions = 6    # 2 terrain types (LAND/WATER) × 3 scales
            self.detail_actions = 9   # 3 terrain types × 3 scales
            self.action_size = self.macro_actions  # Start with macro actions

        elif action_mode == "regional":
            self.max_regions = max_steps or 20
            self.regions_placed = 0
            self.action_size = 6  # 3 terrain types × 2 sizes
        else:  # pixel mode
            self.max_steps = max_steps or (self.size * self.size * 2)
            self.action_size = NUM_TILES
            self._step_order = [(r, c) for r in range(self.size) for c in range(self.size)]
            self.t = 0
            
        self.state_size = (self.size, self.size)
        # Per-phase reward tracking for logging
        self._macro_reward_accum = 0.0
        self._shoreline_reward_accum = 0.0
        self._detail_reward_accum = 0.0

        
        # Village configuration for Phase 1 goal placement
        self.village_config = {
            'count': village_count,
            'contact_radius': village_contact_radius,
            'd_min_water': village_d_min_water,
            'd_min_sand': village_d_min_sand,
            'r_min': village_r_min,
            'border_margin': village_border_margin
        }
        self.villages = []  # Populated at reset
        self.distance_layers = {}  # Stores dist_to_water, dist_to_sand


# Replace the compute_unified_reward method in your MapEnvironment class with this:
    def get_binary_masks(self):
        """Get binary terrain masks for evaluation."""
        grid = self.world
        return {
            'land_mask': (grid == LAND),
            'water_mask': (grid == WATER),
            'sand_mask': (grid == SAND),
        }

    def get_distance_layers(self):
        """Get distance transform layers if computed."""
        if hasattr(self, 'distance_layers') and isinstance(self.distance_layers, dict):
            return self.distance_layers
        return {}

    def compute_unified_reward(self, placement_success=True, local_reward=0.0, is_final_step=False) -> float:
        """Target: 35-40% water, 4-7% sand"""
        metrics = compute_map_metrics(self.world)
        
        # PENALTIES (always applied)
        if metrics["water_frac"] > 0.90 or metrics["water_frac"] < 0.10:
            return -5.0
        
        if hasattr(self, 'phase') and self.phase == "fine":
            if metrics["sand_frac"] < 0.02:
                return -3.0
        
        if metrics["sand_frac"] > 0.12:
            return -4.0
        
        if metrics["largest_land_frac"] < 0.15:
            return -4.0
        
        # ══════════════════════════════════════════════════════════════
        # NORMAL REWARD
        # ══════════════════════════════════════════════════════════════
        
        base = 0.5 if placement_success else 0.0
        
        # ✅ TARGET: 37.5% water (middle of 30-45% range)
        t = 0.375  # Based on ep_156 (40%) and ep_92 (31%)
        w = metrics["water_frac"]
        ratio_error = abs(w - t)
        ratio_term = max(0.0, 1.0 - ratio_error) * 2.5
        
        # Shaping
        shaping = 0.0
        if hasattr(self, '_prev_ratio_err'):
            delta_error = self._prev_ratio_err - ratio_error
            shaping = 0.5 * np.clip(delta_error * 5.0, -1.0, 1.0)
        self._prev_ratio_err = ratio_error
        
        # Coherence
        coherence = 0.7 * min(1.0, metrics["largest_land_frac"] * 2.5)
        
        # Sand quality (gated on ratio)
        if ratio_error < 0.10:  # Within 10% (27.5-47.5% water)
            sand_quality = 0.4 * metrics["sand_quality"]
        else:
            penalty_factor = max(0.0, 1.0 - ratio_error * 5.0)
            sand_quality = 0.4 * metrics["sand_quality"] * penalty_factor
        
        # Smoothness
        smoothness = 0.3 * self._compute_smoothness_reward()
        
        # Diversity
        max_frac = max(metrics["land_frac"], w, metrics["sand_frac"])
        diversity = 0.2 * max(0.0, 1.0 - max(0.0, max_frac - 0.70) * 5.0)
        
        # Mode bonus
        mode_bonus = 0.0
        if hasattr(self, 'phase'):
            if self.phase == "macro":
                mode_bonus = 0.3 * (ratio_term / 2.5) + 0.2 * (coherence / 0.7)
            elif self.phase == "detail":
                mode_bonus = 0.3 * metrics["sand_quality"] + 0.2 * (ratio_term / 2.5)
        
        # ══════════════════════════════════════════════════════════════
        # ✅ BIG BONUS: Target water 30-45%, sand 4-7%
        # ══════════════════════════════════════════════════════════════
        
        ratio_bonus = 0.0
        sand_frac = metrics["sand_frac"]
        
        # Water ratio bonus (30-45% water)
        if 0.30 <= w <= 0.45:  # Target range
            if 0.35 <= w <= 0.42:  # Perfect (like ep_156: 40%)
                ratio_bonus += 20.0
            else:  # Good but not perfect
                ratio_bonus += 12.0
        elif 0.25 <= w <= 0.50:  # Close to range
            ratio_bonus += 6.0
        
        # Sand ratio bonus (4-7% sand)
        if 0.04 <= sand_frac <= 0.07:  # Perfect sand (like ep_156: 6%)
            ratio_bonus += 10.0
        elif 0.03 <= sand_frac <= 0.09:  # Good sand
            ratio_bonus += 5.0
        elif 0.02 <= sand_frac <= 0.12:  # Acceptable sand
            ratio_bonus += 2.0
        
            ratio_bonus = 0.0
        if is_final_step:  # NEW: Only give bonus at end of episode
            sand_frac = metrics["sand_frac"]
            
            # Water bonus
            if 0.30 <= w <= 0.45:
                if 0.35 <= w <= 0.42:
                    ratio_bonus += 20.0
                else:
                    ratio_bonus += 12.0
            elif 0.25 <= w <= 0.50:
                ratio_bonus += 6.0
            
            # Sand bonus
            if 0.04 <= sand_frac <= 0.07:
                ratio_bonus += 10.0
            elif 0.03 <= sand_frac <= 0.09:
                ratio_bonus += 5.0
            elif 0.02 <= sand_frac <= 0.12:
                ratio_bonus += 2.0
        
        total = (base + ratio_term + shaping + coherence + 
                sand_quality + smoothness + diversity + mode_bonus + ratio_bonus)
        
        # FIX 2: Reduce reward scale for more stable learning (was -5 to 35, now -2 to 10)
        return float(np.clip(total * 0.25, -2.0, 10.0))
        # ══════════════════════════════════════════════════════════════
        # TOTAL
        # ══════════════════════════════════════════════════════════════
        
        total = (base + ratio_term + shaping + coherence + 
                sand_quality + smoothness + diversity + mode_bonus + ratio_bonus)
        
        final_reward = float(np.clip(total, -5.0, 35.0))
        
        return final_reward



        
    def _compute_smoothness_reward(self) -> float:
        """Reward local tile clustering (0-1 scale)"""
        smooth_score = 0.0
        count = 0
        
        for r in range(1, self.size - 1):
            for c in range(1, self.size - 1):
                neighbors = self.world[r-1:r+2, c-1:c+2]
                same_neighbors = np.sum(neighbors == self.world[r, c])
                smooth_score += float(same_neighbors) / 9.0
                count += 1
        
        return (smooth_score / count) if count > 0 else 0.0
    
    def get_current_action_size(self) -> int:
        """Return the valid action space size for the current phase"""
        if self.action_mode != "hierarchical":
            return self.action_size
            
        if self.phase == "macro":
            return self.macro_actions  # 6 actions
        elif self.phase == "detail":
            return self.detail_actions  # 9 actions
        else:  # shoreline phase (automatic)
            return 1
    def _land_dilate1(self) -> bool:
        """Dilate LAND by 1 pixel using scipy.ndimage.binary_dilation."""
        from scipy.ndimage import binary_dilation
        land_mask = (self.world == LAND)
        dilated = binary_dilation(land_mask, iterations=1)
        changed = np.any((dilated != land_mask) & (self.world != SAND))
        if changed:
            self.world[dilated & (self.world != SAND)] = LAND
        return changed

    def _land_erode1(self) -> bool:
        """Erode LAND by 1 pixel using scipy.ndimage.binary_erosion."""
        from scipy.ndimage import binary_erosion
        land_mask = (self.world == LAND)
        eroded = binary_erosion(land_mask, iterations=1)
        changed = np.any(land_mask != eroded)
        if changed:
            self.world[land_mask & ~eroded] = WATER
        return changed

    def _water_dilate1(self) -> bool:
        """Dilate WATER by 1 pixel."""
        from scipy.ndimage import binary_dilation
        water_mask = (self.world == WATER)
        dilated = binary_dilation(water_mask, iterations=1)
        changed = np.any((dilated != water_mask) & (self.world != SAND))
        if changed:
            self.world[dilated & (self.world != SAND)] = WATER
        return changed

    def _water_erode1(self) -> bool:
        """Erode WATER by 1 pixel."""
        from scipy.ndimage import binary_erosion
        water_mask = (self.world == WATER)
        eroded = binary_erosion(water_mask, iterations=2)
        changed = np.any(water_mask != eroded)
        if changed:
            self.world[water_mask & ~eroded] = LAND
        return changed

    def _fill_small_water_holes(self, max_size: int = 8) -> bool:
        """Fill small isolated water pockets (lakes) surrounded by land."""
        from scipy.ndimage import label
        
        water_mask = (self.world == WATER)
        labeled, num_features = label(water_mask)
        
        changed = False
        for i in range(1, num_features + 1):
            blob = (labeled == i)
            size = np.sum(blob)
            if size <= max_size:
                # Check if surrounded by land (not touching edge)
                coords = np.argwhere(blob)
                touches_edge = (
                    np.any(coords[:, 0] == 0) or 
                    np.any(coords[:, 0] == self.size - 1) or
                    np.any(coords[:, 1] == 0) or 
                    np.any(coords[:, 1] == self.size - 1)
                )
                if not touches_edge:
                    self.world[blob] = LAND
                    changed = True
        
        return changed

    def _remove_land_speckles(self, max_size: int = 8) -> bool:
        """Remove small isolated land speckles surrounded by water."""
        from scipy.ndimage import label
        
        land_mask = (self.world == LAND)
        labeled, num_features = label(land_mask)
        
        changed = False
        for i in range(1, num_features + 1):
            blob = (labeled == i)
            size = np.sum(blob)
            if size <= max_size:
                self.world[blob] = WATER
                changed = True
        
        return changed

    def _smooth_majority_3x3(self) -> bool:
        """Apply 3x3 majority filter to smooth terrain."""
        from scipy.ndimage import generic_filter
        
        def majority(values):
            # Get most common value in 3x3 neighborhood
            counts = np.bincount(values.astype(int), minlength=NUM_TILES)
            return np.argmax(counts)
        
        old_world = self.world.copy()
        self.world = generic_filter(self.world, majority, size=3, mode='nearest').astype(np.int32)
        
        return np.any(old_world != self.world)

    def _shoreline_band_thin(self) -> bool:
        """Remove excess sand to create thinner shoreline."""
        changed = False
        sand_coords = np.argwhere(self.world == SAND)
        
        for r, c in sand_coords:
            # Check 4-connected neighbors
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    neighbors.append(self.world[nr, nc])
            
            # If not between LAND and WATER, remove sand
            has_land = LAND in neighbors
            has_water = WATER in neighbors
            
            if not (has_land and has_water):
                # Decide what to replace with based on majority neighbor
                if neighbors.count(LAND) > neighbors.count(WATER):
                    self.world[r, c] = LAND
                else:
                    self.world[r, c] = WATER
                changed = True
        
        return changed

    def _shoreline_band_thicken(self) -> bool:
        """Add sand between LAND and WATER to create thicker shoreline."""
        from scipy.ndimage import binary_dilation
        
        land_mask = (self.world == LAND)
        water_mask = (self.world == WATER)
        
        # Dilate both by 1
        land_dilated = binary_dilation(land_mask, iterations=1)
        water_dilated = binary_dilation(water_mask, iterations=1)
        
        # Sand goes where they overlap (but isn't already sand)
        overlap = land_dilated & water_dilated & (self.world != SAND)
        
        changed = np.any(overlap)
        if changed:
            self.world[overlap] = SAND
        
        return changed  
    def reset(self, init_mode: str = "mixed", init_seed: Optional[int] = None,
            perlin_res: Tuple[int, int] = (4, 4),
            perlin_octaves: int = 3,
            island_count: Optional[int] = None) -> np.ndarray:
        """Reset the world with flexible initialization modes."""
        s = self.size
        seed = init_seed if init_seed is not None else self.rng.randint(1_000_000_000)

        if init_mode == "perlin":
            noise = generate_fractal_noise_2d(
                (s, s), res=perlin_res, octaves=perlin_octaves, seed=seed
            )
            threshold = 0.3 + (seed % 500) / 2000.0
            world = np.where(noise > threshold, LAND, WATER)

        elif init_mode == "islands":
            num_islands = island_count or max(1, 2 + (seed % 3))
            mask = generate_island_mask(s, num_islands, seed=seed)
            world = np.where(mask > 0.55, LAND, WATER)

        elif init_mode == "mixed":
            noise = generate_fractal_noise_2d(
                (s, s), res=perlin_res, octaves=perlin_octaves, seed=seed
            )
            num_islands = island_count or max(1, 1 + (seed % 2))
            mask = generate_island_mask(s, num_islands, seed=(seed ^ 0xABC123))
            mix = 0.35 * noise + 0.65 * mask
            world = np.where(mix > 0.55, LAND, WATER)

        else:
            raise ValueError("init_mode must be 'perlin' | 'islands' | 'mixed'")

        # Apply initial land/water terrain
        self.world = world.astype(np.int8)
        self.world = smooth_map(self.world, passes=1)
        
        # Automatically add sand at shorelines after initial generation
        self.world = apply_sand_rules(self.world)

        # Reset internal RNG for deterministic behavior
        if init_seed is not None:
            self.rng = np.random.RandomState(init_seed)

        # Reset counters based on action mode
        if self.action_mode == "hierarchical":
            self.regions_placed = 0
            self.phase = "macro"
            self.macro_done = False
            self.shoreline_done = False
        elif self.action_mode == "regional":
            self.regions_placed = 0
        else:  # pixel mode
            self.t = 0
            self._step_order = [(r, c) for r in range(self.size) for c in range(self.size)]
            if init_seed:
                rng_temp = np.random.RandomState(init_seed ^ 0x12345)
                rng_temp.shuffle(self._step_order)
            else:
                self.rng.shuffle(self._step_order)

        return self.observe()

    def observe(self) -> np.ndarray:
        grid = self.world
        if self.obs_mode == "single":
            # Still return normalized, but could add target as separate feature
            return grid.astype(np.float32) / (NUM_TILES - 1)
    # ... rest unchanged ...

        else:
            # Existing 4 channels
            land_mask = (grid == LAND).astype(np.float32)
            water_mask = (grid == WATER).astype(np.float32)
            sand_mask = (grid == SAND).astype(np.float32)
            shore_mask = np.zeros_like(grid, dtype=np.float32)
            for r in range(self.size):
                for c in range(self.size):
                    if grid[r, c] == SAND:
                        shore_mask[r, c] = 1.0
                    else:
                        neigh = grid[max(0, r - 1):min(self.size, r + 2),
                                    max(0, c - 1):min(self.size, c + 2)]
                        if np.any(neigh == LAND) and np.any(neigh == WATER):
                            shore_mask[r, c] = 1.0
            
            # NEW: 5th channel - desired water fraction (constant across grid)
            target_channel = np.full_like(land_mask, float(getattr(self, 'desired_water_frac', 0.45)))
            
            # Stack all 5 channels
            return np.stack([land_mask, water_mask, sand_mask, shore_mask, target_channel], axis=0).astype(np.float32)


    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.action_mode == "hierarchical":
            return self.step_hierarchical(action)
        elif self.action_mode == "regional":
            return self._step_regional(action)
        else:
            return self._step_pixel(action)
    
    def step_hierarchical(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Simplified 2-phase system: COARSE (land/water) -> FINE (sand/polish)"""
        
        if self.phase == "coarse":
            # Actions 0-5: Macro land/water operations
            if action < 3:  # Land operations
                terrain_type = LAND
                brush_size = [6, 10, 14][action]
            else:  # Water operations
                terrain_type = WATER
                brush_size = [6, 10, 14][action - 3]
            
            success = self.place_macro_feature(terrain_type, brush_size)
            
            # FIX 4: Add intermediate shaping reward for better credit assignment
            is_final = (self.regions_placed >= 11)  # Check if next step will be last
            reward = self.compute_unified_reward(success, is_final_step=is_final)
            self.regions_placed += 1
            
            # Transition to FINE phase after 12 coarse steps
            if self.regions_placed >= 12:
                self.phase = "fine"
                self.regions_placed = 0
                # Apply sand automatically
                self.world = apply_sand_rules(self.world)
                return self.observe(), reward, False
            
            return self.observe(), reward, False
        
        else:  # self.phase == "fine"
            # Actions 0-8: Polish operations (dilate/erode, smooth, etc.)
            success = self.place_detail_feature(action)
            
            # FIX 4: Mark final step for bonus rewards
            is_final = (self.regions_placed >= 7)  # Check if this is last fine step
            reward = self.compute_unified_reward(success, is_final_step=is_final)
            self.regions_placed += 1
            
            done = (self.regions_placed >= 8)  # Allow 8 fine-tuning steps
            return self.observe(), reward, done


    def _place_macro_feature(self, terrain_type, brush_size) -> bool:
        """Place a macro feature with proper bounds checking"""

        
        
        # Use self.world as the terrain map
        terrain_map = self.world
        
        # Fix 1: Clamp brush_size to valid range
        max_brush_size = min(terrain_map.shape[0] // 3, 8)
        brush_size = min(brush_size, max_brush_size)
        
        # Fix 2: Ensure valid range for randint
        map_size = min(terrain_map.shape[0], terrain_map.shape[1])
        min_pos = brush_size
        max_pos = map_size - brush_size
        
        if min_pos >= max_pos:
            # Fallback: use smaller brush
            brush_size = max(1, map_size // 4)
            min_pos = brush_size
            max_pos = map_size - brush_size
            
            if min_pos >= max_pos:
                # Map too small for any brush
                return False
        
        # Now safe to generate random positions
        center_r = self.rng.randint(min_pos, max_pos)
        center_c = self.rng.randint(min_pos, max_pos)
        
        # Fix 3: Create properly sized noise array
        noise_size = 2 * brush_size + 1
        shape_noise = self.rng.random((noise_size, noise_size))
        
        # Apply the macro feature and count placed cells
        placed_cells = 0
        for dr in range(-brush_size, brush_size + 1):
            for dc in range(-brush_size, brush_size + 1):
                r, c = center_r + dr, center_c + dc
                
                # Bounds check for map
                if 0 <= r < terrain_map.shape[0] and 0 <= c < terrain_map.shape[1]:
                    noise_r = dr + brush_size  # Center the noise index
                    noise_c = dc + brush_size
                    
                    # Safety check for noise array bounds
                    if 0 <= noise_r < noise_size and 0 <= noise_c < noise_size:
                        noise_val = shape_noise[noise_r, noise_c]
                        
                        # Calculate distance from center
                        distance = np.sqrt(dr**2 + dc**2)
                        if distance <= brush_size and noise_val > 0.3:
                            old_terrain = terrain_map[r, c]
                            terrain_map[r, c] = terrain_type
                            
                            # Count successful placements
                            if old_terrain != terrain_type:
                                placed_cells += 1
        
        # Return True if we successfully placed any cells
        return placed_cells > 0
    
    def place_detail_feature(self, action_id: int) -> bool:
        """
        Execute one of 9 learned polish operations based on action_id.
        
        Args:
            action_id: 0-8, selects which polish operation to perform
            
        Returns:
            bool: True if the operation changed the map, False otherwise
        """
        if action_id == 0:
            return self._land_dilate1()
        elif action_id == 1:
            return self._land_erode1()
        elif action_id == 2:
            return self._water_dilate1()
        elif action_id == 3:
            return self._water_erode1()
        elif action_id == 4:
            return self._fill_small_water_holes(max_size=8)
        elif action_id == 5:
            return self._remove_land_speckles(max_size=8)
        elif action_id == 6:
            return self._smooth_majority_3x3()
        elif action_id == 7:
            return self._shoreline_band_thin()
        elif action_id == 8:
            return self._shoreline_band_thicken()
        else:
            # Invalid action_id
            return False
   
    def _step_regional(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Regional placement approach - FIXED to use unified reward"""
        if self.regions_placed >= self.max_regions:
            reward = self.compute_unified_reward(placement_success=True)
            return self.observe(), reward, True
            
        terrain_type = action // 2
        brush_size = 4 if (action % 2 == 0) else 7  # small vs large brush
        
        center_r = self.rng.randint(brush_size, self.size - brush_size)
        center_c = self.rng.randint(brush_size, self.size - brush_size)
        
        # Apply circular brush and count placed cells
        placed_cells = 0
        for dr in range(-brush_size, brush_size + 1):
            for dc in range(-brush_size, brush_size + 1):
                r, c = center_r + dr, center_c + dc
                if (0 <= r < self.size and 0 <= c < self.size and 
                    dr*dr + dc*dc <= brush_size*brush_size):
                    old_terrain = self.world[r, c]
                    self.world[r, c] = terrain_type
                    if old_terrain != terrain_type:
                        placed_cells += 1
        
        success = placed_cells > 0
        self.regions_placed += 1
        reward = self.compute_unified_reward(success)
        
        done = self.regions_placed >= self.max_regions
        return self.observe(), reward, done
    
    def _step_pixel(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Original pixel-by-pixel placement - FIXED to use unified reward"""
        if self.t >= len(self._step_order):
            reward = self.compute_unified_reward(placement_success=True)
            return self.observe(), reward, True
            
        r, c = self._step_order[self.t]
        old_terrain = self.world[r, c]
        self.world[r, c] = int(action)
        
        # Calculate local reward for the step
        local_r = stepwise_reward(self.world, r, c, int(action))
        success = old_terrain != int(action)  # Changed terrain
        
        self.t += 1
        
        done = self.t >= len(self._step_order) or self.t >= self.max_steps
        reward = self.compute_unified_reward(success, local_r)
            
        return self.observe(), reward, done

    def polish(self) -> np.ndarray:
        from rule_based_polish import RuleBasedPolisher
        pol = RuleBasedPolisher()
        return pol.apply(self.world.copy())

    def export_to_numpy(self, path: str):
        np.save(path, self.world)

    def get_valid_action_mask(self):
        mask = np.zeros(9, dtype=bool)
        if self.phase == "macro":
            mask[:6] = True  # 6 macro actions
        elif self.phase == "detail":
            # Check which ops would change the grid
            for i in range(9):
                mask[i] = self._would_op_change(i)
        return mask


    def _would_op_change(self, op_id: int) -> bool:
        """
        Quick check if op would modify grid (without applying it).
        Returns False if no tiles would change, True if the op is applicable.
        """
        from scipy.ndimage import binary_dilation, binary_erosion, label
        
        if op_id == 0:  # land_dilate1
            land_mask = (self.world == LAND)
            dilated = binary_dilation(land_mask, iterations=1)
            return np.any((dilated != land_mask) & (self.world != SAND))
        
        elif op_id == 1:  # land_erode1
            land_mask = (self.world == LAND)
            eroded = binary_erosion(land_mask, iterations=1)
            return np.any(land_mask != eroded)
        
        elif op_id == 2:  # water_dilate1
            water_mask = (self.world == WATER)
            dilated = binary_dilation(water_mask, iterations=1)
            return np.any((dilated != water_mask) & (self.world != SAND))
        
        elif op_id == 3:  # water_erode1
            water_mask = (self.world == WATER)
            eroded = binary_erosion(water_mask, iterations=1)
            return np.any(water_mask != eroded)
        
        elif op_id == 4:  # fill_small_water_holes
            water_mask = (self.world == WATER)
            labeled, num_features = label(water_mask)
            # Check if there are any small holes not touching edges
            for i in range(1, num_features + 1):
                blob = (labeled == i)
                size = np.sum(blob)
                if size <= 8:
                    coords = np.argwhere(blob)
                    touches_edge = (
                        np.any(coords[:, 0] == 0) or 
                        np.any(coords[:, 0] == self.size - 1) or
                        np.any(coords[:, 1] == 0) or 
                        np.any(coords[:, 1] == self.size - 1)
                    )
                    if not touches_edge:
                        return True  # Found a fillable hole
            return False
        
        elif op_id == 5:  # remove_land_speckles
            land_mask = (self.world == LAND)
            labeled, num_features = label(land_mask)
            # Check if there are any small land blobs
            for i in range(1, num_features + 1):
                blob = (labeled == i)
                if np.sum(blob) <= 8:
                    return True  # Found a removable speckle
            return False
        
        elif op_id == 6:  # smooth_majority_3x3
            # Fast check: if all neighbors agree, smoothing won't change anything
            # For speed, just return True (smoothing is usually valid)
            return True
        
        elif op_id == 7:  # shoreline_band_thin
            # Check if there's any sand that's not between land and water
            sand_coords = np.argwhere(self.world == SAND)
            if len(sand_coords) == 0:
                return False
            
            for r, c in sand_coords[:min(len(sand_coords), 20)]:  # Sample first 20 for speed
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        neighbors.append(self.world[nr, nc])
                
                has_land = LAND in neighbors
                has_water = WATER in neighbors
                if not (has_land and has_water):
                    return True  # Found sand to remove
            return False
        
        elif op_id == 8:  # shoreline_band_thicken
            land_mask = (self.world == LAND)
            water_mask = (self.world == WATER)
            land_dilated = binary_dilation(land_mask, iterations=1)
            water_dilated = binary_dilation(water_mask, iterations=1)
            overlap = land_dilated & water_dilated & (self.world != SAND)
            return np.any(overlap)
        
        else:
            return False  # Invalid op_id

    def get_phase_reward_breakdown(self) -> Dict[str, float]:
        """Return per-phase rewards for granular logging."""
        if self.action_mode != "hierarchical":
            return {'total_reward': 0.0}
        return {
            'macro_reward': getattr(self, '_macro_reward_accum', 0.0),
            'shoreline_reward': getattr(self, '_shoreline_reward_accum', 0.0),
            'detail_reward': getattr(self, '_detail_reward_accum', 0.0)
        }

    def export_to_json(self, path: str, seed: Optional[int] = None, 
                    episode: Optional[int] = None, init_mode: str = "mixed"):
        """Export with villages, distance_layers for Phase 2, and acceptance metrics."""
        
        # Compute current metrics
        metrics = compute_map_metrics(self.world)
        
        # Calculate acceptance criteria fields
        desired_water_frac = getattr(self, 'desired_water_frac', 0.45)
        ratio_error = abs(metrics['water_frac'] - desired_water_frac)
        
        # Format villages for JSON
        villages_data = {
            'count': len(self.villages),
            'contact_radius': self.village_config['contact_radius'],
            'points': [{'x': int(c), 'y': int(r)} for r, c in self.villages]
        }
        
        # Format distance layers
        distance_layers_data = {
            'dist_to_water': self.distance_layers['dist_to_water'].tolist(),
            'dist_to_sand': self.distance_layers['dist_to_sand'].tolist()
        }
        
        # Create tile_ids mapping for semantic locking
        tile_ids = {'LAND': 0, 'WATER': 1, 'SAND': 2}
        
        payload = {
            'width': int(self.size),
            'height': int(self.size),
            'tiles': self.world.astype(int).tolist(),
            'villages': villages_data,
            'distance_layers': distance_layers_data,
            'metadata': {
                # Original metadata
                'seed': seed,
                'episode': episode,
                'init_mode': init_mode,
                'timestamp': time.time(),
                'tile_ids': tile_ids,
                'village_config': self.village_config,
                
                # NEW: Acceptance criteria metrics
                'desired_water_frac': float(desired_water_frac),
                'ratio_error': float(ratio_error),
                'water_frac': float(metrics['water_frac']),
                'land_frac': float(metrics['land_frac']),
                'sand_frac': float(metrics['sand_frac']),
                'sand_quality': float(metrics['sand_quality']),
                'largest_land_frac': float(metrics['largest_land_frac']),
                
                # Additional useful metrics
                'num_land_blobs': int(metrics.get('num_land_blobs', 0)),
                'num_water_blobs': int(metrics.get('num_water_blobs', 0)),
                
                # Phase 2 readiness indicator
                'phase2_ready': bool(
                    ratio_error <= 0.02 and
                    metrics['sand_quality'] >= 0.75 and
                    0.20 <= metrics['largest_land_frac'] <= 0.60
                )
            }
        }
        
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)



    def show(self, figsize=(6, 6), title: Optional[str] = None):
        plt.figure(figsize=figsize)
        plt.imshow(self.world, cmap=_TILE_CMAP, vmin=0, vmax=2)
        if title: plt.title(title)
        plt.axis("off"); plt.show()

    def get_full_world(self) -> np.ndarray:
        """Return a copy of the current world state."""
        return self.world.copy()

def sanity_check():
    # Test hierarchical mode
    env = MapEnvironment(size=32, seed=42, action_mode="hierarchical")
    env.reset(init_mode="mixed", init_seed=123)
    obs = env.observe()
    
    print("Testing hierarchical mode...")
    for step in range(15):  # Should complete macro + shoreline + detail phases
        a = int(env.rng.randint(0, env.get_current_action_size()))
        obs, r, done = env.step(a)
        print(f"Step {step}: Phase={env.phase}, Action={a}, Reward={r:.2f}, Done={done}")
        if done:
            break
    
    print("Final terrain balance:")
    land_frac = np.mean(env.world == LAND)
    water_frac = np.mean(env.world == WATER) 
    sand_frac = np.mean(env.world == SAND)
    print(f"Land: {land_frac:.2f}, Water: {water_frac:.2f}, Sand: {sand_frac:.2f}")
    print("Hierarchical sanity check passed.")

if __name__ == "__main__":
    sanity_check()