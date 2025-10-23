# map_env.py
import numpy as np
from scipy.ndimage import label, binary_fill_holes

# ------------------------------
# Tile codes
# ------------------------------
LAND, WATER, SAND = 0, 1, 2
NUM_TILES = 3

# ------------------------------
# Stepwise / local reward
# ------------------------------
def stepwise_reward(grid: np.ndarray, r: int, c: int) -> float:
    tile = grid[r, c]
    n = grid.shape[0]
    reward = 0.0
    neighbors = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
    
    land_frac = np.mean(grid == LAND)
    water_frac = np.mean(grid == WATER)
    
    if tile == WATER:
        neighbor_water = sum(1 for nr,nc in neighbors if 0<=nr<n and 0<=nc<n and grid[nr,nc]==WATER)
        reward += 0.03 * neighbor_water
        if water_frac > 0.6:
            reward -= 0.05  # discourage too much water
    elif tile == LAND:
        neighbor_land = sum(1 for nr,nc in neighbors if 0<=nr<n and 0<=nc<n and grid[nr,nc]==LAND)
        reward += 0.08 * neighbor_land
        if land_frac > 0.65:
            reward -= 0.05  # discourage too much land
    elif tile == SAND:
        land_adj = any(0<=nr<n and 0<=nc<n and grid[nr,nc]==LAND for nr,nc in neighbors)
        water_adj = any(0<=nr<n and 0<=nc<n and grid[nr,nc]==WATER for nr,nc in neighbors)
        reward += 0.2 if (land_adj and water_adj) else 0.0

    return reward

# ------------------------------
# Connectivity / blob metrics
# ------------------------------
def count_blobs(grid: np.ndarray, val: int) -> np.ndarray:
    struct = np.ones((3,3))
    labeled,_ = label(grid==val, structure=struct)
    return np.bincount(labeled.flatten())[1:]

def enclosed_land_pockets(grid: np.ndarray) -> np.ndarray:
    land_mask = grid==LAND
    filled = binary_fill_holes(land_mask)
    pockets = filled & (~land_mask)
    labeled,_ = label(pockets)
    return np.bincount(labeled.flatten())[1:]

# ------------------------------
# Island mask
# ------------------------------
def island_mask(size:int, num_islands:int=2) -> np.ndarray:
    mask = np.zeros((size,size))
    radius = size//4
    centers = [(size//2,size//2)]
    if num_islands>=2: centers.append((size//4,size//4))
    if num_islands>=3: centers.append((3*size//4,3*size//4))
    
    for cx,cy in centers[:num_islands]:
        for r in range(size):
            for c in range(size):
                dist = np.sqrt((r-cx)**2 + (c-cy)**2)
                mask[r,c] = max(mask[r,c], max(0.0, 1.0-(dist/radius)))
    return mask

# ------------------------------
# Sand adjacency rules
# ------------------------------
def apply_sand_rules(grid: np.ndarray) -> np.ndarray:
    n = grid.shape[0]
    new_grid = grid.copy()
    for r in range(n):
        for c in range(n):
            if grid[r,c]==LAND:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = r+dr, c+dc
                    if 0<=nr<n and 0<=nc<n and grid[nr,nc]==WATER:
                        new_grid[r,c]=SAND
                        break
    return new_grid

# ------------------------------
# Shoreline reward
# ------------------------------
def shoreline_reward(grid: np.ndarray) -> float:
    n = grid.shape[0]
    reward=0.0
    for r in range(n):
        for c in range(n):
            if grid[r,c]==LAND:
                if any(0<=nr<n and 0<=nc<n and grid[nr,nc]==WATER
                       for nr in [r-1,r,r+1]
                       for nc in [c-1,c,c+1] if not (nr==r and nc==c)):
                    reward += 0.1
            elif grid[r,c]==SAND:
                land_adj = any(0<=nr<n and 0<=nc<n and grid[nr,nc]==LAND for nr in [r-1,r,r+1] for nc in [c-1,c,c+1])
                water_adj = any(0<=nr<n and 0<=nc<n and grid[nr,nc]==WATER for nr in [r-1,r,r+1] for nc in [c-1,c,c+1])
                reward += 0.2 if (land_adj and water_adj) else 0.0
    return reward

# ------------------------------
# Global map reward
# ------------------------------
def compute_map_reward(grid: np.ndarray) -> float:
    reward = 0.0
    n = grid.shape[0]

    # Largest land blob
    sizes = count_blobs(grid, LAND)
    if len(sizes) > 0:
        largest = max(sizes)
        frac = largest / (n*n)
        reward += max(0.0, 5.0*(0.4 - abs(0.4 - frac)))  # reward peak around 0.4 land fraction
        reward -= 0.3*(len(sizes)-1)  # mild penalty for extra islands

    reward += shoreline_reward(grid)*0.2

    # Land-water balance penalties
    land_frac = np.mean(grid==LAND)
    water_frac = np.mean(grid==WATER)
    if water_frac > 0.6: reward -= 3.0
    elif water_frac < 0.25: reward -= 1.0
    if land_frac > 0.65: reward -= 2.0
    elif land_frac < 0.2: reward -= 1.0

    # Reward moderate sand presence
    sand_frac = np.mean(grid==SAND)
    if sand_frac > 0.05: reward += 0.5

    return float(reward)

# ------------------------------
# Map metrics
# ------------------------------
def compute_map_metrics(grid: np.ndarray) -> dict:
    n = grid.shape[0]
    metrics={}
    sizes = count_blobs(grid, LAND)
    metrics["largest_land_frac"] = max(sizes)/(n*n) if len(sizes) else 0.0
    metrics["land_frac"]=np.mean(grid==LAND)
    metrics["water_frac"]=np.mean(grid==WATER)
    metrics["sand_frac"]=np.mean(grid==SAND)
    
    sand_mask = grid==SAND
    total_sand = np.sum(sand_mask)
    quality=0
    if total_sand>0:
        for r,c in zip(*np.where(sand_mask)):
            neighbors = grid[max(0,r-1):min(n,r+2), max(0,c-1):min(n,c+2)]
            land_adj = np.any(neighbors==LAND)
            water_adj = np.any(neighbors==WATER)
            if land_adj and water_adj: quality+=1
        metrics["sand_quality"]=quality/total_sand
    else:
        metrics["sand_quality"]=0.0
    return metrics

# ------------------------------
# Map Environment
# ------------------------------
class MapEnvironment:
    def __init__(self, size:int=32):
        self.size=size
        self.world = np.full((size,size), LAND, dtype=int)
        self.t=0
        self.state_size=(size,size)
        self.action_size=NUM_TILES

    def reset(self) -> np.ndarray:
        self.t=0
        # Simple islands using island mask only
        mask = island_mask(self.size, num_islands=np.random.randint(2,4))
        threshold = 0.5
        self.world[:] = np.where(mask > threshold, LAND, WATER)
        self.world = apply_sand_rules(self.world)
        return self.observe()

    def observe(self) -> np.ndarray:
        return self.world.astype(np.float32)/(NUM_TILES-1)

    def step(self, action:int) -> tuple[np.ndarray,float,bool]:
        r,c = divmod(self.t, self.size)
        self.world[r,c] = int(action)
        reward = stepwise_reward(self.world,r,c)
        self.t += 1
        done = self.t >= self.size*self.size
        if done:
            reward += compute_map_reward(self.world)
        return self.observe(), float(reward), done

    def get_full_world(self) -> np.ndarray:
        return self.world.copy()
