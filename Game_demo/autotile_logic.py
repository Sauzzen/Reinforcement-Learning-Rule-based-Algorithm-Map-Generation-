# autotile_logic.py
"""
Autotiling logic for connecting terrain tiles based on neighbors.
Handles the 2x3 tileset format from your provided sprites.
"""

from typing import Tuple, Dict, List
from enum import IntEnum

class TileType(IntEnum):
    """Terrain type constants"""
    LAND = 0
    WATER = 1
    SAND = 2

class AutotileLogic:
    """Handles autotile selection based on neighbor analysis"""
    
    # Standard 2x3 tileset layout indices
    # Based on common RPG Maker/autotile format
    TILE_INDICES = {
        # Basic tiles (most common)
        'center': 7,        # Middle tile for internal areas
        'isolated': 0,      # Single tile surrounded by different terrain
        
        # Edge pieces
        'top': 1,
        'bottom': 13,
        'left': 6,
        'right': 8,
        
        # Corners
        'top_left': 0,
        'top_right': 2,
        'bottom_left': 12,
        'bottom_right': 14,
        
        # Inner corners (for more complex shapes)
        'inner_tl': 3,
        'inner_tr': 4,
        'inner_bl': 15,
        'inner_br': 16,
        
        # Special cases
        'horizontal': 9,    # Horizontal strip
        'vertical': 5,      # Vertical strip
    }
    
    def __init__(self, map_data, config: Dict = None):
        self.map_data = map_data
        self.config = config or {}
        self.width = map_data.width
        self.height = map_data.height
        
    def get_neighbors(self, x: int, y: int) -> Dict[str, int]:
        """Get terrain types of all 8 neighbors"""
        neighbors = {}
        directions = {
            'n': (0, -1),   'ne': (1, -1),  'e': (1, 0),   'se': (1, 1),
            's': (0, 1),    'sw': (-1, 1),  'w': (-1, 0),  'nw': (-1, -1)
        }
        
        for direction, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            neighbors[direction] = self.map_data.get_tile(nx, ny)
        
        return neighbors
    
    def get_cardinal_neighbors(self, x: int, y: int) -> Dict[str, int]:
        """Get only the 4 cardinal direction neighbors"""
        neighbors = {}
        directions = {
            'n': (0, -1),   'e': (1, 0),
            's': (0, 1),    'w': (-1, 0)
        }
        
        for direction, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            neighbors[direction] = self.map_data.get_tile(nx, ny)
        
        return neighbors
    
    def matches_terrain(self, terrain1: int, terrain2: int) -> bool:
        """Check if two terrain types should connect"""
        if terrain1 == terrain2:
            return True
        
        # Special case: sand connects to both land and water for transitions
        if terrain1 == TileType.SAND:
            return terrain2 in [TileType.LAND, TileType.WATER, TileType.SAND]
        if terrain2 == TileType.SAND:
            return terrain1 in [TileType.LAND, TileType.WATER, TileType.SAND]
        
        return False
    
    def calculate_tile_index(self, x: int, y: int) -> int:
        """Calculate the appropriate tile index for position (x,y)"""
        current_terrain = self.map_data.get_tile(x, y)
        if current_terrain == -1:  # Out of bounds
            return 0
        
        neighbors = self.get_cardinal_neighbors(x, y)
        
        # Count matching neighbors in each direction
        matches = {}
        for direction, neighbor_terrain in neighbors.items():
            matches[direction] = self.matches_terrain(current_terrain, neighbor_terrain)
        
        # Determine tile type based on neighbor pattern
        tile_index = self._select_tile_by_pattern(matches)
        
        # Clamp to safe range (0-15 for 4x4 tileset)
        tile_index = max(0, min(tile_index, 15))
        
        return tile_index
    
    def _select_tile_by_pattern(self, matches: Dict[str, bool]) -> int:
        """Simple mapping using only the clean tiles from your grass tileset"""
        n, e, s, w = matches['n'], matches['e'], matches['s'], matches['w']
        match_count = sum(matches.values())
        
        # Look at your grass tileset and pick indices that have clean, solid grass
        # You'll need to test these values to see which ones look best
        
        if match_count == 4:    # Surrounded by same terrain - use a solid center tile
            return 5    # Try different values: 0, 1, 4, 5, 8, 9, 12, 13
        elif match_count >= 2:  # Some neighbors - use another solid tile
            return 5    # Use same tile for consistency 
        else:                   # Few/no neighbors  
            return 5    # Use same tile for consistency
    
    def get_tile_rect(self, tile_index: int, tileset_cols: int = 6) -> Tuple[int, int]:
        """Convert tile index to (col, row) position in tileset"""
        col = tile_index % tileset_cols
        row = tile_index // tileset_cols
        return col, row
    
    def generate_tile_map(self) -> List[List[int]]:
        """Generate complete tile index map for the terrain"""
        tile_map = []
        
        for y in range(self.height):
            row = []
            for x in range(self.width):
                tile_index = self.calculate_tile_index(x, y)
                row.append(tile_index)
            tile_map.append(row)
        
        return tile_map
    
    def get_transition_tiles(self, terrain_a: int, terrain_b: int) -> List[Tuple[int, int]]:
        """Find all positions where terrain A transitions to terrain B"""
        transitions = []
        
        for y in range(self.height):
            for x in range(self.width):
                current = self.map_data.get_tile(x, y)
                if current == terrain_a:
                    neighbors = self.get_cardinal_neighbors(x, y)
                    
                    # Check if any neighbor is terrain_b
                    for neighbor_terrain in neighbors.values():
                        if neighbor_terrain == terrain_b:
                            transitions.append((x, y))
                            break
        
        return transitions

# Utility functions for common autotiling operations
def create_bitmask(neighbors: Dict[str, bool]) -> int:
    """Create bitmask from neighbor matches for advanced autotiling"""
    # Standard bitmask: N=1, E=2, S=4, W=8
    mask = 0
    if neighbors.get('n', False): mask |= 1
    if neighbors.get('e', False): mask |= 2  
    if neighbors.get('s', False): mask |= 4
    if neighbors.get('w', False): mask |= 8
    return mask

def get_47_tile_index(bitmask: int) -> int:
    """Convert 4-bit bitmask to 47-tile autotile index"""
    # This is for more advanced autotile systems
    # Maps 16 possible neighbor combinations to specific tiles
    bitmask_to_tile = {
        0:  0,   # No connections
        1:  1,   # North only
        2:  2,   # East only  
        3:  3,   # North + East
        4:  4,   # South only
        5:  5,   # North + South
        6:  6,   # East + South
        7:  7,   # North + East + South
        8:  8,   # West only
        9:  9,   # North + West
        10: 10,  # East + West
        11: 11,  # North + East + West
        12: 12,  # South + West
        13: 13,  # North + South + West
        14: 14,  # East + South + West
        15: 15,  # All connections
    }
    return bitmask_to_tile.get(bitmask, 0)

# Example usage and testing
if __name__ == "__main__":
    # Test with dummy map data
    class DummyMapData:
        def __init__(self, width, height, tiles):
            self.width = width
            self.height = height
            self.tiles = tiles
        
        def get_tile(self, x, y):
            if 0 <= x < self.width and 0 <= y < self.height:
                return self.tiles[y][x]
            return -1
    
    # Create test map: small island surrounded by water
    test_tiles = [
        [1, 1, 1, 1, 1],
        [1, 2, 0, 2, 1],
        [1, 0, 0, 0, 1],
        [1, 2, 0, 2, 1],
        [1, 1, 1, 1, 1],
    ]
    
    dummy_map = DummyMapData(5, 5, test_tiles)
    autotiler = AutotileLogic(dummy_map)
    
    print("Test map terrain:")
    for row in test_tiles:
        print([['L', 'W', 'S'][t] for t in row])
    
    print("\nGenerated tile indices:")
    tile_map = autotiler.generate_tile_map()
    for row in tile_map:
        print(row)
    
    print("\nLand-Water transitions:")
    transitions = autotiler.get_transition_tiles(TileType.LAND, TileType.WATER)
    print(f"Found {len(transitions)} transition points: {transitions}")

 #   def calculate_tile_index_debug(self, x: int, y: int) -> int:
 #       """Debug version that prints what's happening"""
 #       current_terrain = self.map_data.get_tile(x, y)
 #       if current_terrain == -1:  # Out of bounds
 #           return self.TILE_INDICES['center']
 #       
 #       neighbors = self.get_cardinal_neighbors(x, y)
 #       
 #       # Count matching neighbors in each direction
 #       matches = {}
 #       for direction, neighbor_terrain in neighbors.items():
 #           matches[direction] = self.matches_terrain(current_terrain, neighbor_terrain)
 #       
 #       # Determine tile type based on neighbor pattern
 #       tile_index = self._select_tile_by_pattern(matches)
 #       
 #       # Debug print for problematic areas
 #       if tile_index >= 24 or tile_index < 0:  # Assuming 6x4 = 24 tiles max
 #           print(f"WARNING: Invalid tile index {tile_index} at ({x},{y})")
 #           print(f"  Terrain: {current_terrain}, Neighbors: {neighbors}")
 #           print(f"  Matches: {matches}")
 #           tile_index = 0  # Use safe fallback
 #       
 #       return tile_index