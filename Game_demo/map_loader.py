# map_loader.py
"""
JSON map loading and validation for RL-generated terrain maps.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

class MapData:
    """Container for map data with validation"""
    def __init__(self, width: int, height: int, tiles: List[List[int]], metadata: Optional[Dict] = None):
        self.width = width
        self.height = height
        self.tiles = tiles
        self.metadata = metadata or {}
        self._validate()
    
    def _validate(self):
        """Validate map data integrity"""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid map dimensions: {self.width}x{self.height}")
        
        if len(self.tiles) != self.height:
            raise ValueError(f"Tile data height mismatch: expected {self.height}, got {len(self.tiles)}")
        
        for row_idx, row in enumerate(self.tiles):
            if len(row) != self.width:
                raise ValueError(f"Row {row_idx} width mismatch: expected {self.width}, got {len(row)}")
            
            for col_idx, tile in enumerate(row):
                if not isinstance(tile, int) or tile < 0:
                    raise ValueError(f"Invalid tile value at ({row_idx}, {col_idx}): {tile}")
    
    def get_tile(self, x: int, y: int) -> int:
        """Get tile value at coordinates (with bounds checking)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return -1  # Out of bounds
    
    def get_terrain_stats(self) -> Dict[int, float]:
        """Calculate terrain type distribution"""
        terrain_counts = {}
        total_tiles = self.width * self.height
        
        for row in self.tiles:
            for tile in row:
                terrain_counts[tile] = terrain_counts.get(tile, 0) + 1
        
        return {terrain: count / total_tiles for terrain, count in terrain_counts.items()}

class MapLoader:
    """Handles loading and parsing of JSON map files"""
    
    @staticmethod
    def load_from_file(filepath: str) -> MapData:
        """Load map from JSON file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Map file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {filepath}: {e}")
        except Exception as e:
            raise IOError(f"Error reading {filepath}: {e}")
        
        return MapLoader._parse_map_data(data, filepath)
    
    @staticmethod
    def _parse_map_data(data: Dict, source_file: str) -> MapData:
        """Parse loaded JSON data into MapData object"""
        required_fields = ['width', 'height', 'tiles']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field '{field}' in {source_file}")
        
        width = int(data['width'])
        height = int(data['height'])
        tiles = data['tiles']
        metadata = data.get('metadata', {})
        
        # Additional validation
        if not isinstance(tiles, list):
            raise ValueError(f"'tiles' must be a list in {source_file}")
        
        # Ensure all tiles are integers (convert if necessary)
        processed_tiles = []
        for row_idx, row in enumerate(tiles):
            if not isinstance(row, list):
                raise ValueError(f"Row {row_idx} must be a list in {source_file}")
            
            processed_row = []
            for col_idx, tile in enumerate(row):
                try:
                    processed_row.append(int(tile))
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid tile value at ({row_idx}, {col_idx}) in {source_file}: {tile}")
            processed_tiles.append(processed_row)
        
        return MapData(width, height, processed_tiles, metadata)
    
    @staticmethod
    def load_multiple_maps(directory: str, pattern: str = "*.json") -> Dict[str, MapData]:
        """Load multiple map files from a directory"""
        import glob
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pattern_path = os.path.join(directory, pattern)
        map_files = glob.glob(pattern_path)
        
        if not map_files:
            raise FileNotFoundError(f"No JSON files found matching pattern: {pattern_path}")
        
        maps = {}
        for filepath in map_files:
            filename = os.path.basename(filepath)
            name = os.path.splitext(filename)[0]  # Remove .json extension
            
            try:
                maps[name] = MapLoader.load_from_file(filepath)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
        
        return maps

# Utility functions for common map operations
def find_spawn_points(map_data: MapData, terrain_type: int = 0, min_area: int = 4) -> List[Tuple[int, int]]:
    """Find suitable spawn points on specified terrain type"""
    spawn_points = []
    
    for y in range(1, map_data.height - 1):  # Avoid edges
        for x in range(1, map_data.width - 1):
            if map_data.get_tile(x, y) == terrain_type:
                # Check if there's enough space around this point
                area_count = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if map_data.get_tile(x + dx, y + dy) == terrain_type:
                            area_count += 1
                
                if area_count >= min_area:
                    spawn_points.append((x, y))
    
    return spawn_points

def validate_map_connectivity(map_data: MapData, walkable_terrains: List[int]) -> bool:
    """Check if walkable areas are connected (basic flood fill)"""
    # Find first walkable tile
    start_pos = None
    for y in range(map_data.height):
        for x in range(map_data.width):
            if map_data.get_tile(x, y) in walkable_terrains:
                start_pos = (x, y)
                break
        if start_pos:
            break
    
    if not start_pos:
        return False  # No walkable tiles
    
    # Flood fill to find all connected tiles
    visited = set()
    stack = [start_pos]
    
    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        
        if map_data.get_tile(x, y) not in walkable_terrains:
            continue
        
        visited.add((x, y))
        
        # Add neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < map_data.width and 0 <= ny < map_data.height and 
                (nx, ny) not in visited):
                stack.append((nx, ny))
    
    # Count total walkable tiles
    total_walkable = 0
    for y in range(map_data.height):
        for x in range(map_data.width):
            if map_data.get_tile(x, y) in walkable_terrains:
                total_walkable += 1
    
    # Consider connected if >80% of walkable tiles are reachable
    return len(visited) >= (total_walkable * 0.8)

# Example usage and testing
if __name__ == "__main__":
    # Test the map loader
    try:
        map_data = MapLoader.load_from_file("maps/final_map_seed0.json")
        print(f"Loaded map: {map_data.width}x{map_data.height}")
        print(f"Terrain distribution: {map_data.get_terrain_stats()}")
        
        spawn_points = find_spawn_points(map_data, terrain_type=0)
        print(f"Found {len(spawn_points)} potential spawn points on land")
        
        connectivity = validate_map_connectivity(map_data, [0, 2])  # Land and sand walkable
        print(f"Map connectivity: {'Good' if connectivity else 'Poor'}")
        
    except Exception as e:
        print(f"Error testing map loader: {e}")