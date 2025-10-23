# tileset_renderer.py
"""
Handles loading and rendering of tilesets with autotile support.
Manages sprite sheets and tile positioning for the demo.
"""

import pygame
import os
import json
from typing import Dict, Tuple, Optional, List
from autotile_logic import AutotileLogic, TileType

class TilesetRenderer:
    """Manages tileset loading and rendering operations"""
    
    def __init__(self, config_path: str = "config/tile_config.json"):
        self.config = self._load_config(config_path)
        self.tilesets: Dict[str, pygame.Surface] = {}
        self.tile_cache: Dict[Tuple[str, int], pygame.Surface] = {}
        
        # Extract config values
        self.tile_size = self.config.get("tile_size", 32)
        self.tileset_layout = self.config.get("tileset_layout", {})
        self.terrain_mapping = self.config.get("terrain_mapping", {})
        
        # Tileset dimensions
        self.tileset_cols = self.tileset_layout.get("columns", 4)
        self.tileset_rows = self.tileset_layout.get("rows", 4)
        self.source_tile_size = self.tileset_layout.get("tile_width", 16)
        
        self._load_tilesets()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            "terrain_mapping": {
                "0": "grass.png",
                "1": "water.png", 
                "2": "sand_light.png"
            },
            "tile_size": 32,
            "tileset_layout": {
                "columns": 6,
                "rows": 4,
                "tile_width": 16,
                "tile_height": 16
            }
        }
    
    def _load_tilesets(self):
        """Load all tileset images"""
        tileset_dir = "assets/tilesets/"
        
        for terrain_id, filename in self.terrain_mapping.items():
            filepath = os.path.join(tileset_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: Tileset not found: {filepath}")
                # Create placeholder tileset
                self.tilesets[terrain_id] = self._create_placeholder_tileset(int(terrain_id))
                continue
            
            try:
                tileset_surface = pygame.image.load(filepath).convert_alpha()
                self.tilesets[terrain_id] = tileset_surface
                print(f"Loaded tileset: {filename}")
            except pygame.error as e:
                print(f"Error loading {filepath}: {e}")
                self.tilesets[terrain_id] = self._create_placeholder_tileset(int(terrain_id))
    
    def _create_placeholder_tileset(self, terrain_type: int) -> pygame.Surface:
        """Create a colored placeholder tileset when image loading fails"""
        colors = {
            0: (34, 139, 34),    # Land - Green
            1: (30, 144, 255),   # Water - Blue
            2: (238, 203, 173),  # Sand - Tan
        }
        
        color = colors.get(terrain_type, (128, 128, 128))
        tileset_width = self.tileset_cols * self.source_tile_size
        tileset_height = self.tileset_rows * self.source_tile_size
        
        surface = pygame.Surface((tileset_width, tileset_height))
        surface.fill(color)
        
        # Add simple border pattern to distinguish tiles
        for row in range(self.tileset_rows):
            for col in range(self.tileset_cols):
                x = col * self.source_tile_size
                y = row * self.source_tile_size
                rect = pygame.Rect(x, y, self.source_tile_size, self.source_tile_size)
                pygame.draw.rect(surface, (255, 255, 255), rect, 1)
        
        return surface
    
    def get_tile_surface(self, terrain_type: str, tile_index: int) -> pygame.Surface:
        """Extract tile from your actual tileset format"""
        cache_key = (terrain_type, tile_index)
        
        if cache_key in self.tile_cache:
            return self.tile_cache[cache_key]
        
        if terrain_type not in self.tilesets:
            terrain_type = list(self.tilesets.keys())[0]
        
        tileset = self.tilesets[terrain_type]
        
        # For your grass tileset, let's examine the actual layout
        # It appears to be organized in a specific autotile pattern
        
        # Limit tile_index to prevent going out of bounds
        tile_index = tile_index % 16  # Assume max 16 tiles available
        
        # Your tileset appears to be roughly 4x4 tiles
        cols = 4
        col = tile_index % cols
        row = tile_index // cols
        
        # Calculate the actual tile size from your tileset
        # Your grass image appears to be about 64x64 total with 4x4 = 16x16 per tile
        actual_tile_size = 16
        
        # Create output surface
        tile_surface = pygame.Surface((self.tile_size, self.tile_size))
        
        # Fallback colors
        terrain_colors = {
            '0': (100, 180, 100),   # Green for land
            '1': (64, 164, 223),    # Blue for water
            '2': (194, 178, 128)    # Tan for sand
        }
        base_color = terrain_colors.get(terrain_type, (128, 128, 128))
        tile_surface.fill(base_color)
        
        try:
            # Extract from the tileset
            source_x = col * actual_tile_size
            source_y = row * actual_tile_size
            
            # Make sure we don't exceed tileset bounds
            tileset_width = tileset.get_width()
            tileset_height = tileset.get_height()
            
            if source_x + actual_tile_size <= tileset_width and source_y + actual_tile_size <= tileset_height:
                source_rect = pygame.Rect(source_x, source_y, actual_tile_size, actual_tile_size)
                
                temp_surface = pygame.Surface((actual_tile_size, actual_tile_size))
                temp_surface.blit(tileset, (0, 0), source_rect)
                
                # Scale to target size
                if self.tile_size != actual_tile_size:
                    tile_surface = pygame.transform.scale(temp_surface, (self.tile_size, self.tile_size))
                else:
                    tile_surface = temp_surface
            
        except Exception as e:
            print(f"Error extracting tile {tile_index} for terrain {terrain_type}: {e}")
            # Keep the solid color fallback
        
        self.tile_cache[cache_key] = tile_surface
        return tile_surface
    
    def render_map(self, screen: pygame.Surface, map_data, autotiler: AutotileLogic, 
                   offset_x: int = 0, offset_y: int = 0, animated_time: float = 0):
        """Render the complete map using autotiles"""
        tile_map = autotiler.generate_tile_map()
        
        for y in range(map_data.height):
            for x in range(map_data.width):
                terrain_type = str(map_data.get_tile(x, y))
                tile_index = tile_map[y][x]
                
                # Get tile surface
                tile_surface = self.get_tile_surface(terrain_type, tile_index)
                
                # Apply water animation if enabled
                if (terrain_type == "1" and 
                    self.config.get("animation", {}).get("water_enabled", False)):
                    tile_surface = self._apply_water_animation(tile_surface, x, y, animated_time)
                
                # Calculate screen position
                screen_x = offset_x + x * self.tile_size
                screen_y = offset_y + y * self.tile_size
                
                screen.blit(tile_surface, (screen_x, screen_y))
    
    def _apply_water_animation(self, tile_surface: pygame.Surface, 
                          tile_x: int, tile_y: int, time: float) -> pygame.Surface:
        """Apply simple color-based water animation"""
        import math
        
        animation_config = self.config.get("animation", {})
        speed = animation_config.get("water_speed", 2.0)
        amplitude = animation_config.get("water_amplitude", 10)
        
        # Create animated surface
        animated_surface = tile_surface.copy()
        
        # Simple color shift animation
        wave_offset = (tile_x + tile_y) * 0.3  # Stagger animation per tile
        wave = math.sin(time * speed + wave_offset)
        color_shift = int(wave * amplitude)
        
        # Apply color modulation using pygame's special flags
        # Create a color overlay surface
        overlay = pygame.Surface(animated_surface.get_size())
        
        # Calculate the color shift - make water shimmer between blue tones
        base_blue = 100
        blue_shift = base_blue + color_shift
        blue_shift = max(50, min(150, blue_shift))  # Clamp between 50-150
        
        overlay.fill((0, 0, blue_shift))
        
        # Blend the overlay with the original surface
        animated_surface.blit(overlay, (0, 0), special_flags=pygame.BLEND_ADD)
        
        # Optional: Add some transparency variation for wave effect
        alpha_variation = int(20 * math.sin(time * speed * 1.5 + wave_offset))
        alpha = 255 + alpha_variation
        alpha = max(200, min(255, alpha))  # Keep it mostly opaque
        
        animated_surface.set_alpha(alpha)
        
        return animated_surface