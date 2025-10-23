# setup_demo.py
"""
Setup script to create necessary files and directories for the map demo.
"""

import os
import json
import pygame

def create_config_file():
    """Create the tile configuration file"""
    config = {
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
        },
        "animation": {
            "water_enabled": True,
            "water_speed": 2.0,
            "water_amplitude": 8
        }
    }
    
    os.makedirs("config", exist_ok=True)
    
    with open("config/tile_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created config/tile_config.json")

def create_placeholder_tilesets():
    """Create simple placeholder tileset images"""
    pygame.init()
    
    os.makedirs("assets/tilesets", exist_ok=True)
    os.makedirs("assets/sprites", exist_ok=True)
    
    # Tileset configuration
    tile_size = 16
    cols, rows = 6, 4
    tileset_width = cols * tile_size
    tileset_height = rows * tile_size
    
    # Color schemes for different terrains
    terrain_colors = {
        "grass.png": {
            "base": (34, 139, 34),      # Forest Green
            "light": (50, 205, 50),     # Lime Green
            "dark": (0, 100, 0)         # Dark Green
        },
        "water.png": {
            "base": (30, 144, 255),     # Dodger Blue
            "light": (135, 206, 255),   # Sky Blue
            "dark": (0, 100, 200)       # Dark Blue
        },
        "sand_light.png": {
            "base": (238, 203, 173),    # Tan
            "light": (255, 228, 196),   # Bisque
            "dark": (210, 180, 140)     # Darker Tan
        }
    }
    
    # Create each tileset
    for filename, colors in terrain_colors.items():
        surface = pygame.Surface((tileset_width, tileset_height))
        
        # Fill with base color
        surface.fill(colors["base"])
        
        # Create tile pattern
        for row in range(rows):
            for col in range(cols):
                x = col * tile_size
                y = row * tile_size
                
                # Create some variation in each tile
                tile_rect = pygame.Rect(x, y, tile_size, tile_size)
                
                # Add some texture/pattern
                if (col + row) % 2 == 0:
                    # Lighter tiles
                    pygame.draw.rect(surface, colors["light"], tile_rect)
                    # Add some detail
                    pygame.draw.rect(surface, colors["dark"], tile_rect, 1)
                else:
                    # Base color tiles with border
                    pygame.draw.rect(surface, colors["base"], tile_rect)
                    pygame.draw.rect(surface, colors["dark"], tile_rect, 1)
                
                # Add some inner detail for variety
                if col % 3 == 1:
                    inner_rect = pygame.Rect(x + 2, y + 2, tile_size - 4, tile_size - 4)
                    pygame.draw.rect(surface, colors["light"], inner_rect, 1)
        
        # Save the tileset
        filepath = f"assets/tilesets/{filename}"
        pygame.image.save(surface, filepath)
        print(f"Created {filepath}")
    
    # Create player sprite
    create_player_sprite()

def create_player_sprite():
    """Create a simple player sprite"""
    size = 28  # Slightly smaller than tile size
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    center = size // 2
    
    # Body (circle)
    pygame.draw.circle(surface, (255, 100, 100), (center, center), center - 2)
    
    # Highlight
    pygame.draw.circle(surface, (255, 200, 200), (center - 4, center - 4), 6)
    
    # Border
    pygame.draw.circle(surface, (200, 50, 50), (center, center), center - 2, 2)
    
    # Simple face
    # Eyes
    pygame.draw.circle(surface, (255, 255, 255), (center - 4, center - 3), 2)
    pygame.draw.circle(surface, (255, 255, 255), (center + 4, center - 3), 2)
    pygame.draw.circle(surface, (0, 0, 0), (center - 4, center - 3), 1)
    pygame.draw.circle(surface, (0, 0, 0), (center + 4, center - 3), 1)
    
    # Simple smile
    pygame.draw.arc(surface, (255, 255, 255), 
                   (center - 6, center - 2, 12, 8), 0, 3.14, 2)
    
    filepath = "assets/sprites/player.png"
    pygame.image.save(surface, filepath)
    print(f"Created {filepath}")

def create_directories():
    """Create necessary directory structure"""
    directories = [
        "config",
        "assets",
        "assets/tilesets",
        "assets/sprites",
        "maps"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Run the setup"""
    print("Setting up map demo...")
    
    create_directories()
    create_config_file()
    create_placeholder_tilesets()
    
    print("\nSetup complete!")
    print("\nYour project structure should now be:")
    print("├── main.py")
    print("├── map_loader.py")
    print("├── autotile_logic.py")
    print("├── tileset_renderer.py")
    print("├── setup_demo.py")
    print("├── config/")
    print("│   └── tile_config.json")
    print("├── maps/")
    print("│   └── final_map_seed0.json")
    print("└── assets/")
    print("    ├── tilesets/")
    print("    │   ├── grass.png")
    print("    │   ├── water.png")
    print("    │   └── sand_light.png")
    print("    └── sprites/")
    print("        └── player.png")
    print("\nRun: python main.py maps/final_map_seed0.json")

if __name__ == "__main__":
    main()