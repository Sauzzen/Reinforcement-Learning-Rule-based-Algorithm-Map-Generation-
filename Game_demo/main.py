
import pygame
import sys
import os
import time
from typing import Tuple, Optional

# Import our modules
from map_loader import MapLoader, find_spawn_points
from autotile_logic import AutotileLogic
from tileset_renderer import TilesetRenderer

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
BACKGROUND_COLOR = (20, 20, 30)

# Player settings
PLAYER_COLOR = (255, 50, 50)
PLAYER_SPEED = 4
PLAYER_SIZE_OFFSET = 4
    
class GameCamera:
    """Enhanced camera with smooth movement and viewport culling"""
    
    def __init__(self, map_width: int, map_height: int, tile_size: int):
        self.x = 0.0
        self.y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.map_pixel_width = map_width * tile_size
        self.map_pixel_height = map_height * tile_size
        self.tile_size = tile_size
        self.smoothing = 0.1  # Camera smoothing factor
        
    def update(self, target_x: int, target_y: int, smooth: bool = True):
        """Update camera to follow target with optional smoothing"""
        # Calculate target camera position
        self.target_x = target_x - WINDOW_WIDTH // 2
        self.target_y = target_y - WINDOW_HEIGHT // 2
        
        # Clamp target to map bounds
        self.target_x = max(0, min(self.target_x, max(0, self.map_pixel_width - WINDOW_WIDTH)))
        self.target_y = max(0, min(self.target_y, max(0, self.map_pixel_height - WINDOW_HEIGHT)))
        
        if smooth:
            # Smooth camera movement
            self.x += (self.target_x - self.x) * self.smoothing
            self.y += (self.target_y - self.y) * self.smoothing
        else:
            # Instant camera movement
            self.x = self.target_x
            self.y = self.target_y
    
    def get_visible_tile_bounds(self) -> Tuple[int, int, int, int]:
        """Get the bounds of tiles that are currently visible"""
        # Add buffer to avoid pop-in
        buffer = 1
        
        start_x = max(0, int(self.x // self.tile_size) - buffer)
        start_y = max(0, int(self.y // self.tile_size) - buffer)
        end_x = min(int((self.x + WINDOW_WIDTH) // self.tile_size) + buffer + 1, 
                   self.map_pixel_width // self.tile_size)
        end_y = min(int((self.y + WINDOW_HEIGHT) // self.tile_size) + buffer + 1,
                   self.map_pixel_height // self.tile_size)
        
        return start_x, start_y, end_x, end_y

class Player:
    """Enhanced player with smoother movement and better collision"""
    
    def __init__(self, x: int, y: int, tile_size: int):
        self.x = float(x)
        self.y = float(y)
        self.tile_size = tile_size
        self.size = tile_size - PLAYER_SIZE_OFFSET
        self.speed = PLAYER_SPEED
        self.last_terrain = 0
        
        # Movement state
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.acceleration = 0.5
        self.friction = 0.8
        
        self.sprite = self._load_sprite()
        
    def _load_sprite(self) -> pygame.Surface:
        """Enhanced sprite loading with better fallback"""
        sprite_paths = [
            "assets/sprites/player.png",
            "assets/player.png",
            "sprites/player.png"
        ]
        
        for path in sprite_paths:
            if os.path.exists(path):
                try:
                    sprite = pygame.image.load(path).convert_alpha()
                    return pygame.transform.scale(sprite, (self.size, self.size))
                except pygame.error:
                    continue
        
        # Enhanced default sprite
        sprite = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        center = self.size // 2
        
        # Body
        pygame.draw.circle(sprite, PLAYER_COLOR, (center, center), center - 2)
        # Highlight
        pygame.draw.circle(sprite, (255, 150, 150), (center - 3, center - 3), center // 3)
        # Border
        pygame.draw.circle(sprite, (255, 255, 255), (center, center), center - 2, 2)
        # Direction indicator
        pygame.draw.circle(sprite, (255, 255, 255), (center, center - 4), 2)
        
        return sprite
    
    def update(self, map_data, keys_pressed, dt: float):
        """Enhanced update with physics-based movement"""
        # Calculate desired movement
        target_vel_x = 0
        target_vel_y = 0
        
        if keys_pressed[pygame.K_LEFT] or keys_pressed[pygame.K_a]:
            target_vel_x = -self.speed
        if keys_pressed[pygame.K_RIGHT] or keys_pressed[pygame.K_d]:
            target_vel_x = self.speed
        if keys_pressed[pygame.K_UP] or keys_pressed[pygame.K_w]:
            target_vel_y = -self.speed
        if keys_pressed[pygame.K_DOWN] or keys_pressed[pygame.K_s]:
            target_vel_y = self.speed
        
        # Apply acceleration/friction
        if target_vel_x != 0:
            self.velocity_x += (target_vel_x - self.velocity_x) * self.acceleration * dt * 60
        else:
            self.velocity_x *= self.friction
        
        if target_vel_y != 0:
            self.velocity_y += (target_vel_y - self.velocity_y) * self.acceleration * dt * 60
        else:
            self.velocity_y *= self.friction
        
        # Apply movement with collision detection
        self._move_with_collision(map_data, dt)
        
        # Update terrain tracking
        self.last_terrain = self.get_current_terrain(map_data)
    
    def _move_with_collision(self, map_data, dt: float):
        """Move with proper collision detection"""
        # Move X axis
        new_x = self.x + self.velocity_x * dt * 60
        if self._is_position_walkable(new_x, self.y, map_data):
            self.x = new_x
        else:
            self.velocity_x = 0
        
        # Move Y axis
        new_y = self.y + self.velocity_y * dt * 60
        if self._is_position_walkable(self.x, new_y, map_data):
            self.y = new_y
        else:
            self.velocity_y = 0
    
    def _is_position_walkable(self, x: float, y: float, map_data) -> bool:
        """Enhanced collision detection"""
        # Check multiple points around player
        check_points = [
            (x + 2, y + 2),                    # Top-left
            (x + self.size - 2, y + 2),        # Top-right
            (x + 2, y + self.size - 2),        # Bottom-left
            (x + self.size - 2, y + self.size - 2),  # Bottom-right
            (x + self.size // 2, y + self.size // 2)  # Center
        ]
        
        for px, py in check_points:
            if px < 0 or py < 0:
                return False
                
            tile_x = int(px // self.tile_size)
            tile_y = int(py // self.tile_size)
            terrain = map_data.get_tile(tile_x, tile_y)
            
            # Water (1) or out of bounds is not walkable
            if terrain == 1 or terrain == -1:
                return False
        
        return True
    
    def render(self, screen: pygame.Surface, camera):
        """Render player with sub-pixel positioning"""
        screen_x = int(self.x - camera.x + PLAYER_SIZE_OFFSET // 2)
        screen_y = int(self.y - camera.y + PLAYER_SIZE_OFFSET // 2)
        screen.blit(self.sprite, (screen_x, screen_y))
    
    def get_current_terrain(self, map_data) -> int:
        """Get terrain at player center"""
        center_x = int((self.x + self.size // 2) // self.tile_size)
        center_y = int((self.y + self.size // 2) // self.tile_size)
        return map_data.get_tile(center_x, center_y)

class MapDemo:
    """Enhanced demo with better performance and features"""
    
    def __init__(self, map_file: str):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Enhanced RL Generated Map Demo")
        self.clock = pygame.time.Clock()
        
        # Load systems
        print("Loading map and initializing systems...")
        self.map_data = MapLoader.load_from_file(map_file)
        self.autotiler = AutotileLogic(self.map_data)
        self.renderer = TilesetRenderer()
        
        self.tile_size = self.renderer.tile_size
        
        # Initialize camera and player
        self._initialize_camera_and_player()
        
        # Game state
        self.running = True
        self.paused = False
        self.show_debug = False
        self.show_minimap = False
        self.camera_smooth = True
        self.start_time = time.time()
        
        # Performance tracking
        self.frame_times = []
        self.tiles_rendered = 0
        
        print("Demo initialized successfully!")
    
    def _initialize_camera_and_player(self):
        """Initialize camera and player with proper positioning"""
        self.camera = GameCamera(
            self.map_data.width,
            self.map_data.height,
            self.tile_size
        )
        
        # Find good spawn point
        spawn_points = find_spawn_points(self.map_data, terrain_type=0, min_area=9)
        if spawn_points:
            # Use spawn point with most surrounding space
            best_spawn = spawn_points[len(spawn_points) // 2]
            spawn_x, spawn_y = best_spawn
            pixel_x = spawn_x * self.tile_size
            pixel_y = spawn_y * self.tile_size
        else:
            # Fallback: find any land tile
            pixel_x = pixel_y = None
            for y in range(self.map_data.height):
                for x in range(self.map_data.width):
                    if self.map_data.get_tile(x, y) == 0:  # Land
                        pixel_x = x * self.tile_size
                        pixel_y = y * self.tile_size
                        break
                if pixel_x is not None:
                    break
            
            # Last resort
            if pixel_x is None:
                pixel_x = self.map_data.width * self.tile_size // 2
                pixel_y = self.map_data.height * self.tile_size // 2
        
        self.player = Player(pixel_x, pixel_y, self.tile_size)
        
        # Set initial camera position
        player_center_x = self.player.x + self.player.size // 2
        player_center_y = self.player.y + self.player.size // 2
        self.camera.update(player_center_x, player_center_y, smooth=False)
    
    def handle_events(self):
        """Enhanced event handling"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_F1:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_F2:
                    self.show_minimap = not self.show_minimap
                elif event.key == pygame.K_F3:
                    self.camera_smooth = not self.camera_smooth
                elif event.key == pygame.K_r:
                    self._initialize_camera_and_player()
    
    def update(self, dt: float):
        """Update with delta time"""
        if not self.paused:
            keys = pygame.key.get_pressed()
            self.player.update(self.map_data, keys, dt)
            
            # Update camera
            player_center_x = int(self.player.x + self.player.size // 2)
            player_center_y = int(self.player.y + self.player.size // 2)
            self.camera.update(player_center_x, player_center_y, self.camera_smooth)
    
    def render(self):
        """Enhanced rendering with viewport culling"""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Get visible area
        start_x, start_y, end_x, end_y = self.camera.get_visible_tile_bounds()
        
        # Render visible tiles only
        current_time = time.time() - self.start_time
        self.tiles_rendered = self._render_visible_tiles(start_x, start_y, end_x, end_y, current_time)
        
        # Render player
        self.player.render(self.screen, self.camera)
        
        # UI
        self._render_ui()
        
        if self.show_debug:
            self._render_debug_info()
        
        if self.show_minimap:
            self._render_minimap()
    
    def _render_visible_tiles(self, start_x: int, start_y: int, end_x: int, end_y: int, current_time: float) -> int:
        """Render only visible tiles for better performance"""
        tile_map = self.autotiler.generate_tile_map()
        tiles_rendered = 0
        
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if 0 <= x < self.map_data.width and 0 <= y < self.map_data.height:
                    terrain_type = str(self.map_data.get_tile(x, y))
                    tile_index = tile_map[y][x]
                    
                    # Get and render tile
                    tile_surface = self.renderer.get_tile_surface(terrain_type, tile_index)
                    
                    # Apply water animation
                    if terrain_type == "1" and self.renderer.config.get("animation", {}).get("water_enabled", False):
                        tile_surface = self.renderer._apply_water_animation(tile_surface, x, y, current_time)
                    
                    screen_x = int(-self.camera.x + x * self.tile_size)
                    screen_y = int(-self.camera.y + y * self.tile_size)
                    
                    self.screen.blit(tile_surface, (screen_x, screen_y))
                    tiles_rendered += 1
        
        return tiles_rendered
    
    def _render_ui(self):
        """Enhanced UI rendering"""
        font = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 24)
        
        # Title
        title = font.render("Enhanced RL Map Demo", True, (255, 255, 255))
        self.screen.blit(title, (10, 10))
        
        # Controls
        y_offset = 45
        controls = [
            "WASD: Move",
            "SPACE: Pause", 
            "F1: Debug",
            "F2: Minimap",
            "F3: Camera Mode",
            "R: Reset",
            "ESC: Quit"
        ]
        
        for control in controls:
            text = font_small.render(control, True, (200, 200, 200))
            self.screen.blit(text, (10, y_offset))
            y_offset += 22
        
        # Status
        terrain_names = {0: "Land", 1: "Water", 2: "Sand", -1: "Out of Bounds"}
        current_terrain = self.player.get_current_terrain(self.map_data)
        
        status_texts = [
            f"Terrain: {terrain_names.get(current_terrain, 'Unknown')}",
            f"Camera: {'Smooth' if self.camera_smooth else 'Instant'}"
        ]
        
        y_offset += 10
        for status in status_texts:
            text = font_small.render(status, True, (255, 255, 100))
            self.screen.blit(text, (10, y_offset))
            y_offset += 22
        
        # Pause indicator
        if self.paused:
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            text_rect = pause_text.get_rect(center=(WINDOW_WIDTH // 2, 50))
            pygame.draw.rect(self.screen, (0, 0, 0), text_rect.inflate(20, 10))
            self.screen.blit(pause_text, text_rect)
    
    def _render_debug_info(self):
        """Enhanced debug information"""
        font = pygame.font.Font(None, 20)
        
        # Calculate average FPS
        current_fps = self.clock.get_fps()
        self.frame_times.append(current_fps)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        avg_fps = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        
        debug_info = [
            f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})",
            f"Player: ({self.player.x:.1f}, {self.player.y:.1f})",
            f"Velocity: ({self.player.velocity_x:.1f}, {self.player.velocity_y:.1f})",
            f"Camera: ({self.camera.x:.1f}, {self.camera.y:.1f})",
            f"Tile: ({int(self.player.x//self.tile_size)}, {int(self.player.y//self.tile_size)})",
            f"Map: {self.map_data.width}x{self.map_data.height}",
            f"Tiles Rendered: {self.tiles_rendered}",
            f"Total Tiles: {self.map_data.width * self.map_data.height}"
        ]
        
        # Background
        bg_height = len(debug_info) * 22 + 10
        debug_bg = pygame.Surface((280, bg_height))
        debug_bg.fill((0, 0, 0))
        debug_bg.set_alpha(180)
        self.screen.blit(debug_bg, (WINDOW_WIDTH - 290, 10))
        
        # Text
        for i, info in enumerate(debug_info):
            color = (255, 255, 255)
            if "FPS:" in info and current_fps < 30:
                color = (255, 100, 100)  # Red for low FPS
            
            text = font.render(info, True, color)
            self.screen.blit(text, (WINDOW_WIDTH - 280, 15 + i * 22))
    
    def _render_minimap(self):
        """Simple minimap implementation"""
        minimap_size = 150
        scale = min(minimap_size / self.map_data.width, minimap_size / self.map_data.height)
        
        # Create minimap surface
        minimap = pygame.Surface((int(self.map_data.width * scale), int(self.map_data.height * scale)))
        
        # Draw terrain
        terrain_colors = {0: (100, 200, 100), 1: (100, 100, 255), 2: (255, 255, 150)}
        
        for y in range(self.map_data.height):
            for x in range(self.map_data.width):
                terrain = self.map_data.get_tile(x, y)
                color = terrain_colors.get(terrain, (128, 128, 128))
                
                rect = pygame.Rect(int(x * scale), int(y * scale), max(1, int(scale)), max(1, int(scale)))
                pygame.draw.rect(minimap, color, rect)
        
        # Draw player position
        player_x = int((self.player.x // self.tile_size) * scale)
        player_y = int((self.player.y // self.tile_size) * scale)
        pygame.draw.circle(minimap, (255, 0, 0), (player_x, player_y), 2)
        
        # Position minimap
        minimap_rect = minimap.get_rect()
        minimap_rect.topright = (WINDOW_WIDTH - 10, 10)
        
        # Background
        bg_rect = minimap_rect.inflate(6, 6)
        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), bg_rect, 2)
        
        self.screen.blit(minimap, minimap_rect)
    
    def run(self):
        """Main game loop with proper timing"""
        print("Starting enhanced demo...")
        print("New controls: F2=Minimap, F3=Camera mode")
        
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            self.handle_events()
            self.update(dt)
            self.render()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        print("Enhanced demo ended.")

def main():
    """Entry point with better error handling"""
    default_map = "maps/final_map_seed0.json"
    map_file = sys.argv[1] if len(sys.argv) > 1 else default_map
    
    print(f"Looking for map file: {map_file}")
    print(f"File exists: {os.path.exists(map_file)}")
    
    
    if not os.path.exists(map_file):
        print(f"Error: Map file '{map_file}' not found!")
        # Try to create a simple test map
        if create_test_map(map_file):
            print(f"Created test map at {map_file}")
        else:
            sys.exit(1)
    
    try:
        demo = MapDemo(map_file)
        demo.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def create_test_map(filepath: str) -> bool:
    """Create a simple test map if none exists"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create simple 20x15 island map
        width, height = 20, 15
        tiles = []
        
        for y in range(height):
            row = []
            for x in range(width):
                # Water border, land center, some sand
                if x == 0 or y == 0 or x == width-1 or y == height-1:
                    row.append(1)  # Water
                elif x == 1 or y == 1 or x == width-2 or y == height-2:
                    row.append(2)  # Sand
                else:
                    row.append(0)  # Land
            tiles.append(row)
        
        map_data = {
            "width": width,
            "height": height,
            "tiles": tiles,
            "metadata": {"generated": True, "type": "test"}
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(map_data, f)
        
        return True
    except Exception as e:
        print(f"Could not create test map: {e}")
        return False

if __name__ == "__main__":
    main()