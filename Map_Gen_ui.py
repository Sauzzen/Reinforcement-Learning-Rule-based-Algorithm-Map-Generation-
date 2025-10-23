"""
Map Generator GUI - Thesis Defense Demo
Compatible with the three-phase hierarchical RL system
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import subprocess
import shutil

# Add hierarchical directory to path for imports
_HIERARCHICAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hierarchical')
if _HIERARCHICAL_PATH not in sys.path:
    sys.path.insert(0, _HIERARCHICAL_PATH)

# Type hints for imported modules (resolved at runtime)
try:
    from map_env import MapEnvironment  # type: ignore
    from train_agent import ImprovedRLAgent  # type: ignore
except ImportError:
    # Will be imported when needed in load_model()
    MapEnvironment = None  # type: ignore
    ImprovedRLAgent = None  # type: ignore

class MapGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Game Map Generator - Hybrid RL + Rule-Based System")
        self.root.geometry("1000x900")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.hierarchical_loaded = False
        self.hybrid_loaded = False
        self.hierarchical_env = None
        self.hierarchical_agent = None
        self.hybrid_env = None
        self.hybrid_agent = None
        self.current_map = None
        
        # Create UI
        self.create_widgets()
        
        # Try to load trained model
        self.attempt_load_model()
    
    def create_widgets(self):
        """Create all UI components"""
        
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="2D Game Map Generator",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Hybrid Reinforcement Learning + Rule-Based System",
            font=("Arial", 12),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        subtitle_label.pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        control_panel = tk.LabelFrame(
            main_container,
            text="Generation Controls",
            font=("Arial", 14, "bold"),
            bg="white",
            padx=15,
            pady=15
        )
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Method selection
        tk.Label(
            control_panel,
            text="Generation Method:",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(anchor="w", pady=(5, 5))
        
        self.method_var = tk.StringVar(value="perlin")
        
        methods = [
            ("RL-Based (3-Phase Hierarchical)", "hierarchical", 
             "Uses trained neural network\nPhase 1: Macro terrain\nPhase 2: Shorelines\nPhase 3: Details"),
            ("Hybrid (RL + Rule-Based)", "hybrid",
             "Combines RL with automatic shorelines\nBalanced approach"),
            ("Perlin Noise (Procedural)", "perlin",
             "Uses fractal Perlin noise\nFast procedural generation")
        ]
        
        for text, value, tooltip in methods:
            rb = tk.Radiobutton(
                control_panel,
                text=text,
                variable=self.method_var,
                value=value,
                font=("Arial", 11),
                bg="white",
                activebackground="white",
                command=self.on_method_change
            )
            rb.pack(anchor="w", pady=3)
        
        tk.Frame(control_panel, height=20, bg="white").pack()  # Spacer
        
        # Seed input
        tk.Label(
            control_panel,
            text="Random Seed:",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(anchor="w", pady=(5, 5))
        
        seed_frame = tk.Frame(control_panel, bg="white")
        seed_frame.pack(fill=tk.X, pady=5)
        
        self.seed_entry = tk.Entry(
            seed_frame,
            font=("Arial", 12),
            width=15
        )
        self.seed_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.seed_entry.insert(0, str(np.random.randint(0, 100000)))
        
        tk.Button(
            seed_frame,
            text="Randomize",
            font=("Arial", 10),
            command=self.randomize_seed,
            width=10
        ).pack(side=tk.LEFT)
        
        tk.Frame(control_panel, height=20, bg="white").pack()  # Spacer
        
        # Generate button
        self.generate_btn = tk.Button(
            control_panel,
            text="Generate Map",
            font=("Arial", 14, "bold"),
            bg="#27ae60",
            fg="white",
            activebackground="#229954",
            activeforeground="white",
            command=self.generate_map,
            height=2,
            cursor="hand2"
        )
        self.generate_btn.pack(fill=tk.X, pady=10)
        
        # Export button
        self.export_btn = tk.Button(
            control_panel,
            text="Export Map",
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            command=self.export_map,
            state=tk.DISABLED
        )
        self.export_btn.pack(fill=tk.X, pady=5)
        
        # Play Map button
        self.play_btn = tk.Button(
            control_panel,
            text="Play This Map",
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            command=self.play_map,
            state=tk.DISABLED,
            cursor="hand2"
        )
        self.play_btn.pack(fill=tk.X, pady=5)
        
        tk.Frame(control_panel, height=20, bg="white").pack()  # Spacer
        
        # Status indicator
        self.status_frame = tk.Frame(control_panel, bg="white")
        self.status_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            self.status_frame,
            text="System Status:",
            font=("Arial", 11, "bold"),
            bg="white"
        ).pack(anchor="w")
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Initializing...",
            font=("Arial", 10),
            bg="white",
            fg="gray",
            wraplength=200,
            justify=tk.LEFT
        )
        self.status_label.pack(anchor="w", pady=(5, 0))
        
        # Right panel - Display
        display_panel = tk.Frame(main_container, bg="white")
        display_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Map canvas
        canvas_frame = tk.LabelFrame(
            display_panel,
            text="Generated Map (64×64)",
            font=("Arial", 12, "bold"),
            bg="white",
            padx=10,
            pady=10
        )
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=640,
            height=640,
            bg="#ecf0f1",
            highlightthickness=1,
            highlightbackground="#bdc3c7"
        )
        self.canvas.pack()
        
        # Initial placeholder
        self.canvas.create_text(
            320, 320,
            text="Click 'Generate Map' to start",
            font=("Arial", 16),
            fill="#7f8c8d"
        )
        
        # Metrics panel
        metrics_frame = tk.LabelFrame(
            display_panel,
            text="Quality Metrics",
            font=("Arial", 12, "bold"),
            bg="white",
            padx=15,
            pady=10
        )
        metrics_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.metrics_container = tk.Frame(metrics_frame, bg="white")
        self.metrics_container.pack()
        
        # Initial metrics placeholder
        tk.Label(
            self.metrics_container,
            text="No metrics available yet",
            font=("Arial", 11),
            bg="white",
            fg="gray"
        ).pack()
    
    def attempt_load_model(self):
        """Try to load trained RL models"""
        # Try loading hierarchical model
        hierarchical_path = "hierarchical/checkpoints/best_model.pth"
        if os.path.exists(hierarchical_path):
            try:
                self.load_model(hierarchical_path, model_type='hierarchical')
            except Exception as e:
                print(f"Failed to load hierarchical model: {e}")
        
        # Try loading hybrid model
        hybrid_path = "Hybrid/checkpoints/best_model.pth"
        if os.path.exists(hybrid_path):
            try:
                self.load_model(hybrid_path, model_type='hybrid')
            except Exception as e:
                print(f"Failed to load hybrid model: {e}")
        
        # Update status
        if self.hierarchical_loaded or self.hybrid_loaded:
            models = []
            if self.hierarchical_loaded:
                models.append("Hierarchical")
            if self.hybrid_loaded:
                models.append("Hybrid")
            
            self.status_label.config(
                text=f"✓ Models loaded:\n{', '.join(models)}",
                fg="#27ae60"
            )
        else:
            print("\n" + "="*60)
            print("  ⚠ NO TRAINED MODELS FOUND")
            print("  Running in PERLIN NOISE MODE")
            print("="*60 + "\n")
            
            self.status_label.config(
                text="⚠ No trained models found\nUsing Perlin noise generation",
                fg="#e67e22"
            )
    
    def load_model(self, model_path, model_type='hierarchical'):
        """Load trained RL model"""
        print(f"Loading {model_type} model from: {model_path}")
        
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Import torch
            import torch
            import importlib
            
            # Determine input channels based on model type
            if model_type == 'hierarchical':
                input_channels = 5
                subfolder = 'hierarchical'
            else:  # hybrid
                input_channels = 4
                subfolder = 'Hybrid'
            
            # Clear any cached imports and set correct path
            module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), subfolder)
            
            # Temporarily modify sys.path to import from correct folder
            original_path = sys.path.copy()
            sys.path.insert(0, module_path)
            
            # Force reload to get the correct version
            if 'map_env' in sys.modules:
                del sys.modules['map_env']
            if 'train_agent' in sys.modules:
                del sys.modules['train_agent']
            
            from map_env import MapEnvironment  # type: ignore
            from train_agent import ImprovedRLAgent  # type: ignore
            
            # Initialize environment
            env = MapEnvironment(
                size=64,
                action_mode='hierarchical',
                max_steps=20 if model_type == 'hierarchical' else 15
            )

            # Initialize agent with correct parameters
            agent = ImprovedRLAgent(
                state_size=(64, 64),
                action_size=9,
                input_channels=input_channels,
                device=torch.device('cpu'),
                gamma=0.98,
                lr=5e-4,
                batch_size=32,
                eps_start=0.0,  # No exploration for inference
                eps_end=0.0,
                eps_decay_episodes=1,
                eps_schedule="linear"
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Load the policy network state
            if isinstance(checkpoint, dict):
                if 'policy_state' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['policy_state'])
                elif 'model_state_dict' in checkpoint:
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Assume whole checkpoint is the state_dict
                    agent.policy_net.load_state_dict(checkpoint)
            else:
                agent.policy_net.load_state_dict(checkpoint)
            
            agent.policy_net.eval()
            
            # Restore original sys.path
            sys.path = original_path
            
            # Store based on model type
            if model_type == 'hierarchical':
                self.hierarchical_env = env
                self.hierarchical_agent = agent
                self.hierarchical_loaded = True
            else:  # hybrid
                self.hybrid_env = env
                self.hybrid_agent = agent
                self.hybrid_loaded = True
            
            print(f"✓ {model_type.capitalize()} model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading {model_type} model: {e}")
            import traceback
            traceback.print_exc()
            # Restore path even on error
            if 'original_path' in locals():
                sys.path = original_path


    
    def on_method_change(self):
        """Handle method selection change"""
        method = self.method_var.get()
        
        if method == "hierarchical" and not self.hierarchical_loaded:
            messagebox.showwarning(
                "Model Not Available",
                "Hierarchical model not found. Switching to Perlin noise generation."
            )
            self.method_var.set("perlin")
        elif method == "hybrid" and not self.hybrid_loaded:
            messagebox.showwarning(
                "Model Not Available",
                "Hybrid model not found. Switching to Perlin noise generation."
            )
            self.method_var.set("perlin")
    
    def randomize_seed(self):
        """Generate random seed"""
        self.seed_entry.delete(0, tk.END)
        self.seed_entry.insert(0, str(np.random.randint(0, 100000)))
    
    def generate_map(self):
        """Generate a map using selected method"""
        import time
        
        self.status_label.config(text="⏳ Generating map...", fg="#3498db")
        self.generate_btn.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            method = self.method_var.get()
            seed = int(self.seed_entry.get())
            
            # Track generation time and steps
            start_time = time.time()
            
            if method == "hierarchical" and self.hierarchical_loaded:
                # Use Hierarchical RL generation
                world, steps = self.generate_rl(seed, self.hierarchical_env, self.hierarchical_agent)
            elif method == "hybrid" and self.hybrid_loaded:
                # Use Hybrid RL generation
                world, steps = self.generate_rl(seed, self.hybrid_env, self.hybrid_agent)
            else:
                # Use Perlin noise generation
                world, steps = self.generate_perlin(seed)
            
            generation_time = time.time() - start_time
            
            # Store current map and metadata
            self.current_map = world
            self.current_generation_time = generation_time
            self.current_steps = steps
            
            # Display map
            self.display_map(world)
            
            # Compute and display metrics
            metrics = self.compute_metrics(world)
            self.display_metrics(metrics, method, seed, generation_time, steps)
            
            # Enable export and play buttons
            self.export_btn.config(state=tk.NORMAL)
            self.play_btn.config(state=tk.NORMAL)
            
            self.status_label.config(
                text="✓ Generation complete!",
                fg="#27ae60"
            )
            
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))
            self.status_label.config(
                text=f"✗ Error: {str(e)[:40]}...",
                fg="#e74c3c"
            )
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.generate_btn.config(state=tk.NORMAL)
    
    def generate_rl(self, seed, env, agent):
        """Generate map using RL model"""
        import torch
        
        # Reset environment with proper signature
        np.random.seed(seed)
        state = env.reset(init_mode='perlin', init_seed=seed)
        done = False
        steps = 0
        max_steps = env.max_regions if hasattr(env, 'max_regions') else 20
        
        print(f"Generating with RL (seed={seed})...")
        
        while not done and steps < max_steps:
            # Get valid action mask for current phase
            action_mask = env.get_valid_action_mask()
            
            # Select action (deterministic)
            with torch.no_grad():
                action = agent.act(state, action_mask, deterministic=True)
            
            # Take step
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1
            
            # Update progress
            if steps % 5 == 0:
                progress = min(100, int(steps / max_steps * 100))
                self.status_label.config(
                    text=f"⏳ Generating... {progress}%",
                    fg="#3498db"
                )
                self.root.update()
        
        print(f"✓ Generated in {steps} steps")
        return env.world, steps
    
    def generate_perlin(self, seed):
        """Generate map using Perlin noise"""
        from scipy.ndimage import gaussian_filter, label
        
        print(f"Generating with Perlin noise (seed={seed})...")
        
        np.random.seed(seed)
        size = 64
        steps = 0  # Perlin is single-step generation
        
        # Generate fractal Perlin noise (inline implementation)
        noise = self._generate_fractal_noise(size, seed)
        steps += 1
        
        # Apply Gaussian smoothing
        noise = gaussian_filter(noise, sigma=1.5)
        
        # Threshold to create land/water (target ~40% water)
        water_threshold = np.percentile(noise, 60)
        world = np.where(noise > water_threshold, 1, 0)  # 1=WATER, 0=LAND
        
        # Extract largest land mass
        land_mask = (world == 0)
        labeled, num_features = label(land_mask)
        
        if num_features > 0:
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest_idx = np.argmax(sizes) + 1
            world[labeled != largest_idx] = 1  # Convert small islands to water
        
        # Apply automatic shorelines (inline implementation)
        world = self._apply_shorelines(world)
        
        print("✓ Generated with Perlin noise")
        return world, steps
    
    def _generate_fractal_noise(self, size, seed):
        """Generate fractal Perlin-like noise"""
        np.random.seed(seed)
        noise = np.zeros((size, size), dtype=np.float32)
        
        # Multi-octave noise generation
        octaves = 4
        persistence = 0.5
        amplitude = 1.0
        frequency = 1.0
        
        for octave in range(octaves):
            # Generate random gradients
            grid_size = max(2, int(4 * frequency))
            gradients = np.random.randn(grid_size, grid_size)
            
            # Interpolate to full size
            from scipy.ndimage import zoom
            zoom_factor = size / grid_size
            octave_noise = zoom(gradients, zoom_factor, order=1)
            
            # Ensure correct size
            octave_noise = octave_noise[:size, :size]
            
            # Add to cumulative noise
            noise += amplitude * octave_noise
            
            # Update for next octave
            amplitude *= persistence
            frequency *= 2.0
        
        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
        return noise
    
    def _apply_shorelines(self, world):
        """Apply sand shorelines between land and water"""
        n = world.shape[0]
        new_world = world.copy()
        
        for r in range(n):
            for c in range(n):
                if world[r, c] == 1:  # WATER tile
                    # Check if adjacent to land
                    has_land = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n and world[nr, nc] == 0:
                            has_land = True
                            break
                    
                    if has_land:
                        new_world[r, c] = 2  # SAND
        
        return new_world
    
    def display_map(self, world):
        """Display map on canvas"""
        # Color scheme
        colors = {
            0: (34, 139, 34),    # LAND - Forest Green
            1: (30, 144, 255),   # WATER - Dodger Blue
            2: (238, 214, 175)   # SAND - Wheat/Tan
        }
        
        # Create RGB image
        h, w = world.shape
        img_array = np.zeros((h, w, 3), dtype=np.uint8)
        
        for tile_type, color in colors.items():
            mask = (world == tile_type)
            img_array[mask] = color
        
        # Scale up to 640x640
        img = Image.fromarray(img_array, mode='RGB')
        img = img.resize((640, 640), Image.NEAREST)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(320, 320, image=self.photo)
    
    def compute_metrics(self, world):
        """Compute quality metrics"""
        total_tiles = world.size
        
        # Count tile types
        water_count = np.sum(world == 1)
        sand_count = np.sum(world == 2)
        land_count = np.sum(world == 0)
        
        # Ratios
        water_ratio = water_count / total_tiles
        sand_ratio = sand_count / total_tiles
        land_ratio = land_count / total_tiles
        
        # Water ratio error (target: 35-45%)
        water_target = 0.40
        water_error = abs(water_ratio - water_target)
        
        # Sand quality (% of sand tiles at boundaries)
        sand_quality = self.compute_sand_quality(world)
        
        # Island coherence (largest island size)
        island_coherence = self.compute_island_coherence(world)
        
        # Acceptance criteria
        accepted = (
            water_error <= 0.02 and
            sand_quality >= 0.75 and
            0.20 <= island_coherence <= 0.60
        )
        
        return {
            'water_ratio': water_ratio,
            'sand_ratio': sand_ratio,
            'land_ratio': land_ratio,
            'water_error': water_error,
            'sand_quality': sand_quality,
            'island_coherence': island_coherence,
            'accepted': accepted
        }
    
    def compute_sand_quality(self, world):
        """Compute % of sand tiles properly positioned"""
        sand_tiles = (world == 2)
        if np.sum(sand_tiles) == 0:
            return 0.0
        
        # Check each sand tile
        valid_sand = 0
        total_sand = 0
        
        h, w = world.shape
        for r in range(h):
            for c in range(w):
                if world[r, c] == 2:
                    total_sand += 1
                    # Check if adjacent to both land and water
                    neighbors = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            neighbors.append(world[nr, nc])
                    
                    has_land = 0 in neighbors
                    has_water = 1 in neighbors
                    
                    if has_land and has_water:
                        valid_sand += 1
        
        return valid_sand / total_sand if total_sand > 0 else 0.0
    
    def compute_island_coherence(self, world):
        """Compute largest island as % of map"""
        from scipy.ndimage import label
        
        land_mask = (world == 0)
        labeled, num_features = label(land_mask)
        
        if num_features == 0:
            return 0.0
        
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        largest = max(sizes)
        
        return largest / world.size
    
    def display_metrics(self, metrics, method, seed, generation_time, steps):
        """Display quality metrics"""
        # Clear previous metrics
        for widget in self.metrics_container.winfo_children():
            widget.destroy()
        
        # Header
        header = tk.Frame(self.metrics_container, bg="white")
        header.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            header,
            text=f"Method: {method.upper()} | Seed: {seed}",
            font=("Arial", 11, "bold"),
            bg="white"
        ).pack(side=tk.LEFT)
        
        # Acceptance status
        status_color = "#27ae60" if metrics['accepted'] else "#e74c3c"
        status_text = "✓ PASS" if metrics['accepted'] else "✗ FAIL"
        
        tk.Label(
            header,
            text=status_text,
            font=("Arial", 11, "bold"),
            bg="white",
            fg=status_color
        ).pack(side=tk.RIGHT)
        
        # Performance metrics section
        perf_frame = tk.Frame(self.metrics_container, bg="white")
        perf_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            perf_frame,
            text=f"Generation Time: {generation_time:.3f}s | Steps: {steps}",
            font=("Arial", 10, "bold"),
            bg="white",
            fg="#3498db"
        ).pack()
        
        # Metrics grid
        metrics_grid = tk.Frame(self.metrics_container, bg="white")
        metrics_grid.pack(fill=tk.X)
        
        metric_data = [
            ("Water Ratio:", f"{metrics['water_ratio']*100:.1f}%", 
             f"(Target: 35-45%, Error: {metrics['water_error']*100:.1f}%)"),
            ("Sand Quality:", f"{metrics['sand_quality']*100:.1f}%", 
             "(Target: ≥75%)"),
            ("Island Size:", f"{metrics['island_coherence']*100:.1f}%", 
             "(Target: 20-60%)"),
            ("Land Ratio:", f"{metrics['land_ratio']*100:.1f}%", ""),
            ("Sand Ratio:", f"{metrics['sand_ratio']*100:.1f}%", "")
        ]
        
        for i, (label, value, target) in enumerate(metric_data):
            row = tk.Frame(metrics_grid, bg="white")
            row.pack(fill=tk.X, pady=2)
            
            tk.Label(
                row,
                text=label,
                font=("Arial", 10),
                bg="white",
                width=15,
                anchor="w"
            ).pack(side=tk.LEFT)
            
            tk.Label(
                row,
                text=value,
                font=("Arial", 10, "bold"),
                bg="white",
                width=10,
                anchor="w"
            ).pack(side=tk.LEFT)
            
            if target:
                tk.Label(
                    row,
                    text=target,
                    font=("Arial", 9),
                    bg="white",
                    fg="gray"
                ).pack(side=tk.LEFT)
    
    def export_map(self):
        """Export current map as image and JSON"""
        if self.current_map is None:
            return
        
        try:
            import json
            import time
            
            # Create export directory
            os.makedirs("exports", exist_ok=True)
            
            # Get method name for filename
            method = self.method_var.get()
            method_prefix = {
                "hierarchical": "hierarchical",
                "hybrid": "hybrid",
                "perlin": "perlin"
            }.get(method, "map")
            
            # Find next available number for this method
            existing_files = [f for f in os.listdir("exports") if f.startswith(method_prefix) and f.endswith(".png")]
            next_num = 1
            if existing_files:
                # Extract numbers from existing files
                numbers = []
                for f in existing_files:
                    try:
                        # Extract number between method prefix and .png
                        num_part = f.replace(method_prefix + "_map_", "").replace(".png", "")
                        numbers.append(int(num_part))
                    except:
                        pass
                if numbers:
                    next_num = max(numbers) + 1
            
            # Generate filename base with method and number
            png_filename = f"exports/{method_prefix}_map_{next_num}.png"
            json_filename = f"exports/{method_prefix}_map_{next_num}.json"
            
            # Save map visualization (PNG)
            colors = {
                0: (34, 139, 34),
                1: (30, 144, 255),
                2: (238, 214, 175)
            }
            
            h, w = self.current_map.shape
            img_array = np.zeros((h, w, 3), dtype=np.uint8)
            
            for tile_type, color in colors.items():
                mask = (self.current_map == tile_type)
                img_array[mask] = color
            
            img = Image.fromarray(img_array, mode='RGB')
            img = img.resize((640, 640), Image.NEAREST)
            img.save(png_filename)
            
            # Save map data (JSON)
            map_data = {
                "width": int(w),
                "height": int(h),
                "tiles": self.current_map.tolist(),
                "metadata": {
                    "generation_method": self.method_var.get(),
                    "seed": int(self.seed_entry.get()),
                    "generation_time_seconds": getattr(self, 'current_generation_time', 0.0),
                    "generation_steps": getattr(self, 'current_steps', 0),
                    "tile_legend": {
                        "0": "LAND",
                        "1": "WATER",
                        "2": "SAND"
                    }
                }
            }
            
            with open(json_filename, 'w') as f:
                json.dump(map_data, f, indent=2)
            
            messagebox.showinfo(
                "Export Successful",
                f"Map saved to:\n{png_filename}\n{json_filename}"
            )
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    
    def play_map(self):
        """Launch the Game Demo with the current map"""
        if self.current_map is None:
            messagebox.showwarning("No Map", "Please generate a map first!")
            return
        
        try:
            import json
            
            # Create Game_demo/maps directory if it doesn't exist
            game_maps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Game_demo', 'maps')
            os.makedirs(game_maps_dir, exist_ok=True)
            
            # Save current map to game's maps folder
            current_map_path = os.path.join(game_maps_dir, 'current_map.json')
            
            h, w = self.current_map.shape
            map_data = {
                "width": int(w),
                "height": int(h),
                "tiles": self.current_map.tolist(),
                "metadata": {
                    "generation_method": self.method_var.get(),
                    "seed": int(self.seed_entry.get()),
                    "generation_time_seconds": getattr(self, 'current_generation_time', 0.0),
                    "generation_steps": getattr(self, 'current_steps', 0),
                    "tile_legend": {
                        "0": "LAND",
                        "1": "WATER",
                        "2": "SAND"
                    }
                }
            }
            
            with open(current_map_path, 'w') as f:
                json.dump(map_data, f, indent=2)
            
            # Launch the game
            game_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Game_demo', 'main.py')
            game_demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Game_demo')
            
            if not os.path.exists(game_main_path):
                messagebox.showerror("Game Not Found", f"Could not find Game Demo at:\n{game_main_path}")
                return
            
            # Launch game as subprocess with correct working directory and relative map path
            subprocess.Popen(
                [sys.executable, 'main.py', 'maps/current_map.json'],
                cwd=game_demo_dir
            )
            
            messagebox.showinfo(
                "Game Launched",
                "Game Demo is starting!\nYour generated map will load automatically."
            )
            
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch game:\n{str(e)}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    
    # Launch GUI
    root = tk.Tk()
    app = MapGeneratorGUI(root)
    root.mainloop()
