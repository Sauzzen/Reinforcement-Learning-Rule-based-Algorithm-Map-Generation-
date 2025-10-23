"""
Map Generator GUI - Using MapGenerator class
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageTk
import os

class MapGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Game Map Generator - Thesis Demo")
        self.root.geometry("1000x900")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.generator = None
        self.current_map = None
        
        # Create UI
        self.create_widgets()
        
        # Try to load MapGenerator
        self.load_generator()
    
    def create_widgets(self):
        """Create all UI components"""
        
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🗺️ 2D Game Map Generator",
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
        
        # Seed input
        tk.Label(
            control_panel,
            text="Random Seed:",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(anchor="w", pady=(5, 5))
        
        seed_frame = tk.Frame(control_panel, bg="white")
        seed_frame.pack(fill=tk.X, pady=5)
        
        self.seed_entry = tk.Entry(seed_frame, font=("Arial", 12), width=15)
        self.seed_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.seed_entry.insert(0, str(np.random.randint(0, 100000)))
        
        tk.Button(
            seed_frame,
            text="🎲",
            font=("Arial", 10),
            command=self.randomize_seed,
            width=3
        ).pack(side=tk.LEFT)
        
        tk.Frame(control_panel, height=20, bg="white").pack()
        
        # Polish checkbox
        self.polish_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            control_panel,
            text="Apply rule-based polish",
            variable=self.polish_var,
            font=("Arial", 11),
            bg="white"
        ).pack(anchor="w", pady=5)
        
        tk.Frame(control_panel, height=20, bg="white").pack()
        
        # Generate button
        self.generate_btn = tk.Button(
            control_panel,
            text="🚀 Generate Map",
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
        
        # Export buttons
        export_frame = tk.Frame(control_panel, bg="white")
        export_frame.pack(fill=tk.X, pady=5)
        
        self.export_json_btn = tk.Button(
            export_frame,
            text="💾 JSON",
            font=("Arial", 10),
            bg="#3498db",
            fg="white",
            command=self.export_json,
            state=tk.DISABLED,
            width=8
        )
        self.export_json_btn.pack(side=tk.LEFT, padx=2)
        
        self.export_png_btn = tk.Button(
            export_frame,
            text="🖼️ PNG",
            font=("Arial", 10),
            bg="#3498db",
            fg="white",
            command=self.export_png,
            state=tk.DISABLED,
            width=8
        )
        self.export_png_btn.pack(side=tk.LEFT, padx=2)
        
        tk.Frame(control_panel, height=20, bg="white").pack()
        
        # Status
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
            text="Generated Map",
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
        
        tk.Label(
            self.metrics_container,
            text="No metrics available yet",
            font=("Arial", 11),
            bg="white",
            fg="gray"
        ).pack()
    
    def load_generator(self):
        """Load MapGenerator from generate_map.py"""
        try:
            from generate_map import MapGenerator
            
            model_path = "checkpoints/best_model.pth"
            config_path = "checkpoints/config.json"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model checkpoint not found")
            
            # Initialize generator
            self.generator = MapGenerator(
                model_path=model_path,
                config_path=config_path if os.path.exists(config_path) else None
            )
            
            self.status_label.config(
                text="✓ Model loaded\nReady to generate",
                fg="#27ae60"
            )
            print("✓ MapGenerator loaded successfully")
            
        except Exception as e:
            print(f"⚠ Could not load MapGenerator: {e}")
            self.generator = None
            self.status_label.config(
                text="⚠ Model not loaded\nUsing fallback generation",
                fg="#e67e22"
            )
    
    def randomize_seed(self):
        """Generate random seed"""
        self.seed_entry.delete(0, tk.END)
        self.seed_entry.insert(0, str(np.random.randint(0, 100000)))
    
    def generate_map(self):
        """Generate a map"""
        self.status_label.config(text="⏳ Generating...", fg="#3498db")
        self.generate_btn.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            seed = int(self.seed_entry.get())
            apply_polish = self.polish_var.get()
            
            if self.generator:
                # Use trained model
                world, metrics = self.generator.generate_map(
                    seed=seed,
                    apply_polish=apply_polish,
                    deterministic=True,
                    verbose=True
                )
            else:
                # Fallback to simple rule-based
                world, metrics = self.generate_fallback(seed)
            
            # Store
            self.current_map = world
            
            # Display
            self.display_map(world)
            self.display_metrics(metrics, seed)
            
            # Enable exports
            self.export_json_btn.config(state=tk.NORMAL)
            self.export_png_btn.config(state=tk.NORMAL)
            
            self.status_label.config(text="✓ Complete!", fg="#27ae60")
            
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))
            self.status_label.config(text="✗ Error", fg="#e74c3c")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.generate_btn.config(state=tk.NORMAL)
    
    def generate_fallback(self, seed):
        """Simple fallback generation"""
        from rulebased_maps import rule_based_generate
        
        world = rule_based_generate(64, 64, water_target=0.40, seed=seed)
        
        metrics = {
            'land_frac': float(np.mean(world == 0)),
            'water_frac': float(np.mean(world == 1)),
            'sand_frac': float(np.mean(world == 2)),
            'seed': seed,
            'method': 'fallback_rulebased'
        }
        
        return world, metrics
    
    def display_map(self, world):
        """Display map on canvas"""
        colors = {
            0: (34, 139, 34),
            1: (30, 144, 255),
            2: (238, 214, 175)
        }
        
        h, w = world.shape
        img_array = np.zeros((h, w, 3), dtype=np.uint8)
        
        for tile_type, color in colors.items():
            mask = (world == tile_type)
            img_array[mask] = color
        
        img = Image.fromarray(img_array, mode='RGB')
        img = img.resize((640, 640), Image.NEAREST)
        
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(320, 320, image=self.photo)
    
    def display_metrics(self, metrics, seed):
        """Display metrics"""
        for widget in self.metrics_container.winfo_children():
            widget.destroy()
        
        # Header
        tk.Label(
            self.metrics_container,
            text=f"Seed: {seed}",
            font=("Arial", 11, "bold"),
            bg="white"
        ).pack(pady=5)
        
        # Metrics
        metric_data = [
            ("Water:", f"{metrics.get('water_frac', 0)*100:.1f}%"),
            ("Land:", f"{metrics.get('land_frac', 0)*100:.1f}%"),
            ("Sand:", f"{metrics.get('sand_frac', 0)*100:.1f}%"),
            ("Steps:", str(metrics.get('steps', 'N/A')))
        ]
        
        for label, value in metric_data:
            row = tk.Frame(self.metrics_container, bg="white")
            row.pack(fill=tk.X, pady=2)
            
            tk.Label(
                row,
                text=label,
                font=("Arial", 10),
                bg="white",
                width=10,
                anchor="w"
            ).pack(side=tk.LEFT)
            
            tk.Label(
                row,
                text=value,
                font=("Arial", 10, "bold"),
                bg="white"
            ).pack(side=tk.LEFT)
    
    def export_json(self):
        """Export as JSON using MapGenerator"""
        if self.current_map is None or self.generator is None:
            messagebox.showwarning("Export", "No map to export or generator not available")
            return
        
        try:
            os.makedirs("exports", exist_ok=True)
            seed = int(self.seed_entry.get())
            filename = f"exports/map_{seed}.json"
            
            self.generator.export_json(
                self.current_map,
                filename,
                metadata={'seed': seed}
            )
            
            messagebox.showinfo("Export", f"Saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    
    def export_png(self):
        """Export as PNG using MapGenerator"""
        if self.current_map is None or self.generator is None:
            messagebox.showwarning("Export", "No map to export or generator not available")
            return
        
        try:
            os.makedirs("exports", exist_ok=True)
            seed = int(self.seed_entry.get())
            filename = f"exports/map_{seed}.png"
            
            self.generator.export_png(self.current_map, filename)
            
            messagebox.showinfo("Export", f"Saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    
    root = tk.Tk()
    app = MapGeneratorGUI(root)
    root.mainloop()
