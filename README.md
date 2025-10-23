# Rigel's Guide for The map Generation

## What is this?

This is a **smart map generator** that creates 2D game worlds automatically! It uses **Artificial Intelligence (AI)** to make realistic-looking islands with land, water, and beaches - just like in video games!

Think of it like a robot artist that draws maps for you. You just press a button, and it creates a whole new world! 

---

## Quick Start (Super Easy!)

### Step 1: Open the Map Generator
1. Find the file called `Map_Gen_ui.py`
2. Double-click it (or run it with Python)
3. A colorful window will pop up!

### Step 2: Generate Your Map
1. Choose how you want to make your map:
   - **RL-Based (3-Phase Hierarchical)** - The smartest AI (takes 8-20 steps)
   - **Hybrid (RL + Rule-Based)** - Mix of AI and rules (takes ~15 steps)
   - **Perlin Noise (Procedural)** - Super fast random terrain (1 step)

2. Pick a **Random Seed** number (any number you want!)
   - Same number = Same map every time
   - Different number = Different map
   - Click the Dice button to get a random number

3. Click the big green button: ** Generate Map**

4. Wait a few seconds... BOOM! Your map appears! 

### Step 3: Play Your Map!
1. Click the red button: ** Play This Map**
2. A game window opens with YOUR map!
3. Use **Arrow Keys** to walk around
4. Press **ESC** to quit the game

### Step 4: Save Your Map (Optional)
- Click **Export Map** to save it as a picture and data file
- Find it in the `exports/` folder

---

##  What Do the Colors Mean?

When you look at your map, you'll see three colors:

-  **Green = Land** - Walk here! Safe and solid ground
-  **Blue = Water** - Can't walk here! It's the ocean
-  **Sand/Beach = Yellow** - The border between land and water

---

##  What Are the Three Methods?

### RL-Based (Hierarchical) - The Smart AI
**What it is:** An AI that learned by playing 50,000+ practice games!

**How it works:**
- **Phase 1:** Makes big islands (macro terrain)
- **Phase 2:** Draws beaches around islands (shorelines)
- **Phase 3:** Adds tiny details (fine-tuning)

**Best for:** Realistic-looking maps with natural shapes
**Speed:** Medium (8-20 steps)

### 2 Hybrid (RL + Rule-Based) - The Balanced One
**What it is:** Mix of AI brain + automatic rules

**How it works:**
- AI decides where to put land and water
- Computer automatically adds beaches
- Faster than pure AI

**Best for:** Good quality maps, faster generation
**Speed:** Medium (15 steps)

###  Perlin Noise - The Speed Demon
**What it is:** Mathematical formula that makes random terrain

**How it works:**
- Uses fancy math (fractal noise)
- No AI thinking, just calculation
- Creates natural-looking randomness instantly

**Best for:** Quick testing or procedural generation
**Speed:** SUPER FAST (1 step, like 0.01 seconds!)

---

##  Understanding the Stats

After generating a map, you'll see numbers at the bottom:

### Terrain Distribution
- **Land:** How much green (usually 30-40%)
- **Water:** How much blue (usually 40-50%)
- **Sand:** How much yellow/beach (usually 10-20%)

### Quality Metrics (0-5 stars)
- **Connectivity:** Are the islands connected? Higher = better
- **Shore Smoothness:** Are beaches smooth? Higher = better
- **Balance:** Is land/water balanced? Higher = better
- **Overall Quality:** Average of everything

### Performance
- **Generation Time:** How long it took (seconds)
- **Steps:** How many AI decisions (or 1 for Perlin)

---

##  Tips & Tricks

###  Fun Things to Try:
1. **Use seed 42** - Try this famous number!
2. **Use seed 0, 1, 2, 3, 4** - These already have saved game maps
3. **Compare methods** - Generate with all 3 methods using the same seed
4. **Make a series** - Use seeds 100, 101, 102 to make similar maps

###  What Makes a Good Map?
- **30-40% land** - Not too much, not too little
- **Smooth beaches** - Natural-looking coastlines
- **Connected islands** - You can walk to most places
- **Balance** - Mix of exploration and open space

###  Speed vs Quality:
- **Need it fast?** → Use Perlin Noise
- **Want it pretty?** → Use RL-Based
- **Want both?** → Use Hybrid

---

## Project Structure (Where Everything Lives)

```
Map Generator/
├── Map_Gen_ui.py          ← Main program (double-click this!)
├── README.md              ← You are here!
│
├── hierarchical/          ← Smart AI #1
│   ├── map_env.py        (The AI's "world")
│   ├── train_agent.py    (The AI's "brain")
│   └── checkpoints/      (Saved AI memory)
│       └── best_model.pth
│
├── Hybrid/                ← Smart AI #2
│   ├── map_env.py
│   ├── train_agent.py
│   └── checkpoints/
│       └── best_model.pth
│
├── Game_demo/             ← The playable game!
│   ├── main.py           (Game engine)
│   ├── map_loader.py     (Loads your maps)
│   ├── autotile_logic.py (Makes tiles look good)
│   ├── tileset_renderer.py (Draws the graphics)
│   ├── maps/             (Where maps are saved for game)
│   │   └── current_map.json
│   └── assets/           (Pictures and graphics)
│
└── exports/               ← Your saved maps go here!
    ├── map_20251023_143022.png
    └── map_20251023_143022.json
```

---

##  How to Install (First Time Setup)

### You Need:
1. **Python 3.11** (or newer)
2. **Some Python packages** (libraries)

### Installation Steps:

#### Windows:
```powershell
# Install Python packages
pip install torch numpy pillow scipy pygame
```

#### Mac/Linux:
```bash
# Install Python packages
pip3 install torch numpy pillow scipy pygame
```

### Check if it worked:
```powershell
python Map_Gen_ui.py
```

If a window pops up, you're ready!

---


---

##  Troubleshooting (If Something Goes Wrong)

### Problem: "Module not found" error
**Solution:** Install the missing package
```powershell
pip install [package-name]
```

### Problem: Map looks weird or broken
**Solution:** Try a different seed number or generation method

### Problem: Game won't start
**Solution:** 
1. Make sure you generated a map first
2. Check if `Game_demo/main.py` exists
3. Try generating a new map and playing again

### Problem: It's too slow!
**Solution:** 
- Use **Perlin Noise** method - it's instant!
- Close other programs to free up computer power

### Problem: AI models not loading
**Solution:** 
- Check if `hierarchical/checkpoints/best_model.pth` exists
- Check if `Hybrid/checkpoints/best_model.pth` exists
- You might need to train the models first

---

## 🎮 Game Controls (When Playing Your Map)

| Key | What It Does |
|-----|--------------|
| **↑ ↓ ← →** | Move your character |
| **Arrow Keys** | Walk around the map |
| **ESC** | Quit the game |
| **F2** | Toggle minimap |
| **F3** | Change camera mode |


---

##  Technical Details (For Advanced Users)

### AI Architecture:
- **Type:** Deep Q-Network (DQN) with Dueling architecture
- **Input:** 5 channels (hierarchical) or 4 channels (hybrid)
- **Output:** Action values for tile placement decisions
- **Training:** 50,000+ episodes with epsilon-greedy exploration
- **Reward System:** 0-5 scale based on connectivity, smoothness, balance

### Map Format:
- **Size:** 64×64 tiles (4,096 total tiles)
- **Tile Values:** 0=Land, 1=Water, 2=Sand
- **Export Format:** JSON + PNG
- **Compatibility:** Works with Pygame-based game engine

### Performance:
- **Hierarchical RL:** 0.5-2 seconds (8-20 steps)
- **Hybrid RL:** 0.4-1 second (15 steps)
- **Perlin Noise:** 0.01-0.05 seconds (1 step)

---


This project was created for a thesis on **Reinforcement Learning for Procedural Content Generation**.

**Technologies Used:**
- Python 3.11
- PyTorch (Deep Learning)
- NumPy (Math)
- Pygame (Game Engine)
- Tkinter (User Interface)
- Pillow (Image Processing)

---

---

Remember: There are **no bad maps**, only different adventures! Every map you generate is unique and can be fun to explore. Try different seeds, compare methods, and see what amazing worlds the AI creates!

**Questions? Problems? Cool discoveries?** 
Just remember - even professional game developers started by playing around with tools like this! 

Happy Map Generating!

---

**Version:** 1.0  
**Last Updated:** October 23, 2025  
**Made with:** AI +  Passion for Games
