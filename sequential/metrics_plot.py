# metrics_plot.py
import pandas as pd
import matplotlib.pyplot as plt
import os

class MetricsLogger:
    def __init__(self):
        self.episodes = []
        self.total_rewards = []
        self.land_frac = []
        self.water_frac = []
        self.sand_frac = []
        self.largest_land_frac = []
        self.sand_quality = []

    def log(self, episode, grid, total_reward):
        from map_env import compute_map_metrics
        metrics = compute_map_metrics(grid)

        self.episodes.append(episode)
        self.total_rewards.append(total_reward)
        self.land_frac.append(metrics["land_frac"])
        self.water_frac.append(metrics["water_frac"])
        self.sand_frac.append(metrics["sand_frac"])
        self.largest_land_frac.append(metrics["largest_land_frac"])
        self.sand_quality.append(metrics["sand_quality"])
        
    def save_to_csv(self, filepath):
            """Save all metrics to CSV file for thesis analysis"""
            df = pd.DataFrame({
                'episode': self.episodes,
                'total_reward': self.total_rewards,
                'land_frac': self.land_frac,
                'water_frac': self.water_frac,
                'sand_frac': self.sand_frac,
                'largest_land_frac': self.largest_land_frac,
                'sand_quality': self.sand_quality
            })
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            df.to_csv(filepath, index=False)
            print(f"💾 Metrics saved to {filepath}")    
    def plot(self, show=True, save_path=None):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(self.episodes, self.total_rewards, label="Total Reward")
        plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.episodes, self.land_frac, label="Land")
        plt.plot(self.episodes, self.water_frac, label="Water")
        plt.plot(self.episodes, self.sand_frac, label="Sand")
        plt.xlabel("Episode"); plt.ylabel("Fraction"); plt.legend(); plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(self.episodes, self.largest_land_frac, label="Largest Land Fraction")
        plt.xlabel("Episode"); plt.ylabel("Fraction"); plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(self.episodes, self.sand_quality, label="Sand Quality")
        plt.xlabel("Episode"); plt.ylabel("Quality"); plt.grid(True)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
