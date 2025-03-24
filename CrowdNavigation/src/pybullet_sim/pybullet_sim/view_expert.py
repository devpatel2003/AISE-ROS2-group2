import pickle
import time
from IRL_maze_gym import CrowdAvoidanceEnv

# Path to the saved expert file
expert_path = "expert_trajectories/expert_001.pkl"

# Load expert data
with open(expert_path, "rb") as f:
    data = pickle.load(f)

trajectory = data["trajectory"]
grid_map = data["grid_map"]
start = data["start"]
goal = data["goal"]


# Create environment using preset map and positions
env = CrowdAvoidanceEnv(
    use_gui=True,
    preset_grid=grid_map,
    preset_start=start,
    preset_goal=goal
)

obs, _ = env.reset()

# Replay the expert trajectory
print("🎮 Replaying expert trajectory...")

for (obs, action) in trajectory:
    obs, reward, done, _, _ = env.step(action)
    time.sleep(1.0 / 10000.0)  # Slow down for visualization
    if done:
        print("✅ Replay complete (goal reached or collision).")
        break

env.close()