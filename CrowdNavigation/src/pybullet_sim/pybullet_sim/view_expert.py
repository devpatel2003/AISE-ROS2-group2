import pickle
import time
from IRL_maze_gym import CrowdAvoidanceEnv

# Path to the saved expert file
expert_path = "expert_trajectories2/expert_003.pkl"

# Load expert data
with open(expert_path, "rb") as f:
    traj = pickle.load(f)

infos = traj.infos
# The last step:
final_info = infos[-1]
grid_map = final_info["grid_map"]
start = final_info["start"]
goal = final_info["goal"]

# Create environment using preset map and positions
env = CrowdAvoidanceEnv(use_gui=True,
                        preset_grid=grid_map,
                        preset_start=start,
                        preset_goal=goal)
obs = env.reset()


# Replay the expert trajectory
print("🎮 Replaying expert trajectory...")

for j in range(10000):
    for i, action in enumerate(traj.acts):
        obs, reward, done, info = env.step(action)
        #time.sleep(1/240)# maybe a short sleep
        if done :
            break
    env.reset()  

env.close()