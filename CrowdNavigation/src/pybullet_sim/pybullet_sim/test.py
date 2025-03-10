import gymnasium as gym
import pybullet as p
from stable_baselines3 import PPO
from crowd_gym_obs import CrowdAvoidanceEnv
import time

# Load trained model
model = PPO.load("./ppo_obstacle_models/best_model")

# Create environment with GUI
env = CrowdAvoidanceEnv(use_gui=True)

# Reset environment
obs, _ = env.reset()

# Run for 1000 steps (or until the episode ends)
for _ in range(1000000):
    env.render()  # Ensure PyBullet GUI is active
    action, _ = model.predict(obs)  # Get action from trained model
    obs, reward, done, _, _ = env.step(action)
    
    print(f"Reward: {reward:.3f}")  # ? Print reward to track behavior
    #print(f"Action: {action}")  # ? Print action to track behavior
    if done:  
        print("Restarting environment.")
        obs, _ = env.reset()

    #time.sleep(0.05)  # ? Slow down simulation for visibility

# ? Keep the environment open
print("?? Testing Complete! Close window manually.")
while True:
    time.sleep(1)

