import gymnasium as gym
import pybullet as p
import pybullet_data
from IRL_maze_gym import CrowdAvoidanceEnv
import time
import keyboard
import matplotlib.pyplot as plt
import numpy as np



import torch 
import torch.nn as nn

class RewardNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_size = observation_space.shape[0]
        act_size = action_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(obs_size + act_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, acts, next_obs=None, dones=None):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        x = torch.cat([obs, acts], dim=1)
        return self.network(x).squeeze(dim=1)

    def predict_processed(self, obs, acts, next_obs=None, dones=None):
        with torch.no_grad():
            rew = self.forward(obs, acts)
        return rew.cpu().numpy()

    @property
    def device(self):
        return next(self.parameters()).device

# === Load AIRL Model ===


# === Control Parameters ===
MAX_LINEAR_SPEED = 1
MAX_ANGULAR_SPEED = 1
show_lidar = False

env = CrowdAvoidanceEnv(use_gui=True)
# Instantiate reward net
reward_net = RewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space
)
reward_net.load_state_dict(torch.load("airl_models/learned_reward.pt"))
reward_net.eval()
obs = env.reset()

print("\n **Manual Drive Mode (WASD Controls)** ")
print("[W] = Forward | [S] = Backward | [A] = Left | [D] = Right | [ESC] = Exit")

def plot_lidar_scan(scan):
    plt.clf()
    angles = np.linspace(-np.pi, np.pi, len(scan))
    x = scan * np.cos(angles)
    y = scan * np.sin(angles)
    plt.scatter(x, y, s=2, color='red')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Real-time LiDAR Scan")
    plt.pause(0.00001)

if show_lidar:
    plt.ion()
    fig = plt.figure()

# === Main Loop ===
while True:
    

    linear_speed = 0.0
    angular_speed = 0.0

    if keyboard.is_pressed("up"):
        linear_speed = MAX_LINEAR_SPEED
    if keyboard.is_pressed("down"):
        linear_speed = -MAX_LINEAR_SPEED
    if keyboard.is_pressed("left"):
        angular_speed = MAX_ANGULAR_SPEED
    if keyboard.is_pressed("right"):
        angular_speed = -MAX_ANGULAR_SPEED
    if keyboard.is_pressed("f"):
        linear_speed = 10
    if keyboard.is_pressed("g"):
        angular_speed = 10
    if keyboard.is_pressed("esc"):
        print("\n Exiting manual control mode.")
        break

    action = [linear_speed, angular_speed]
    obs, reward, done, _, = env.step(action)

    lidar_scan = obs[7:79]

    if show_lidar:
        plot_lidar_scan(lidar_scan)

    # === Compute AIRL Reward ===
    obs_tensor = torch.tensor([obs], dtype=torch.float32)
    act_tensor = torch.tensor([action], dtype=torch.float32)

    irl_reward = reward_net.predict_processed(obs_tensor, act_tensor)[0]

    print(f"Action: {action} | Env Reward: {reward:.3f} | AIRL Reward: {irl_reward:.3f}")


    if done:
        print("\n Episode ended (Goal reached or collision). Restarting...")
        obs, _ = env.reset()

    time.sleep(0.0000001)

env.close()
