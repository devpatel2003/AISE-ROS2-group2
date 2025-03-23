import gymnasium as gym
import pybullet as p
import pybullet_data
from static_gym import CrowdAvoidanceEnv
import time
import keyboard  # ✅ Used for key detection
import matplotlib.pyplot as plt
import numpy as np

# Define movement speeds
MAX_LINEAR_SPEED = 1  # m/s
MAX_ANGULAR_SPEED = 1  # rad/s
show_lidar = False

# Create environment with GUI
env = CrowdAvoidanceEnv(use_gui=True)

# Reset environment
obs, _ = env.reset()

print("\n🚗 **Manual Drive Mode (WASD Controls)** ")
print("[W] = Forward | [S] = Backward | [A] = Left | [D] = Right | [ESC] = Exit")

def plot_lidar_scan(scan):
    plt.clf()  # Clear previous frame
    angles = np.linspace(-np.pi, np.pi, len(scan))  # Angles from -pi to pi
    x = scan * np.cos(angles)  # Convert polar to cartesian
    y = scan * np.sin(angles)
    plt.scatter(x, y, s=2, color='red')  # Plot scan points
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Real-time LiDAR Scan")
    plt.pause(0.00001)  # Small pause to update plot

if show_lidar:
    plt.ion()  # Interactive plotting mode
    fig = plt.figure()  # Create figure for LiDAR plot



# Loop indefinitely (exit when `ESC` is pressed)
while True:
    env.render()  # Ensure PyBullet GUI is active

    

    # Default no movement
    linear_speed = 0.0
    angular_speed = 0.0

    # Read keyboard inputs
    if keyboard.is_pressed("up"):  # Forward
        linear_speed = MAX_LINEAR_SPEED
    if keyboard.is_pressed("down"):  # Backward
        linear_speed = -MAX_LINEAR_SPEED
    if keyboard.is_pressed("left"):  # Turn Left
        angular_speed = MAX_ANGULAR_SPEED
    if keyboard.is_pressed("right"):  # Turn Right
        angular_speed = -MAX_ANGULAR_SPEED
    if keyboard.is_pressed("f"): 
        linear_speed = 10
    if keyboard.is_pressed("g"):  # Turn Left
        angular_speed = 10

    # Exit simulation if `ESC` is pressed
    if keyboard.is_pressed("esc"):
        print("\n Exiting manual control mode.")
        break

    # Send action to environment
    action = [linear_speed, angular_speed]  # [v, w] format
    obs, reward, done, _, _ = env.step(action)

    lidar_scan = obs[7:79]

    if show_lidar:
        plot_lidar_scan(lidar_scan)  # Update plot

    # Print debug info (optional)
    print(f"Action: {action} | Reward: {reward:.3f}")


    # Reset environment if done
    if done:
        print("\n Episode ended (Goal reached or collision). Restarting...")
        obs, _ = env.reset()

    time.sleep(0.0000001)  # Control refresh rate (adjust for smoothness)

# Close the environment
env.close()