import gymnasium as gym
import pybullet as p
import pybullet_data
from race_track_enviornment import CrowdAvoidanceEnv
import time
import keyboard  # ✅ Used for key detection

# Define movement speeds
MAX_LINEAR_SPEED = 10  # m/s
MAX_ANGULAR_SPEED = 10  # rad/s

# Create environment with GUI
env = CrowdAvoidanceEnv(use_gui=True)

# Reset environment
obs, _ = env.reset()

print("\n **Manual Drive Mode (WASD Controls)** ")
print("[^] = Forward | [v] = Backward | [<] = Left | [>] = Right | [ESC] = Exit")

# Loop indefinitely (exit when `ESC` is pressed)
while True:
    env.render()  # Ensure PyBullet GUI is active

    # Default no movement
    linear_speed = 0.0
    angular_speed = 0.0

    # Read keyboard inputs
    if keyboard.is_pressed("up"):    # Forward
        linear_speed = MAX_LINEAR_SPEED
    if keyboard.is_pressed("down"):  # Backward
        linear_speed = -MAX_LINEAR_SPEED
    if keyboard.is_pressed("left"):  # Turn Left
        angular_speed = MAX_ANGULAR_SPEED
    if keyboard.is_pressed("right"): # Turn Right
        angular_speed = -MAX_ANGULAR_SPEED

    # Exit simulation if `ESC` is pressed
    if keyboard.is_pressed("esc"):
        print("\n Exiting manual control mode.")
        break

    # Send action to environment
    action = [linear_speed, angular_speed]  # [v, w] format
    obs, reward, done, _, _ = env.step(action)

    # Print debug info (optional)
    print(f"Action: {action} | Reward: {reward:.3f}")

    # Reset environment if done
    if done:
        print("\n Episode ended (Goal reached or collision). Restarting...")
        obs, _ = env.reset()

    time.sleep(0.05)  # Control refresh rate (adjust for smoothness)

# Close the environment
env.close()