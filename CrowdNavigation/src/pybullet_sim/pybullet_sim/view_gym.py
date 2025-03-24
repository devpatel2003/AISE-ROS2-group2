import gymnasium as gym
from avoid_gym import CrowdAvoidanceEnv


def main():
    """Create and visualize the CrowdAvoidanceEnv."""
    env = CrowdAvoidanceEnv(use_gui=True)
    env.reset()
    
    while True:
        action = [1,0]  # Sample a random action
        observation, reward, done, _, _ = env.step(action)
        print("Observation shape:", observation.shape)
        
        
        if done:
            env.reset()

if __name__ == "__main__":
    main()