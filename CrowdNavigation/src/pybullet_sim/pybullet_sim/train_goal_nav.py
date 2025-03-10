import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv #dont use with lstm
import torch.nn as nn
from crowd_gym_obs import CrowdAvoidanceEnv
import time  # Needed for rendering
import pybullet as p
import shutil
import os
import torch

torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
torch.backends.cudnn.enabled = True 
 
def make_env():
    return Monitor(CrowdAvoidanceEnv(use_gui=False))  # Disable GUI for faster training


if __name__ == "__main__":
    num_envs = 4
    env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])
    eval_env = make_env()

    TIMESTEPS = 5_000_000



    # Enable TensorBoard logging
    log_dir = "./ppo_logs/"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # Deletes previous logs
    os.makedirs(log_dir, exist_ok=True)  # Creates a fresh log directory
    logger = configure(log_dir, ["stdout", "tensorboard"])

    # Set up evaluation callback
    eval_callback = EvalCallback(eval_env, best_model_save_path="./ppo_models/",
                                log_path="./logs/", eval_freq= int(TIMESTEPS*0.1))

    class MLPFeatureExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space):
            super(MLPFeatureExtractor, self).__init__(observation_space, features_dim=512)  # MLP output size

            # Define MLP for goal-seeking
            self.mlp_fc = nn.Sequential(
                nn.Linear(7, 512),  
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU()
            )

        def forward(self, observations):
            non_lidar_features = observations[:, :7]  # Extract only first 7 inputs (goal + IMU)
            return self.mlp_fc(non_lidar_features)  # Pass through MLP

    # Define SAC model with proper parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,       # Learning rate (default: 3e-4)
        batch_size=512,           # Batch size for training     
        clip_range=0.2,
        ent_coef=0.02, 
        n_steps = 1024,# Train after every 5s
        policy_kwargs = {
            "net_arch": [],
            "features_extractor_class": MLPFeatureExtractor,  
        },
        verbose=1,                # Logging level (1 = info, 0 = silent)
        tensorboard_log="./ppo_logs/",
        #device = "cuda"
    )

    
    print("CUDA Available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")


    print("Model Created. Training begins...")


    #model = PPO.load("./ppo_models/ppo_goal_nav_2", env = env, ent_coef = 0.05) #retrain model
    model.set_logger(logger)
    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, progress_bar=True, reset_num_timesteps=True)

    # Save final trained model
    model.save("./ppo_models/ppo_goal_nav")
    print("Training Complete! Model Saved.")

    #1 just target
    #2 lidar added
    #3 cnn added