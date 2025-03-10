import torch.nn as nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import shutil
from avoid_gym import CrowdAvoidanceEnv

# CNN for LiDAR input
class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_filters=64):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim=512)

        # ✅ CNN layers to process 1D LiDAR data
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Reduce dimensionality
            nn.Flatten()
        )

        self.linear = nn.Linear(num_filters * 2, 512)  

    def forward(self, observations):
        lidar_data = observations[:, 7:]  # ✅Extract only LiDAR data
        latest_lidar_scan = lidar_data[:, -450:]  # Use only the most recent scan

        latest_lidar_scan = latest_lidar_scan.unsqueeze(1)  
        lidar_features = self.cnn(latest_lidar_scan)
        return self.linear(lidar_features)

if __name__ == "__main__":
    # Create multiple environments for parallel training
    num_envs = 8  
    envs = SubprocVecEnv([lambda: Monitor(CrowdAvoidanceEnv(use_gui=False)) for _ in range(num_envs)])

    num_eval_envs = 1  # Use one process for evaluation
    eval_env = SubprocVecEnv([lambda: Monitor(CrowdAvoidanceEnv(use_gui=False)) for _ in range(num_eval_envs)])

    # Ensure fresh logs before training
    log_dir = "./ppo_obstacle_logs/"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # Delete previous logs
    os.makedirs(log_dir, exist_ok=True)  # Create fresh log directory


    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./ppo_obstacle_models/",
        log_path="./logs/",
        eval_freq=250_000  # Evaluates every 250K steps
    )

    # ✅ Create PPO model with CNN Feature Extractor
    obstacle_avoidance_model = PPO(
        "CnnPolicy",
        envs,
        learning_rate=3e-4,
        batch_size=512,
        n_steps=2048,  # Longer sequences to improve learning
        ent_coef=0.05,  # More exploration to handle obstacles
        gamma=0.98,  # Short-term + long-term learning balance
        gae_lambda=0.98,
        clip_range=0.2,
        vf_coef=1.0,  # Helps PPO learn better value estimation
        policy_kwargs={
            "features_extractor_class": CNNFeatureExtractor,  # Use CNN for LiDAR
            "features_extractor_kwargs": {"num_filters": 64},
        },
        verbose=1,
        tensorboard_log="./ppo_obstacle_logs/",
        device="cuda"
    )
    

    print("Training begins...")

    # Train the obstacle avoidance model
    obstacle_avoidance_model.learn(total_timesteps= 10_000_000, callback=eval_callback, progress_bar=True  )

    # Save the trained model
    obstacle_avoidance_model.save("./ppo_obstacle_models/ppo_obstacle_avoidance_1")
    print("🎯 Training Complete! Model Saved.")
