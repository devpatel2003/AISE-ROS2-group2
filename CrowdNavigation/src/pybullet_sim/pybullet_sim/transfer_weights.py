import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv #dont use with lstm
#from stable_baselines3.common.envs import DummyVecEnv
import torch.nn as nn
from CrowdNavigation.src.pybullet_sim.pybullet_sim.goal_nav_gym import CrowdAvoidanceEnv
import time  # Needed for rendering
import pybullet as p
import shutil
import os
import torch

# Load the pretrained SAC model
pretrained_model_path = "./sac_models/sac_crowd_avoidance_1_100k_3090"  # Update with your model path
pretrained_sac = SAC.load(pretrained_model_path)

# Extract the MLP feature extractor
pretrained_mlp = pretrained_sac.actor.mu

# Custom Feature Extractor that uses only the first 7 elements of observations
class MLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(MLPFeatureExtractor, self).__init__(observation_space, features_dim=512)  # MLP output size

        # Wrap the extracted MLP from the pretrained policy
        self.mlp_fc = nn.Sequential(
            nn.Linear(7, 512),  # Match the first layer to the new input size
            nn.ReLU(),
            *list(pretrained_mlp.children())[1:]  # Copy remaining layers from pretrained model
        )

    def forward(self, observations):
        non_lidar_features = observations[:, :7]  # Extract only first 7 inputs (goal + IMU)
        return self.mlp_fc(non_lidar_features)  # Pass through extracted MLP

# Initialize new environment with expanded observation space
env = CrowdAvoidanceEnv(use_gui=False)  # Replace with your actual environment

# Define policy with the custom feature extractor
policy_kwargs = dict(
    features_extractor_class=MLPFeatureExtractor,
    features_extractor_kwargs={},  # No need to specify feature_dim explicitly
    net_arch=[]  # No extra layers after feature extraction
)

# Create a new SAC model using the custom feature extractor
new_model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)


# Save the updated model
new_model.save("sac_goal_nav_mlp")