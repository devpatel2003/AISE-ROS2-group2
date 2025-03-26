import os
import pickle
import numpy as np
import torch as th
import torch.nn as nn

from imitation.algorithms.adversarial.airl import AIRL
from imitation.util.util import make_vec_env
from imitation.util.networks import RunningNorm 
from imitation.data.wrappers import RolloutInfoWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from IRL_maze_gym import CrowdAvoidanceEnv  # Your environment


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
        obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        acts = th.as_tensor(acts, dtype=th.float32, device=self.device)
        x = th.cat([obs, acts], dim=1)
        
        return self.network(x).squeeze(dim=1)


    def predict_processed(self, obs, acts, next_obs=None, dones=None):
        with th.no_grad():
            rew = self.forward(obs, acts)
        return rew.cpu().numpy()
    
    def preprocess(self, obs, acts, next_obs, dones):
        obs_th = th.tensor(obs, dtype=th.float32)
        acts_th = th.tensor(acts, dtype=th.float32)
        next_obs_th = th.tensor(next_obs, dtype=th.float32)
        dones_th = th.tensor(dones, dtype=th.float32)
        return obs_th, acts_th, next_obs_th, dones_th
    
    @property
    def device(self):
        return next(self.parameters()).device


# === Load expert trajectories ===
def load_trajectories(directory, limit=None):
    demos = []
    files = sorted(os.listdir(directory))[:limit]
    for fname in files:
        with open(os.path.join(directory, fname), "rb") as f:
            trajectory = pickle.load(f)  
            demos.append(trajectory)   
    return demos 


# === Wrap your env ===
def make_env():
    return CrowdAvoidanceEnv()

venv = DummyVecEnv([make_env])
venv = RolloutInfoWrapper(venv)

# === Load expert data ===
expert_demos = load_trajectories("expert_trajectories1", limit=100)

# === Set up AIRL ===
reward_net = RewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
)

airl_trainer = AIRL(
    demonstrations=expert_demos,
    demo_batch_size=32,
    n_disc_updates_per_round=4,
    gen_algo=PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
    ),
    reward_net=reward_net,
    venv=venv
)

# === Train AIRL ===
airl_trainer.train(total_timesteps=1_000_000)

# === Save policy and reward ===
airl_trainer.gen_algo.save("airl_models/airl_policy")
reward_net.save("airl_models/learned_reward.pt")
