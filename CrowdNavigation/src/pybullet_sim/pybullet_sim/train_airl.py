import os
import pickle
import numpy as np

from imitation.algorithms.adversarial import airl
from imitation.data.types import TrajectoryWithRew
from imitation.util.util import make_vec_env
from imitation.util.networks import RunningNorm
from imitation.data.wrappers import RolloutInfoWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from IRL_maze_gym import CrowdAvoidanceEnv  # Your environment


# === Load expert trajectories ===
def load_trajectories(directory, limit=None):
    demos = []
    files = sorted(os.listdir(directory))[:limit]
    for fname in files:
        with open(os.path.join(directory, fname), "rb") as f:
            data = pickle.load(f)
            trajectory = data["trajectory"]  # Assuming this is a list of TrajectoryWithRew objects
            
            # Access the obs and acts attributes directly
            obs = [traj.obs for traj in trajectory]
            acts = [traj.acts for traj in trajectory]
            
            demos.append(TrajectoryWithRew(
                obs=np.array(obs),
                acts=np.array(acts),
                rews=None,
                terminal=True
            ))

# === Wrap your env ===
def make_env():
    return CrowdAvoidanceEnv()

venv = DummyVecEnv([make_env])
venv = RolloutInfoWrapper(venv)

# === Load expert data ===
expert_demos = load_trajectories("expert_trajectories1", limit=100)

# === Set up AIRL ===
reward_net = airl.BasicRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    use_state_only=True
)

airl_trainer = airl.AIRL(
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
