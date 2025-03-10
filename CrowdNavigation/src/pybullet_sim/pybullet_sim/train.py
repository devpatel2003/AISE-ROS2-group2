import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from crowd_gym_obs import CrowdAvoidanceEnv
import time  # Needed for rendering
import pybullet as p
import shutil
import os
import torch

torch.backends.cudnn.benchmark = True  # Optimize CUDA performance



class CustomCNNLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_filters=64, lstm_hidden_size=256, train_cnn=False, train_lstm=False):
        super(CustomCNNLSTM, self).__init__(observation_space, features_dim=512)

        # 1st: Define MLP for goal-seeking
        self.mlp_fc = nn.Sequential(
                nn.Linear(7, 512),  
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU()
        )

        # 2nd: CNN with BatchNorm for static obsticals 
        self.cnn = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # 3rd: LSTM for dynamic obstacle tracking
        self.lstm = nn.LSTM(input_size=num_filters * 2, hidden_size=lstm_hidden_size, batch_first=True)

        # "Freeze" CNN if `train_cnn=False`
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False  # Prevents CNN updates

        # "Freeze" LSTM if `train_lstm=False`
        if not train_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False  # Prevents LSTM updates

        # Final Linear layer (LSTM output + 7 extra input features)
        self.linear = nn.Linear(lstm_hidden_size + 7, 512)

    def forward(self, observations):

        lidar_data = observations[:, 7:].unsqueeze(1)  # Extract LiDAR features
        extra_features = observations[:, :7]  # Extract 7 scalar features
        mlp_features = self.mlp_fc(extra_features)  # Pass through MLP

        cnn_features = self.cnn(lidar_data)  # CNN processes LiDAR
        cnn_features = cnn_features.unsqueeze(1)  # Add time dimension for LSTM

        lstm_out, _ = self.lstm(cnn_features)  # Process with LSTM
        lstm_features = lstm_out[:, -1, :]  # Take the last hidden state

        # Combine LSTM output with extra input features
        combined_features = torch.cat((mlp_features ,lstm_features, extra_features), dim=1)

        return self.linear(combined_features).linear(combined_features)








# Create training environment with GUI
env = CrowdAvoidanceEnv()


# Enable TensorBoard logging
log_dir = "./sac_logs/"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)  # ✅ Deletes previous logs
os.makedirs(log_dir, exist_ok=True)  # ✅ Creates a fresh log directory
logger = configure(log_dir, ["stdout", "tensorboard"])

# Set up evaluation callback
eval_env = CrowdAvoidanceEnv()
eval_callback = EvalCallback(eval_env, best_model_save_path="./sac_models/",
                             log_path="./logs/", eval_freq=50_000)

# Load previous model
old_model_path = "./sac_models/sac_crowd_avoidance_1_100k_3090"  # Adjust path
old_model = SAC.load(old_model_path, print_system_info=True)

# Define SAC model with proper parameters
model = SAC(
    "MlpPolicy",
    env,
    ent_coef="auto_0.2",
    learning_rate=3e-4,       # Learning rate (default: 3e-4)
    batch_size=512,           # Batch size for training
    buffer_size=1_000_000,      # Replay buffer size
    tau=0.005,                # Target network update rate
    gamma=0.98,               # Discount factor
    train_freq=(100, "step"),# Train after every 5s
    gradient_steps=8,         # Number of gradient updates per step
    policy_kwargs = {
        "net_arch": [512, 512],  
        "features_extractor_class": CustomCNNLSTM,  
        "features_extractor_kwargs": {"num_filters": 64, "lstm_hidden_size": 256, "train_cnn": False, "train_lstm": False},  
    },
    verbose=1,                # Logging level (1 = info, 0 = silent)
    tensorboard_log="./sac_logs/",
    device = "cuda"
)


print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")


# Load the old model's state_dict
old_state_dict = old_model.policy.state_dict()
new_state_dict = model.policy.state_dict()

# Remove mismatched input layers
for key in old_state_dict:
    if any(sub in key for sub in ["latent_pi.0.weight", "latent_pi.0.bias",
                                  "qf0.0.weight", "qf0.0.bias",
                                  "qf1.0.weight", "qf1.0.bias",
                                  "critic_target.qf0.0.weight", "critic_target.qf0.0.bias"]):
        continue  # Skip mismatched layers
    if key in new_state_dict:
        new_state_dict[key] = old_state_dict[key]

# Load the modified state dict
#model.policy.load_state_dict(new_state_dict, strict=False)


#model = SAC.load("src/pybullet_sim/pybullet_sim/sac_models/sac_crowd_avoidance_1_50k", env=env)
# Attach logger to model
model.set_logger(logger)

print("Model Created. Training begins...")

# Train model with proper logging
TIMESTEPS = 5000000 #500 ts = 1s

model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, progress_bar=True, reset_num_timesteps=False)

# Save final trained model
model.save("sac_crowd_avoidance_2_5mil")
print("Training Complete! Model Saved.")

#1 just target
#2 lidar added
#3 cnn added