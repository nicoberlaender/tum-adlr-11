import gymnasium as gym
import numpy as np
import wandb
import matplotlib.pyplot as plt
import sys
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from PIL import Image
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from types import SimpleNamespace
from torchvision import transforms
from torchvision.transforms import ToTensor

# Add the path to the 'model' folder (or wherever your 'my_unet_model.py' is located)
# Use the current working directory to get the model folder
sys.path.append(os.path.join(os.getcwd(), 'model'))
sys.path.append(os.path.join(os.getcwd(), 'dataset'))
sys.path.append(os.path.join(os.getcwd(), 'visualization'))
sys.path.append(os.path.join(os.getcwd(), 'loss'))
sys.path.append(os.path.join(os.getcwd(),'reinforcment_learning'))

# Import the UNet class
from model.my_unet_model import UNet
from visualization.view_result import visual3
from loss.losses import WeightedBCE, FocalLoss
from dataset.preprocessing import sample_pixels, segmap_to_binary, binary_to_image
from reinforcment_learning.enviroment import RayEnviroment , ActionNormWrapper, RunningRewardCallback
from reinforcment_learning.agent import Agent
from dataset.create_dataset_file import ImageDataset


# Specify the device
device = 'cpu'


# Load the model and map it to the GPU
model = torch.load("2D_Shape_Completion/model_full.pth", map_location=device)


# Set the model to evaluation mode
model.eval()

print("Model loaded onto", device)

# Step 3: Reload Dataset and DataLoader with the Updated Transform
dataset = ImageDataset('2D_Shape_Completion/data', num_samples=50, len_dataset=200, transform=transforms.Compose([ToTensor()]))


training_period=5
image_shape = (224,224)
env = RayEnviroment(image_shape,
    model=model,
    loss = torch.nn.BCELoss(), 
    max_number_rays = 15,
    dataset=dataset, 
    device=device, 
    render_mode='rgb_array',
    ) 
    
# Required for video recording)
env = RecordVideo(env, video_folder="video", name_prefix="training",
                  episode_trigger=lambda x: x % training_period == 0)
env = Monitor(env, filename= '1')
#env = ActionNormWrapper(env)
#env = RecordEpisodeStatistics(env)

agent = PPO("MlpPolicy", env, verbose=1)

# Create the callback to track the running average of rewards
callback = RunningRewardCallback(window_size=10)

agent.learn(total_timesteps=15000, callback=callback)

agent.save("PPO")

# Ottieni le ricompense cumulative per ciascun episodio
episode_rewards = env.get_episode_rewards()
print("Ricompense degli episodi:", episode_rewards)

# Ottieni le lunghezze degli episodi
episode_lengths = env.get_episode_lengths()
print("Lunghezza degli episodi:", episode_lengths)
