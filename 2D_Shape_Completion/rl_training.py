import gymnasium as gym
import numpy as np
import wandb
import matplotlib.pyplot as plt
import sys
import os
import torch

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
from reinforcment_learning.enviroment import RayEnviroment
from reinforcment_learning.agent import Agent
from dataset.create_dataset_file import ImageDataset


# Specify the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'

# Load the model and map it to the GPU
model = torch.load("model_full.pth", map_location=device, weights_only=True)

# Set the model to evaluation mode
model.eval()

print("Model loaded onto", device)


# hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

# Define hyperparameters
hyperparameters = {
    # hyperparameters
    "learning_rate": 0.01,
    "n_episodes": 100_000,
    "start_epsilon": 1.0,
    "epsilon_decay": 1.0 / (100_000 / 2),  # reduce the exploration over time
    "final_epsilon": 0.1,
}

# Control whether to use wandb
use_wandb = True

# Step 3: Reload Dataset and DataLoader with the Updated Transform
dataset = ImageDataset('data', num_samples=400, len_dataset=2500, transform=transforms.Compose([ToTensor()]))

if use_wandb:
    # Initialize wandb with project configuration
    wandb.init(project='unet-training', name='shape_compleetion', config=hyperparameters)
    config = wandb.config  # Directly use wandb.config
else:
    # Create a SimpleNamespace for offline configuration
    config = SimpleNamespace(**hyperparameters)

training_period=200
image_shape = (224,224)
env = RayEnviroment(image_shape, model=model, loss = torch.nn.BCELoss(), max_number_rays = 15)
env = RecordVideo(env, video_folder="video", name_prefix="training",
                  episode_trigger=lambda x: x % training_period == 0)
env = RecordEpisodeStatistics(env)

agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

for episode in range(n_episodes):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    wandb.log({"episode": episode, "reward": info["episode"]["r"], "length": info["episode"]["l"]})


    agent.decay_epsilon()