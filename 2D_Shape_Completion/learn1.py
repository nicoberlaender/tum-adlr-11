import gymnasium as gym
import numpy as np
import sys
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from PIL import Image
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from torchvision import transforms
from torchvision.transforms import ToTensor

# Add the path to the 'model' folder (or wherever your 'my_unet_model.py' is located)
# Use the current working directory to get the model folder
sys.path.append(os.path.join(os.getcwd(), 'dataset'))
sys.path.append(os.path.join(os.getcwd(),'reinforcment_learning'))


from stable_baselines3.common.callbacks import ProgressBarCallback
from reinforcment_learning.enviroment import RayEnviroment , ActionNormWrapper, RunningRewardCallback
from reinforcment_learning.agent import Agent
from dataset.create_dataset_file import ImageDataset


# Specify the device
device = 'cpu'


# Load the model and map it to the GPU
model = torch.load("2D_Shape_Completion/saved_models/model_full.pth", map_location=device)


# Set the model to evaluation mode
model.eval()

print("Model loaded onto", device)

# Step 3: Reload Dataset and DataLoader with the Updated Transform
dataset = ImageDataset('2D_Shape_Completion/data', num_samples=100, len_dataset=10, transform=transforms.Compose([ToTensor()]))
print(len(dataset))

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
                  episode_trigger=lambda x: x % training_period == 100)
env = Monitor(env, filename= '1')
#env = ActionNormWrapper(env)
#env = RecordEpisodeStatistics(env)

agent = PPO("MlpPolicy", env, verbose=1, device=device, n_steps=8, batch_size=64, n_epochs=1)

# Create the callback to track the running average of rewards
callback = RunningRewardCallback(window_size=10)

# Add a progress bar callback
progress_bar_callback = ProgressBarCallback()

# Combine your custom callback and the progress bar callback
combined_callbacks = [callback, progress_bar_callback]

# Train the agent with the progress bar
agent.learn(total_timesteps=100, callback=combined_callbacks, reset_num_timesteps=False)