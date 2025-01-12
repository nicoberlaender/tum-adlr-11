import wandb
import torch
from stable_baselines3 import PPO
from test_environment import TestEnvironment
from enviroment import RayEnviroment
from wandb.integration.sb3 import WandbCallback
from dataset.create_dataset_file import ImageDataset
from torchvision.transforms import ToTensor
from torchvision import transforms
import os
import sys

# Add the 2D_Shape_Completion directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2D_Shape_Completion'))
sys.path.append(project_root)
#env = TestEnvironment((224, 224), 15, "./data_new/data_new")
# Define the device (GPU or mps or cpu)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
# Load the model and map it to the GPU
Unet = torch.load("Reinforcement_Learning/saved_models/model_full.pth", map_location=device)

env = RayEnviroment((224,224),Unet, 15,"data_new/data_new", device )
config = {
    "env": env,
    "total_timesteps": 200000,
    "policy": "MlpPolicy"
}

run = wandb.init(project='ppo_hit_objects', config=config, sync_tensorboard=True, monitor_gym=True, save_code=True)

model = PPO(config["policy"], config["env"], learning_rate=1e-3, verbose=1, tensorboard_log=f"runs/{run.id}")


# Change the model save path to a directory where you have write permissions
model_save_path = os.path.join(os.path.expanduser("~"), "models", run.id)

# Ensure the directory exists
os.makedirs(model_save_path, exist_ok=True)

callback = WandbCallback(
    gradient_save_freq=1000,
    model_save_freq=10000,
    model_save_path=model_save_path,
    verbose=2
)

model.learn(total_timesteps=config["total_timesteps"], callback=callback)
run.finish()