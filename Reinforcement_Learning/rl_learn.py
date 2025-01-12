import gymnasium as gym
import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt


from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from PIL import Image
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from torchvision import transforms
from torchvision.transforms import ToTensor
from stable_baselines3.common.logger import configure

# Add the path to the 'model' folder (or wherever your 'my_unet_model.py' is located)
# Use the current working directory to get the model folder
sys.path.append(os.path.join(os.getcwd(), 'dataset'))
sys.path.append(os.path.join(os.getcwd(),'reinforcment_learning'))


from stable_baselines3.common.callbacks import ProgressBarCallback
from enviroment import RayEnviroment  
from dataset.create_dataset_file import ImageDataset


# Specify the device
device = 'cpu'


# Load the model and map it to the GPU
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
training_period = 10000

env = Monitor(env, filename= '1')
#env = ActionNormWrapper(env)
#env = RecordEpisodeStatistics(env)

# Directory for TensorBoard logs and loss file
log_dir = './ppo_tensorboard_logs'

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

agent = PPO("MlpPolicy", env, verbose=1, device=device, n_steps=32, batch_size=32, n_epochs=1, tensorboard_log=log_dir)



# Add a progress bar callback
progress_bar_callback = ProgressBarCallback()



# Parametri di training
total_timesteps = 128000
training_period = 1028  # ogni 100 episodi
total_len_episode_lengths=0
len_ep_lengths = []
len_ep_lengths.append(0)

# Set new logger
agent.set_logger(new_logger)

# L'iteratore per l'addestramento
for step in range(0, total_timesteps, training_period):
    # Inizia l'addestramento
    agent.learn(total_timesteps=training_period, callback=progress_bar_callback, reset_num_timesteps = False,)

    print(f"Addestramento in corso: Iterazione {step}/{total_timesteps}")

    # Salva il modello
    agent.save(f"PPO")  # Salva con il passo corrente per evitare sovrascrittura

    # Supponiamo che tu stia ottenendo le ricompense cumulative per ogni episodio
    episode_rewards = env.get_episode_rewards()

    # Calcolare e visualizzare la media mobile
    window_size = 100  # Imposta la finestra della media mobile
    moving_avg_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')

    plt.plot(moving_avg_rewards)
    plt.title(f"Reward con Media Mobile (Iterazione {step})")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.savefig(f"moving_avg_rewards.png")  
    plt.close()  # Chiudi la figura corrente

    # Ottieni le lunghezze degli episodi
    episode_lengths = env.get_episode_lengths()
 
    print("Lunghezza degli episodi:", episode_lengths[total_len_episode_lengths:])
    print("Lunghezza totale sommata", np.sum(episode_lengths[total_len_episode_lengths:]))

    len_ep_lengths.append(len(episode_lengths) - total_len_episode_lengths)

    total_len_episode_lengths = len(episode_lengths)

    # Create a bar chart where each column represents an individual element in len_ep_lengths
    plt.figure(figsize=(10, 6))

    # Create the bars for each value in len_ep_lengths
    plt.bar(range(len(len_ep_lengths)), len_ep_lengths, color='blue', edgecolor='black')

    # Adding title and labels
    plt.title('Episode Length Differences (Bar Chart)')
    plt.xlabel('Episode Index')
    plt.ylabel('Difference in Episode Length')

    # Save the bar chart as an image file
    plt.savefig('episode_length_bar_chart.png')