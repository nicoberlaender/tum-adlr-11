import wandb
from stable_baselines3 import PPO
from test_environment import TestEnvironment
from test_env2 import TestEnvironment2
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import CallbackList
import sys
import os

# Aggiungere il percorso al sys.path
sys.path.append(os.path.abspath('./Reinforcement_Learning'))

env = TestEnvironment2((224, 224), 15, "data_new/data_new", render_mode = 'rgb_array')

# Wrap the environment with DummyVecEnv (required by VecVideoRecorder)
vec_env = DummyVecEnv([lambda: env])


# Configure video recording
video_folder = "videos/"
video_length = 200  # Record the first 200 steps of each episode
vec_env = VecVideoRecorder(
    vec_env, 
    video_folder=video_folder, 
    record_video_trigger=lambda x: x % 100 == 0,  # Record every 10,000 steps
    video_length=video_length,
    name_prefix="ppo_agent"
)

# Add a progress bar callback
progress_bar_callback = ProgressBarCallback()

config = {
    "env": env,
    "total_timesteps" : 200000,
    "policy": "MlpPolicy"
}

run = wandb.init(project='ppo_hit_objects_new', config=config, sync_tensorboard=True, monitor_gym=True, save_code=True)

model = PPO(config["policy"], config["env"], learning_rate=1e-3, verbose=1, tensorboard_log=f"runs/{run.id}")

callback = WandbCallback(
    gradient_save_freq=1000,
    model_save_freq=100,
    model_save_path=f"models/{run.id}",
    verbose=2
)

# Combine callbacks into a CallbackList
callback_list = CallbackList([progress_bar_callback, callback])

model.learn(total_timesteps=config["total_timesteps"], callback=callback_list)
run.finish()
