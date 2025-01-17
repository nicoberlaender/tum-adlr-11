import wandb
from stable_baselines3 import PPO
from test_env2 import TestEnvironment2, VideoLoggingCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import CallbackList
import sys
import os




# Setup paths
current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
data_path = os.path.join(parent_path, "data_new/data_new")
video_folder = "videos/"

# Ensure video directory exists
os.makedirs(video_folder, exist_ok=True)

env1 = TestEnvironment2((224, 224), 15, data_path, render_mode='rgb_array')
   

# Create vectorized environment
env = DummyVecEnv([lambda : env1])

# Add video recording wrapper
env = VecVideoRecorder(
    env,
    video_folder=video_folder,
    record_video_trigger=lambda x: x % 2048 * 10== 0,  # Record every 2048 steps
    video_length=100,  # Record 300 frames per video
    name_prefix="ppo_agent"
)

# Add a progress bar callback
progress_bar_callback = ProgressBarCallback()

# Configure training
config = {
    "env": env,
    "total_timesteps": 400000,
    "policy": "MlpPolicy"
}

# Initialize wandb
run = wandb.init(
    project='ppo_hit_objects_new',
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True
)


# Create model
model = PPO(
    config["policy"],
    config["env"],
    learning_rate=1e-3,
    verbose=1,
    tensorboard_log=f"runs/{run.id}"
)

print("Run id is :", run.id)

callback = WandbCallback(
    gradient_save_freq=1000,
    model_save_freq=100,
    model_save_path=f"models/{run.id}",
    verbose=2
)

video_logging_callback = VideoLoggingCallback(video_path=os.path.join(current_path, 'videos'), log_freq=2048*10)

# Combine callbacks into a CallbackList
callback_list = CallbackList([progress_bar_callback, callback,video_logging_callback ])

model.learn(total_timesteps=config["total_timesteps"], callback=callback_list)
run.finish()
env.close()

