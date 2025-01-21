import wandb
from stable_baselines3 import PPO
from test_env2 import TestEnvironment2, VideoLoggingCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import CallbackList
import os

# Setup paths
current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
data_path = os.path.join(parent_path, "data_new")
video_folder = "videos/"

# Ensure video directory exists
os.makedirs(video_folder, exist_ok=True)

env1 = TestEnvironment2((224, 224), 15, data_path, render_mode='rgb_array', wand= True)
   

# Create vectorized environment
env = DummyVecEnv([lambda : env1])

# Define recording interval
RECORD_INTERVAL = 2048 * 10  # 20,480 steps

# Add video recording wrapper with fixed trigger
env = VecVideoRecorder(
    env,
    video_folder=video_folder,
    record_video_trigger=lambda x: (x % RECORD_INTERVAL) == 0,  # Add parentheses to fix precedence
    video_length=100,  
    name_prefix="ppo_agent"
)
# Add a progress bar callback
progress_bar_callback = ProgressBarCallback()

# Configure training
config = {
    "env": env,
    "total_timesteps": 300000,
    "policy": "CnnPolicy"
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
    ent_coef=0.1,
    batch_size=256,
    use_sde=True,
    sde_sample_freq=4,
    verbose=1,
    tensorboard_log=f"runs/{run.id}"
)

print("Run id is :", run.id)

callback = WandbCallback(
    gradient_save_freq=1000,
    model_save_freq=50000,
    model_save_path=f"models/{run.id}",
    verbose=2
)

video_logging_callback = VideoLoggingCallback(video_path=os.path.join(current_path, 'videos'), log_freq=2048*10)

# Combine callbacks into a CallbackList
callback_list = CallbackList([progress_bar_callback, callback, video_logging_callback ])

model.learn(total_timesteps=config["total_timesteps"], callback=callback_list)

env.close()

"""
env2 = TestEnvironment2((224, 224), 15, data_path, render_mode='human', wand = True)
env2 = DummyVecEnv([lambda: env2]) 
num_episodes = 5
counter = 0

print(f"Starting environment rendering for {num_episodes} episodes...")
obs= env2.reset()
env2.render()
while counter < num_episodes:
    # Get action from the model based on the current observation
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done,  info = env2.step(action)

    env2.render()  # Render the environment for visualization

    # Provide feedback to the user
    print(f"Episode {counter + 1}:")
    print(f"  Action Taken: {action}")
    print(f"  Reward Received: {reward}")
    print(f"  Done: {done}")

    if done :
        print(f"Episode {counter + 1} finished. Resetting environment...")
        counter += 1
        obs = env2.reset()
        env2.render()

print("All episodes completed. Exiting...")

env2.close()
"""

run.finish()