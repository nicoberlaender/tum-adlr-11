import os
import sys
import wandb
from stable_baselines3 import PPO
from cnn_environment import CNN_Environment
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import CallbackList

def main(observation_type):
    # Setup paths
    current_path = os.getcwd()

    data_path = os.path.join(current_path, "data", "train")

    env1 = CNN_Environment((224, 224), 15, data_path, render_mode='rgb_array', wand= True, observation_type=observation_type)

    # Create vectorized environment
    env = DummyVecEnv([lambda : env1])

    policy = "MultiInputPolicy"
    if observation_type == "heatmap":
        policy = "CnnPolicy"

    # Add a progress bar callback
    progress_bar_callback = ProgressBarCallback()

    # Configure training
    config = {
        "env": env,
        "total_timesteps": 200000,
        "policy": policy
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
        model_save_freq=50000,
        model_save_path=f"models/{run.id}",
        verbose=2
    )

    # Combine callbacks into a CallbackList
    callback_list = CallbackList([progress_bar_callback, callback])

    model.learn(total_timesteps=config["total_timesteps"], callback=callback_list)

    model.save("models/200kJaccard1e-3" + observation_type)

    env.close()

    run.finish()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ppo_cnn.py <observation_type>")
        sys.exit(1)
    main(sys.argv[1])