import wandb
from stable_baselines3 import PPO
from test_environment import TestEnvironment
from wandb.integration.sb3 import WandbCallback

env = TestEnvironment((224, 224), 15, "../data_new")

config = {
    "env": env,
    "total_timesteps": 200000,
    "policy": "MlpPolicy"
}

run = wandb.init(project='ppo_hit_objects', config=config, sync_tensorboard=True, monitor_gym=True, save_code=True)

model = PPO(config["policy"], config["env"], learning_rate=1e-3, verbose=1, tensorboard_log=f"runs/{run.id}", device="cpu")

callback = WandbCallback(
    gradient_save_freq=1000,
    model_save_freq=10000,
    model_save_path=f"models/{run.id}",
    verbose=2
)

model.learn(total_timesteps=config["total_timesteps"], callback=callback)
run.finish()