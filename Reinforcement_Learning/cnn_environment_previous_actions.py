import os
import torch
import numpy as np
import gymnasium as gym
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import os
import wandb
import sys
import math
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Reinforcement_Learning.utils import converter, plotter_with_ray, plotter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2D_Shape_Completion'))
sys.path.append(project_root)

class TestEnvironment2(gym.Env):
    def __init__(self, image_shape, number_rays, data_location, render_mode = 'rgb_array', wand = False):

        self.wand = wand

        self.shape = image_shape

        self.height, self.width = image_shape

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        #self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, self.width, self.height), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({
            'past_actions': gym.spaces.Box(low=-1, high=1, shape=(number_rays, 2), dtype=np.float32),
            'current_rays': gym.spaces.Discrete(number_rays)
        })

        self.metadata = {'render_mode': render_mode,
                          'render_fps': 30}
        
        self.render_mode = render_mode

        #Max number of rays the model can shoot
        self.number_rays = number_rays

        #Store path to data
        self.data_location = data_location
        
        #Load random image from data_location
        self.image_files = [os.path.join(self.data_location, f) for f in os.listdir(self.data_location) if f.endswith('.png')]

        #Model and loss to calculate reward
        # Define the device (GPU or mps or cpu)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        # Load the model and map it to the GPU
        self.unet = torch.load("saved_models/model_full_old.pth", map_location=self.device)
        self.unet.eval()

        self.loss = torch.nn.BCELoss()

        self.episode_rewards = []

        self.current_similarity = 0

        self.past_actions = [(0, 0)] * number_rays


    def _get_obs(self):
        # Add channel dimension to observation
        image_observation = self.obs[None, :, :] * 255
        image_observation = image_observation.astype(np.uint8)
        return {
            'past_actions': np.array(self.past_actions),
            'current_rays': self.current_rays
        }
        #return observation.astype(np.uint8)
    
    def _get_info(self):
        return {
            'current_similarity': self.current_similarity,
            'current_loss': self.current_loss,
        }
    
    def reset(self, seed=None, options= None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_rays = 0
        
        # Load new image
        idx = np.random.randint(len(self.image_files))
        image_path = self.image_files[idx]

        #Initialize the image
        self.image = torchvision.io.read_image(image_path)

        #Convert image to numpy array with two axis and 0-1 values
        self.image = self.image[0].numpy().astype(np.float32) / 255

        #Input should be black image
        self.input = np.zeros_like(self.image, dtype=np.float32)

        model_prediction = self.unet(torch.from_numpy(self.input).to(self.device).float().unsqueeze(0).unsqueeze(0))
        self.current_loss = self.loss(model_prediction, torch.from_numpy(self.image).to(self.device).float().unsqueeze(0).unsqueeze(0)).item()

        self.obs = self.input

        self.current_episode_reward = 0

        self.action = None

        self.current_similarity = 0

        self.past_actions = [(0, 0)] * self.number_rays

        # Must return observation and info
        return self._get_obs(), self._get_info()
    

    def step(self, action):
        self.action = action
        self.past_actions[self.current_rays] = action
        x, y= self._shoot_ray(action)
        self.x = x
        self.y = y

        self.current_rays += 1

        if (x is not None and y is not None):
            self.input[x][y] = 1     
            
            # Feed the input to the model
            transformer_input = torch.from_numpy(self.input).to(self.device).float()
            transformer_input = transformer_input.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                output = self.unet(transformer_input)

            # Use probability map directly as observation, but cut away the batch dimension
            self.obs = output.cpu().detach().numpy().squeeze()

            # Convert prediction to binary mask using threshold
            pred_mask = (output > 0.5).float()

            # convert to numpy array
            pred_mask = pred_mask.cpu().detach().numpy().squeeze()

            #calculate intersection between pred_mask and self.image
            intersection = (pred_mask * self.image).sum()
            union = pred_mask.sum() + self.image.sum() - intersection
            jaccard = intersection / (union + 1e-6)

            reward = jaccard
            self.current_similarity = reward

            model_prediction = self.unet(torch.from_numpy(self.input).to(self.device).float().unsqueeze(0).unsqueeze(0))
            self.current_loss = self.loss(model_prediction, torch.from_numpy(self.image).to(self.device).float().unsqueeze(0).unsqueeze(0)).item()
        else:
            # Penalize missing the object
            reward = 0

        self.current_episode_reward += reward

        done = self.current_rays >= self.number_rays

        if done:
            # Append float value to list
            self.episode_rewards.append(float(self.current_episode_reward))
            episode_rew_mean = np.mean(self.episode_rewards)
            if self.wand:
                wandb.log({
                    "Current similarity": float(self.current_similarity),
                    "Episode Reward Mean": float(episode_rew_mean),
                    "Episode reward": float(self.current_episode_reward)
                })
        else:
            if self.wand:
                wandb.log({
                    "Current similarity": self.current_similarity,
                    "Action_position": float(action[0]),
                    "Action_angle": float(action[1]),
                    "Action_magnitude": float(np.linalg.norm(action))
                })

        return self._get_obs(), reward, done, False, self._get_info()
    
    def render(self):     
        # Convert tensors to numpy arrays and scale to 0-255 for visualization and  Duplicate channels for RGB display
        predict_rgb = (self.obs >= 0.5) * 255
        input_rgb = self.input * 255
        grount_truth_rgb = self.image * 255

        #For visualization
        if self.metadata["render_mode"] == 'human':            
            if ( self.action is not None):
                position, angle = self.action

                plotter_with_ray(input_rgb, predict_rgb, grount_truth_rgb, "Input", "Prediction", "Ground Truth", self._value_to_circle_pixel(position), angle, (self.y, self.x), self.wand, self.current_rays)
                
            else:
                plotter(input_rgb, predict_rgb, grount_truth_rgb, "Input", "Prediction", "Ground Truth", self.wand, self.current_rays)
            
            print("Current similarity :", self.current_similarity)
            
        elif self.metadata["render_mode"] == 'rgb_array':
            # Ensure proper shape for video recording (height, width, channels)
            render_img = np.expand_dims(predict_rgb, axis=-1)  # Add channel dimension
            render_img = np.repeat(render_img, 3, axis=-1)  # Repeat to 3 channels
            render_img = render_img.astype(np.uint8)  # Convert to uint8
            return render_img  # Return format: (height, width, 3)
        
        
    import math

    def _shoot_ray(self, action):
        border, angle = action
        x_start, y_start = self._value_to_circle_pixel(border)

        # Calculate center and radial direction
        cx = (self.width - 1) / 2
        cy = (self.height - 1) / 2
        radial_x = cx - x_start
        radial_y = cy - y_start
        radial_length = math.hypot(radial_x, radial_y)

        if radial_length == 0:
            return None, None  # Edge case

        # Calculate maximum safe angle deviation
        half_width = (self.width - 1) / 2
        half_height = (self.height - 1) / 2
        radius = math.hypot(half_width, half_height)
        max_deviation = math.atan(max(half_width, half_height) / radius)

        # Map [-1, 1] to [-max_deviation, max_deviation]
        angle_dev = angle * max_deviation

        # Calculate direction vector with constrained angle
        cos_a = math.cos(angle_dev)
        sin_a = math.sin(angle_dev)
        dx = (radial_x * cos_a - radial_y * sin_a) / radial_length
        dy = (radial_x * sin_a + radial_y * cos_a) / radial_length

        # Tracing with guaranteed hit
        x, y = x_start, y_start
        for _ in range(int(2 * radius * 2)):  # Sufficient steps to cross image
            x += dx
            y += dy
            x_int = int(round(x))
            y_int = int(round(y))

            if 0 <= x_int < self.width and 0 <= y_int < self.height:
                if self.image[x_int, y_int] == 1:
                    return x_int, y_int
    
        return None, None  # Fallback (shouldn't reach here)
    
    def _value_to_circle_pixel(self, position):
        # Convert action from [-1, 1] to angle in [0, 2π)
        theta = (position + 1) * math.pi  # Scales to 0-2π

        # Calculate image center coordinates
        cx = (self.width - 1) / 2  # Center x-coordinate
        cy = (self.height - 1) / 2  # Center y-coordinate

        # Calculate radius to image corners
        half_width = (self.width - 1) / 2
        half_height = (self.height - 1) / 2
        radius = math.sqrt(half_width**2 + half_height**2)

        # Convert polar coordinates to Cartesian coordinates
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)

        return x, y
    


from stable_baselines3.common.callbacks import BaseCallback
import glob

class VideoLoggingCallback(BaseCallback):
    def __init__(self, video_path, log_freq=2048, verbose=0):
        super().__init__(verbose)
        self.video_path = video_path
        self.log_freq = log_freq
        self.logged_videos = set()

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            # Get a list of all video files in the folder
            video_files = glob.glob(os.path.join(self.video_path, "*.mp4"))
            
            # Log only new videos
            new_videos = [v for v in video_files if v not in self.logged_videos]
            for video in new_videos:
                wandb.log({"video": wandb.Video(video, fps=30, format="mp4")}, step=self.n_calls)
                self.logged_videos.add(video)

        return True