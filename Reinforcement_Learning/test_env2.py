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

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, self.width, self.height), dtype=np.uint8)

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


    def _get_obs(self):
        # Add channel dimension to observation
        observation = self.obs[None, :, :] * 255
        return observation.astype(np.uint8)
    
    def _get_info(self):
        return {}
    
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

        self.obs = self.input

        self.current_loss = 0

        # Convert self.obs to tensor to feed into the model
        transformer_input = torch.from_numpy(self.obs).to(self.device).float()
        transformer_input = transformer_input.unsqueeze(0)
        transformer_input = transformer_input.unsqueeze(0)

        with torch.no_grad():
            #Get prediction from model and found points
            output = self.unet(transformer_input)
        # Convert the model output to a probability map and binary mask
        self.current_loss = float(self.loss(output, transformer_input).cpu().detach())

        self.current_episode_reward = 0

        self.action = None

        # Must return observation and info
        return self._get_obs(), self._get_info()
    

    def step(self, action):
        previous_loss = self.current_loss

        self.action = action
        x, y= self._shoot_ray(action)
        self.x = x
        self.y = y

        self.current_rays += 1

        if (x is not None and y is not None):
            self.input[x][y] = 1     
            
            # Feed the input to the model
            transformer_input = torch.from_numpy(self.input).to(self.device).float()
            transformer_input = transformer_input.unsqueeze(0)
            transformer_input = transformer_input.unsqueeze(0)

            with torch.no_grad():
                output = self.unet(transformer_input)

            # Use probability map directly as observation, but cut away the batch dimension
            self.obs = output.cpu().detach().numpy().squeeze()

            # Calculate new loss
            new_loss = float(self.loss(output, transformer_input).cpu().detach())

            # Calculate reward as improvement in loss
            reward = max(0, (previous_loss - new_loss) / previous_loss)

            # Update current loss for next step
            self.current_loss = new_loss
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
                    "Current loss": float(self.current_loss),
                    "Episode Reward Mean": float(episode_rew_mean),
                    "Episode reward": float(self.current_episode_reward)
                })
        else:
            if self.wand:
                wandb.log({"Current loss": self.current_loss})

        return self._get_obs(), reward, done, False, self._get_info()
    
    def render(self):     
        # Convert tensors to numpy arrays and scale to 0-255 for visualization and  Duplicate channels for RGB display
        predict_rgb = self.obs * 255
        input_rgb = self.input * 255
        grount_truth_rgb = self.image * 255

        #For visualization
        if self.metadata["render_mode"] == 'human':            
            if ( self.action is not None):
                border, angle = self.action

                angle = (angle + 1) * 180 +180

                plotter_with_ray(input_rgb, predict_rgb, grount_truth_rgb, "Input", "Prediction", "Ground Truth", self._value_to_border_pixel(border), angle, (self.y, self.x), self.wand, self.current_rays)
                
            else:
                plotter(input_rgb, predict_rgb, grount_truth_rgb, "Input", "Prediction", "Ground Truth", self.wand, self.current_rays)
            
            print("Current loss :", self.current_loss)
            
        elif self.metadata["render_mode"] == 'rgb_array':
            # Ensure proper shape for video recording (height, width, channels)
            render_img = np.expand_dims(predict_rgb, axis=-1)  # Add channel dimension
            render_img = np.repeat(render_img, 3, axis=-1)  # Repeat to 3 channels
            render_img = render_img.astype(np.uint8)  # Convert to uint8
            return render_img  # Return format: (height, width, 3)
        
        
    def _shoot_ray(self, action):
        #Get two different actions, the angle is already econded in angle_action since we are taking 360 degrees
        border, angle = action

        x,y = self._value_to_border_pixel(border)

        angle_action = (angle + 1) * 180

        #Conver angle tso radiants
        angle_action = np.radians(angle_action)

        # Compute step direction for ray tracing 
        dx = np.cos(angle_action)
        dy = np.sin(angle_action)

         # Ray tracing loop
        while True:
            x += dx
            y += dy
        
            # Round x and y to nearest integer positions
            x_int = int(round(x))
            y_int = int(round(y))
        
            # Check if the ray goes out of bounds
            if x_int < 0 or x_int >= self.width or y_int < 0 or y_int >= self.height:
                return None, None
        
            # Check if the ray hits an obstacle (assumed to be represented by 1)
            elif self.image[x_int, y_int] == 1:
                return x_int, y_int
    
    def _value_to_border_pixel(self, border):
        border = (border + 1) / 2

         # Convert border value to pixel position
        total_perimeter = 2 * (self.width + self.height)
        position = border * total_perimeter

        # Top edge
        if position < self.width:
            x = position
            y = 0
        # Right edge
        elif position < self.width + self.height:
            x = self.width - 1
            y = position - self.width
        # Bottom edge
        elif position < 2 * self.width + self.height:
            x = 2 * self.width + self.height - position - 1
            y = self.height - 1
        # Left edge
        else:
            x = 0
            y = total_perimeter - position
        
        return x,y
    


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