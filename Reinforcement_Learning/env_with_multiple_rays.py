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

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,4), dtype=np.float32)

        self.metadata = {'render_mode': render_mode,
                          'render_fps': 30}
        
        self.render_mode = render_mode
        #Obseervations are the points so fare known
        self.observation_space = gym.spaces.MultiBinary([self.height, self.width])

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

        # Initialize variables for performance
        self.episode_rewards = []
        self.episode_rewards2 = []
        self.current_losses2 = []
        self.current_losses = []

        self.num_wandb_steps = 0

        self.total_num_steps = 0

        self.different_steps = 0

        self.num_resets = 0

    def _get_obs(self):
        return self.obs
    
    def _get_info(self):
        return {}
    
    def reset(self, seed=None, options= None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.num_resets += 1
        self.current_rays = 0
        
        # Load new image
        idx = np.random.randint(len(self.image_files))
        image_path = self.image_files[idx]

        #Initialize the image
        self.image = torchvision.io.read_image(image_path)
        self.image = self.image[0] > 0

        self.input = self.image > np.inf

        self.obs = self.input
        self.total_episode_reward = 0

        transformer_input = self.input.unsqueeze(0).to(self.device).float()
        transformer_input = transformer_input.unsqueeze(0)
        transformer_truth = self.image.unsqueeze(0).to(self.device).float()
        transformer_truth = transformer_truth.unsqueeze(0)  
        with torch.no_grad():
            #Get prediction from model and found points
            output = self.unet(transformer_input)
        # Convert the model output to a probability map and binary mask
        self.current_loss = float(self.loss(output, transformer_truth).cpu().detach())

        self.reward = 0

        self.action = None
        if self.wand:
            wandb.log({"Current loss": self.current_loss, 
                        }, step = self.total_num_steps)
        # Must return observation and info
        return self._get_obs(), self._get_info()

    def step(self, action):

        for i in range(4):
            self.total_num_steps += 1
            #If agent performs actions means sending ray on the image, finds a point hopefully and reuse alghorithm 
            self.action = action[:,i]
            x, y= self._shoot_ray(self.action)
            self.x = x
            self.y = y

            self.current_rays += 1

            if ( x is  not None and y is not None):
                
                self.input[x][y]= 1          
                transformer_input = self.input.unsqueeze(0).to(self.device).float()
                transformer_input = transformer_input.unsqueeze(0) 
                transformer_truth = self.image.unsqueeze(0).to(self.device).float()
                transformer_truth = transformer_truth.unsqueeze(0) 


                with torch.no_grad():
                    output = self.unet(transformer_input)

                # Convert the model output to a probability map and binary mask
                output_image = output[0][0].cpu()  
                self.obs = (output_image > 0.5)  # Thresholding to create a binary mask
                
                # Convert loss to CPU float
                self.current_loss = float(self.loss(output, transformer_truth).cpu().detach())
            self.current_episode_reward = -self.current_loss 

            self.total_episode_reward += self.current_episode_reward

            done = self.current_rays >= self.number_rays
            
            if done and self.wand:
                # Append float value to list
                self.current_losses.append(float(self.current_loss))
                loss_mean = np.mean(self.current_losses)
                wandb.log({
                    "Current loss": self.current_loss,
                    "Loss Mean": float(loss_mean),
                }, step = self.total_num_steps,
                )
            elif self.wand:
                wandb.log({
                    "Current loss": self.current_loss,
                    "Loss Mean": float(loss_mean),
                           }
                        , step=self.total_num_steps)
                
            if self.num_resets % 3000 == 0 and self.wand:
                predict_rgb =converter(self.obs) 
                input_rgb = converter(self.input) 
                ground_truth_rgb = converter(self.image) 
            
                if self.action is not None:
                    border, angle = self.action
                    angle = (angle + 1) * 180 + 180
                    plotter_with_ray(input_rgb, predict_rgb, ground_truth_rgb, 
                                "Input", "Prediction", "Ground Truth", 
                                self._value_to_border_pixel(border), angle, 
                                (self.x, self.y), self.wand, 
                                self.total_num_steps, -self.current_loss)
                else:
                    plotter(input_rgb, predict_rgb, ground_truth_rgb, 
                        "Input", "Prediction", "Ground Truth", 
                        self.wand, self.total_num_steps, -self.current_loss)
                    
        self.different_steps+=1
        self.reward = -self.current_loss + self.reward

        if self.wand:
            # Append float value to list
            self.episode_rewards2.append(float(self.reward))
            episode_rew_mean2 = np.mean(self.episode_rewards2)
            wandb.log({
                "Episode Reward Mean2": float(episode_rew_mean2),
                "Episode reward2": float(self.reward),
            }, step = self.different_steps,)
        return self._get_obs(), self.reward, done, False, self._get_info()
    
    def render(self):     
        # Convert tensors to numpy arrays and scale to 0-255 for visualization and  Duplicate channels for RGB display
        predict_rgb =converter(self.obs) 
        input_rgb = converter(self.input) 
        ground_truth_rgb = converter(self.image) 

        #For visualization
        if self.metadata["render_mode"] == 'human':
            self.num_wandb_steps += 1            
            if self.action is not None:
                border, angle = self.action
                angle = (angle + 1) * 180 + 180
                plotter_with_ray(input_rgb, predict_rgb, ground_truth_rgb, 
                            "Input", "Prediction", "Ground Truth", 
                            self._value_to_border_pixel(border), angle, 
                            (self.x, self.y), self.wand, 
                            self.num_wandb_steps, -self.current_loss)
            else:
                plotter(input_rgb, predict_rgb, ground_truth_rgb, 
                    "Input", "Prediction", "Ground Truth", 
                    self.wand, self.num_wandb_steps, -self.current_loss)
        
        print("Current loss:", self.current_loss)
            
        # Ensure correct shape (H,W,C) and add batch dimension (N,H,W,C)
        if len(predict_rgb.shape) == 2:
            predict_rgb = np.stack([predict_rgb] * 3, axis=-1)  # Add channels
        predict_rgb = np.expand_dims(predict_rgb, axis=0)  # Add batch dimension
        
        return predict_rgb
        
        
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