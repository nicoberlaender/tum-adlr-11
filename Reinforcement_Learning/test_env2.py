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

        self.metadata = {'render_mode': render_mode,
                          'render_fps': 30}
        
        self.render_mode = render_mode
        #Obseervations are the points so fare known
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, self.width, self.height), dtype=np.uint8)

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
        self.current_similarity = 0.0

        self.episode_rewards = []
        self.current_losses = []

        self.num_wandb_steps = 0

        self.total_num_steps = 0

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

        #Convert image to numpy array with two axis and 0-1 values
        self.image = self.image[0].numpy().astype(np.float32) / 255

        #Input should be black image
        self.input = np.zeros_like(self.image, dtype=np.float32)

        self.obs = self.input
        self.total_reward = 0
        self.current_loss = 0

        transformer_input = torch.tensor(self.input).unsqueeze(0).to(self.device).float()
        transformer_input = transformer_input.unsqueeze(0)
        transformer_truth = torch.tensor(self.image).unsqueeze(0).to(self.device).float()
        transformer_truth = transformer_truth.unsqueeze(0)  
        with torch.no_grad():
            #Get prediction from model and found points
            output = self.unet(transformer_input)
        # Convert the model output to a probability map and binary mask
        self.current_loss = float(self.loss(output, transformer_truth).cpu().detach())

        self.current_episode_reward = 0

        self.action = None
        if self.wand:
            wandb.log({"Current loss": self.current_loss, 
                        }, step = self.total_num_steps)
        # Must return observation and info
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.total_num_steps += 1
        #If agent performs actions means sending ray on the image, finds a point hopefully and reuse alghorithm 
        self.action = action
        x, y= self._shoot_ray(action)
        self.x = x
        self.y = y

        self.current_rays += 1

        if ( x is  not None and y is not None):
            
            self.input[x][y]= 1          
            transformer_input = torch.tensor(self.input).unsqueeze(0).to(self.device).float()
            transformer_input = transformer_input.unsqueeze(0) 

            transformer_truth = torch.tensor(self.image).unsqueeze(0).to(self.device).float()
            transformer_truth = transformer_truth.unsqueeze(0)  

            with torch.no_grad():
                output = self.unet(transformer_input)

            # Use probability map directly as observation, but cut away the batch dimension
            self.obs = output.cpu().detach().numpy().squeeze()
            # Convert the model output to a probability map and binary mask
            pred_mask = (output > 0.5).float()
            pred_mask = pred_mask.cpu().detach().numpy().squeeze()
            
            # Convert loss to CPU float
            # Convert the model output to a probability map and binary mask
            self.current_loss = float(self.loss(output, transformer_truth).cpu().detach())

            #calculate intersection between pred_mask and self.image
            intersection = (pred_mask * self.image).sum()
            union = pred_mask.sum() + self.image.sum() - intersection
            jaccard = intersection / (union + 1e-6)


            self.current_similarity = jaccard

        self.current_episode_reward = -self.current_loss 
        self.total_reward += self.current_episode_reward

        done = self.current_rays >= self.number_rays
        
        if self.wand:
            # Append float value to list
            self.episode_rewards.append(float(self.current_episode_reward))
            self.current_losses.append(float(self.current_loss))
            episode_rew_mean = np.mean(self.episode_rewards)
            loss_mean = np.mean(self.current_losses)
            wandb.log({
                "Current loss": float(self.current_loss),
                "Episode Reward Mean": float(episode_rew_mean),
                "Episode reward": float(self.current_episode_reward),
                "Loss Mean": float(loss_mean),              

                    "Current similarity": self.current_similarity,
                    "Action_border": float(action[0]),
                    "Action_angle": float(action[1]),
                    "Action_magnitude": float(np.linalg.norm(action))
                }, step = self.total_num_steps,)

            

        if done:
            wandb.log({"Total Reward": self.total_reward}, step = self.total_num_steps)
            
        if self.num_resets % 100 == 0 and self.wand:
            predict_rgb =self.obs
            input_rgb = np.stack([self.input * 255] * 3, axis=-1)
            ground_truth_rgb = np.stack([self.image * 255] * 3, axis=-1)
         
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
        

        return self._get_obs(), self.current_episode_reward, done, False, self._get_info()
    
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
    


