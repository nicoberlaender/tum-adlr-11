import numpy as np
import gymnasium as gym
import torch
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import ActionWrapper
from typing import Optional


from dataset.preprocessing import  path_to_tensor



class RayEnviroment(gym.Env):
    def __init__(self, shape_image, model, max_number_rays, data_location, device):
        self.shape = shape_image
        self.height, self.width = shape_image
        self.device = device
        self.data_location = data_location
        #Load random image from data_location
        self.image_files = [os.path.join(self.data_location, f) for f in os.listdir(self.data_location) if f.endswith('.png')]

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        #Obseervations are the points so fare known
        self.observation_space = gym.spaces.MultiBinary([self.height, self.width])

        #Inizialize l'immage ground truth used in reset plus the tensor of itself
        self.image = np.zeros(shape_image)
        self.tensor_image = transforms.ToTensor()(self.image).unsqueeze(0).to(device)

        #Model and loss to calculate reward
        self.model = model
        self.model.eval()


        #Max number of rays the model can shoot
        self.max_number_rays = max_number_rays

        #Number current rays
        self.number_rays= 0

        self.input = np.zeros(self.shape)



    def _get_obs(self):
        return self.input
    
    def _get_info(self):
        return {}
    
    def reset(self, seed:Optional[int], options= None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Load new image
        idx = np.random.randint(len(self.image_files))
        image_path = self.image_files[idx]

        #Initialize the image
        image = Image.open(image_path).convert('L')
        self.image = np.array(image)
        self.tensor_image = transforms.ToTensor()(self.image).unsqueeze(0).to(self.device) #Makes it a tensor and unsqueezes

        self.input = np.zeros(self.shape)
        #Reset number rays and terimanted
        self.number_rays = 0
        # Must return observation and info
        return self._get_obs(), self._get_info()

    def step(self, action):
        #If agent performs actions means sending ray on the image, finds a point hopefully and reuse alghorithm 
        x, y= self._shoot_ray(action)
        #Given the point found by the ray (still loses all the info about the fact that there are no points in betweeen)
        #Step for the enviroment
        if (x is not None and y is not None):
            #print(f"Found a point at iteration _{self.number_rays}")
            self.input[x][y]= 1          
            transformer_input = transforms.ToTensor()(self.input).unsqueeze(0).to(self.device)

        else:
            #print(f"Not found anything at iteration _{self.number_rays}")
            #reward = -2.5 * (self.max_number_rays - self.number_rays)
            return self._get_obs(), 0 , False, False, self._get_info()
        
        
        with torch.no_grad:
            #Get prediction from model and found points
            output = self.model(transformer_input)

        # Convert the model output to a probability map and binary mask
        output_image = output[0][0].cpu().numpy()  # Get the first output channel as a numpy array
        self.input = (output_image > 0.5)  # Thresholding to create a binary mask

        #Output
        info = self._get_info()

        #When i did too many reys terminate
        self.number_rays += 1
        if (self.number_rays >= self.max_number_rays):
            return self._get_obs(), +1, True, False, info

        return self._get_obs(), +1, False, False, info
    

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
    




    """
    def render(self):
        if self.metadata["render_mode"] == 'rgb_array':
            # Ensure `predict` and `sampled_image` are in the correct format
            predict_bw = (self.predict * 255).astype(np.uint8)  # Black and white (0 or 255)

            # Convert the black and white image to RGB by repeating the single channel across all 3 channels
            predict_rgb = np.stack([predict_bw] * 3, axis=-1)  # Duplicate across the 3 channels (R, G, B)

            return predict_rgb
        else:
            raise ValueError("Render mode is not set to 'rgb_array'.")
"""
