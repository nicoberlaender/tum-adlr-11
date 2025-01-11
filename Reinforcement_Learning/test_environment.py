import os

import numpy as np
import gymnasium as gym
import torchvision

class TestEnvironment(gym.Env):
    def __init__(self, image_shape, number_rays, data_location):
        self.shape = image_shape
        self.height, self.width = image_shape

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        #Obseervations are the points so fare known
        self.observation_space = gym.spaces.MultiBinary([self.height, self.width])

        #Max number of rays the model can shoot
        self.number_rays = number_rays

        #Number current rays
        self.current_rays = 0

        #Store path to data
        self.data_location = data_location
        
        #Load random image from data_location
        # Create dataset from data_location using a custom dataset
        #self.image_files = [os.path.join(self.data_location, f) for f in os.listdir(self.data_location) if f.endswith('.png')]

        #Randomly sample a path of the image
        #idx = np.random.randint(len(self.image_files))

        #Get path and load image
        #image_path = self.image_files[idx]
        image_path = os.path.join(self.data_location, "0.png")
        self.image = torchvision.io.read_image(image_path)

        #Convert image to np array with values in {0, 1}
        self.image = self.image[0] > 0

    def _get_obs(self):
        return self.image
    
    def _get_info(self):
        return {}
    
    def reset(self, seed=None, options= None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_rays = 0
        
        # Load new image
        #idx = np.random.randint(len(self.image_files))
        #image_path = self.image_files[idx]
        #self.image = torchvision.io.read_image(image_path)
        #self.image = self.image[0] > 0
        
        # Must return observation and info
        return self._get_obs(), self._get_info()

    def step(self, action):
        #If agent performs actions means sending ray on the image, finds a point hopefully and reuse alghorithm 
        x, y= self._shoot_ray(action)

        #Reward
        reward = 1 if x is not None and y is not None else 0
        
        #Output
        info = self._get_info()
        
        self.current_rays += 1
        done = self.current_rays >= self.number_rays

        return self._get_obs(), reward, done, False, info


    def _shoot_ray(self, action):
        #Get two different actions, the angle is already econded in angle_action since we are taking 360 degrees
 
        border, angle = action

        x,y = self._value_to_border_pixel(border)

        angle_action = angle * 360

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