import os
import torch
import numpy as np
import gymnasium as gym
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import os
import sys

# Add the 2D_Shape_Completion directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2D_Shape_Completion'))
sys.path.append(project_root)

class TestEnvironment2(gym.Env):
    def __init__(self, image_shape, number_rays, data_location, render_mode = 'rgb_array'):
        self.shape = image_shape
        self.height, self.width = image_shape
        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.metadata = {'render_modes': ['human', 'rgb_array'],
                          'render_fps': 30}
        #Obseervations are the points so fare known
        self.observation_space = gym.spaces.MultiBinary([self.height, self.width])

        #Max number of rays the model can shoot
        self.number_rays = number_rays

        #Number current rays
        self.current_rays = 0

        #Store path to data
        self.data_location = data_location
        
        #Load random image from data_location
        self.image_files = [os.path.join(self.data_location, f) for f in os.listdir(self.data_location) if f.endswith('.png')]

        #Randomly sample a path of the image
        idx = np.random.randint(len(self.image_files))

        #Get path and load image
        image_path = self.image_files[idx]
        self.image = torchvision.io.read_image(image_path)

        #Convert image to np array with values in {0, 1}
        self.image = self.image[0] > 0

        #Model and loss to calculate reward
        # Define the device (GPU or mps or cpu)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        # Load the model and map it to the GPU
        self.unet = torch.load("Reinforcement_Learning/saved_models/model_full.pth", map_location=self.device)
        self.unet.eval()

        self.input = self.image > np.inf

        self.obs = self.input

        self.loss = torch.nn.BCELoss()

        self.current_loss = 0


    def _get_obs(self):
        return self.obs
    
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
        self.image = self.image[0] > 0

        self.input = self.image > np.inf

        self.obs = self.input

        self.current_loss = -2

        # Must return observation and info
        return self._get_obs(), self._get_info()

    def step(self, action):
        #If agent performs actions means sending ray on the image, finds a point hopefully and reuse alghorithm 
        x, y= self._shoot_ray(action)

        self.current_rays += 1

        reward = 0
        if ( x is  not None and y is not None):

            self.input[x][y]= 1          
            transformer_input = self.input.unsqueeze(0).to(self.device).float()
            transformer_input = transformer_input.unsqueeze(0) 

            with torch.no_grad():
                #Get prediction from model and found points
                output = self.unet(transformer_input)

            # Convert the model output to a probability map and binary mask
            output_image = output[0][0].cpu()  
            self.obs = (output_image > 0.5)  # Thresholding to create a binary mask
            
            self.current_loss = - self.loss(output, transformer_input)
            
        reward = self.current_loss

        done = self.current_rays >= self.number_rays

        #Output
        info = self._get_info()

        return self._get_obs(), reward, done, False, info

    def render(self):
        # Convert tensors to numpy arrays and scale to 0-255 for visualization
        predict_bw = (self.obs.cpu().numpy() * 255).astype(np.uint8)  # Convert self.obs
        input_bw = (self.input.cpu().numpy() * 255).astype(np.uint8)  # Convert self.input
        grount_truth_bw = (self.image.cpu().numpy() * 255).astype(np.uint8)

        # Duplicate channels for RGB display
        predict_rgb = np.stack([predict_bw] * 3, axis=-1)  # Convert self.obs to RGB
        input_rgb = np.stack([input_bw] * 3, axis=-1)  # Convert self.input to RGB
        grount_truth_rgb = np.stack([grount_truth_bw]* 3 , axis=-1)
        if self.render_mode == "human":
            # Display both images side by side using matplotlib
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            
            # Plot self.input
            axes[0].imshow(input_rgb)
            axes[0].set_title("Input")
            axes[0].axis("off")
            
            # Plot self.obs
            axes[1].imshow(predict_rgb)
            axes[1].set_title("Prediction")
            axes[1].axis("off")

            #PLot self.image
            axes[2].imshow(grount_truth_rgb)
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")
            
            # Show the combined plot
            plt.tight_layout()
            plt.show()
        elif self.render_mode == "rgb_array":
            # Return the image for external rendering
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