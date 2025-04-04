import os
import torch
import numpy as np
import gymnasium as gym
import torchvision
import wandb
import sys
from utils import plotter, plotter_with_ray, value_to_circle_pixel, shoot_ray

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2D_Shape_Completion'))
sys.path.append(project_root)

class CNN_Environment(gym.Env):
    def __init__(self, image_shape, number_rays, data_location, render_mode = 'rgb_array', wand = True, observation_type = "full", random_perfect = False):

        self.wand = wand

        self.shape = image_shape

        self.height, self.width = image_shape

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.observation_type = observation_type

        if observation_type == "full":
            self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(low=0, high=255, shape=(1, self.width, self.height), dtype=np.uint8),
                'past_actions': gym.spaces.Box(low=-1, high=1, shape=(number_rays, 2), dtype=np.float32),
                'current_rays': gym.spaces.Discrete(number_rays)
            })
        elif observation_type == "heatmap":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(5, self.width, self.height), dtype=np.uint8)
        elif observation_type == "past_actions":
            self.observation_space = gym.spaces.Dict({
                'past_actions': gym.spaces.Box(low=-1, high=1, shape=(number_rays, 2), dtype=np.float32),
                'current_rays': gym.spaces.Discrete(number_rays)
            })
        else:
            raise ValueError("Observations must be 'full', 'heatmap' or 'past_actions'")

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
        self.unet = torch.load("shape_completion_models/model.pth", map_location=self.device)
        self.unet.eval()

        self.loss = torch.nn.BCELoss()

        self.episode_rewards = []

        self.current_similarity = 0

        self.past_actions = [(0, 0)] * number_rays

        self.hit =False

        self.random_perfect = random_perfect

        self.x = None
        self.y = None

    def _get_obs(self):
        # Add channel dimension to observation
        image_observation = self.obs[None, :, :] * 255
        image_observation = image_observation.astype(np.uint8)
        if self.observation_type == "full":
            return {
                'image': image_observation,
                'past_actions': np.array(self.past_actions),
                'current_rays': self.current_rays
            }
        elif self.observation_type == "heatmap":
            return image_observation
        elif self.observation_type == "past_actions":
            return {
                'past_actions': np.array(self.past_actions),
                'current_rays': self.current_rays
            }
    
    def _get_info(self):
        return {
            'current_similarity': self.current_similarity,
            'current_loss': self.current_loss,
            'ray_hit': self.hit,
            'hit_point': (self.x, self.y)
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

        self.hit =False

        # Must return observation and info
        return self._get_obs(), self._get_info()
    

    def step(self, action):
        x, y= shoot_ray(action,  self.image, self.width, self.height,)
        if self.random_perfect and (x is None or y is None):
            self.hit = False
            return self._get_obs(), 0, False, False, self._get_info()
        self.action = action
        self.past_actions[self.current_rays] = action
        
        self.x = x
        self.y = y

        self.current_rays += 1
        self.hit =False
        if (x is not None and y is not None):
            self.input[x][y] = 1   
            self.hit =True  
            
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

                plotter_with_ray(input_rgb, predict_rgb, grount_truth_rgb, "Input", "Prediction", "Ground Truth", value_to_circle_pixel(position), angle, (self.y, self.x), self.wand, self.current_rays)
                
            else:
                plotter(input_rgb, predict_rgb, grount_truth_rgb, "Input", "Prediction", "Ground Truth", self.wand, self.current_rays)
            
            print("Current similarity :", self.current_similarity)
            
        elif self.metadata["render_mode"] == 'rgb_array':
            # Ensure proper shape for video recording (height, width, channels)
            render_img = np.expand_dims(predict_rgb, axis=-1)  # Add channel dimension
            render_img = np.repeat(render_img, 3, axis=-1)  # Repeat to 3 channels
            render_img = render_img.astype(np.uint8)  # Convert to uint8
            return render_img  # Return format: (height, width, 3)