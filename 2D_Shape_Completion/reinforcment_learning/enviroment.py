import numpy as np
import gymnasium as gym
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import ActionWrapper
from typing import Optional


from dataset.preprocessing import  path_to_tensor



class RayEnviroment(gym.Env):
    def __init__(self, shape_image, model, loss, max_number_rays, dataset, device, render_mode=None):
        self.shape = shape_image
        self.height, self.length = shape_image

        self.device = device

        # Validate render_mode
        if render_mode not in {None, 'rgb_array'}:
            raise ValueError("Invalid render_mode. Supported modes are: None, 'rgb_array'.")

        self.render_mode = render_mode
        self.metadata = {"render_modes": ['rgb_array'], "render_fps": 30}

        

        self.action_space = gym.spaces.MultiDiscrete([2*(self.height+self.length), 360])

        #Getting the border action and translating it in a border point
        self.action_to_border = {x: self.fun_action_to_border(x, self.length, self.height) for x in range(2 * (self.height + self.length))}


        #Obseervations are the points so fare known
        self.size = self.height * self.length
        self.observation_space = gym.spaces.MultiBinary(self.size)

        #Lista contenente i punti ottenuti
        self._sampled_point = np.zeros(self.size, dtype = np.int8)

        #Inizialize l'immage ground truth used in reset plus the tensor of itself
        self.image = np.zeros(shape_image)
        self.tensor_image = transforms.ToTensor()(self.image).unsqueeze(0).to(device)

        #Information on how to render
        self.metadata = {
            "render_mode": render_mode,
        }

        #Model and loss to calculate reward
        self.model = model
        self.loss = loss

        #Max number of rays the model can shoot
        self.max_number_rays = max_number_rays

        #Number current rays
        self.number_rays= 0

        #Predict image for rendering
        self.predict = np.zeros(shape_image)

        #Image with only the sampled points
        self.sampled_image = np.zeros(shape_image)

        self.dataset = dataset

    def _get_obs(self):
        return self._sampled_point
    
    def _get_info(self):
        return {}
    
    def reset(self, seed:Optional[int], options= None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #Randomly sample a path of the image
        x = np.random.randint(len(self.dataset))
        
        #Get path
        image_path = self.dataset.data[x]
        print(image_path)

        #Initialize the image
        self.image = path_to_tensor(image_path=image_path, device= self.device)
        self.tensor_image = transforms.ToTensor()(self.image).unsqueeze(0).to(self.device) #Makes it a tensor and unsqueezes

        #Initialize self._sampled_point to zero since we have no info yet
        self._sampled_point = np.zeros(self.size, dtype = np.int8)

        #Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        #Reset number rays and terimanted
        self.number_rays = 0

        #Reset sampled image
        self.predict = np.zeros(self.shape)

        return observation, info

    def step(self, action):
        #If agent performs actions means sending ray on the image, finds a point hopefully and reuse alghorithm 
        x, y= self._shoot_ray(action)

        #Given the point found by the ray (still loses all the info about the fact that there are no points in betweeen)
        #Step for the enviroment
        if (x is not None and y is not None):
            #print(f"Found a point at iteration _{self.number_rays}")
            self._sampled_point[self.length *x +y]= 1          
            self.sampled_image[x][y]=1
        else:
            #print(f"Not found anything at iteration _{self.number_rays}")
            return self._sampled_point, -2.5 * (self.max_number_rays - self.number_rays) , True, False, self._get_info()
        
        #Reward is the loss of the model 
        input_tensor = torch.tensor(self.sampled_image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.unsqueeze(1).to(self.device)  # Add channel dimension

        #Get prediction from model and found points
        output = self.model(input_tensor)

        # Convert the model output to a probability map and binary mask
        output_image = output[0][0].cpu().detach().numpy()  # Get the first output channel as a numpy array
        self.predict = (output_image > 0.5).astype(np.uint8)  # Thresholding to create a binary mask


        if callable(self.loss):
            reward = -self.loss(output, self.tensor_image) 
        else:
            print("Errore: self.loss not callable")
        
        #Output
        info = self._get_info()
        
        #When i did too many reys terminate
        self.number_rays += 1
        if (self.number_rays >= self.max_number_rays):
            return self._sampled_point, reward, True, False, info

        return self._sampled_point, reward, False, False, info
    

    def render(self):
        if self.metadata["render_mode"] == 'rgb_array':
            # Ensure `predict` and `sampled_image` are in the correct format
            predict_bw = (self.predict * 255).astype(np.uint8)  # Black and white (0 or 255)

            # Convert the black and white image to RGB by repeating the single channel across all 3 channels
            predict_rgb = np.stack([predict_bw] * 3, axis=-1)  # Duplicate across the 3 channels (R, G, B)

            return predict_rgb
        else:
            raise ValueError("Render mode is not set to 'rgb_array'.")


    def _shoot_ray(self, action):
        #Get two different actions, the angle is already econded in angle_action since we are taking 360 degrees
 
        border_action, angle_action = action
        
        #Find point found by ray
        x , y = self.action_to_border[border_action]

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
            if x_int < 0 or x_int >= self.length or y_int < 0 or y_int >= self.height:
                return None, None
        
            # Check if the ray hits an obstacle (assumed to be represented by 1)
            elif self.image[y_int, x_int] == 1:
                return x_int, y_int



    def fun_action_to_border(self, x, length, height):
        if x <= length:
            return (0, x)  #Top border
        elif x <= length + height:
            return (x - length, length - 1)  # Right border
        elif x <= 2 * length + height:
            return (height - 1, 2 * length + height - x)  # Bottom border
        else:
            return (2 * (length + height) - x, 0)  # Left border







class ActionNormWrapper(ActionWrapper):
    

    def action(self, action):
        border_action, angle_action = action
        border_action = border_action / (2 * (224 + 224))

        angle_action = angle_action / 360

        return [border_action, angle_action]
    



class RunningRewardCallback(BaseCallback):
    def __init__(self, window_size=100, verbose=0):
        super(RunningRewardCallback, self).__init__(verbose)
        self.window_size = window_size
        self.episode_rewards = []  # List to store rewards of each episode
        self.running_avg_rewards = []  # List to store running average of rewards

    def _on_step(self) -> bool:
        # Store the episode reward for the current step
        reward = self.locals.get('rewards', None)
        if reward is not None:
            self.episode_rewards.append(reward)
        
        # Calculate and store the running average
        avg_reward = np.mean(self.episode_rewards[-self.window_size:])
        self.running_avg_rewards.append(avg_reward)
        
        # Print the running average of the reward
        #print(f"Running average reward: {avg_reward}")
        
        # Optionally, you can plot the rewards as well
        plt.plot(self.running_avg_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Running Average Reward')
        plt.title('Running Average of Rewards during Training')

        # Save plot as image
        plt.savefig('Running_Average_Reward_Plot.png')
        plt.close()
        return True
    


