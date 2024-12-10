import numpy as np
import gymnasium as gym

from PIL import Image
from typing import Optional

from dataset.preprocessing import sample_pixels, segmap_to_binary, binary_to_image

class RayEnviroment(gym.Env):
    def __init__(self, shape_image, model, loss, max_number_rays, dataset, render_mode=None):
        self.height, self.length = shape_image

        # Validate render_mode
        if render_mode not in {None, 'rgb_array'}:
            raise ValueError("Invalid render_mode. Supported modes are: None, 'rgb_array'.")

        self.render_mode = render_mode
        self.metadata = {"render_modes": ['rgb_array'], "render_fps": 30}

        # (Other initialization code remains unchanged)

        self.action_space = gym.spaces.Dict(
            {
                #Possible actions are choosing the border pixel and choosing the degree of the ray 
                "action_space_border" : gym.spaces.Discrete(2*(self.height+self.length)),
                "action_space_angle" : gym.spaces.Discrete(360),

            }
        )
        

        #Getting the border action and translating it in a border point
        self.action_to_border = {x: self.fun_action_to_border(x, self.length, self.height) for x in range(2 * (self.height + self.length))}


        #Obseervations are the points so fare known
        self.size = self.height * self.length
        self.observation = gym.spaces.Dict(
            {
                "sampled_point" : gym.spaces.Box(0, self.size-1, shape=(self.size,), dtype=int)
            }
        )

        #Observation array (flattened image cointaining 1 if point was sampled or found, 0 else)
        self._sampled_point = np.zeros(self.size, dtype=np.int8)

        #Inizialize l'immage ground truth used in reset
        self.image = np.zeros(shape_image)

        #Information on how to render
        self.metadata = {
            "render_mode": 'rgb_array',
        }

        #Model and loss to calculate reward
        self.model = model
        self.loss = loss

        #Max number of rays the model can shoot
        self.max_number_rays = max_number_rays

        #Number current rays and terminated
        self.terminated = False
        self.number_rays= 0

        #Predict image for rendering
        self.predict = np.zeros(shape_image)
        self.sampled_image = np.zeros(shape_image)

        self.dataset = dataset

    def _get_obs(self):
        return {"sampled_point": self._sampled_point}
    
    def _get_info(self):
        return None
    
    def reset(self, seed:Optional[int]):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #Randomly sample a path of the image
        x = np.random.randint(len(self.dataset))
        
        #Get path
        image_path = self.dataset.data[x]

        #Initialize the image
        self.image = Image.open(image_path).convert('L')  # Convert to grayscale
        self.image = np.array(self.image)
        self.image = segmap_to_binary(self.image)

        #Initialize self._sampled_point to zero since we have no info yet
        self._sampled_point = np.zeros(self.size, dtype = np.int8)

        #Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        #Reset number rays and terimanted
        self.number_rays = 0
        self.terminated = False

        
        return observation, info

    def step(self, action):
        #If agent performs actions means sending ray on the image, finds a point hopefully and reuse alghorithm 
        x, y= self._shoot_ray(self, action)

        #Given the point found by the ray (still loses all the info about the fact that there are no points in betweeen)
        #Step for the enviroment
        if (x!=None and y!=None):
            self._sampled_point[self.length * x + y] = 1

        #Make the image from samplepoint
        self.sampled_image = np.reshape(self._sampled_point, (self.height, self.length))

        #Reward is the loss of the model 
        self.predict = self.model(self.sampled_image)
        reward = self.loss(self.predict, self.image)

        #Output
        observation = self._get_obs()
        info = self._get_info()
        
        #When i did too many reys terminate
        self.number_rays += 1
        if (self.number_rays >= self.max_number_rays):
            self.terminated = True

        truncated = None

        return observation, reward, self.terminated, truncated, info
    

    def render(self):
        if self.metadata["render_mode"] == 'rgb_array':
            # Ensure `predict` and `sampled_image` are in the correct format
            predict_bw = (self.predict * 255).astype(np.uint8)  # Black and white (0 or 255)
            sampled_image_bw = (self.sampled_image * 255).astype(np.uint8)  # Black and white (0 or 255)

            # Convert arrays to PIL images
            predict_image = Image.fromarray(predict_bw, mode="L")
            sampled_image = Image.fromarray(sampled_image_bw, mode="L")
            
            # Combine the images side-by-side
            combined_width = predict_image.width + sampled_image.width
            combined_height = max(predict_image.height, sampled_image.height)
            combined_image = Image.new("L", (combined_width, combined_height))  # Black and white canvas
            
            combined_image.paste(predict_image, (0, 0))  # Place `predict` on the left
            combined_image.paste(sampled_image, (predict_image.width, 0))  # Place `sampled_image` on the right
            
            # Return the combined image as an array
            return np.array(combined_image)
        else:
            raise ValueError("Render mode is not set to 'rgb_array'.")


    def _shoot_ray(self, action):
        #Get two different actions, the angle is already econded in angle_action since we are taking 360 degrees
        border_action, angle_action = action
    
        #Find point found by ray
        x , y = self.action_to_border[border_action]

        #Conver angle to radiants
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
