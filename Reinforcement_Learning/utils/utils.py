import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import math

# Convert tensors to numpy arrays and scale to 0-255 for visualization and  Duplicate channels for RGB display
def converter(tensor):
    tensor= (tensor.cpu().numpy() * 255).astype(np.uint8)
    return  np.stack([tensor] * 3, axis=-1)  # Convert tensor to RGB



#PLot 3 images( usually input, output and ground truth and add title, point and arrow)
def plotter_with_ray(image1, image2, image3, title1, title2, title3, 
                    border_points, angle_action, hit_point, wand=False, step=0):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    
    # Plot input image with ray
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis("off")

    # Get coordinates
    start_x, start_y = border_points
    hit_x, hit_y = hit_point

    width = image1.shape[1]
    height = image1.shape[0]
    
    # Calculate direction vector components
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    
    # Original radial vector (points toward center)
    radial_x = center_x - start_x
    radial_y = center_y - start_y
    
    # Calculate angle deviation (same logic as in _shoot_ray)
    half_width = (width - 1) / 2
    half_height = (height - 1) / 2
    radius = math.hypot(half_width, half_height)
    max_dev = math.atan(max(half_width, half_height)/radius)
    angle_dev = angle_action * max_dev  # angle_action is [-1, 1]
    
    # Rotate radial vector by angle_dev
    cos_a = math.cos(angle_dev)
    sin_a = math.sin(angle_dev)
    dir_x = radial_x * cos_a - radial_y * sin_a
    dir_y = radial_x * sin_a + radial_y * cos_a
    
    # Normalize and scale direction for visualization
    length = math.hypot(dir_x, dir_y)
    if length > 0:
        dir_x /= length
        dir_y /= length
    
    # Plot elements
    axes[0].scatter(start_y, start_x, c='green', s=20)  # Border point
    axes[0].scatter(hit_x, hit_y, c='blue', s=20)       # Hit point
    axes[0].arrow(start_y, start_x, 
                 dir_y * 30, dir_x * 30,  # Note y/x swap for matplotlib coordinates
                 head_width=5, head_length=5, 
                 fc='red', ec='red')
    # Also draw a circle around the image
    circle = plt.Circle((center_x, center_y), radius, color='grey', fill=False)
    axes[0].add_artist(circle)

    # Plot other images
    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis("off")

    axes[2].imshow(image3)
    axes[2].set_title(title3)
    axes[2].axis("off")

    plt.tight_layout()
    if wand:
        wandb.log({"Ray Tracing Visualization": wandb.Image(plt), "Step": step})
    else:
        plt.show()

#PLot 3 images( usually input, output and ground truth and add title, point and arrow)
def plotter (image1, image2, imgae3, title1, title2, title3, wand= False, step= 0):
    # Display both images side by side using matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            
     # Plot self.input
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis("off")

    # Plot self.obs
    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis("off")

    #PLot self.image
    axes[2].imshow(imgae3)
    axes[2].set_title(title3)
    axes[2].axis("off")

    # Show the combined plot
    plt.tight_layout()
    # Show the combined plot
    plt.tight_layout()
    if wand:
        wandb.log({"Image": wandb.Image(plt), 
                   "Step" : step})
    else:
        plt.show()