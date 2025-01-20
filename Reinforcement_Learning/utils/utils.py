import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Convert tensors to numpy arrays and scale to 0-255 for visualization and  Duplicate channels for RGB display
def converter(tensor):
    tensor= (tensor.cpu().numpy() * 255).astype(np.uint8)
    return  np.stack([tensor] * 3, axis=-1)  # Convert tensor to RGB



#PLot 3 images( usually input, output and ground truth and add title, point and arrow)
def plotter_with_ray (image1, image2, imgae3, title1, title2, title3, border_points, angle, point, wand = False, step= 0):
    # Display both images side by side using matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
      
    # Plot self.input
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis("off")

    bord_x,bord_y = border_points

    x,y = point

    # Add a green point at coordinates (self.x, self.y)
    axes[0].scatter(bord_y, bord_x, c='green', s=20)  # s controls the size of the point
    axes[0].scatter(x, y, c='blue', s=20)  # s controls the size of the point
                
    #Show arrow from border point with the angle (angle is between 0 and 360)
    axes[0].arrow(bord_y, bord_x, -30*np.sin(np.deg2rad(angle)), -30*np.cos(np.deg2rad(angle)), head_width=5, head_length=5, fc='red', ec='red')

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
    if wand:
        wandb.log({"Image": wandb.Image(plt), 
                   "Step" : step})
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