import matplotlib.pyplot as plt
import numpy as np


def visual3(binary_prediction, sampled_surface, true_image):
    # Plot the images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Display the predicted image
    ax[0].imshow(binary_prediction, cmap="gray")
    ax[0].set_title("Prediction")
    # Assuming sampled_surface has shape [1, 1, height, width]
    sampled_surface = sampled_surface.squeeze(0).squeeze(0)  # Removes the first two dimensions

    # Display the ground truth image
    ax[1].imshow(sampled_surface, cmap="gray")  # Ground truth
    ax[1].set_title("Input Image")

    # Display the original input image
    ax[2].imshow(true_image, cmap='gray')  # Original input
    ax[2].set_title("Ground Truth")

    # Show the plot
    plt.show()
    return 0

def ray_shooting(border, angle, length, height, image, max_steps=1000):
    x, y = border

    # Convert angle to radians
    angle_action = np.radians(angle)

    # Compute step direction for ray tracing 
    dx = np.cos(angle_action)
    dy = np.sin(angle_action)

    # Ray tracing loop with maximum steps to avoid infinite loop
    steps = 0
    while steps < max_steps:
        x += dx
        y += dy
        x_int = int(round(x))
        y_int = int(round(y))
        
        # Check if the ray goes out of bounds
        if x_int < 0 or x_int >= length or y_int < 0 or y_int >= height:
            return None, None
        
        # Check if the ray hits an obstacle (assumed to be represented by 1)
        elif image[x_int, y_int] >0:
            return x_int, y_int
        
        steps += 1
    
    return None, None

