import matplotlib.pyplot as plt



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