import numpy as np

def get_surface_pixels(image):
    """
    Extract surface pixels from a binary image.
    
    Args:
        image: 2D numpy array with values 0 (background) and 1 (object)
    
    Returns:
        2D numpy array where 1 indicates surface pixels and 0 indicates non-surface pixels
    """
    # Ensure input is binary
    image = (image > 0).astype(np.uint8)
    
    # Create a padded version of the image to handle borders
    padded = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    
    # Initialize surface array
    surface = np.zeros_like(image)
    
    # Get image dimensions
    rows, cols = image.shape
    
    # Check each pixel and its neighbors
    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 1:  # If pixel is part of object
                # Check 8-neighborhood
                neighborhood = padded[i:i+3, j:j+3]
                # If any neighbor is 0, this is a surface pixel
                if np.any(neighborhood == 0):
                    surface[i, j] = 1
    
    return surface

def binary_to_image(binary_array):
    """
    Convert a binary array to a displayable image.
    
    Args:
        binary_array: 2D numpy array with values 0 and 1
        
    Returns:
        2D numpy array with values 0 (black) and 255 (white)
    """
    # Convert to uint8 and scale to 0-255
    return (binary_array * 255).astype(np.uint8)