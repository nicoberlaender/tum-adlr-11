import numpy as np
from PIL import Image

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

def sample_pixels(binary_image, n):
    """
    Randomly sample n active pixels (value 1) from a binary image and return an image marking those pixels.
    
    Args:
        binary_image: 2D numpy array with values 0 and 1
        n: Number of pixels to sample
        
    Returns:
        2D numpy array of same dimensions as input, with 1s at sampled locations and 0s elsewhere
    """
    # Initialize output image
    output = np.zeros_like(binary_image)
    
    # Get active pixel coordinates
    active_pixels = np.argwhere(binary_image > 0)
    n = min(n, len(active_pixels))
    
    # Sample pixels
    sampled_indices = np.random.choice(len(active_pixels), size=n, replace=False)
    sampled_points = active_pixels[sampled_indices]
    
    # Mark sampled pixels in output image
    output[sampled_points[:, 0], sampled_points[:, 1]] = 1
    
    return output

def segmap_to_binary(image):
    """
    Convert a 2D numpy array to a binary image.
    
    Args:
        image: 2D numpy array
        
    Returns:
        2D numpy array with values 0 and 1
    """
    return (image == 255).astype(np.uint8)

def pil_to_binary(image):
    """
    Convert a black/white PIL image to a binary numpy array.
    
    Args:
        image: PIL Image object in black/white mode ('1' or 'L')
        
    Returns:
        2D numpy array with values 0 and 1
    """
    # Convert image to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy array
    np_image = np.array(image)
    
    # Convert to binary
    binary_image = (np_image > 127).astype(np.int8)
    
    return binary_image

def path_to_tensor (image_path, device):
    """
    Converts an image file to a binary tensor.

    Args:
        image_path (str): Path to the image file
        device (torch.device): Device to which the tensor should be moved (not currently used)

    Returns:
        torch.Tensor: Binary tensor where original white pixels (255) are converted to 1 and black pixels (0) remain 0
    """
    

    image = Image.open(image_path).convert('L')  # Convert to grayscale

    image = np.array(image) #Makes it np.array

    return segmap_to_binary(image) #From 255 to 1 values

    