import numpy as np
import cv2

def draw_circle(radius=50, center=None):
    """
    Draw a black circle on a white canvas.
    
    Args:
        radius (int): Radius of the circle in pixels
        center (tuple): Center coordinates of the circle (x, y). If None, circle will be centered.
    
    Returns:
        numpy.ndarray: 256x256 image with white background (0) and black circle (1)
    """
    # Create white canvas (0s)
    canvas = np.zeros((256, 256), dtype=np.uint8)
    
    # Set center point if not provided
    if center is None:
        center = (128, 128)  # Center of the image
    
    # Draw black circle (255 for cv2, will be converted to 1 later)
    cv2.circle(canvas, center, radius, 255, -1)
    
    # Convert to binary (0 and 1) where circle is 1 (black) and background is 0 (white)
    binary_image = (canvas > 0).astype(np.uint8)
    
    return binary_image