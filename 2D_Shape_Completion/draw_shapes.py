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

def draw_donut(outer_radius=50, inner_radius=25, center=None):
    """
    Draw a black donut (annulus) on a white canvas.
    
    Args:
        outer_radius (int): Outer radius of the donut in pixels
        inner_radius (int): Inner radius of the donut in pixels
        center (tuple): Center coordinates of the donut (x, y). If None, donut will be centered.
    
    Returns:
        numpy.ndarray: 256x256 image with white background (0) and black donut (1)
    """
    # Create white canvas
    canvas = np.zeros((256, 256), dtype=np.uint8)
    
    if center is None:
        center = (128, 128)
    
    # Draw outer circle
    cv2.circle(canvas, center, outer_radius, 255, -1)
    # Draw inner circle (erases center)
    cv2.circle(canvas, center, inner_radius, 0, -1)
    
    # Convert to binary
    binary_image = (canvas > 0).astype(np.uint8)
    
    return binary_image

def draw_spiral(revolutions=3, max_radius=100, line_thickness=5, center=None):
    """
    Draw a black spiral on a white canvas.
    
    Args:
        revolutions (float): Number of complete revolutions
        max_radius (int): Maximum radius of the spiral in pixels
        line_thickness (int): Thickness of the spiral line in pixels
        center (tuple): Center coordinates of the spiral (x, y). If None, spiral will be centered.
    
    Returns:
        numpy.ndarray: 256x256 image with white background (0) and black spiral (1)
    """
    canvas = np.zeros((256, 256), dtype=np.uint8)
    
    if center is None:
        center = (128, 128)
        
    t = np.linspace(0, revolutions * 2 * np.pi, 1000)
    r = t * max_radius / (revolutions * 2 * np.pi)
    x = center[0] + r * np.cos(t)
    y = center[1] + r * np.sin(t)
    
    points = np.column_stack((x, y)).astype(np.int32)
    cv2.polylines(canvas, [points], False, 255, line_thickness)
    
    binary_image = (canvas > 0).astype(np.uint8)
    
    return binary_image