import numpy as np

def action_to_border(x, length, height):
        if x <= length:
            return (0, x)  #Top border
        elif x <= length + height:
            return (x - length, length - 1)  # Right border
        elif x <= 2 * length + height:
            return (height - 1, 2 * length + height - x)  # Bottom border
        else:
            return (2 * (length + height) - x, 0)  # Left border

def shoot_ray(binary_image, border, angle):
    x, y = action_to_border(border, 224, 224)
    
    # Check if initial position is valid
    if (x < 0 or x >= binary_image.shape[1] or 
        y < 0 or y >= binary_image.shape[0]):
        return [], None
        
    # Check if starting on obstacle
    if binary_image[y, x] == 1:
        return [], (x, y)

    dx = np.cos(angle)
    dy = np.sin(angle)
    pixels = [(int(round(x)), int(round(y)))]

    while True:
        x += dx
        y += dy
    
        x_int = int(round(x))
        y_int = int(round(y))
    
        if (x_int < 0 or x_int >= binary_image.shape[1] or 
            y_int < 0 or y_int >= binary_image.shape[0]):
            return pixels, None
    
        elif binary_image[y_int, x_int] == 1:
            return pixels, (x_int, y_int)
        
        else:
            pixels.append((x_int, y_int))