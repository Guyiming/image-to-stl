import cv2
import numpy as np

def create_test_chart(size=800):
    """Create a test image for subtractive color mixing using OpenCV
    
    Args:
        size (int): Width and height of the square test chart
    
    Returns:
        np.ndarray: BGR image array of shape (size, size, 3)
    """
    # Create white background
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    section_size = size // 4
    
    # 1. Pure CMY blocks (note: OpenCV uses BGR)
    colors_cmy = [
        (255, 255, 0),  # Cyan    (BGR)
        (255, 0, 255),  # Magenta (BGR)
        (0, 255, 255),  # Yellow  (BGR)
    ]
    for i, color in enumerate(colors_cmy):
        x1, y1 = i * section_size, 0
        x2, y2 = (i + 1) * section_size, section_size
        img[y1:y2, x1:x2] = color
    
    # 2. Pure RGB blocks
    colors_rgb = [
        (0, 0, 255),    # Red   (BGR)
        (0, 255, 0),    # Green (BGR)
        (255, 0, 0),    # Blue  (BGR)
    ]
    for i, color in enumerate(colors_rgb):
        x1, y1 = i * section_size, section_size
        x2, y2 = (i + 1) * section_size, 2 * section_size
        img[y1:y2, x1:x2] = color
    
    # 3. Grayscale gradient (two sections tall)
    for i in range(4 * section_size):
        value = int((i / (4 * section_size)) * 255)
        y = i #+ section_size  # Start from second row (y=section_size)
        img[y, 3*section_size:4*section_size] = [value, value, value]
    
    # 4. Color wheel
    center = (size // 2, 3 * size // 4)
    radius = section_size
    
    # Create meshgrid for vectorized calculations
    y, x = np.ogrid[0:size, 0:size]
    
    # Calculate distances and angles for all points
    dx = x - center[0]
    dy = y - center[1]
    distances = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)
    
    # Convert angles to degrees (0-360)
    angles = np.degrees(angles)
    angles += 180  # Shift from [-180, 180] to [0, 360]
    
    # Create mask for points within radius
    mask = distances <= radius
    
    # Normalize distances for saturation
    saturations = distances[mask] / radius
    
    # Convert angles to hue (0-180 for OpenCV)
    hues = (angles[mask] / 2).astype(np.uint8)
    
    # Create HSV array
    wheel = np.zeros((mask.sum(), 3), dtype=np.uint8)
    wheel[:, 0] = hues
    wheel[:, 1] = (saturations * 255).astype(np.uint8)
    wheel[:, 2] = 255
    
    # Convert to BGR
    wheel_bgr = cv2.cvtColor(wheel.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR)
    
    # Place the color wheel pixels
    y_coords, x_coords = np.where(mask)
    img[y_coords, x_coords] = wheel_bgr.reshape(-1, 3)
    
    # Add test gradient bars in bottom right
    gradient_width = section_size // 3
    
    # Add pure CMY gradients
    gradients = [
        (255, 255, 0),  # Cyan gradient
        (255, 0, 255),  # Magenta gradient
        (0, 255, 255),  # Yellow gradient
    ]
    
    for i, color in enumerate(gradients):
        x1 = i * gradient_width
        x2 = x1 + gradient_width
        for j in range(2 * section_size):
            alpha = j / (2 * section_size)
            mixed_color = tuple(int(c * alpha) for c in color)
            y = 2 * section_size + j
            img[y, x1:x2] = mixed_color
    
    return img

if __name__ == "__main__":
    # Create and save test chart
    test_image = create_test_chart()
    cv2.imwrite('examples/color_test_chart.png', test_image)