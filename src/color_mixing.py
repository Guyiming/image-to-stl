import numpy as np
from Models import LayerType, IntensityChannels, StlConfig, FilamentProperties
from ImageAnalyzer import ImageAnalyzer
from dataclasses import dataclass
from typing import Dict, Tuple

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    # Remove the hash symbol if present
    hex_color = hex_color.lstrip('#')
    
    # Check if the hex code is valid
    if len(hex_color) not in (3, 6):
        raise ValueError("Invalid hex color code. Must be 3 or 6 characters long.")
    
    # If it's a 3-digit hex code, expand it to 6 digits
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return (max(r,1), max(g,1), max(b,1))

def calculate_exact_thicknesses(
    target_rgb: np.ndarray,
    filaments: Dict[LayerType, FilamentProperties]
) -> Tuple[float, float, float]:
    """
    Calculate exact layer thicknesses using measured RGB values of filaments
    
    For each wavelength (R,G,B), the transmission through a layer is:
    T = e^(-t/d) where:
    T is transmission ratio for that wavelength (0-1)
    t is thickness
    d is max transmission distance
    
    For multiple layers, multiply transmission ratios per wavelength
    """
    # Convert target RGB (0-255) to transmission values (0-1)
    target_t = target_rgb / 255.0
    
    # Get filament properties
    f1 = filaments[LayerType.CYAN]
    f2 = filaments[LayerType.MAGENTA]
    f3 = filaments[LayerType.YELLOW]
    
    # Normalize filament RGB values to [0, 1] for logarithmic calculations
    f1_rgb = np.array(f1.rgb) / 255.0
    f2_rgb = np.array(f2.rgb) / 255.0
    f3_rgb = np.array(f3.rgb) / 255.0

    # Prevent log(0) errors
    epsilon = 1e-10
    target_t = np.clip(target_t, epsilon, 1.0)
    
    # Set up the system of equations
    # For each wavelength (R,G,B):
    # target_T = T1^a * T2^b * T3^c
    # where T1,T2,T3 are the base transmission ratios of each filament
    # Taking log of both sides:
    # log(target_T) = a*log(T1) + b*log(T2) + c*log(T3)
    
    # Create matrix A for the system Ax = b
    A = np.array([
        [np.log(f1_rgb[0]), np.log(f2_rgb[0]), np.log(f3_rgb[0])],
        [np.log(f1_rgb[1]), np.log(f2_rgb[1]), np.log(f3_rgb[1])],
        [np.log(f1_rgb[2]), np.log(f2_rgb[2]), np.log(f3_rgb[2])]
    ])
    
    # Create vector b
    b = np.array([
        np.log(target_t[0]),
        np.log(target_t[1]),
        np.log(target_t[2])
    ])
    #print(A,b)
    
    # Solve for x (the relative thicknesses)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # If matrix is singular, use least squares
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Convert relative thicknesses to actual thicknesses
    thickness_1 = x[0] * f1.max_distance
    thickness_2 = x[1] * f2.max_distance
    thickness_3 = x[2] * f3.max_distance
    
    # Clip to physical constraints
    thickness_1 = np.clip(thickness_1, f1.min_thickness, f1.max_distance)
    thickness_2 = np.clip(thickness_2, f2.min_thickness, f2.max_distance)
    thickness_3 = np.clip(thickness_3, f3.min_thickness, f3.max_distance)
    
    return thickness_1, thickness_2, thickness_3

def extract_and_invert_channels(img: ImageAnalyzer, config: StlConfig) -> IntensityChannels:
    # Create FilamentProperties from config
    filaments = {
        LayerType.CYAN: FilamentProperties(
            rgb=config.filament_colors[LayerType.CYAN],
            max_distance=config.layer_heights[LayerType.CYAN],
            min_thickness=config.layer_mins[LayerType.CYAN]
        ),
        LayerType.MAGENTA: FilamentProperties(
            rgb=config.filament_colors[LayerType.MAGENTA],
            max_distance=config.layer_heights[LayerType.MAGENTA],
            min_thickness=config.layer_mins[LayerType.MAGENTA]
        ),
        LayerType.YELLOW: FilamentProperties(
            rgb=config.filament_colors[LayerType.YELLOW],
            max_distance=config.layer_heights[LayerType.YELLOW],
            min_thickness=config.layer_mins[LayerType.YELLOW]
        )
    }
    
    # Initialize output arrays
    shape = img.pixelated.shape[:2]
    c_channel = np.zeros(shape)
    y_channel = np.zeros(shape)
    m_channel = np.zeros(shape)
    
    # Process each pixel
    for i in range(shape[0]):
        for j in range(shape[1]):
            c, m, y = calculate_exact_thicknesses(img.pixelated[i,j], filaments)
            print(img.pixelated[i,j], c,m,y)
            c_channel[i,j] = c
            m_channel[i,j] = m
            y_channel[i,j] = y
    
    # Calculate intensity map
    avg_pixels = (img.pixelated[:, :, 0] + img.pixelated[:, :, 1] + img.pixelated[:, :, 2]) / 3.0
    intensity_map = normalize_thickness_linear(
        avg_pixels,
        config.layer_heights[LayerType.KEY],
        config.layer_mins[LayerType.KEY]
    )
    
    return IntensityChannels(
        c_channel=c_channel,
        y_channel=y_channel,
        m_channel=m_channel,
        intensity_map=intensity_map
    )

def normalize_thickness_linear(intensity: np.ndarray, max_distance: float, min_thickness: float) -> np.ndarray:
    """
    Convert intensity values (0-255) to thickness values
    Uses linear mapping where:
    - intensity 0 (black) = max_distance (fully blocks light)
    - intensity 255 (white) = min_thickness (maximum light transmission)
    """
    # Normalize intensity to [0, 1]
    normalized = intensity / 255.0
    
    # Linear interpolation between min_thickness and max_distance
    thickness = (1 - normalized) * (max_distance - min_thickness) + min_thickness
    
    return thickness

def extract_and_invert_channels_linear(img: ImageAnalyzer, config: StlConfig) -> IntensityChannels:
    c_channel = normalize_thickness_linear(img.pixelated[:, :, 0], config.layer_heights[LayerType.CYAN], config.layer_mins[LayerType.CYAN])
    y_channel = normalize_thickness_linear(img.pixelated[:, :, 1], config.layer_heights[LayerType.YELLOW], config.layer_mins[LayerType.YELLOW])
    m_channel = normalize_thickness_linear(img.pixelated[:, :, 2], config.layer_heights[LayerType.MAGENTA], config.layer_mins[LayerType.MAGENTA])
    
    # For intensity map, use average of RGB then scale to KEY layer heights
    avg_pixels = (img.pixelated[:, :, 0] + img.pixelated[:, :, 1] + img.pixelated[:, :, 2]) / 3.0
    intensity_map = normalize_thickness_linear(avg_pixels, config.layer_heights[LayerType.KEY], config.layer_mins[LayerType.KEY])
    
    return IntensityChannels(
        c_channel=c_channel,
        y_channel=y_channel,
        m_channel=m_channel,
        intensity_map=intensity_map
    ) 