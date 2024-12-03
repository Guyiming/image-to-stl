import numpy as np
from Models import LayerType, IntensityChannels, LuminanceConfig, StlConfig, FilamentProperties
from ImageAnalyzer import ImageAnalyzer
from dataclasses import dataclass
from typing import Dict, Tuple
from pydantic import BaseModel, Field



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
    
    return (max(r,0), max(g,0), max(b,0))

def calculate_color_thicknesses(
    target_rgb: np.ndarray,
    filaments: Dict[LayerType, FilamentProperties],
    luminance_config: LuminanceConfig,
) -> Tuple[float, float, float, float]:
    """
    Calculate thicknesses for CMY layers using CMYK conversion and filament properties.
    Returns (cyan, magenta, yellow, black) thicknesses.
    """
    # Convert RGB [0-255] to [0-1] scale
    rgb = target_rgb / 255.0
    
    # Convert RGB to CMYK
    k = 1 - np.max(rgb)
    c = (1 - rgb[0] - k) / (1 - k + 1e-10)  # Add small epsilon to prevent division by zero
    m = (1 - rgb[1] - k) / (1 - k + 1e-10)
    y = (1 - rgb[2] - k) / (1 - k + 1e-10)
    
    # Get filament properties
    f_cyan = filaments[LayerType.CYAN]
    f_magenta = filaments[LayerType.MAGENTA]
    f_yellow = filaments[LayerType.YELLOW]
    f_white = filaments[LayerType.WHITE]
    
    # Calculate base thicknesses using CMYK values
    target_thickness_cym = luminance_config.cym_target_thickness  # Base target thickness, can be adjusted
    target_thickness_white = luminance_config.white_target_thickness  # Base target thickness, can be adjusted
    
    # Calculate K thickness
    
    # Scale thicknesses by filament properties and transmission distance
    cyan_thickness = c * target_thickness_cym * f_cyan.transmission_distance
    magenta_thickness = m * target_thickness_cym * f_magenta.transmission_distance
    yellow_thickness = y * target_thickness_cym * f_yellow.transmission_distance
    white_thickness = k * target_thickness_white * f_white.transmission_distance
    
    # Apply K (black) component to all layers
    darkness_boost = k * 0.3  # Adjust factor as needed
    cyan_thickness *= (1.0 + darkness_boost)
    magenta_thickness *= (1.0 + darkness_boost)
    yellow_thickness *= (1.0 + darkness_boost)
    white_thickness *= (1.0 + darkness_boost)
    # Clip to physical constraints
    cyan_thickness = np.clip(cyan_thickness, 0, f_cyan.transmission_distance)
    magenta_thickness = np.clip(magenta_thickness, 0, f_magenta.transmission_distance)
    yellow_thickness = np.clip(yellow_thickness, 0, f_yellow.transmission_distance)
    white_thickness = np.clip(white_thickness, 0, f_white.transmission_distance)
    
    return cyan_thickness, magenta_thickness, yellow_thickness, white_thickness

def calculate_white_thickness(
    target_rgb: np.ndarray,
    filaments: Dict[LayerType, FilamentProperties],
    luminance_config: LuminanceConfig
) -> float:
    """Calculate white layer thickness based on luminance"""
    # Calculate perceived brightness
    brightness = (0.299 * target_rgb[0] + 
                 0.587 * target_rgb[1] + 
                 0.114 * target_rgb[2]) / 255.0
    
    # Calculate saturation
    max_rgb = np.max(target_rgb)
    min_rgb = np.min(target_rgb)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-10)
    
    # Get white filament properties
    f_white = filaments[LayerType.WHITE]
    max_white_thickness = f_white.transmission_distance * (1 - luminance_config.target_max_luminance)
    
    # Calculate white thickness
    white_thickness = ((1.0 - brightness) * 
                      (1.0 - saturation * 0.8) *  # Reduce white more in saturated areas
                      max_white_thickness)
    
    # Add extra white for very dark colors
    if brightness < 0.2:
        white_thickness *= 1.5  # Boost dark areas
    
    return np.clip(white_thickness, 0, max_white_thickness)

def calculate_exact_thicknesses(
    target_rgb: np.ndarray,
    filaments: Dict[LayerType, FilamentProperties],
    luminance_config: LuminanceConfig
) -> Tuple[float, float, float, float]:
    """Calculate all layer thicknesses"""
    c, m, y, k = calculate_color_thicknesses(target_rgb, filaments, luminance_config)
    #w = calculate_white_thickness(target_rgb, filaments, luminance_config)
    return c, m, y, k

def extract_and_invert_channels(img: ImageAnalyzer, config: StlConfig) -> IntensityChannels:
    """Process entire image"""
    shape = img.pixelated.shape[:2]
    c_channel = np.zeros(shape)
    y_channel = np.zeros(shape)
    m_channel = np.zeros(shape)
    w_channel = np.zeros(shape)
    
    # Process each pixel
    for i in range(shape[0]):
        for j in range(shape[1]):
            c, m, y, w = calculate_exact_thicknesses(
                img.pixelated[i,j], 
                config.filament_library, 
                config.luminance_config
            )
            c_channel[i,j] = c
            m_channel[i,j] = m
            y_channel[i,j] = y
            w_channel[i,j] = w
    
    return IntensityChannels(
        c_channel=c_channel,
        y_channel=y_channel,
        m_channel=m_channel,
        intensity_map=w_channel
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
    c_channel = normalize_thickness_linear(img.pixelated[:, :, 0], config.filament_library[LayerType.CYAN].transmission_distance, 0)
    y_channel = normalize_thickness_linear(img.pixelated[:, :, 1], config.filament_library[LayerType.YELLOW].transmission_distance, 0) 
    m_channel = normalize_thickness_linear(img.pixelated[:, :, 2], config.filament_library[LayerType.MAGENTA].transmission_distance, 0)
    
    # For intensity map, use average of RGB then scale to KEY layer heights
    avg_pixels = (img.pixelated[:, :, 0] + img.pixelated[:, :, 1] + img.pixelated[:, :, 2]) / 3.0
    intensity_map = normalize_thickness_linear(avg_pixels, config.filament_library[LayerType.WHITE].transmission_distance, config.intensity_min_height)
    
    return IntensityChannels(
        c_channel=c_channel,
        y_channel=y_channel,
        m_channel=m_channel,
        intensity_map=intensity_map
    ) 