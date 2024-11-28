from dataclasses import dataclass
import numpy as np
from stl.mesh import Mesh
from typing import Dict, Tuple
from pydantic import BaseModel, Field
from enum import Enum

class LayerType(Enum):
    CYAN = "cyan"
    YELLOW = "yellow"
    MAGENTA = "magenta"
    KEY = "key"  # For intensity/black
    BASE = "base"

class IntensityChannels(BaseModel):
    c_channel: np.ndarray
    y_channel: np.ndarray
    m_channel: np.ndarray
    intensity_map: np.ndarray

    model_config = {
        'arbitrary_types_allowed': True
    }

class StlConfig(BaseModel):
    """Configuration for STL generation"""
    base_height: float = Field(default=0.2, description="Height of the base plate in mm")
    pixel_size: float = Field(default=1.0, description="Size of each pixel in mm")
    height_step_mm: float = Field(default=0.0, description="Height quantization step. 0 for continuous height")
    layer_heights: Dict[LayerType, float] = Field(
        default_factory=lambda: {
            LayerType.CYAN: 1.6,
            LayerType.YELLOW: 1.6,
            LayerType.MAGENTA: 1.6,
            LayerType.KEY: 1.6,
        },
        description="Maximum height for each layer in mm"
    )
    layer_mins: Dict[LayerType, float] = Field(
        default_factory=lambda: {
            LayerType.CYAN: 0,
            LayerType.YELLOW: 0,
            LayerType.MAGENTA: 0,
            LayerType.KEY: 0,
        },
    )
    filament_colors: Dict[LayerType, Tuple[float, float, float]] = Field(
        default_factory=lambda: {
            LayerType.CYAN: (1, 255, 255),     # RGB for Cyan
            LayerType.YELLOW: (255, 255, 1),   # RGB for Yellow
            LayerType.MAGENTA: (255, 1, 255),  # RGB for Magenta
            LayerType.KEY: (255, 255, 255),          # RGB for White
        },
        description="RGB values for each filament color"
    )

@dataclass
class FilamentProperties:
    """Properties of a specific filament color"""
    rgb: Tuple[float, float, float]  # Measured RGB values (0-1)
    max_distance: float              # Maximum transmission distance
    min_thickness: float            # Minimum printable thickness