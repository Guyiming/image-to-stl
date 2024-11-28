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
    WHITE = "white"  # For intensity/black
    BASE = "base"

class Filament(BaseModel):
    manufacturer: str
    filament_type: str
    color_name: str
    hex_value: str
    transmission_distance: int

@dataclass
class FilamentProperties:
    """Properties of a specific filament color"""
    rgb: Tuple[float, float, float]  # Measured RGB values (0-1)
    max_distance: float              # Maximum transmission distance
    min_thickness: float            # Minimum printable thickness

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
    filament_library: Dict[LayerType, Filament] = Field(default_factory=lambda: {})
