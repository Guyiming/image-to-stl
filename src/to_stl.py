import os
from ImageAnalyzer import ImageAnalyzer
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
        description="Maximum height for each layer in mm"
    )

def extract_and_invert_channels(img: ImageAnalyzer) -> IntensityChannels:
    c_channel = 255 - img.pixelated[:, :, 0]
    y_channel = 255 - img.pixelated[:, :, 1]
    m_channel = 255 - img.pixelated[:, :, 2]
    intensity_map = (c_channel + y_channel + m_channel) / 3.0
    return IntensityChannels(
        c_channel=c_channel,
        y_channel=y_channel,
        m_channel=m_channel,
        intensity_map=intensity_map
    )

def create_layer_mesh(height_map: np.ndarray,
                     height_step_mm: float,
                     pixel_size: float,
                     previous_heights: np.ndarray = None,
                     override_max_height: float = None,
                     min_layer_height: float = 0,
                     flat_top: bool = False) -> Tuple[Mesh, np.ndarray]:
    vertices = []
    faces = []
    y_pixels, x_pixels = height_map.shape
    
    next_heights = np.zeros_like(height_map, dtype=float)
    
    if flat_top:
        max_intensity_height = np.max(previous_heights)
        target_height = max_intensity_height + height_step_mm
    
    for y in range(y_pixels):
        for x in range(x_pixels):
            start_height = previous_heights[y, x] if previous_heights is not None else 0
            
            if flat_top:
                z = target_height - start_height
            else:
                height_value = float(height_map[y, x]) / 255.0
                z = height_value * override_max_height
                if height_step_mm > 0:
                    z = round(z / height_step_mm) * height_step_mm
                z = max(min_layer_height, z)
            
            z += start_height
            next_heights[y, x] = z
            
            v0 = [x * pixel_size, y * pixel_size, start_height]
            v1 = [(x + 1) * pixel_size, y * pixel_size, start_height]
            v2 = [(x + 1) * pixel_size, (y + 1) * pixel_size, start_height]
            v3 = [x * pixel_size, (y + 1) * pixel_size, start_height]
            v4 = [x * pixel_size, y * pixel_size, z]
            v5 = [(x + 1) * pixel_size, y * pixel_size, z]
            v6 = [(x + 1) * pixel_size, (y + 1) * pixel_size, z]
            v7 = [x * pixel_size, (y + 1) * pixel_size, z]
            
            base_idx = len(vertices)
            vertices.extend([v0, v1, v2, v3, v4, v5, v6, v7])
            
            faces.extend([
                [base_idx + 0, base_idx + 2, base_idx + 1],
                [base_idx + 0, base_idx + 3, base_idx + 2],
                [base_idx + 4, base_idx + 5, base_idx + 6],
                [base_idx + 4, base_idx + 6, base_idx + 7],
                [base_idx + 0, base_idx + 1, base_idx + 5],
                [base_idx + 0, base_idx + 5, base_idx + 4],
                [base_idx + 2, base_idx + 3, base_idx + 7],
                [base_idx + 2, base_idx + 7, base_idx + 6],
                [base_idx + 0, base_idx + 4, base_idx + 7],
                [base_idx + 0, base_idx + 7, base_idx + 3],
                [base_idx + 1, base_idx + 2, base_idx + 6],
                [base_idx + 1, base_idx + 6, base_idx + 5]
            ])

    vertices = np.array(vertices)
    faces = np.array(faces)
    
    stl_mesh = Mesh(np.zeros(len(faces), dtype=Mesh.dtype))
    
    for i, face in enumerate(faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        if np.any(normal):
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.array([0, 0, 1])
        
        stl_mesh.vectors[i] = np.array([v0, v1, v2])
        stl_mesh.normals[i] = normal
    
    return stl_mesh, next_heights

def create_base_plate(x_pixels: int, y_pixels: int, config: StlConfig) -> Mesh:
    height_map = np.full((y_pixels, x_pixels), 255, dtype=np.uint8)
    
    base_mesh, _ = create_layer_mesh(
        height_map=height_map,
        height_step_mm=config.height_step_mm,
        pixel_size=config.pixel_size,
        previous_heights=np.zeros((y_pixels, x_pixels)),
        override_max_height=config.base_height
    )
    
    return base_mesh

def create_color_layer(height_map: np.ndarray, 
                      previous_heights: np.ndarray,
                      config: StlConfig,
                      layer_type: LayerType) -> Tuple[Mesh, np.ndarray]:
    return create_layer_mesh(
        height_map=height_map,
        height_step_mm=config.height_step_mm,
        pixel_size=config.pixel_size,
        previous_heights=previous_heights,
        min_layer_height = config.layer_mins[layer_type],
        override_max_height=config.layer_heights[layer_type]
    )

class StlCollection(BaseModel):
    meshes: Dict[str, Mesh]
    
    class Config:
        arbitrary_types_allowed = True
    
    def __getitem__(self, key: str) -> Mesh:
        return self.meshes[key]
    
    def __iter__(self):
        return iter(self.meshes.values())
    
    def items(self):
        return self.meshes.items()

    def save_to_folder(self, output_dir):
        for k,v in self.meshes.items():
            v.save(os.path.join(output_dir, f"{k}.stl"))

def to_stl_cym(img: ImageAnalyzer, config: StlConfig = None) -> StlCollection:
    if config is None:
        config = StlConfig()
        
    if len(img.pixelated.shape) != 3 or img.pixelated.shape[2] != 3:
        raise ValueError("Image must have 3 channels (CYM)")

    intensity_channels = extract_and_invert_channels(img)
    y_pixels, x_pixels = img.pixelated.shape[:2]
    
    print("creating stl: white_base_mesh.stl")
    base_mesh = create_base_plate(x_pixels, y_pixels, config)
    base_heights = np.full_like(intensity_channels.c_channel, config.base_height, dtype=float)
    
    layers = {
        'cyan_mesh': (intensity_channels.c_channel, base_heights, LayerType.CYAN),
        'yellow_mesh': (intensity_channels.y_channel, None, LayerType.YELLOW),
        'magenta_mesh': (intensity_channels.m_channel, None, LayerType.MAGENTA),
        'white_intensity_mesh': (intensity_channels.intensity_map, None, LayerType.KEY)
    }
    
    previous_heights = base_heights
    meshes = {'white_base_mesh': base_mesh}
    
    for name, (height_map, _, layer_type) in layers.items():
        print("creating stl: " + name + ".stl")
        mesh, previous_heights = create_color_layer(
            height_map=height_map,
            previous_heights=previous_heights,
            config=config,
            layer_type=layer_type
        )
        meshes[name] = mesh
    
    return StlCollection(meshes=meshes)