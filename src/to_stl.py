import os
from ImageAnalyzer import ImageAnalyzer
import numpy as np
from stl.mesh import Mesh
from typing import Dict, Tuple
from pydantic import BaseModel
import numpy as np

class IntensityChannels(BaseModel):
    c_channel: np.ndarray
    y_channel: np.ndarray
    m_channel: np.ndarray
    intensity_map: np.ndarray

    model_config = {
        'arbitrary_types_allowed': True
    }


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
                      previous_heights: np.ndarray = None, 
                      pixel_size: float = 1.0, 
                      max_height: float = 1.6, 
                      height_step_mm: float = 0.0,
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
            # Get starting height from previous layer (or 0 if first layer)
            start_height = previous_heights[y, x] if previous_heights is not None else 0
            
            # Calculate height
            if flat_top:
                #current_intensity_height = float(height_map[y, x]) / 255.0 * max_height
                z = target_height - start_height
            else:
                height_value = float(height_map[y, x]) / 255.0
                z = height_value * max_height
                if height_step_mm > 0:
                    z = round(z / height_step_mm) * height_step_mm
            
            # Add start_height to z for stacking
            z += start_height
            
            # Store this height for the next layer
            next_heights[y, x] = z
            
            # Define vertices for cube
            v0 = [x * pixel_size, y * pixel_size, start_height]  # bottom
            v1 = [(x + 1) * pixel_size, y * pixel_size, start_height]
            v2 = [(x + 1) * pixel_size, (y + 1) * pixel_size, start_height]
            v3 = [x * pixel_size, (y + 1) * pixel_size, start_height]
            v4 = [x * pixel_size, y * pixel_size, z]  # top
            v5 = [(x + 1) * pixel_size, y * pixel_size, z]
            v6 = [(x + 1) * pixel_size, (y + 1) * pixel_size, z]
            v7 = [x * pixel_size, (y + 1) * pixel_size, z]
            
            # Add vertices
            base_idx = len(vertices)
            vertices.extend([v0, v1, v2, v3, v4, v5, v6, v7])
            
            # Define faces (12 triangles = 6 sides)
            faces.extend([
                # Bottom face
                [base_idx + 0, base_idx + 2, base_idx + 1],
                [base_idx + 0, base_idx + 3, base_idx + 2],
                # Top face
                [base_idx + 4, base_idx + 5, base_idx + 6],
                [base_idx + 4, base_idx + 6, base_idx + 7],
                # Front face
                [base_idx + 0, base_idx + 1, base_idx + 5],
                [base_idx + 0, base_idx + 5, base_idx + 4],
                # Back face
                [base_idx + 2, base_idx + 3, base_idx + 7],
                [base_idx + 2, base_idx + 7, base_idx + 6],
                # Left face
                [base_idx + 0, base_idx + 4, base_idx + 7],
                [base_idx + 0, base_idx + 7, base_idx + 3],
                # Right face
                [base_idx + 1, base_idx + 2, base_idx + 6],
                [base_idx + 1, base_idx + 6, base_idx + 5]
            ])

    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create the mesh
    stl_mesh = Mesh(np.zeros(len(faces), dtype=Mesh.dtype))
    
    # Transfer faces to the mesh with proper normal calculation
    for i, face in enumerate(faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        if np.any(normal):  # Avoid zero-length normals
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.array([0, 0, 1])
        
        stl_mesh.vectors[i] = np.array([v0, v1, v2])
        stl_mesh.normals[i] = normal
    
    return stl_mesh, next_heights

def create_base_plate(x_pixels: int, y_pixels: int, pixel_size: float, base_height: float) -> Mesh:
    height_map = np.full((y_pixels, x_pixels), 255, dtype=np.uint8)
    
    base_mesh, _ = create_layer_mesh(
        height_map,
        previous_heights=np.zeros((y_pixels, x_pixels)),
        pixel_size=pixel_size,
        max_height=base_height,
        height_step_mm=0.0
    )
    
    return base_mesh

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


def to_stl_cym(img: ImageAnalyzer, pixel_size: float = 1.0, max_height: float = 1.6, 
               height_step_mm: float = 0.0, base_height: float = 0.2, intensity_height: float = 1.6 ) -> Tuple[Mesh, Mesh, Mesh, Mesh, Mesh]:
    # Extract CYM channels
    if len(img.pixelated.shape) != 3 or img.pixelated.shape[2] != 3:
        raise ValueError("Image must have 3 channels (CYM)")

    intensity_channels = extract_and_invert_channels(img)

    y_pixels, x_pixels = img.pixelated.shape[:2]
    
    base_mesh = create_base_plate(x_pixels, y_pixels, pixel_size, base_height)
    cyan_mesh, cyan_heights = create_layer_mesh(intensity_channels.c_channel, previous_heights=np.full_like(intensity_channels.c_channel, base_height, dtype=float), pixel_size=pixel_size, max_height=max_height, height_step_mm=height_step_mm)
    yellow_mesh, yellow_heights = create_layer_mesh(intensity_channels.y_channel, previous_heights=cyan_heights, pixel_size=pixel_size, max_height=max_height, height_step_mm=height_step_mm)
    magenta_mesh, magenta_heights = create_layer_mesh(intensity_channels.m_channel, previous_heights=yellow_heights, pixel_size=pixel_size, max_height=max_height, height_step_mm=height_step_mm)
    intensity_mesh, _ = create_layer_mesh(intensity_channels.intensity_map, previous_heights=magenta_heights, pixel_size=pixel_size, max_height=intensity_height, height_step_mm=height_step_mm)    

    return StlCollection(meshes={
        'base_mesh': base_mesh,
        'cyan_mesh': cyan_mesh,
        'yellow_mesh': yellow_mesh,
        'magenta_mesh': magenta_mesh,
        'intensity_map': intensity_mesh
    })