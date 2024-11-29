from ImageAnalyzer import ImageAnalyzer
import numpy as np
from stl.mesh import Mesh
from typing import Tuple
from Models import ColorCorrection, LayerType, StlConfig, StlCollection
from color_mixing import extract_and_invert_channels, extract_and_invert_channels_linear

def create_layer_mesh(height_map: np.ndarray,
                     height_step_mm: float,
                     pixel_size: float,
                     previous_heights: np.ndarray = None,
                     min_height: float = 0,
                     flat_top: bool = False,
                     ) -> Tuple[Mesh, np.ndarray]:
    vertices = []
    faces = []
    y_pixels, x_pixels = height_map.shape
    
    next_heights = np.zeros_like(height_map, dtype=float)
    
    # Calculate the target height for flat top
    max_height = np.max(previous_heights) + min_height if flat_top else 0
    
    for y in range(y_pixels):
        for x in range(x_pixels):
            start_height = previous_heights[y, x] if previous_heights is not None else 0
            
            if flat_top:
                # Calculate the height needed to reach max_height
                z = max_height - previous_heights[y, x]
            else:
                z = height_map[y, x]
                if height_step_mm > 0:
                    z = round(z / height_step_mm) * height_step_mm
                z = max(z, min_height)
            
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
    height_map = np.full((y_pixels, x_pixels), config.base_height, dtype=float)
    
    base_mesh, _ = create_layer_mesh(
        height_map=height_map,
        height_step_mm=config.height_step_mm,
        pixel_size=config.pixel_size,
        previous_heights=np.zeros((y_pixels, x_pixels))
    )
    
    return base_mesh

def create_color_layer(height_map: np.ndarray, 
                      previous_heights: np.ndarray,
                      config: StlConfig,
                      layer_type: LayerType,
                      flat_top: bool = False) -> Tuple[Mesh, np.ndarray]:
    return create_layer_mesh(
        height_map=height_map,
        height_step_mm=config.height_step_mm,
        pixel_size=config.pixel_size,
        previous_heights=previous_heights,
        min_height=config.intensity_min_height if layer_type == LayerType.WHITE else 0,
        flat_top=flat_top
      
    )


def to_stl_cym(img: ImageAnalyzer, config: StlConfig = None) -> StlCollection:
    if config is None:
        config = StlConfig()
        
    if len(img.pixelated.shape) != 3 or img.pixelated.shape[2] != 3:
        raise ValueError("Image must have 3 channels (CYM)")

    intensity_channels = extract_and_invert_channels(img, config) if config.color_correction == ColorCorrection.LUMINANCE else extract_and_invert_channels_linear(img, config)
    y_pixels, x_pixels = img.pixelated.shape[:2]
    
    print("creating stl: white_base_mesh.stl")
    base_mesh = create_base_plate(x_pixels, y_pixels, config)
    base_heights = np.full((y_pixels, x_pixels), config.base_height, dtype=float)
    print("base_heights " + str(config.base_height))

    layers = {
        'cyan_mesh': (intensity_channels.c_channel, base_heights, LayerType.CYAN),
        'yellow_mesh': (intensity_channels.y_channel, None, LayerType.YELLOW),
        'magenta_mesh': (intensity_channels.m_channel, None, LayerType.MAGENTA),
        'clear_mesh': (intensity_channels.intensity_map, None, LayerType.CLEAR),
        'white_intensity_mesh': (intensity_channels.intensity_map, None, LayerType.WHITE)
    }
    
    previous_heights = base_heights
    meshes = {'white_base_mesh': base_mesh}
    
    for name, (height_map, _, layer_type) in layers.items():
        print("creating stl: " + name + ".stl")
        mesh, previous_heights = create_color_layer(
            height_map=height_map,
            previous_heights=previous_heights,
            config=config,
            layer_type=layer_type,
            flat_top=layer_type == LayerType.CLEAR,
        )
        meshes[name] = mesh
    
    return StlCollection(meshes=meshes)