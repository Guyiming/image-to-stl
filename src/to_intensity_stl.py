# Import numpy for array operations
from ImageAnalyzer import ImageAnalyzer
import numpy as np
from stl.mesh import Mesh
import cv2
from typing import Tuple


def to_stl_triange(img: ImageAnalyzer, resolution: float, base_height: float = 0.5, max_height: float = 3.0, height_step_mm: float = 0.0) -> Mesh:
    # Convert to grayscale if needed
    if len(img.pixelated.shape) == 3:
        height_map = cv2.cvtColor(img.pixelated, cv2.COLOR_RGB2GRAY)
    else:
        height_map = img.pixelated

    # Invert the height map (darker areas become higher)
    height_map = 255 - height_map

    vertices = []
    faces = []
    y_pixels, x_pixels = height_map.shape
    
    # Add a small buffer around the edges
    padded_height_map = np.pad(height_map, pad_width=1, mode='edge')
    
    # Create vertices for top surface (including edges)
    for y in range(y_pixels + 1):
        for x in range(x_pixels + 1):
            # Calculate initial height value
            height_value = float(padded_height_map[y, x]) / 255.0
            z = base_height + (height_value * (max_height - base_height))
            
            # If step size is specified, round to nearest step
            if height_step_mm > 0:
                z = base_height + (round((z - base_height) / height_step_mm) * height_step_mm)
                
            vertices.append([x * resolution, y * resolution, z])  # Top vertices
            vertices.append([x * resolution, y * resolution, 0])  # Bottom vertices
    
    vertices_per_row = (x_pixels + 1) * 2
    
    # Create faces for top surface
    for y in range(y_pixels):
        for x in range(x_pixels):
            # Calculate vertex indices for this quad (top surface)
            v0 = y * vertices_per_row + (x * 2)
            v1 = v0 + 2
            v2 = (y + 1) * vertices_per_row + (x * 2)
            v3 = v2 + 2
            
            # Create two triangles for top surface
            faces.append([v0, v1, v2])
            faces.append([v2, v1, v3])
            
            # Create two triangles for bottom surface (reversed orientation)
            v0b = v0 + 1
            v1b = v1 + 1
            v2b = v2 + 1
            v3b = v3 + 1
            faces.append([v0b, v2b, v1b])
            faces.append([v2b, v3b, v1b])
            
            # Create side faces for front and back if at edges
            if y == 0:  # Front edge
                faces.append([v0b, v1b, v0])
                faces.append([v0, v1b, v1])
            if y == y_pixels - 1:  # Back edge
                faces.append([v2, v3, v2b])
                faces.append([v2b, v3, v3b])
            
            # Create side faces for left and right if at edges
            if x == 0:  # Left edge
                faces.append([v0b, v0, v2b])
                faces.append([v2b, v0, v2])
            if x == x_pixels - 1:  # Right edge
                faces.append([v1, v1b, v3])
                faces.append([v3, v1b, v3b])
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create the mesh
    stl_mesh = Mesh(np.zeros(len(faces), dtype=Mesh.dtype))
    
    # Transfer faces to the mesh with proper normal calculation
    for i, face in enumerate(faces):
        # Get vertices for this face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Calculate normal vector
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        if np.any(normal):  # Avoid zero-length normals
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.array([0, 0, 1])
        
        # Assign vertices and normal to the mesh
        stl_mesh.vectors[i] = np.array([v0, v1, v2])
        stl_mesh.normals[i] = normal
    
    return stl_mesh