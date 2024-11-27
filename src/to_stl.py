# Import numpy for array operations
from ImageAnalyzer import ImageAnalyzer
import numpy as np
from stl.mesh import Mesh
import cv2


def to_stl_triange(img: ImageAnalyzer, resolution: float, base_height: float = 0.5, max_height: float = 3.0, height_step_mm: float = 0.0):
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


def to_stl_cym(img, pixel_size: float = 1.0, max_height: float = 1.6, 
               height_step_mm: float = 0.0, clear_height: float = 5.0, base_height: float = 0.2):
    # Extract CYM channels
    if len(img.pixelated.shape) != 3 or img.pixelated.shape[2] != 3:
        raise ValueError("Image must have 3 channels (CYM)")
    
    def create_layer_mesh(height_map, previous_heights=None, is_clear=False):
        vertices = []
        faces = []
        y_pixels, x_pixels = height_map.shape
        
        # Store the heights for the next layer
        next_heights = np.zeros_like(height_map, dtype=float)
        
        for y in range(y_pixels):
            for x in range(x_pixels):
                # Get starting height from previous layer (or 0 if first layer)
                start_height = previous_heights[y, x] if previous_heights is not None else 0
                
                # Calculate height
                if is_clear:
                    z = clear_height
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

    def create_base_plate():
        y_pixels, x_pixels = img.pixelated.shape[:2]
        
        # Create vertices for base plate
        vertices = [
            [0, 0, 0],  # bottom
            [x_pixels * pixel_size, 0, 0],
            [x_pixels * pixel_size, y_pixels * pixel_size, 0],
            [0, y_pixels * pixel_size, 0],
            [0, 0, base_height],  # top
            [x_pixels * pixel_size, 0, base_height],
            [x_pixels * pixel_size, y_pixels * pixel_size, base_height],
            [0, y_pixels * pixel_size, base_height]
        ]
        
        # Define faces (12 triangles = 6 sides)
        faces = [
            # Bottom
            [0, 2, 1], [0, 3, 2],
            # Top
            [4, 5, 6], [4, 6, 7],
            # Sides
            [0, 1, 5], [0, 5, 4],  # Front
            [1, 2, 6], [1, 6, 5],  # Right
            [2, 3, 7], [2, 7, 6],  # Back
            [3, 0, 4], [3, 4, 7]   # Left
        ]
        
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        # Create the mesh
        base_mesh = Mesh(np.zeros(len(faces), dtype=Mesh.dtype))
        
        # Transfer faces to the mesh
        for i, face in enumerate(faces):
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            
            base_mesh.vectors[i] = np.array([v0, v1, v2])
            base_mesh.normals[i] = normal
        
        return base_mesh

    # Extract and invert channels
    c_channel = 255 - img.pixelated[:, :, 0]
    y_channel = 255 - img.pixelated[:, :, 1]
    m_channel = 255 - img.pixelated[:, :, 2]
    
    # Create a dummy height map for clear layer (all same height)
    clear_map = np.zeros_like(c_channel)
    
    # Create base plate and layer meshes
    base_mesh = create_base_plate()
    # Start cyan layer at base_height
    cyan_mesh, cyan_heights = create_layer_mesh(c_channel, previous_heights=np.full_like(c_channel, base_height, dtype=float))
    yellow_mesh, yellow_heights = create_layer_mesh(y_channel, previous_heights=cyan_heights)
    magenta_mesh, magenta_heights = create_layer_mesh(m_channel, previous_heights=yellow_heights)
    clear_mesh, _ = create_layer_mesh(clear_map, previous_heights=magenta_heights, is_clear=True)
    
    return base_mesh, cyan_mesh, yellow_mesh, magenta_mesh, clear_mesh