#!/usr/bin/python3

from ImageAnalyzer import ImageAnalyzer
from to_stl import to_stl_cym
import os
import cv2

# SHOW IMAGES
show_images = False 
# SHOW IMAGES

ifile = 'data/girl-with-pearl-earings.png'
ofile = 'data/girl-with-pearl-earings_pixelated.png'
x_len = 20 #mm
resolution = float(0.4)

if __name__ == '__main__':
        
    img = ImageAnalyzer(ifile)
    img_info = img.get_image_info()
    x_pixels = img_info.get('width')
    y_pixels = img_info.get('height')
    y_len = int(float(y_pixels) * float(x_pixels) ) * x_len

    n_blocks =  int(x_len / resolution)
    block_size = int(x_pixels / n_blocks)

    img.pixelate(block_size)

    img.save_processed_image(ofile)
    # Display the original and processed images
    if show_images:
        cv2.imshow('Original Image', cv2.cvtColor(img.original, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.imshow('Processed Image', cv2.cvtColor(img.pixelated, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    base_mesh, cyan_mesh, yellow_mesh, magenta_mesh, clear_mesh = to_stl_cym(
        img,
        pixel_size=1.0,
        base_height=0.2,
        max_height=0.2,
        clear_height=0.6
    )

    # Save each mesh separately
    base_mesh.save('./data/girl-w-pearl/base_layer.stl')
    cyan_mesh.save('./data/girl-w-pearl/cyan_layer.stl')
    yellow_mesh.save('./data/girl-w-pearl/yellow_layer.stl')
    magenta_mesh.save('./data/girl-w-pearl/magenta_layer.stl')
    clear_mesh.save('./data/girl-w-pearl/clear_layer.stl')
