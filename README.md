# Color Lithophane project

This project helps you prepare and slice 3D Color Lithophanes.

A couple of points to consider:
* Simple CMYK colors and simple subtractive mixing
* Support both 0.2mm and 0.4mm nozzle side

## Examples

![Example 1](./examples/example_printed.png)

## Setup

1. Initialize UV environment:

```bash
uv venv
source .venv/bin/activate # For Unix/MacOS
```

## Running the Project

Execute the main script from the UV environment:

```bash
python src/main.py
```

## Options for generation

```bash
$ python src/main.py  --help
usage: main.py [-h] [--show-images] [--input INPUT] [--output-image OUTPUT_IMAGE] [--width WIDTH] [--resolution RESOLUTION]
               [--stl-output STL_OUTPUT]

Process an image into CMYK 3D printable layers

options:
  -h, --help            show this help message and exit
  --show-images         Display the original and processed images
  --input, -i INPUT     Input image file path
  --output-image, -o OUTPUT_IMAGE
                        Output pixelated image file path
  --width, -w WIDTH     Desired width in mm
  --resolution, -r RESOLUTION
                        Resolution in mm per pixel/block
  --stl-output STL_OUTPUT
                        Output directory for STL files
```



## Bambu Studio Setup

1. Launch Bambu Studio
2. File → Import → Select example files (supports .stl, .obj, .3mf)
3. Select all 4 generated files for import
   - Click "Yes" when prompted to load all files as a single object with multiple parts.

### Print Settings
Configure the following settings for optimal results:

- Nozzle diameter: 0.4mm
- First layer height: 0.2mm
- Subsequent layer heights: 0.1mm

## Project TODO

### Mixing algorithm
- [x] Implement transmission distance model
- [ ] Implement more optimal subtractive color mixing model

### Usability
- [ ] Add yaml config for filament parameters
- [ ] Replace resolution with nozzle sizes (0.2mm and 0.4mm)
- [ ] Implement error handling for invalid input images
- [ ] Add progress bar for long operations
- [ ] Optimize memory usage for large images


### Communication
- [ ] Better documentation for parameters
- [ ] Create library of commonly used filaments
- [ ] Create example gallery with sample prints
- [ ] Add calibration pattern generation
- [ ] Create troubleshooting guide

### General
- [ ] Add test suite for core functionality
- [ ] Add support for web server hosting
- [ ] Refactor/clean-up code