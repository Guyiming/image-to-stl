import cv2
import numpy as np
import os
from typing import Tuple, Dict, List, Set

class ImageAnalyzer:
    # Supported file extensions (lowercase)
    SUPPORTED_FORMATS: Set[str] = {
        '.jpg', '.jpeg', '.png', '.bmp', 
        '.tiff', '.tif', '.webp', 
        '.ppm', '.pgm', '.pbm',
        '.sr', '.ras'
    }

    def __init__(self, image_path: str):
        """
        Initialize with an image path.
        Raises:
            ValueError: If file format is not supported or file cannot be loaded
            FileNotFoundError: If the image file doesn't exist
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Check file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_ext}\n"
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Try to load the image
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image: {image_path}. File may be corrupted.")

        # Convert BGR to RGB
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        
        # Store image info
        self.height, self.width = self.original.shape[:2]
        self.image_path = image_path

    @classmethod
    def is_supported_format(cls, filepath: str) -> bool:
        """Check if the file format is supported."""
        return os.path.splitext(filepath)[1].lower() in cls.SUPPORTED_FORMATS

    def get_image_info(self) -> Dict:
        """Return basic information about the loaded image."""
        return {
            'path': self.image_path,
            'width': self.width,
            'height': self.height,
            'aspect_ratio': self.width / self.height,
            'size_bytes': os.path.getsize(self.image_path),
            'format': os.path.splitext(self.image_path)[1].lower()
        }
    
    def pixelate(self, block_size: int) -> np.ndarray:
        """
        Pixelate the image using the specified block size.
        Returns the pixelated image at reduced dimensions.
        """
        # Validate block size
        if block_size <= 0:
            raise ValueError("Block size must be positive")
        if block_size > min(self.height, self.width):
            raise ValueError("Block size too large for image dimensions")
        
        # Calculate new dimensions
        new_width = self.width // block_size
        new_height = self.height // block_size
        
        # Resize down to create pixelation effect
        self.pixelated = cv2.resize(self.original, (new_width, new_height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Update instance dimensions to match new size
        self.width = new_width
        self.height = new_height
        
        return self.pixelated
    
    def analyze_pixels(self) -> List[Dict]:
        """
        Analyze each unique pixel in the pixelated image.
        Returns list of dictionaries containing color and intensity information.
        Raises:
            RuntimeError: If called before pixelate()
        """
        if not hasattr(self, 'pixelated'):
            raise RuntimeError("Must call pixelate() before analyzing pixels")

        # Get unique pixels and their counts
        pixels = self.pixelated.reshape(-1, 3)
        unique_pixels, counts = np.unique(pixels, axis=0, return_counts=True)
        
        results = []
        for pixel, count in zip(unique_pixels, counts):
            r, g, b = pixel
            # Calculate intensity (brightness) - using standard luminance formula
            intensity = 0.299 * r + 0.587 * g + 0.114 * b
            
            results.append({
                'color': (int(r), int(g), int(b)),
                'intensity': float(intensity),
                'frequency': int(count),
                'percentage': float(count / len(pixels) * 100)
            })
        
        # Sort by frequency
        return sorted(results, key=lambda x: x['frequency'], reverse=True)
    
    def get_intensity_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate intensity histogram data.
        Returns (bin_edges, histogram_values).
        Raises:
            RuntimeError: If called before pixelate()
        """
        if not hasattr(self, 'pixelated'):
            raise RuntimeError("Must call pixelate() before getting histogram")

        # Convert to grayscale for intensity analysis
        gray = cv2.cvtColor(self.pixelated, cv2.COLOR_RGB2GRAY)
        
        # Calculate histogram with 256 bins (0-255 intensity levels)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        bins = np.arange(257)  # bin edges
        
        return bins, hist.flatten()
    
    def get_channel_histograms(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate histograms for each color channel.
        Returns dictionary with RGB channel histograms.
        Raises:
            RuntimeError: If called before pixelate()
        """
        if not hasattr(self, 'pixelated'):
            raise RuntimeError("Must call pixelate() before getting channel histograms")

        channels = {'red': 0, 'green': 1, 'blue': 2}
        results = {}
        
        for channel_name, channel_idx in channels.items():
            hist = cv2.calcHist([self.pixelated], [channel_idx], None, [256], [0, 256])
            bins = np.arange(257)
            results[channel_name] = (bins, hist.flatten())
            
        return results

    def save_processed_image(self, output_path: str):
        """
        Save the processed (pixelated) image.
        Raises:
            RuntimeError: If called before pixelate()
            ValueError: If output format is not supported
        """
        if not hasattr(self, 'pixelated'):
            raise RuntimeError("Must call pixelate() before saving")

        # Check output format
        output_ext = os.path.splitext(output_path)[1].lower()
        if output_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported output format: {output_ext}")

        # Convert back to BGR for saving
        output = cv2.cvtColor(self.pixelated, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(output_path, output)
        
        if not success:
            raise RuntimeError(f"Failed to save image to {output_path}")
