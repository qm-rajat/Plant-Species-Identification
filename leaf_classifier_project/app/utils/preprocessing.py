"""
Data Preprocessing Module for Plant Species Identification
Handles image preprocessing including noise injection and denoising
"""

import cv2
import numpy as np
from skimage import filters, restoration
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class ImagePreprocessor:
    """Handles all image preprocessing operations"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale
        
        Args:
            image: Input RGB image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    def resize_image(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            size: Target size (width, height), uses self.target_size if None
            
        Returns:
            Resized image
        """
        if size is None:
            size = self.target_size
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    def add_gaussian_noise(self, image: np.ndarray, noise_level: str = 'low') -> np.ndarray:
        """
        Add Gaussian noise to image
        
        Args:
            image: Input image
            noise_level: 'low', 'medium', or 'high'
            
        Returns:
            Noisy image
        """
        noise_params = {
            'low': {'mean': 0, 'std': 10},
            'medium': {'mean': 0, 'std': 25},
            'high': {'mean': 0, 'std': 50}
        }
        
        params = noise_params.get(noise_level, noise_params['low'])
        noise = np.random.normal(params['mean'], params['std'], image.shape)
        
        # Add noise and clip values
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def add_impulse_noise(self, image: np.ndarray, noise_level: str = 'low') -> np.ndarray:
        """
        Add salt and pepper noise to image
        
        Args:
            image: Input image
            noise_level: 'low', 'medium', or 'high'
            
        Returns:
            Noisy image
        """
        noise_params = {
            'low': 0.01,
            'medium': 0.05,
            'high': 0.1
        }
        
        prob = noise_params.get(noise_level, noise_params['low'])
        noisy_image = image.copy()
        
        # Salt noise
        salt_coords = tuple([np.random.randint(0, i - 1, int(prob * image.size * 0.5))
                            for i in image.shape])
        noisy_image[salt_coords] = 255
        
        # Pepper noise
        pepper_coords = tuple([np.random.randint(0, i - 1, int(prob * image.size * 0.5))
                              for i in image.shape])
        noisy_image[pepper_coords] = 0
        
        return noisy_image
    
    def denoise_median(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filtering for denoising
        
        Args:
            image: Noisy image
            kernel_size: Size of median filter kernel
            
        Returns:
            Denoised image
        """
        return cv2.medianBlur(image, kernel_size)
    
    def denoise_gaussian(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian filtering for denoising
        
        Args:
            image: Noisy image
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Denoised image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def denoise_bilateral(self, image: np.ndarray, d: int = 9, sigma_color: float = 75, 
                         sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filtering for edge-preserving denoising
        
        Args:
            image: Noisy image
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Denoised image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def preprocess_pipeline(self, image_path: str, add_noise: bool = False, 
                           noise_type: str = 'gaussian', noise_level: str = 'low',
                           denoise: bool = True, denoise_method: str = 'bilateral') -> dict:
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to input image
            add_noise: Whether to add noise
            noise_type: 'gaussian' or 'impulse'
            noise_level: 'low', 'medium', or 'high'
            denoise: Whether to apply denoising
            denoise_method: 'median', 'gaussian', or 'bilateral'
            
        Returns:
            Dictionary containing processed images at different stages
        """
        results = {}
        
        # Load and convert to grayscale
        original = self.load_image(image_path)
        if original is None:
            return None
        
        results['original'] = original
        
        # Convert to grayscale
        gray = self.to_grayscale(original)
        results['grayscale'] = gray
        
        # Resize
        resized = self.resize_image(gray)
        results['resized'] = resized
        
        current_image = resized
        
        # Add noise if requested
        if add_noise:
            if noise_type == 'gaussian':
                current_image = self.add_gaussian_noise(current_image, noise_level)
            elif noise_type == 'impulse':
                current_image = self.add_impulse_noise(current_image, noise_level)
            results['noisy'] = current_image
        
        # Denoise if requested
        if denoise:
            if denoise_method == 'median':
                current_image = self.denoise_median(current_image)
            elif denoise_method == 'gaussian':
                current_image = self.denoise_gaussian(current_image)
            elif denoise_method == 'bilateral':
                current_image = self.denoise_bilateral(current_image)
            results['denoised'] = current_image
        
        results['final'] = current_image
        return results
    
    def visualize_preprocessing(self, results: dict, save_path: Optional[str] = None):
        """
        Visualize preprocessing results
        
        Args:
            results: Results from preprocess_pipeline
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        stages = ['original', 'grayscale', 'resized', 'noisy', 'denoised', 'final']
        titles = ['Original', 'Grayscale', 'Resized', 'Noisy', 'Denoised', 'Final']
        
        for i, (stage, title) in enumerate(zip(stages, titles)):
            if stage in results:
                if len(results[stage].shape) == 3:
                    axes[i].imshow(results[stage])
                else:
                    axes[i].imshow(results[stage], cmap='gray')
                axes[i].set_title(title)
                axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def batch_preprocess(image_paths: List[str], output_dir: str, preprocessor: ImagePreprocessor,
                    **kwargs) -> List[str]:
    """
    Batch preprocess multiple images
    
    Args:
        image_paths: List of image paths
        output_dir: Output directory
        preprocessor: ImagePreprocessor instance
        **kwargs: Additional arguments for preprocessing
        
    Returns:
        List of output paths
    """
    import os
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    
    for i, image_path in enumerate(tqdm(image_paths, desc="Preprocessing images")):
        try:
            results = preprocessor.preprocess_pipeline(image_path, **kwargs)
            if results is not None:
                output_path = os.path.join(output_dir, f"processed_{i:04d}.png")
                cv2.imwrite(output_path, results['final'])
                output_paths.append(output_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return output_paths


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor()
    
    # Test with a sample image (you'll need to provide an actual image path)
    # results = preprocessor.preprocess_pipeline("sample_leaf.jpg", add_noise=True, noise_level='medium')
    # if results:
    #     preprocessor.visualize_preprocessing(results)