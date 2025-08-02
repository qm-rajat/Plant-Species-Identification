"""
Edge Detection Module for Plant Species Identification
Implements various edge detection techniques for leaf analysis
"""

import cv2
import numpy as np
from skimage import filters, feature, morphology
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class EdgeDetector:
    """Implements various edge detection algorithms"""
    
    def __init__(self):
        """Initialize edge detector"""
        self.methods = {
            'sobel': self.sobel_edge_detection,
            'prewitt': self.prewitt_edge_detection,
            'canny': self.canny_edge_detection,
            'log': self.log_edge_detection,
            'laplacian': self.laplacian_edge_detection
        }
    
    def sobel_edge_detection(self, image: np.ndarray, ksize: int = 3) -> Dict[str, np.ndarray]:
        """
        Apply Sobel edge detection
        
        Args:
            image: Input grayscale image
            ksize: Kernel size (1, 3, 5, 7)
            
        Returns:
            Dictionary with 'x', 'y', and 'magnitude' edge maps
        """
        # Sobel in X and Y directions
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Calculate magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        
        return {
            'x': np.uint8(np.absolute(sobel_x)),
            'y': np.uint8(np.absolute(sobel_y)),
            'magnitude': magnitude
        }
    
    def prewitt_edge_detection(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply Prewitt edge detection
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary with 'x', 'y', and 'magnitude' edge maps
        """
        # Prewitt kernels
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        
        # Apply convolution
        prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
        
        # Calculate magnitude
        magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        
        return {
            'x': np.uint8(np.absolute(prewitt_x)),
            'y': np.uint8(np.absolute(prewitt_y)),
            'magnitude': magnitude
        }
    
    def canny_edge_detection(self, image: np.ndarray, low_threshold: int = 50, 
                           high_threshold: int = 150, aperture_size: int = 3,
                           l2_gradient: bool = False) -> np.ndarray:
        """
        Apply Canny edge detection
        
        Args:
            image: Input grayscale image
            low_threshold: Lower threshold for edge linking
            high_threshold: Upper threshold for edge linking
            aperture_size: Aperture size for Sobel operator
            l2_gradient: Use L2 gradient calculation
            
        Returns:
            Binary edge map
        """
        return cv2.Canny(image, low_threshold, high_threshold, 
                        apertureSize=aperture_size, L2gradient=l2_gradient)
    
    def log_edge_detection(self, image: np.ndarray, sigma: float = 1.0, 
                          threshold: float = 0.1) -> np.ndarray:
        """
        Apply Laplacian of Gaussian edge detection
        
        Args:
            image: Input grayscale image
            sigma: Standard deviation for Gaussian kernel
            threshold: Threshold for zero crossings
            
        Returns:
            Binary edge map
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Find zero crossings
        edges = np.zeros_like(laplacian, dtype=np.uint8)
        
        # Simple zero crossing detection
        for i in range(1, laplacian.shape[0] - 1):
            for j in range(1, laplacian.shape[1] - 1):
                # Check for sign changes in neighborhood
                neighbors = laplacian[i-1:i+2, j-1:j+2]
                if (neighbors.max() - neighbors.min()) > threshold:
                    if (neighbors.max() > 0 and neighbors.min() < 0):
                        edges[i, j] = 255
        
        return edges
    
    def laplacian_edge_detection(self, image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Apply Laplacian edge detection
        
        Args:
            image: Input grayscale image
            ksize: Kernel size
            
        Returns:
            Edge magnitude map
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
        return np.uint8(np.absolute(laplacian))
    
    def adaptive_canny(self, image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        """
        Apply adaptive Canny edge detection with automatic threshold selection
        
        Args:
            image: Input grayscale image
            sigma: Sigma value for threshold calculation
            
        Returns:
            Binary edge map
        """
        # Compute median of the single channel pixel intensities
        median = np.median(image)
        
        # Apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        
        return cv2.Canny(image, lower, upper)
    
    def multi_scale_edge_detection(self, image: np.ndarray, scales: List[float] = [1.0, 2.0, 4.0]) -> np.ndarray:
        """
        Apply multi-scale edge detection
        
        Args:
            image: Input grayscale image
            scales: List of scales for edge detection
            
        Returns:
            Combined edge map
        """
        combined_edges = np.zeros_like(image, dtype=np.float32)
        
        for scale in scales:
            # Apply Gaussian blur at different scales
            blurred = cv2.GaussianBlur(image, (0, 0), scale)
            
            # Apply Canny edge detection
            edges = self.adaptive_canny(blurred)
            
            # Add to combined edges with weight
            weight = 1.0 / scale
            combined_edges += weight * edges.astype(np.float32)
        
        # Normalize and convert to uint8
        combined_edges = (combined_edges / combined_edges.max() * 255).astype(np.uint8)
        return combined_edges
    
    def apply_all_methods(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Apply all edge detection methods to an image
        
        Args:
            image: Input grayscale image
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary with results from all methods
        """
        results = {}
        
        # Sobel
        sobel_result = self.sobel_edge_detection(image, kwargs.get('sobel_ksize', 3))
        results['sobel'] = sobel_result['magnitude']
        results['sobel_x'] = sobel_result['x']
        results['sobel_y'] = sobel_result['y']
        
        # Prewitt
        prewitt_result = self.prewitt_edge_detection(image)
        results['prewitt'] = prewitt_result['magnitude']
        results['prewitt_x'] = prewitt_result['x']
        results['prewitt_y'] = prewitt_result['y']
        
        # Canny
        results['canny'] = self.canny_edge_detection(
            image, 
            kwargs.get('canny_low', 50),
            kwargs.get('canny_high', 150)
        )
        
        # Adaptive Canny
        results['adaptive_canny'] = self.adaptive_canny(image, kwargs.get('canny_sigma', 0.33))
        
        # LoG
        results['log'] = self.log_edge_detection(
            image, 
            kwargs.get('log_sigma', 1.0),
            kwargs.get('log_threshold', 0.1)
        )
        
        # Laplacian
        results['laplacian'] = self.laplacian_edge_detection(image, kwargs.get('laplacian_ksize', 3))
        
        # Multi-scale
        results['multiscale'] = self.multi_scale_edge_detection(
            image, 
            kwargs.get('scales', [1.0, 2.0, 4.0])
        )
        
        return results
    
    def compare_edge_methods(self, image: np.ndarray, ground_truth: Optional[np.ndarray] = None) -> Dict:
        """
        Compare different edge detection methods
        
        Args:
            image: Input grayscale image
            ground_truth: Optional ground truth edge map for quantitative comparison
            
        Returns:
            Dictionary with comparison results
        """
        results = self.apply_all_methods(image)
        comparison = {'methods': results}
        
        if ground_truth is not None:
            # Calculate PSNR and SSIM for each method
            metrics = {}
            for method, edge_map in results.items():
                if 'x' in method or 'y' in method:  # Skip directional components
                    continue
                    
                # Ensure same data type and range
                edge_map_norm = edge_map.astype(np.float32) / 255.0
                gt_norm = ground_truth.astype(np.float32) / 255.0
                
                try:
                    psnr = peak_signal_noise_ratio(gt_norm, edge_map_norm, data_range=1.0)
                    ssim = structural_similarity(gt_norm, edge_map_norm, data_range=1.0)
                    
                    metrics[method] = {
                        'psnr': psnr,
                        'ssim': ssim
                    }
                except Exception as e:
                    print(f"Error calculating metrics for {method}: {e}")
                    metrics[method] = {'psnr': 0, 'ssim': 0}
            
            comparison['metrics'] = metrics
        
        return comparison
    
    def visualize_edge_detection(self, image: np.ndarray, results: Dict[str, np.ndarray], 
                               save_path: Optional[str] = None):
        """
        Visualize edge detection results
        
        Args:
            image: Original input image
            results: Results from edge detection methods
            save_path: Path to save visualization
        """
        # Filter out directional components for main visualization
        main_methods = {k: v for k, v in results.items() if not ('_x' in k or '_y' in k)}
        
        n_methods = len(main_methods) + 1  # +1 for original
        cols = 3
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # Show original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show edge detection results
        for i, (method, edge_map) in enumerate(main_methods.items(), 1):
            axes[i].imshow(edge_map, cmap='gray')
            axes[i].set_title(f'{method.upper()} Edge Detection')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(main_methods) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def extract_edge_features(self, edge_map: np.ndarray) -> Dict[str, float]:
        """
        Extract features from edge map
        
        Args:
            edge_map: Binary or grayscale edge map
            
        Returns:
            Dictionary with edge features
        """
        # Ensure binary edge map
        if edge_map.max() > 1:
            binary_edges = (edge_map > edge_map.mean()).astype(np.uint8)
        else:
            binary_edges = edge_map.astype(np.uint8)
        
        # Calculate features
        total_pixels = edge_map.shape[0] * edge_map.shape[1]
        edge_pixels = np.sum(binary_edges > 0)
        
        features = {
            'edge_density': edge_pixels / total_pixels,
            'total_edge_pixels': edge_pixels,
            'edge_strength_mean': np.mean(edge_map[edge_map > 0]) if edge_pixels > 0 else 0,
            'edge_strength_std': np.std(edge_map[edge_map > 0]) if edge_pixels > 0 else 0,
            'edge_connectivity': self._calculate_connectivity(binary_edges),
            'edge_orientation_entropy': self._calculate_orientation_entropy(edge_map)
        }
        
        return features
    
    def _calculate_connectivity(self, binary_edges: np.ndarray) -> float:
        """Calculate edge connectivity using connected components"""
        num_labels, _ = cv2.connectedComponents(binary_edges)
        total_edge_pixels = np.sum(binary_edges > 0)
        if total_edge_pixels == 0:
            return 0
        return (total_edge_pixels - num_labels + 1) / total_edge_pixels
    
    def _calculate_orientation_entropy(self, edge_map: np.ndarray) -> float:
        """Calculate orientation entropy of edges"""
        # Calculate gradients
        grad_x = cv2.Sobel(edge_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(edge_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate orientation
        orientation = np.arctan2(grad_y, grad_x)
        
        # Quantize orientations into bins
        n_bins = 8
        hist, _ = np.histogram(orientation.flatten(), bins=n_bins, range=(-np.pi, np.pi))
        
        # Calculate entropy
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]  # Remove zero probabilities
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy


def batch_edge_detection(image_paths: List[str], output_dir: str, detector: EdgeDetector,
                        method: str = 'canny', **kwargs) -> List[str]:
    """
    Apply edge detection to multiple images
    
    Args:
        image_paths: List of image paths
        output_dir: Output directory
        detector: EdgeDetector instance
        method: Edge detection method to use
        **kwargs: Additional parameters for edge detection
        
    Returns:
        List of output paths
    """
    import os
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    
    for i, image_path in enumerate(tqdm(image_paths, desc=f"Applying {method} edge detection")):
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            
            # Apply edge detection
            if method in detector.methods:
                if method == 'sobel':
                    result = detector.sobel_edge_detection(image, **kwargs)
                    edge_map = result['magnitude']
                elif method == 'prewitt':
                    result = detector.prewitt_edge_detection(image)
                    edge_map = result['magnitude']
                else:
                    edge_map = detector.methods[method](image, **kwargs)
            else:
                edge_map = detector.adaptive_canny(image)
            
            # Save result
            output_path = os.path.join(output_dir, f"{method}_edges_{i:04d}.png")
            cv2.imwrite(output_path, edge_map)
            output_paths.append(output_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return output_paths


if __name__ == "__main__":
    # Example usage
    detector = EdgeDetector()
    
    # Test with a sample image (you'll need to provide an actual image path)
    # image = cv2.imread("sample_leaf.jpg", cv2.IMREAD_GRAYSCALE)
    # if image is not None:
    #     results = detector.apply_all_methods(image)
    #     detector.visualize_edge_detection(image, results)
    #     
    #     # Extract features from Canny edges
    #     canny_features = detector.extract_edge_features(results['canny'])
    #     print("Canny Edge Features:", canny_features)