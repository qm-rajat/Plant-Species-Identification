"""
Feature Extraction Module for Plant Species Identification
Extracts comprehensive features from leaf images for classification
"""

import cv2
import numpy as np
from skimage import morphology, measure, feature, filters, segmentation
from scipy import ndimage, spatial
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import math


class FeatureExtractor:
    """Extracts various features from leaf images"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_categories = {
            'geometric': self.extract_geometric_features,
            'vein': self.extract_vein_features,
            'margin': self.extract_margin_features,
            'texture': self.extract_texture_features,
            'color': self.extract_color_features,
            'shape_descriptors': self.extract_shape_descriptors
        }
    
    def extract_geometric_features(self, image: np.ndarray, binary_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract geometric descriptors (shape, aspect ratio, area, etc.)
        
        Args:
            image: Input grayscale image
            binary_mask: Optional binary mask of the leaf
            
        Returns:
            Dictionary of geometric features
        """
        if binary_mask is None:
            # Create binary mask using Otsu's thresholding
            _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._get_empty_geometric_features()
        
        # Get the largest contour (main leaf)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        bounding_area = w * h
        
        # Minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
        enclosing_circle_area = np.pi * radius * radius
        
        # Convex hull
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        
        # Fitted ellipse
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
        else:
            ellipse_area = area
            major_axis = max(w, h)
            minor_axis = min(w, h)
        
        # Calculate geometric features
        features = {
            # Basic measurements
            'area': area,
            'perimeter': perimeter,
            'width': w,
            'height': h,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            
            # Shape ratios and descriptors
            'aspect_ratio': w / h if h > 0 else 0,
            'rectangularity': area / bounding_area if bounding_area > 0 else 0,
            'circularity': (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0,
            'roundness': (4 * area) / (np.pi * major_axis * major_axis) if major_axis > 0 else 0,
            'compactness': (perimeter * perimeter) / area if area > 0 else 0,
            
            # Convexity measures
            'solidity': area / hull_area if hull_area > 0 else 0,
            'convexity': hull_perimeter / perimeter if perimeter > 0 else 0,
            'convex_hull_ratio': hull_area / area if area > 0 else 0,
            
            # Ellipse-based features
            'ellipse_variance': (major_axis - minor_axis) / (major_axis + minor_axis) if (major_axis + minor_axis) > 0 else 0,
            'eccentricity': self._calculate_eccentricity(major_axis, minor_axis),
            'elongation': major_axis / minor_axis if minor_axis > 0 else 0,
            
            # Other shape measures
            'extent': area / bounding_area if bounding_area > 0 else 0,
            'equivalent_diameter': np.sqrt(4 * area / np.pi),
            'form_factor': (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0,
            'shape_index': perimeter / (2 * np.sqrt(np.pi * area)) if area > 0 else 0,
            
            # Centroid and moments
            'centroid_x': cx,
            'centroid_y': cy,
        }
        
        # Hu moments
        moments = cv2.moments(main_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        for i, hu in enumerate(hu_moments):
            features[f'hu_moment_{i+1}'] = -np.sign(hu) * np.log10(np.abs(hu)) if hu != 0 else 0
        
        return features
    
    def extract_vein_features(self, image: np.ndarray, binary_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract vein pattern features using skeletonization and analysis
        
        Args:
            image: Input grayscale image
            binary_mask: Optional binary mask of the leaf
            
        Returns:
            Dictionary of vein features
        """
        if binary_mask is None:
            _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Enhance vein patterns using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Apply top-hat transform to enhance veins
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Apply Frangi vesselness filter for vein enhancement
        veins_enhanced = self._frangi_filter(image)
        
        # Combine enhanced images
        combined = cv2.addWeighted(tophat, 0.5, veins_enhanced, 0.5, 0)
        
        # Threshold to get vein binary image
        _, vein_binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply mask to focus on leaf area only
        vein_binary = cv2.bitwise_and(vein_binary, binary_mask)
        
        # Skeletonization (thinning)
        skeleton = morphology.skeletonize(vein_binary > 0)
        skeleton = skeleton.astype(np.uint8) * 255
        
        # Calculate vein features
        total_leaf_area = np.sum(binary_mask > 0)
        total_vein_pixels = np.sum(skeleton > 0)
        
        # Vein density
        vein_density = total_vein_pixels / total_leaf_area if total_leaf_area > 0 else 0
        
        # Find vein endpoints and junctions
        endpoints, junctions = self._find_vein_endpoints_junctions(skeleton)
        
        # Vein length estimation
        vein_length = total_vein_pixels  # Approximation
        
        # Vein orientation analysis
        vein_orientations = self._calculate_vein_orientations(skeleton)
        
        # Distance transform for vein thickness analysis
        dist_transform = cv2.distanceTransform(vein_binary, cv2.DIST_L2, 5)
        avg_vein_thickness = np.mean(dist_transform[dist_transform > 0]) if np.any(dist_transform > 0) else 0
        
        features = {
            'vein_density': vein_density,
            'total_vein_length': vein_length,
            'vein_endpoints': len(endpoints),
            'vein_junctions': len(junctions),
            'avg_vein_thickness': avg_vein_thickness,
            'vein_complexity': (len(endpoints) + len(junctions)) / total_leaf_area if total_leaf_area > 0 else 0,
            'vein_area_ratio': total_vein_pixels / total_leaf_area if total_leaf_area > 0 else 0,
        }
        
        # Add orientation features
        if vein_orientations.size > 0:
            features.update({
                'vein_orientation_mean': np.mean(vein_orientations),
                'vein_orientation_std': np.std(vein_orientations),
                'vein_orientation_entropy': self._calculate_orientation_entropy(vein_orientations),
            })
        else:
            features.update({
                'vein_orientation_mean': 0,
                'vein_orientation_std': 0,
                'vein_orientation_entropy': 0,
            })
        
        return features
    
    def extract_margin_features(self, image: np.ndarray, binary_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract margin/contour features including chain codes and curvature
        
        Args:
            image: Input grayscale image
            binary_mask: Optional binary mask of the leaf
            
        Returns:
            Dictionary of margin features
        """
        if binary_mask is None:
            _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return self._get_empty_margin_features()
        
        # Get the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        contour_points = main_contour.reshape(-1, 2)
        
        # Chain code analysis
        chain_code = self._calculate_chain_code(contour_points)
        chain_code_features = self._analyze_chain_code(chain_code)
        
        # Curvature analysis
        curvatures = self._calculate_curvature(contour_points)
        
        # Smoothness analysis
        smoothness = self._calculate_contour_smoothness(contour_points)
        
        # Fourier descriptors
        fourier_descriptors = self._calculate_fourier_descriptors(contour_points)
        
        # Margin roughness
        perimeter = cv2.arcLength(main_contour, True)
        convex_hull = cv2.convexHull(main_contour)
        convex_perimeter = cv2.arcLength(convex_hull, True)
        roughness = perimeter / convex_perimeter if convex_perimeter > 0 else 0
        
        # Lobedness (number of significant protrusions)
        lobedness = self._calculate_lobedness(contour_points)
        
        features = {
            # Basic margin properties
            'margin_roughness': roughness,
            'margin_smoothness': smoothness,
            'lobedness': lobedness,
            
            # Curvature features
            'curvature_mean': np.mean(curvatures) if len(curvatures) > 0 else 0,
            'curvature_std': np.std(curvatures) if len(curvatures) > 0 else 0,
            'curvature_max': np.max(curvatures) if len(curvatures) > 0 else 0,
            'curvature_min': np.min(curvatures) if len(curvatures) > 0 else 0,
            'curvature_range': (np.max(curvatures) - np.min(curvatures)) if len(curvatures) > 0 else 0,
            
            # Chain code features
            **chain_code_features,
            
            # Fourier descriptor features (first 10 coefficients)
            **{f'fourier_desc_{i}': fourier_descriptors[i] if i < len(fourier_descriptors) else 0 
               for i in range(10)},
        }
        
        return features
    
    def extract_texture_features(self, image: np.ndarray, binary_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract texture features using various methods
        
        Args:
            image: Input grayscale image
            binary_mask: Optional binary mask to focus on leaf area
            
        Returns:
            Dictionary of texture features
        """
        # Apply mask if provided
        if binary_mask is not None:
            masked_image = cv2.bitwise_and(image, binary_mask)
            roi = masked_image[binary_mask > 0]
        else:
            roi = image.flatten()
        
        if len(roi) == 0:
            return self._get_empty_texture_features()
        
        features = {}
        
        # Statistical texture features
        features.update({
            'texture_mean': np.mean(roi),
            'texture_std': np.std(roi),
            'texture_variance': np.var(roi),
            'texture_skewness': skew(roi),
            'texture_kurtosis': kurtosis(roi),
            'texture_energy': np.sum(roi ** 2) / len(roi),
            'texture_entropy': self._calculate_entropy(roi),
        })
        
        # Local Binary Pattern (LBP)
        lbp = feature.local_binary_pattern(image, 24, 3, method='uniform')
        if binary_mask is not None:
            lbp_roi = lbp[binary_mask > 0]
        else:
            lbp_roi = lbp.flatten()
        
        lbp_hist, _ = np.histogram(lbp_roi, bins=26, range=(0, 26))
        lbp_hist = lbp_hist.astype(float) / (np.sum(lbp_hist) + 1e-7)
        
        for i, val in enumerate(lbp_hist):
            features[f'lbp_bin_{i}'] = val
        
        # Gray Level Co-occurrence Matrix (GLCM)
        try:
            glcm = feature.graycomatrix(image, distances=[1], angles=[0, 45, 90, 135], 
                                      levels=256, symmetric=True, normed=True)
            
            glcm_features = {
                'glcm_contrast': feature.graycoprops(glcm, 'contrast').mean(),
                'glcm_dissimilarity': feature.graycoprops(glcm, 'dissimilarity').mean(),
                'glcm_homogeneity': feature.graycoprops(glcm, 'homogeneity').mean(),
                'glcm_energy': feature.graycoprops(glcm, 'energy').mean(),
                'glcm_correlation': feature.graycoprops(glcm, 'correlation').mean(),
                'glcm_asm': feature.graycoprops(glcm, 'ASM').mean(),
            }
            features.update(glcm_features)
        except:
            features.update({
                'glcm_contrast': 0, 'glcm_dissimilarity': 0, 'glcm_homogeneity': 0,
                'glcm_energy': 0, 'glcm_correlation': 0, 'glcm_asm': 0
            })
        
        return features
    
    def extract_color_features(self, image: np.ndarray, binary_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract color-based features (for RGB images)
        
        Args:
            image: Input image (RGB or grayscale)
            binary_mask: Optional binary mask to focus on leaf area
            
        Returns:
            Dictionary of color features
        """
        features = {}
        
        if len(image.shape) == 3:
            # RGB image
            for i, channel in enumerate(['red', 'green', 'blue']):
                ch = image[:, :, i]
                if binary_mask is not None:
                    roi = ch[binary_mask > 0]
                else:
                    roi = ch.flatten()
                
                if len(roi) > 0:
                    features.update({
                        f'color_{channel}_mean': np.mean(roi),
                        f'color_{channel}_std': np.std(roi),
                        f'color_{channel}_skewness': skew(roi),
                        f'color_{channel}_kurtosis': kurtosis(roi),
                        f'color_{channel}_min': np.min(roi),
                        f'color_{channel}_max': np.max(roi),
                        f'color_{channel}_range': np.max(roi) - np.min(roi),
                    })
                else:
                    features.update({
                        f'color_{channel}_mean': 0, f'color_{channel}_std': 0,
                        f'color_{channel}_skewness': 0, f'color_{channel}_kurtosis': 0,
                        f'color_{channel}_min': 0, f'color_{channel}_max': 0,
                        f'color_{channel}_range': 0,
                    })
            
            # Color ratios and relationships
            if binary_mask is not None:
                r_roi = image[:, :, 0][binary_mask > 0]
                g_roi = image[:, :, 1][binary_mask > 0]
                b_roi = image[:, :, 2][binary_mask > 0]
            else:
                r_roi = image[:, :, 0].flatten()
                g_roi = image[:, :, 1].flatten()
                b_roi = image[:, :, 2].flatten()
            
            if len(r_roi) > 0:
                r_mean, g_mean, b_mean = np.mean(r_roi), np.mean(g_roi), np.mean(b_roi)
                total = r_mean + g_mean + b_mean
                
                if total > 0:
                    features.update({
                        'color_r_ratio': r_mean / total,
                        'color_g_ratio': g_mean / total,
                        'color_b_ratio': b_mean / total,
                        'color_rg_ratio': r_mean / g_mean if g_mean > 0 else 0,
                        'color_rb_ratio': r_mean / b_mean if b_mean > 0 else 0,
                        'color_gb_ratio': g_mean / b_mean if b_mean > 0 else 0,
                    })
                else:
                    features.update({
                        'color_r_ratio': 0, 'color_g_ratio': 0, 'color_b_ratio': 0,
                        'color_rg_ratio': 0, 'color_rb_ratio': 0, 'color_gb_ratio': 0,
                    })
            
            # Convert to other color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            for color_space, img_cs in [('hsv', hsv), ('lab', lab)]:
                for i, ch_name in enumerate(['ch1', 'ch2', 'ch3']):
                    ch = img_cs[:, :, i]
                    if binary_mask is not None:
                        roi = ch[binary_mask > 0]
                    else:
                        roi = ch.flatten()
                    
                    if len(roi) > 0:
                        features.update({
                            f'{color_space}_{ch_name}_mean': np.mean(roi),
                            f'{color_space}_{ch_name}_std': np.std(roi),
                        })
                    else:
                        features.update({
                            f'{color_space}_{ch_name}_mean': 0,
                            f'{color_space}_{ch_name}_std': 0,
                        })
        
        else:
            # Grayscale image
            if binary_mask is not None:
                roi = image[binary_mask > 0]
            else:
                roi = image.flatten()
            
            if len(roi) > 0:
                features.update({
                    'color_gray_mean': np.mean(roi),
                    'color_gray_std': np.std(roi),
                    'color_gray_skewness': skew(roi),
                    'color_gray_kurtosis': kurtosis(roi),
                })
            else:
                features.update({
                    'color_gray_mean': 0, 'color_gray_std': 0,
                    'color_gray_skewness': 0, 'color_gray_kurtosis': 0,
                })
        
        return features
    
    def extract_shape_descriptors(self, image: np.ndarray, binary_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract advanced shape descriptors
        
        Args:
            image: Input grayscale image
            binary_mask: Optional binary mask of the leaf
            
        Returns:
            Dictionary of shape descriptor features
        """
        if binary_mask is None:
            _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._get_empty_shape_descriptors()
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Zernike moments
        try:
            zernike_moments = self._calculate_zernike_moments(binary_mask)
        except:
            zernike_moments = [0] * 10
        
        # Shape context (simplified)
        shape_context = self._calculate_shape_context(main_contour)
        
        # Radial distance features
        radial_features = self._calculate_radial_features(main_contour)
        
        features = {
            # Zernike moments
            **{f'zernike_{i}': zernike_moments[i] if i < len(zernike_moments) else 0 
               for i in range(10)},
            
            # Shape context features
            'shape_context_mean': np.mean(shape_context) if len(shape_context) > 0 else 0,
            'shape_context_std': np.std(shape_context) if len(shape_context) > 0 else 0,
            
            # Radial features
            **radial_features,
        }
        
        return features
    
    def extract_all_features(self, image: np.ndarray, binary_mask: Optional[np.ndarray] = None,
                           categories: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract all feature categories
        
        Args:
            image: Input image
            binary_mask: Optional binary mask of the leaf
            categories: List of feature categories to extract (default: all)
            
        Returns:
            Dictionary with all extracted features
        """
        if categories is None:
            categories = list(self.feature_categories.keys())
        
        all_features = {}
        
        # Convert to grayscale if needed for some features
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        for category in categories:
            if category in self.feature_categories:
                try:
                    if category == 'color':
                        # Use original image for color features
                        features = self.feature_categories[category](image, binary_mask)
                    else:
                        # Use grayscale for other features
                        features = self.feature_categories[category](gray_image, binary_mask)
                    
                    # Add category prefix to feature names
                    prefixed_features = {f"{category}_{k}": v for k, v in features.items()}
                    all_features.update(prefixed_features)
                    
                except Exception as e:
                    print(f"Error extracting {category} features: {e}")
                    continue
        
        return all_features
    
    # Helper methods
    def _get_empty_geometric_features(self) -> Dict[str, float]:
        """Return zero-filled geometric features"""
        return {f'geometric_{k}': 0 for k in [
            'area', 'perimeter', 'width', 'height', 'major_axis', 'minor_axis',
            'aspect_ratio', 'rectangularity', 'circularity', 'roundness', 'compactness',
            'solidity', 'convexity', 'convex_hull_ratio', 'ellipse_variance',
            'eccentricity', 'elongation', 'extent', 'equivalent_diameter',
            'form_factor', 'shape_index', 'centroid_x', 'centroid_y'
        ] + [f'hu_moment_{i}' for i in range(1, 8)]}
    
    def _get_empty_margin_features(self) -> Dict[str, float]:
        """Return zero-filled margin features"""
        base_features = ['margin_roughness', 'margin_smoothness', 'lobedness',
                        'curvature_mean', 'curvature_std', 'curvature_max',
                        'curvature_min', 'curvature_range']
        chain_features = [f'chain_code_{i}' for i in range(8)]
        fourier_features = [f'fourier_desc_{i}' for i in range(10)]
        return {k: 0 for k in base_features + chain_features + fourier_features}
    
    def _get_empty_texture_features(self) -> Dict[str, float]:
        """Return zero-filled texture features"""
        base_features = ['texture_mean', 'texture_std', 'texture_variance',
                        'texture_skewness', 'texture_kurtosis', 'texture_energy',
                        'texture_entropy']
        lbp_features = [f'lbp_bin_{i}' for i in range(26)]
        glcm_features = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
                        'glcm_energy', 'glcm_correlation', 'glcm_asm']
        return {k: 0 for k in base_features + lbp_features + glcm_features}
    
    def _get_empty_shape_descriptors(self) -> Dict[str, float]:
        """Return zero-filled shape descriptors"""
        zernike_features = [f'zernike_{i}' for i in range(10)]
        other_features = ['shape_context_mean', 'shape_context_std',
                         'radial_mean', 'radial_std', 'radial_variance']
        return {k: 0 for k in zernike_features + other_features}
    
    def _calculate_eccentricity(self, major_axis: float, minor_axis: float) -> float:
        """Calculate eccentricity from major and minor axes"""
        if major_axis == 0:
            return 0
        return np.sqrt(1 - (minor_axis / major_axis) ** 2)
    
    def _frangi_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Frangi vesselness filter for vein enhancement"""
        # Simplified Frangi filter implementation
        # In practice, you might want to use skimage.filters.frangi
        try:
            from skimage.filters import frangi
            return (frangi(image) * 255).astype(np.uint8)
        except:
            # Fallback: use simple edge enhancement
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(image, -1, kernel)
            return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _find_vein_endpoints_junctions(self, skeleton: np.ndarray) -> Tuple[List, List]:
        """Find endpoints and junctions in skeleton"""
        # Kernel for counting neighbors
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        
        # Count neighbors for each skeleton pixel
        neighbor_count = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
        
        # Endpoints have 1 neighbor, junctions have 3+ neighbors
        endpoints = np.where((skeleton > 0) & (neighbor_count == 1))
        junctions = np.where((skeleton > 0) & (neighbor_count >= 3))
        
        return list(zip(endpoints[0], endpoints[1])), list(zip(junctions[0], junctions[1]))
    
    def _calculate_vein_orientations(self, skeleton: np.ndarray) -> np.ndarray:
        """Calculate orientations of vein segments"""
        # Calculate gradients
        grad_x = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate orientations
        orientations = np.arctan2(grad_y, grad_x)
        
        # Return orientations only for skeleton pixels
        return orientations[skeleton > 0]
    
    def _calculate_orientation_entropy(self, orientations: np.ndarray) -> float:
        """Calculate entropy of orientation distribution"""
        if len(orientations) == 0:
            return 0
        
        hist, _ = np.histogram(orientations, bins=8, range=(-np.pi, np.pi))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
    
    def _calculate_chain_code(self, contour_points: np.ndarray) -> List[int]:
        """Calculate 8-directional chain code"""
        if len(contour_points) < 2:
            return []
        
        # Direction vectors for 8-connectivity
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        chain_code = []
        for i in range(len(contour_points) - 1):
            dx = contour_points[i + 1][0] - contour_points[i][0]
            dy = contour_points[i + 1][1] - contour_points[i][1]
            
            # Find closest direction
            min_dist = float('inf')
            best_dir = 0
            
            for j, (dir_x, dir_y) in enumerate(directions):
                dist = (dx - dir_x) ** 2 + (dy - dir_y) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_dir = j
            
            chain_code.append(best_dir)
        
        return chain_code
    
    def _analyze_chain_code(self, chain_code: List[int]) -> Dict[str, float]:
        """Analyze chain code to extract features"""
        if not chain_code:
            return {f'chain_code_{i}': 0 for i in range(8)}
        
        # Calculate histogram of chain codes
        hist, _ = np.histogram(chain_code, bins=8, range=(0, 8))
        hist = hist.astype(float) / len(chain_code)
        
        return {f'chain_code_{i}': hist[i] for i in range(8)}
    
    def _calculate_curvature(self, contour_points: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Calculate curvature along contour"""
        if len(contour_points) < window_size:
            return np.array([])
        
        curvatures = []
        half_window = window_size // 2
        
        for i in range(half_window, len(contour_points) - half_window):
            # Get points in window
            p1 = contour_points[i - half_window]
            p2 = contour_points[i]
            p3 = contour_points[i + half_window]
            
            # Calculate curvature using three points
            curvature = self._point_curvature(p1, p2, p3)
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def _point_curvature(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate curvature at point p2 using three points"""
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Cross product magnitude
        cross = np.cross(v1, v2)
        
        # Dot product
        dot = np.dot(v1, v2)
        
        # Calculate curvature
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        
        return cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def _calculate_contour_smoothness(self, contour_points: np.ndarray) -> float:
        """Calculate smoothness of contour"""
        if len(contour_points) < 3:
            return 0
        
        # Calculate second derivatives (acceleration)
        second_derivs = []
        for i in range(1, len(contour_points) - 1):
            d2x = contour_points[i+1][0] - 2*contour_points[i][0] + contour_points[i-1][0]
            d2y = contour_points[i+1][1] - 2*contour_points[i][1] + contour_points[i-1][1]
            second_derivs.append(np.sqrt(d2x**2 + d2y**2))
        
        return 1.0 / (1.0 + np.mean(second_derivs)) if second_derivs else 0
    
    def _calculate_fourier_descriptors(self, contour_points: np.ndarray, n_descriptors: int = 10) -> np.ndarray:
        """Calculate Fourier descriptors of contour"""
        if len(contour_points) < 4:
            return np.zeros(n_descriptors)
        
        # Create complex representation
        complex_contour = contour_points[:, 0] + 1j * contour_points[:, 1]
        
        # Apply FFT
        fft_result = np.fft.fft(complex_contour)
        
        # Take magnitude of first n_descriptors coefficients (skip DC component)
        descriptors = np.abs(fft_result[1:n_descriptors+1])
        
        # Normalize by first coefficient to achieve scale invariance
        if descriptors[0] != 0:
            descriptors = descriptors / descriptors[0]
        
        return descriptors
    
    def _calculate_lobedness(self, contour_points: np.ndarray) -> int:
        """Calculate number of lobes/protrusions in leaf"""
        if len(contour_points) < 10:
            return 0
        
        # Calculate distance from centroid
        centroid = np.mean(contour_points, axis=0)
        distances = np.array([np.linalg.norm(p - centroid) for p in contour_points])
        
        # Find local maxima (peaks) in distance function
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(distances, height=np.mean(distances), distance=len(distances)//10)
        
        return len(peaks)
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data"""
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
    
    def _calculate_zernike_moments(self, binary_image: np.ndarray, max_order: int = 8) -> List[float]:
        """Calculate Zernike moments (simplified implementation)"""
        # This is a placeholder for Zernike moments
        # Full implementation would require complex Zernike polynomial calculations
        moments = []
        
        # Calculate basic moments as approximation
        m = cv2.moments(binary_image)
        
        # Normalized central moments
        if m['m00'] != 0:
            mu20 = m['mu20'] / m['m00']
            mu02 = m['mu02'] / m['m00']
            mu11 = m['mu11'] / m['m00']
            mu30 = m['mu30'] / m['m00']
            mu03 = m['mu03'] / m['m00']
            mu21 = m['mu21'] / m['m00']
            mu12 = m['mu12'] / m['m00']
            
            moments = [mu20, mu02, mu11, mu30, mu03, mu21, mu12,
                      mu20 + mu02, mu30 + mu12, mu03 + mu21]
        else:
            moments = [0] * 10
        
        return moments[:max_order]
    
    def _calculate_shape_context(self, contour: np.ndarray, n_points: int = 50) -> np.ndarray:
        """Calculate shape context descriptor (simplified)"""
        if len(contour) < n_points:
            return np.array([])
        
        # Sample points uniformly
        indices = np.linspace(0, len(contour) - 1, n_points, dtype=int)
        sampled_points = contour[indices].reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = spatial.distance_matrix(sampled_points, sampled_points)
        
        # Return flattened upper triangle (excluding diagonal)
        return distances[np.triu_indices_from(distances, k=1)]
    
    def _calculate_radial_features(self, contour: np.ndarray) -> Dict[str, float]:
        """Calculate radial distance features"""
        if len(contour) == 0:
            return {'radial_mean': 0, 'radial_std': 0, 'radial_variance': 0}
        
        # Calculate centroid
        contour_points = contour.reshape(-1, 2)
        centroid = np.mean(contour_points, axis=0)
        
        # Calculate radial distances
        radial_distances = np.array([np.linalg.norm(p - centroid) for p in contour_points])
        
        return {
            'radial_mean': np.mean(radial_distances),
            'radial_std': np.std(radial_distances),
            'radial_variance': np.var(radial_distances),
        }


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    # Test with a sample image (you'll need to provide an actual image path)
    # image = cv2.imread("sample_leaf.jpg")
    # if image is not None:
    #     features = extractor.extract_all_features(image)
    #     print(f"Extracted {len(features)} features")
    #     print("Sample features:", list(features.keys())[:10])