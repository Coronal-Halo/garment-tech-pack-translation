"""
Design Pack Image Detector Module

This module detects and identifies design pack image regions that should
be preserved (not translated) in tech pack images.

Design pack images are typically:
- Colorful artistic graphics
- High texture/entropy regions
- Complex edge patterns
- Located in specific areas of the tech pack
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import entropy

from .models import DesignPackRegion, BoundingBox

logger = logging.getLogger(__name__)


class DesignPackDetector:
    """
    Detector for design pack image regions in tech pack images.
    
    Uses multiple feature analysis techniques:
    1. Color variance analysis - design packs are typically colorful
    2. Edge density analysis - design packs have complex edges
    3. Texture entropy analysis - design packs have high entropy
    4. Region proposal and scoring
    """
    
    def __init__(
        self,
        color_variance_threshold: float = 0.3,
        edge_density_threshold: float = 0.15,
        entropy_threshold: float = 6.0,
        min_region_ratio: float = 0.05,
        max_region_ratio: float = 0.5,
        margin: int = 10,
        detection_method: str = "multi_feature"
    ):
        """
        Initialize the design pack detector.
        
        Args:
            color_variance_threshold: Threshold for color variance detection
            edge_density_threshold: Threshold for edge density detection
            entropy_threshold: Threshold for entropy-based detection
            min_region_ratio: Minimum region size as ratio of image size
            max_region_ratio: Maximum region size as ratio of image size
            margin: Margin to add around detected regions (pixels)
            detection_method: Detection method ('multi_feature', 'color', 'edge', 'entropy')
        """
        self.color_variance_threshold = color_variance_threshold
        self.edge_density_threshold = edge_density_threshold
        self.entropy_threshold = entropy_threshold
        self.min_region_ratio = min_region_ratio
        self.max_region_ratio = max_region_ratio
        self.margin = margin
        self.detection_method = detection_method
        
        logger.info(f"DesignPackDetector initialized with method: {detection_method}")
    
    def detect(self, image: np.ndarray) -> List[DesignPackRegion]:
        """
        Detect design pack image regions in the input image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of DesignPackRegion objects
        """
        logger.info("Starting design pack detection...")
        
        h, w = image.shape[:2]
        min_area = int(h * w * self.min_region_ratio)
        max_area = int(h * w * self.max_region_ratio)
        
        # Generate candidate regions using multiple methods
        candidates = []
        
        if self.detection_method in ["multi_feature", "color"]:
            candidates.extend(self._detect_by_color_variance(image))
        
        if self.detection_method in ["multi_feature", "edge"]:
            candidates.extend(self._detect_by_edge_density(image))
        
        if self.detection_method in ["multi_feature", "entropy"]:
            candidates.extend(self._detect_by_entropy(image))
        
        # Also try connected component analysis on high-saturation regions
        candidates.extend(self._detect_by_saturation(image))
        
        if not candidates:
            logger.info("No design pack regions detected")
            return []
        
        # Score and filter candidates
        scored_regions = []
        for bbox, method in candidates:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Filter by size
            if area < min_area or area > max_area:
                continue
            
            # Calculate confidence score
            features = self._calculate_region_features(image, bbox)
            confidence = self._calculate_confidence(features)
            
            if confidence > 0.5:  # Only keep high-confidence detections
                region = DesignPackRegion(
                    bbox=BoundingBox(
                        x1=max(0, bbox[0] - self.margin),
                        y1=max(0, bbox[1] - self.margin),
                        x2=min(w, bbox[2] + self.margin),
                        y2=min(h, bbox[3] + self.margin)
                    ),
                    confidence=confidence,
                    detection_method=method,
                    features=features
                )
                scored_regions.append(region)
        
        # Non-maximum suppression to remove duplicates
        final_regions = self._non_max_suppression(scored_regions)
        
        logger.info(f"Detected {len(final_regions)} design pack regions")
        
        return final_regions
    
    def _detect_by_color_variance(
        self, 
        image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """Detect regions with high color variance."""
        candidates = []
        
        # Convert to float for variance calculation
        img_float = image.astype(np.float32) / 255.0
        
        # Calculate local color variance using sliding window
        kernel_size = 50
        
        # Compute variance in each channel
        variance_maps = []
        for c in range(3):
            channel = img_float[:, :, c]
            # Local mean
            local_mean = cv2.blur(channel, (kernel_size, kernel_size))
            # Local variance
            local_var = cv2.blur(channel**2, (kernel_size, kernel_size)) - local_mean**2
            variance_maps.append(local_var)
        
        # Combined variance map
        variance_map = np.mean(variance_maps, axis=0)
        
        # Threshold to find high-variance regions
        high_variance_mask = (variance_map > self.color_variance_threshold).astype(np.uint8) * 255
        
        # Clean up the mask
        kernel = np.ones((10, 10), np.uint8)
        high_variance_mask = cv2.morphologyEx(high_variance_mask, cv2.MORPH_CLOSE, kernel)
        high_variance_mask = cv2.morphologyEx(high_variance_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            high_variance_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:  # Minimum size filter
                candidates.append(((x, y, x + w, y + h), "color_variance"))
        
        return candidates
    
    def _detect_by_edge_density(
        self, 
        image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """Detect regions with high edge density."""
        candidates = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate local edge density
        kernel_size = 50
        edge_density = cv2.blur(edges.astype(np.float32), (kernel_size, kernel_size))
        edge_density = edge_density / 255.0  # Normalize
        
        # Threshold to find high-density regions
        high_density_mask = (edge_density > self.edge_density_threshold).astype(np.uint8) * 255
        
        # Clean up the mask
        kernel = np.ones((15, 15), np.uint8)
        high_density_mask = cv2.morphologyEx(high_density_mask, cv2.MORPH_CLOSE, kernel)
        high_density_mask = cv2.morphologyEx(high_density_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            high_density_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:
                candidates.append(((x, y, x + w, y + h), "edge_density"))
        
        return candidates
    
    def _detect_by_entropy(
        self, 
        image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """Detect regions with high texture entropy."""
        candidates = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local entropy using block processing
        h, w = gray.shape
        block_size = 50
        entropy_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = gray[i:i+block_size, j:j+block_size]
                # Calculate histogram
                hist, _ = np.histogram(block, bins=256, range=(0, 256))
                hist = hist / hist.sum()
                # Calculate entropy
                block_entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropy_map[i:i+block_size, j:j+block_size] = block_entropy
        
        # Threshold to find high-entropy regions
        high_entropy_mask = (entropy_map > self.entropy_threshold).astype(np.uint8) * 255
        
        # Clean up the mask
        kernel = np.ones((20, 20), np.uint8)
        high_entropy_mask = cv2.morphologyEx(high_entropy_mask, cv2.MORPH_CLOSE, kernel)
        high_entropy_mask = cv2.morphologyEx(high_entropy_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            high_entropy_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:
                candidates.append(((x, y, x + w, y + h), "entropy"))
        
        return candidates
    
    def _detect_by_saturation(
        self, 
        image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """Detect regions with high color saturation."""
        candidates = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # High saturation regions (colorful areas)
        high_sat_mask = (saturation > 100).astype(np.uint8) * 255
        
        # Clean up the mask
        kernel = np.ones((15, 15), np.uint8)
        high_sat_mask = cv2.morphologyEx(high_sat_mask, cv2.MORPH_CLOSE, kernel)
        high_sat_mask = cv2.morphologyEx(high_sat_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            high_sat_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 80 and h > 80:  # Larger minimum for saturation-based detection
                candidates.append(((x, y, x + w, y + h), "saturation"))
        
        return candidates
    
    def _calculate_region_features(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Dict[str, float]:
        """Calculate feature scores for a region."""
        x1, y1, x2, y2 = bbox
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return {"color_variance": 0, "edge_density": 0, "entropy": 0, "saturation": 0}
        
        # Color variance
        color_variance = np.var(region.astype(np.float32) / 255.0)
        
        # Edge density
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Entropy
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        region_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Saturation
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        return {
            "color_variance": float(color_variance),
            "edge_density": float(edge_density),
            "entropy": float(region_entropy),
            "saturation": float(mean_saturation)
        }
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate overall confidence score from features."""
        # Weighted combination of features
        weights = {
            "color_variance": 0.25,
            "edge_density": 0.25,
            "entropy": 0.25,
            "saturation": 0.25
        }
        
        # Normalize features to 0-1 range
        normalized = {
            "color_variance": min(1.0, features["color_variance"] / 0.1),
            "edge_density": min(1.0, features["edge_density"] / 0.3),
            "entropy": min(1.0, features["entropy"] / 7.0),
            "saturation": features["saturation"]
        }
        
        confidence = sum(weights[k] * normalized[k] for k in weights)
        return float(confidence)
    
    def _non_max_suppression(
        self, 
        regions: List[DesignPackRegion],
        iou_threshold: float = 0.5
    ) -> List[DesignPackRegion]:
        """Apply non-maximum suppression to remove duplicate detections."""
        if len(regions) <= 1:
            return regions
        
        # Sort by confidence (highest first)
        regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        
        keep = []
        used = set()
        
        for i, region in enumerate(regions):
            if i in used:
                continue
            
            keep.append(region)
            
            # Mark overlapping regions as used
            for j, other in enumerate(regions[i+1:], start=i+1):
                if j in used:
                    continue
                
                iou = region.bbox.iou(other.bbox)
                if iou >= iou_threshold:
                    used.add(j)
            
            used.add(i)
        
        return keep
    
    def generate_exclusion_mask(
        self, 
        image_shape: Tuple[int, int, int],
        regions: List[DesignPackRegion]
    ) -> np.ndarray:
        """
        Generate a binary mask of regions to exclude from translation.
        
        Args:
            image_shape: Shape of the original image (H, W, C)
            regions: List of detected design pack regions
            
        Returns:
            Binary mask where 255 = exclude, 0 = include
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for region in regions:
            bbox = region.bbox
            mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = 255
        
        return mask
