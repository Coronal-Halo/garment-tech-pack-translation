"""
Image Processing Module

This module handles image loading, text inpainting, and Chinese text rendering.
"""

import logging
import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .models import TextRegion, BoundingBox, TextType

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles image processing operations including:
    - Image loading and saving
    - Text region inpainting (removal)
    - Chinese text rendering
    - Font management
    """
    
    # Default Chinese fonts to try (in order of preference)
    DEFAULT_CHINESE_FONTS = [
        "NotoSansSC-Regular.ttf",
        "NotoSansCJK-Regular.ttc",
        "SimHei.ttf",
        "Microsoft YaHei.ttf",
        "PingFang.ttc",
        "WenQuanYi Micro Hei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    
    def __init__(
        self,
        font_path: str = None,
        default_font_size: int = 12,
        min_font_size: int = 8,
        max_font_size: int = 48,
        font_color: Tuple[int, int, int] = (0, 0, 0),
        inpainting_method: str = "telea",
        inpainting_radius: int = 3,
        mask_dilation: int = 2,
        background_bias: str = "white"
    ):
        """
        Initialize the image processor.
        
        Args:
            font_path: Path to Chinese font file
            default_font_size: Default font size for rendering
            min_font_size: Minimum font size
            max_font_size: Maximum font size
            font_color: Default font color (RGB)
            inpainting_method: Inpainting method ('telea' or 'ns')
            inpainting_radius: Radius for inpainting algorithm
            mask_dilation: Dilation amount for text masks
            background_bias: Bias for background fill color ('white' or 'local')
        """
        self.font_path = font_path or self._find_chinese_font()
        self.default_font_size = default_font_size
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.font_color = font_color
        self.inpainting_method = inpainting_method
        self.inpainting_radius = inpainting_radius
        self.mask_dilation = mask_dilation
        self.background_bias = background_bias
        
        # Preload font
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        
        logger.info(f"ImageProcessor initialized with font: {self.font_path} (Bias: {background_bias})")
    
    def _find_chinese_font(self) -> Optional[str]:
        """Find an available Chinese font on the system."""
        # Check assets folder first
        assets_fonts = [
            "./assets/fonts/NotoSansSC-Regular.ttf",
            "assets/fonts/NotoSansSC-Regular.ttf",
        ]
        
        for font_path in assets_fonts + self.DEFAULT_CHINESE_FONTS:
            if os.path.exists(font_path):
                return font_path
        
        # Try to find system fonts
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            "/System/Library/Fonts",  # macOS
            "C:\\Windows\\Fonts",  # Windows
        ]
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if any(name in file.lower() for name in ['noto', 'cjk', 'chinese', 'simsun', 'simhei']):
                            return os.path.join(root, file)
        
        logger.warning("No Chinese font found, text rendering may fail")
        return None
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a font object for the specified size (with caching)."""
        if size in self._font_cache:
            return self._font_cache[size]
        
        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, size)
            else:
                # Fallback to default font
                font = ImageFont.load_default()
                logger.warning(f"Using default font, Chinese characters may not render correctly")
        except Exception as e:
            logger.error(f"Failed to load font: {e}")
            font = ImageFont.load_default()
        
        self._font_cache[size] = font
        return font
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array (BGR format)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.info(f"Loaded image: {image_path} (shape: {image.shape})")
        return image
    
    def save_image(
        self, 
        image: np.ndarray, 
        output_path: str,
        quality: int = 95
    ) -> str:
        """
        Save an image to file.
        
        Args:
            image: Image as numpy array (BGR format)
            output_path: Path to save the image
            quality: JPEG quality (for JPEG format)
            
        Returns:
            Path to saved image
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Determine format from extension
        ext = os.path.splitext(output_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(output_path, image)
        
        logger.info(f"Saved image: {output_path}")
        return output_path
    
    def create_text_mask(
        self, 
        image_shape: Tuple[int, int, int],
        text_regions: List[TextRegion],
        dilation: int = None
    ) -> np.ndarray:
        """
        Create a binary mask of text regions for inpainting.
        
        Args:
            image_shape: Shape of the image (H, W, C)
            text_regions: List of text regions to mask
            dilation: Amount to dilate the mask (for better inpainting)
            
        Returns:
            Binary mask (255 = text region, 0 = background)
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        dilation = dilation if dilation is not None else self.mask_dilation
        
        for region in text_regions:
            if region.text_type == TextType.TRANSLATABLE and region.translated_text:
                if region.polygon:
                    # Use exact polygon if available for clean masking
                    poly_pts = np.array(region.polygon, dtype=np.int32)
                    cv2.fillPoly(mask, [poly_pts], 255)
                else:
                    # Fallback to rectangular bbox
                    bbox = region.bbox
                    mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = 255
        
        # Dilate the mask for better inpainting coverage
        if dilation > 0:
            kernel = np.ones((dilation * 2 + 1, dilation * 2 + 1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def inpaint_text_regions(
        self, 
        image: np.ndarray,
        text_regions: List[TextRegion]
    ) -> np.ndarray:
        """
        Remove text from image using smart background filling.
        
        Args:
            image: Input image (BGR format)
            text_regions: List of text regions to inpaint
            
        Returns:
            Image with text regions cleaned
        """
        logger.info("Cleaning text regions...")
        result_img = image.copy()
        
        cleaned_count = 0
        for region in text_regions:
            if region.text_type == TextType.TRANSLATABLE and region.translated_text:
                if region.polygon:
                    # Get the background color of the region
                    # We sample the edges of the bbox to get the background color
                    bbox = region.bbox
                    bg_color = self._estimate_background_color(image, bbox)
                    
                    # Fill the exact polygon with the solid background color (Clean Fill)
                    poly_pts = np.array(region.polygon, dtype=np.int32)
                    cv2.fillPoly(result_img, [poly_pts], bg_color)
                    cleaned_count += 1
                else:
                    # Fallback to bbox if polygon is missing
                    bbox = region.bbox
                    bg_color = self._estimate_background_color(image, bbox)
                    cv2.rectangle(result_img, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), bg_color, -1)
                    cleaned_count += 1
        
        logger.info(f"Cleaned {cleaned_count} text regions using Smart Fill")
        return result_img

    def _estimate_background_color(self, image: np.ndarray, bbox: BoundingBox) -> Tuple[int, int, int]:
        """Estimate the background color of a region by sampling its edges."""
        if self.background_bias == "white":
            # If biased towards white, we still sample the local color but blend it with white
            # or simply return white if the local color is light enough
            h, w = image.shape[:2]
            y1, y2 = max(0, bbox.y1-2), min(h, bbox.y2+2)
            x1, x2 = max(0, bbox.x1-2), min(w, bbox.x2+2)
            region = image[y1:y2, x1:x2]
            if region.size == 0: return (255, 255, 255)
            
            # Median color
            local_median = np.median(region.reshape(-1, 3), axis=0)
            
            # If the local color is already light (avg > 200), just use white for maximum cleanliness
            if np.mean(local_median) > 200:
                return (255, 255, 255)
            
            # Otherwise, blend local color with white (70% white, 30% local)
            blended = (0.7 * 255 + 0.3 * local_median).astype(int)
            return tuple(blended.tolist())
            
        h, w = image.shape[:2]
        # Expand slightly to get surrounding pixels
        y1, y2 = max(0, bbox.y1-2), min(h, bbox.y2+2)
        x1, x2 = max(0, bbox.x1-2), min(w, bbox.x2+2)
        
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return (255, 255, 255)
            
        # Use median color of the region as a robust estimate of background
        # (Assuming text takes up less than 50% of the pixels)
        median_color = np.median(region.reshape(-1, 3), axis=0).astype(int)
        return tuple(median_color.tolist())
    
    def calculate_optimal_font_size(
        self, 
        text: str, 
        bbox: BoundingBox,
        max_iterations: int = 10
    ) -> int:
        """
        Calculate the optimal font size to fit text within a bounding box.
        
        Args:
            text: Text to render
            bbox: Target bounding box
            max_iterations: Maximum iterations for binary search
            
        Returns:
            Optimal font size
        """
        target_width = bbox.width
        target_height = bbox.height
        
        # Binary search for optimal size
        low = self.min_font_size
        high = min(self.max_font_size, target_height)
        optimal_size = self.default_font_size
        
        for _ in range(max_iterations):
            mid = (low + high) // 2
            font = self._get_font(mid)
            
            # Get text size
            try:
                # Use getbbox for newer Pillow versions
                dummy_img = Image.new('RGB', (1, 1))
                draw = ImageDraw.Draw(dummy_img)
                bbox_result = draw.textbbox((0, 0), text, font=font)
                text_width = bbox_result[2] - bbox_result[0]
                text_height = bbox_result[3] - bbox_result[1]
            except AttributeError:
                # Fallback for older Pillow versions
                text_width, text_height = font.getsize(text)
            
            # Check if text fits
            if text_width <= target_width and text_height <= target_height:
                optimal_size = mid
                low = mid + 1
            else:
                high = mid - 1
            
            if low > high:
                break
        
        return optimal_size
    
    def render_text_on_image(
        self, 
        image: np.ndarray,
        text_regions: List[TextRegion],
        use_optimal_sizing: bool = True
    ) -> np.ndarray:
        """
        Render translated text on the image.
        
        Args:
            image: Input image (BGR format)
            text_regions: List of text regions with translations
            use_optimal_sizing: Whether to auto-size text to fit bounding boxes
            
        Returns:
            Image with rendered text
        """
        logger.info("Rendering translated text...")
        
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        rendered_count = 0
        
        for region in text_regions:
            # Only render translatable regions with translations
            if region.text_type != TextType.TRANSLATABLE:
                continue
            
            if not region.translated_text:
                continue
            
            text = region.translated_text
            bbox = region.bbox
            
            # Calculate font size
            if use_optimal_sizing:
                font_size = self.calculate_optimal_font_size(text, bbox)
            else:
                font_size = region.font_size or self.default_font_size
            
            font = self._get_font(font_size)
            
            # Get text color
            text_color = region.text_color or self.font_color
            # Convert BGR to RGB if needed
            if len(text_color) == 3:
                text_color = (text_color[2], text_color[1], text_color[0])  # BGR to RGB
            
            # Calculate text position (center in bbox)
            try:
                bbox_result = draw.textbbox((0, 0), text, font=font)
                text_width = bbox_result[2] - bbox_result[0]
                text_height = bbox_result[3] - bbox_result[1]
            except AttributeError:
                text_width, text_height = font.getsize(text)
            
            x = bbox.x1 + (bbox.width - text_width) // 2
            y = bbox.y1 + (bbox.height - text_height) // 2
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, pil_image.width - text_width))
            y = max(0, min(y, pil_image.height - text_height))
            
            # Draw text
            try:
                draw.text((x, y), text, font=font, fill=text_color)
                rendered_count += 1
            except Exception as e:
                logger.error(f"Failed to render text '{text}': {e}")
        
        # Convert back to BGR
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        logger.info(f"Rendered {rendered_count} text regions")
        return result
    
    def process_image(
        self, 
        image: np.ndarray,
        text_regions: List[TextRegion],
        exclusion_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Complete image processing pipeline:
        1. Inpaint original text
        2. Render translated text
        
        Args:
            image: Input image (BGR format)
            text_regions: List of text regions with translations
            exclusion_mask: Optional mask of regions to exclude
            
        Returns:
            Processed image with translated text
        """
        logger.info("Starting image processing pipeline...")
        
        # Step 1: Inpaint text regions
        inpainted = self.inpaint_text_regions(image, text_regions)
        
        # Step 2: Render translated text
        result = self.render_text_on_image(inpainted, text_regions)
        
        logger.info("Image processing complete")
        return result
    
    def create_comparison_image(
        self, 
        original: np.ndarray, 
        processed: np.ndarray,
        orientation: str = "horizontal"
    ) -> np.ndarray:
        """
        Create a side-by-side comparison image.
        
        Args:
            original: Original image
            processed: Processed image
            orientation: 'horizontal' or 'vertical'
            
        Returns:
            Comparison image
        """
        if orientation == "horizontal":
            # Add labels
            h, w = original.shape[:2]
            label_height = 30
            
            # Create labeled images
            original_labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
            original_labeled[label_height:, :] = original
            original_labeled[:label_height, :] = (255, 255, 255)
            
            processed_labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
            processed_labeled[label_height:, :] = processed
            processed_labeled[:label_height, :] = (255, 255, 255)
            
            # Add text labels
            cv2.putText(original_labeled, "Original", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(processed_labeled, "Translated", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            return np.hstack([original_labeled, processed_labeled])
        else:
            return np.vstack([original, processed])
    
    def visualize_detections(
        self, 
        image: np.ndarray,
        text_regions: List[TextRegion],
        design_regions: List = None
    ) -> np.ndarray:
        """
        Create a visualization of detected regions.
        
        Args:
            image: Original image
            text_regions: Detected text regions
            design_regions: Detected design pack regions
            
        Returns:
            Image with visualized detections
        """
        vis = image.copy()
        
        # Draw text regions
        for region in text_regions:
            bbox = region.bbox
            if region.text_type == TextType.TRANSLATABLE:
                color = (0, 255, 0)  # Green for translatable
            else:
                color = (0, 165, 255)  # Orange for preserved
            
            cv2.rectangle(vis, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
        
        # Draw design pack regions
        if design_regions:
            for region in design_regions:
                bbox = region.bbox
                cv2.rectangle(vis, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), 
                             (255, 0, 0), 3)  # Blue for design pack
        
        return vis
