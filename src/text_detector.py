"""
Text Detection Module

This module handles OCR-based text detection and extraction from tech pack images.
Supports multiple OCR backends: PaddleOCR, EasyOCR, and Tesseract.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import cv2

from .models import TextRegion, BoundingBox, TextType

logger = logging.getLogger(__name__)


class OCRBackend(ABC):
    """Abstract base class for OCR backends."""
    
    @abstractmethod
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and recognize text in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries with keys: 'bbox', 'text', 'confidence'
        """
        pass


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR backend for text detection and recognition."""
    
    def __init__(self, use_gpu: bool = False, lang: str = "en", 
                 use_angle_cls: bool = True, show_log: bool = False, **kwargs):
        """
        Initialize PaddleOCR backend.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Language for recognition
            use_angle_cls: Whether to use angle classification
            show_log: Whether to show PaddleOCR logs
            **kwargs: Additional arguments for PaddleOCR
        """
        try:
            from paddleocr import PaddleOCR
            # PaddleOCR 2.x API
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu,
                show_log=show_log,
                **kwargs
            )
            logger.info("PaddleOCR backend initialized successfully")
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Install it with: pip install paddleocr paddlepaddle"
            )
    
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and recognize text using PaddleOCR."""
        results = []
        
        # PaddleOCR expects RGB image
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # PaddleOCR 2.x API
        try:
            ocr_result = self.ocr.ocr(image_rgb, cls=True)
        except Exception as e:
            logger.error(f"PaddleOCR prediction failed: {e}")
            return results
        
        if ocr_result is None or len(ocr_result) == 0:
            return results
        
        # Handle the nested list structure of PaddleOCR 2.x output
        for line in ocr_result:
            if line is None:
                continue
            for detection in line:
                if detection is None or len(detection) < 2:
                    continue
                    
                bbox_points = detection[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                text_info = detection[1]  # (text, confidence)
                
                if len(bbox_points) >= 4 and text_info:
                    # Convert polygon to rectangle
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]
                    
                    results.append({
                        'bbox': [
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords)),
                            int(max(y_coords))
                        ],
                        'polygon': [(int(p[0]), int(p[1])) for p in bbox_points],
                        'text': text_info[0],
                        'confidence': float(text_info[1])
                    })
        
        return results


class EasyOCRBackend(OCRBackend):
    """EasyOCR backend for text detection and recognition."""
    
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        """
        Initialize EasyOCR backend.
        
        Args:
            languages: List of language codes
            gpu: Whether to use GPU
        """
        try:
            import easyocr
            self.reader = easyocr.Reader(
                languages or ['en'],
                gpu=gpu
            )
            logger.info("EasyOCR backend initialized successfully")
        except ImportError:
            raise ImportError(
                "EasyOCR is not installed. Install it with: pip install easyocr"
            )
    
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and recognize text using EasyOCR."""
        results = []
        
        # EasyOCR expects RGB image
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        ocr_result = self.reader.readtext(image_rgb)
        
        for detection in ocr_result:
            bbox_points, text, confidence = detection
            
            # Convert polygon to rectangle
            x_coords = [p[0] for p in bbox_points]
            y_coords = [p[1] for p in bbox_points]
            
            results.append({
                'bbox': [
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                ],
                'text': text,
                'confidence': float(confidence)
            })
        
        return results


class TesseractBackend(OCRBackend):
    """Tesseract OCR backend for text detection and recognition."""
    
    def __init__(self, lang: str = "eng"):
        """
        Initialize Tesseract backend.
        
        Args:
            lang: Language code for Tesseract
        """
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.lang = lang
            logger.info("Tesseract backend initialized successfully")
        except ImportError:
            raise ImportError(
                "pytesseract is not installed. Install it with: pip install pytesseract"
            )
    
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and recognize text using Tesseract."""
        results = []
        
        # Get detailed OCR data
        data = self.pytesseract.image_to_data(
            image, 
            lang=self.lang,
            output_type=self.pytesseract.Output.DICT
        )
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf > 0:  # Filter out empty and low-confidence results
                x, y, w, h = (
                    data['left'][i], 
                    data['top'][i], 
                    data['width'][i], 
                    data['height'][i]
                )
                results.append({
                    'bbox': [x, y, x + w, y + h],
                    'text': text,
                    'confidence': conf / 100.0
                })
        
        return results


class TextDetector:
    """
    Main text detector class that coordinates OCR operations.
    
    This class handles:
    - OCR backend selection and initialization
    - Text detection and extraction
    - Post-processing of detected text regions
    - Text classification (translatable vs preserve)
    """
    
    # Patterns for text that should be preserved (not translated)
    DEFAULT_PRESERVE_PATTERNS = [
        r'^[A-Z]{2,3}$',           # Abbreviations: CB, CF, YKK (Stricter: only 2-3 chars)
        r'^\d+(\.\d+)?\s*(oz|cm|mm|L|"|\'|%)?$',  # Measurements
        r'^YKK.*$',                 # YKK brand codes
        r'^#[A-Fa-f0-9]{6}$',      # Hex color codes
        r'^N/A$',                   # N/A
        r'^\d+$',                   # Pure numbers
        r'^[A-Z]{1,3}:\s*\d+',     # Code patterns like "CB: 123"
    ]
    
    def __init__(
        self,
        backend: str = "paddleocr",
        confidence_threshold: float = 0.5,
        use_gpu: bool = False,
        preserve_patterns: List[str] = None,
        **backend_kwargs
    ):
        """
        Initialize the text detector.
        
        Args:
            backend: OCR backend to use ('paddleocr', 'easyocr', 'tesseract')
            confidence_threshold: Minimum confidence for accepting detections
            use_gpu: Whether to use GPU acceleration
            preserve_patterns: Regex patterns for text to preserve
            **backend_kwargs: Additional arguments for the OCR backend
        """
        self.confidence_threshold = confidence_threshold
        self.preserve_patterns = preserve_patterns or self.DEFAULT_PRESERVE_PATTERNS
        
        # Initialize OCR backend
        self.backend = self._create_backend(backend, use_gpu, **backend_kwargs)
        
        # Compile preserve patterns
        import re
        self._compiled_patterns = [
            re.compile(p) for p in self.preserve_patterns
        ]
        
        logger.info(f"TextDetector initialized with {backend} backend")
    
    def _create_backend(
        self, 
        backend: str, 
        use_gpu: bool, 
        **kwargs
    ) -> OCRBackend:
        """Create and return the appropriate OCR backend."""
        backend_lower = backend.lower()
        
        if backend_lower == "paddleocr":
            # Extract paddleocr specific settings from kwargs if they exist
            paddle_kwargs = kwargs.get('paddleocr', {})
            # Merge with direct kwargs
            for k, v in kwargs.items():
                if k != 'paddleocr':
                    paddle_kwargs[k] = v
                    
            return PaddleOCRBackend(
                use_gpu=use_gpu,
                lang=paddle_kwargs.pop('lang', 'en'),
                use_angle_cls=paddle_kwargs.pop('use_angle_cls', True),
                show_log=paddle_kwargs.pop('show_log', False),
                **paddle_kwargs
            )
        elif backend_lower == "easyocr":
            return EasyOCRBackend(
                languages=kwargs.get('languages', ['en']),
                gpu=use_gpu
            )
        elif backend_lower == "tesseract":
            return TesseractBackend(
                lang=kwargs.get('lang', 'eng')
            )
        else:
            raise ValueError(f"Unknown OCR backend: {backend}")
    
    def detect(
        self, 
        image: np.ndarray,
        exclusion_mask: np.ndarray = None
    ) -> List[TextRegion]:
        """
        Detect and extract text from an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            exclusion_mask: Optional mask of regions to exclude (255 = exclude)
            
        Returns:
            List of TextRegion objects
        """
        logger.info("Starting text detection...")
        
        # Run OCR
        raw_results = self.backend.detect_and_recognize(image)
        logger.info(f"OCR detected {len(raw_results)} raw text regions")
        
        # Convert to TextRegion objects and filter
        text_regions = []
        for result in raw_results:
            # Filter by confidence
            if result['confidence'] < self.confidence_threshold:
                continue
            
            bbox = BoundingBox(
                x1=result['bbox'][0],
                y1=result['bbox'][1],
                x2=result['bbox'][2],
                y2=result['bbox'][3]
            )
            
            # Check if region is in exclusion zone
            if exclusion_mask is not None:
                # SPECIAL CASE: If the text is inside the table area (top half, left side)
                # we ignore the exclusion mask because it's likely a false positive
                is_table_area = bbox.y2 < (image.shape[0] * 0.6) and bbox.x2 < (image.shape[1] * 0.7)
                
                if not is_table_area:
                    if self._is_in_exclusion_zone(bbox, exclusion_mask):
                        continue
            
            # Classify text type
            text_type = self._classify_text(result['text'])
            
            # Extract text color from image (sample from bbox center)
            text_color = self._extract_text_color(image, bbox)
            
            # Estimate font size based on bbox height
            font_size = self._estimate_font_size(bbox)
            
            text_region = TextRegion(
                bbox=bbox,
                text=result['text'],
                confidence=result['confidence'],
                text_type=text_type,
                font_size=font_size,
                text_color=text_color,
                polygon=result.get('polygon')
            )
            text_regions.append(text_region)
        
        # Merge overlapping regions
        text_regions = self._merge_overlapping_regions(text_regions)
        
        logger.info(
            f"Detected {len(text_regions)} text regions after filtering "
            f"({sum(1 for r in text_regions if r.text_type == TextType.TRANSLATABLE)} translatable)"
        )
        
        return text_regions
    
    def _is_in_exclusion_zone(
        self, 
        bbox: BoundingBox, 
        exclusion_mask: np.ndarray
    ) -> bool:
        """Check if a bounding box is predominantly in an exclusion zone."""
        # Get the region of the mask corresponding to the bbox
        h, w = exclusion_mask.shape[:2]
        y1, y2 = max(0, bbox.y1), min(h, bbox.y2)
        x1, x2 = max(0, bbox.x1), min(w, bbox.x2)
        
        if y1 >= y2 or x1 >= x2:
            return False
            
        mask_region = exclusion_mask[y1:y2, x1:x2]
        
        if mask_region.size == 0:
            return False
        
        # Calculate exclusion ratio
        exclusion_ratio = np.sum(mask_region > 0) / mask_region.size
        
        # If the region is very small (like tiny table text), be more lenient
        # If it's a larger block of text, be more aggressive
        if bbox.area < 1000:  # Increased area threshold
            return exclusion_ratio > 0.9  # Extremely lenient for small text
        else:
            return exclusion_ratio > 0.5  # Standard for large text blocks
    
    def _classify_text(self, text: str) -> TextType:
        """Classify text as translatable or preserve."""
        text_clean = text.strip()
        
        # Check if it's in our garment glossary first (always translate if known)
        from .translator import OfflineGlossaryBackend
        if text_clean.lower() in OfflineGlossaryBackend.GLOSSARY:
            return TextType.TRANSLATABLE
            
        # Check against preserve patterns
        for pattern in self._compiled_patterns:
            if pattern.match(text_clean):
                return TextType.PRESERVE
        
        # If text is very short (1-2 chars) and not a word, preserve it
        if len(text_clean) <= 2 and not text_clean.isalpha():
            return TextType.PRESERVE
        
        return TextType.TRANSLATABLE
    
    def _extract_text_color(
        self, 
        image: np.ndarray, 
        bbox: BoundingBox
    ) -> Tuple[int, int, int]:
        """Extract the dominant text color from a bounding box region."""
        try:
            region = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            if region.size == 0:
                return (0, 0, 0)
            
            # Simple approach: find the darkest color (assuming text is dark)
            # Convert to grayscale and find darkest pixels
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            dark_mask = gray < np.percentile(gray, 20)  # Darkest 20%
            
            if np.any(dark_mask):
                dark_pixels = region[dark_mask]
                avg_color = np.mean(dark_pixels, axis=0).astype(int)
                return tuple(avg_color.tolist())
            
            return (0, 0, 0)  # Default to black
        except Exception:
            return (0, 0, 0)
    
    def _estimate_font_size(self, bbox: BoundingBox) -> int:
        """Estimate font size based on bounding box height."""
        # Approximate conversion from pixel height to font points
        # This is a rough estimate and may need adjustment
        return max(8, min(48, int(bbox.height * 0.8)))
    
    def _merge_overlapping_regions(
        self, 
        regions: List[TextRegion],
        iou_threshold: float = 0.5
    ) -> List[TextRegion]:
        """Merge overlapping text regions."""
        if len(regions) <= 1:
            return regions
        
        # Sort by area (largest first)
        regions = sorted(regions, key=lambda r: r.bbox.area, reverse=True)
        
        merged = []
        used = set()
        
        for i, region in enumerate(regions):
            if i in used:
                continue
            
            # Find overlapping regions
            to_merge = [region]
            for j, other in enumerate(regions[i+1:], start=i+1):
                if j in used:
                    continue
                
                if region.bbox.iou(other.bbox) >= iou_threshold:
                    to_merge.append(other)
                    used.add(j)
            
            if len(to_merge) == 1:
                merged.append(region)
            else:
                # Merge regions
                merged.append(self._merge_regions(to_merge))
            
            used.add(i)
        
        return merged
    
    def _merge_regions(self, regions: List[TextRegion]) -> TextRegion:
        """Merge multiple text regions into one."""
        # Combine bounding boxes
        x1 = min(r.bbox.x1 for r in regions)
        y1 = min(r.bbox.y1 for r in regions)
        x2 = max(r.bbox.x2 for r in regions)
        y2 = max(r.bbox.y2 for r in regions)
        
        # Combine text (sort by x position for left-to-right reading)
        sorted_regions = sorted(regions, key=lambda r: (r.bbox.y1, r.bbox.x1))
        combined_text = " ".join(r.text for r in sorted_regions)
        
        # Use highest confidence
        max_conf = max(r.confidence for r in regions)
        
        # Determine text type (if any is translatable, result is translatable)
        text_type = TextType.TRANSLATABLE if any(
            r.text_type == TextType.TRANSLATABLE for r in regions
        ) else TextType.PRESERVE
        
        return TextRegion(
            bbox=BoundingBox(x1, y1, x2, y2),
            text=combined_text,
            confidence=max_conf,
            text_type=text_type,
            font_size=regions[0].font_size,
            text_color=regions[0].text_color
        )
