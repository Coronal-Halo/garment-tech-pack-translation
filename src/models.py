"""
Data Models for Tech Pack Translation System

This module defines the data structures used throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np


class TextType(Enum):
    """Classification of text types for translation decisions."""
    TRANSLATABLE = "translatable"
    PRESERVE = "preserve"  # Codes, measurements, abbreviations
    DESIGN_PACK = "design_pack"  # Text within design pack image
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside the bounding box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def intersection(self, other: 'BoundingBox') -> Optional['BoundingBox']:
        """Calculate intersection with another bounding box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x1 < x2 and y1 < y2:
            return BoundingBox(x1, y1, x2, y2)
        return None
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another box."""
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0
        
        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0


@dataclass
class TextRegion:
    """Represents a detected text region with all associated data."""
    bbox: BoundingBox
    text: str
    confidence: float
    text_type: TextType = TextType.UNKNOWN
    translated_text: Optional[str] = None
    font_size: Optional[int] = None
    text_color: Tuple[int, int, int] = (0, 0, 0)
    background_color: Optional[Tuple[int, int, int]] = None
    rotation_angle: float = 0.0
    polygon: Optional[List[Tuple[int, int]]] = None  # Exact points from OCR
    
    def should_translate(self) -> bool:
        """Determine if this text region should be translated."""
        return self.text_type == TextType.TRANSLATABLE
    
    def get_display_text(self) -> str:
        """Get the text to display (translated if available, else original)."""
        if self.translated_text and self.text_type == TextType.TRANSLATABLE:
            return self.translated_text
        return self.text


@dataclass
class DesignPackRegion:
    """Represents a detected design pack image region."""
    bbox: BoundingBox
    confidence: float
    detection_method: str
    features: Dict[str, float] = field(default_factory=dict)
    
    def get_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate a binary mask for this region."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        mask[self.bbox.y1:self.bbox.y2, self.bbox.x1:self.bbox.x2] = 255
        return mask


@dataclass
class ProcessingResult:
    """Contains the complete result of processing a tech pack image."""
    original_image: np.ndarray
    processed_image: np.ndarray
    text_regions: List[TextRegion]
    design_pack_regions: List[DesignPackRegion]
    exclusion_mask: Optional[np.ndarray] = None
    processing_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_text_regions(self) -> int:
        return len(self.text_regions)
    
    @property
    def translated_regions(self) -> int:
        return sum(1 for r in self.text_regions if r.translated_text is not None)
    
    @property
    def preserved_regions(self) -> int:
        return sum(1 for r in self.text_regions if r.text_type == TextType.PRESERVE)
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate a summary of the processing result."""
        return {
            "total_text_regions": self.total_text_regions,
            "translated_regions": self.translated_regions,
            "preserved_regions": self.preserved_regions,
            "design_pack_regions": len(self.design_pack_regions),
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "image_dimensions": {
                "width": self.original_image.shape[1],
                "height": self.original_image.shape[0],
            }
        }


@dataclass 
class TranslationTask:
    """Represents a batch translation task."""
    texts: List[str]
    source_language: str = "en"
    target_language: str = "zh-CN"
    preserve_indices: List[int] = field(default_factory=list)
    
    def get_translatable_texts(self) -> List[Tuple[int, str]]:
        """Get list of (index, text) tuples for texts that should be translated."""
        return [
            (i, text) for i, text in enumerate(self.texts) 
            if i not in self.preserve_indices
        ]
