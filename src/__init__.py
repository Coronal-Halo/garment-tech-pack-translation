"""
Tech Pack Image Translation System
Crystal International - Technical Assessment

A computer vision system for translating text in garment industry 
tech pack images from English to Chinese while preserving design graphics.
"""

__version__ = "1.0.0"
__author__ = "Yuxiang Huang"

from .text_detector import TextDetector
from .design_pack_detector import DesignPackDetector
from .translator import Translator
from .image_processor import ImageProcessor
from .pipeline import TechPackTranslationPipeline

__all__ = [
    "TextDetector",
    "DesignPackDetector", 
    "Translator",
    "ImageProcessor",
    "TechPackTranslationPipeline",
]
