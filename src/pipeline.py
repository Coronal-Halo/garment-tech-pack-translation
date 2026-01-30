"""
Tech Pack Translation Pipeline

This module orchestrates the complete translation pipeline,
coordinating all components to process tech pack images.
"""

import logging
import time
import os
import json
from typing import Optional, Dict, Any
from dataclasses import asdict
import numpy as np
import yaml

from .models import ProcessingResult, TextType
from .text_detector import TextDetector
from .design_pack_detector import DesignPackDetector
from .translator import Translator
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


def _ensure_writable(file_path: str) -> None:
    """
    Ensure a file path is writable.
    
    If the file exists but is not writable (e.g., created by Docker as root),
    attempt to remove it first so we can create a new file.
    """
    if os.path.exists(file_path):
        if not os.access(file_path, os.W_OK):
            try:
                os.remove(file_path)
                logger.info(f"Removed non-writable file: {file_path}")
            except PermissionError:
                raise PermissionError(
                    f"Cannot write to '{file_path}'. "
                    f"The file exists but is owned by another user (likely root from Docker). "
                    f"Please remove it manually with: sudo rm '{file_path}'"
                )



class TechPackTranslationPipeline:
    """
    Main orchestrator for the tech pack translation pipeline.
    
    This class coordinates all components:
    1. Design pack detection (to identify regions to preserve)
    2. Text detection (OCR)
    3. Translation
    4. Image processing (inpainting and rendering)
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        config_path: str = None
    ):
        """
        Initialize the translation pipeline.
        
        Args:
            config: Configuration dictionary
            config_path: Path to YAML configuration file
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or self._get_default_config()
        
        # Initialize components
        self._init_components()
        
        logger.info("TechPackTranslationPipeline initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "general": {
                "output_dir": "./outputs",
                "save_intermediate": True,
            },
            "ocr": {
                "engine": "paddleocr",
                "confidence_threshold": 0.5,
                "use_gpu": False,
            },
            "design_pack_detection": {
                "enabled": True,
                "method": "multi_feature",
                "margin": 10,
            },
            "translation": {
                "service": "google",
                "source_language": "en",
                "target_language": "zh-CN",
            },
            "rendering": {
                "default_font_size": 12,
                "font_color": [0, 0, 0],
            },
            "inpainting": {
                "method": "telea",
                "radius": 3,
            },
        }
    
    def _init_components(self):
        """Initialize all pipeline components."""
        ocr_config = self.config.get("ocr", {})
        design_config = self.config.get("design_pack_detection", {})
        trans_config = self.config.get("translation", {})
        render_config = self.config.get("rendering", {})
        inpaint_config = self.config.get("inpainting", {})
        
        # Text detector
        # Create a copy of ocr_config and remove keys already passed explicitly
        text_detector_config = ocr_config.copy()
        backend = text_detector_config.pop("engine", "paddleocr")
        conf_thresh = text_detector_config.pop("confidence_threshold", 0.5)
        use_gpu = text_detector_config.pop("use_gpu", False)
        
        self.text_detector = TextDetector(
            backend=backend,
            confidence_threshold=conf_thresh,
            use_gpu=use_gpu,
            **text_detector_config
        )
        
        # Design pack detector
        self.design_detector = DesignPackDetector(
            detection_method=design_config.get("method", "multi_feature"),
            margin=design_config.get("margin", 10),
        )
        
        # Translator
        self.translator = Translator(
            service=trans_config.get("service", "google"),
            source_language=trans_config.get("source_language", "en"),
            target_language=trans_config.get("target_language", "zh-CN"),
            use_api=trans_config.get("use_api", True),
            use_local_glossary=trans_config.get("use_local_glossary", True),
            use_smart_correction=trans_config.get("use_smart_correction", True),
        )
        
        # Image processor
        self.image_processor = ImageProcessor(
            font_path=render_config.get("font_path"),
            default_font_size=render_config.get("default_font_size", 12),
            font_color=tuple(render_config.get("font_color", [0, 0, 0])),
            inpainting_method=inpaint_config.get("method", "telea"),
            inpainting_radius=inpaint_config.get("radius", 3),
            background_bias=render_config.get("background_bias", "white"),
        )
    
    def process(
        self,
        image_path: str,
        output_path: str = None,
        save_intermediate: bool = None
    ) -> ProcessingResult:
        """
        Process a tech pack image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            save_intermediate: Whether to save intermediate results
            
        Returns:
            ProcessingResult containing all processing data
        """
        start_time = time.time()
        
        logger.info(f"Processing image: {image_path}")
        
        # Determine output settings
        if save_intermediate is None:
            save_intermediate = self.config.get("general", {}).get("save_intermediate", True)
        
        if output_path is None:
            output_dir = self.config.get("general", {}).get("output_dir", "./outputs")
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_translated.png")
        
        # Step 1: Load image
        logger.info("Step 1: Loading image...")
        original_image = self.image_processor.load_image(image_path)
        
        # Step 2: Detect design pack regions
        logger.info("Step 2: Detecting design pack regions...")
        design_regions = []
        exclusion_mask = None
        
        if self.config.get("design_pack_detection", {}).get("enabled", True):
            design_regions = self.design_detector.detect(original_image)
            if design_regions:
                exclusion_mask = self.design_detector.generate_exclusion_mask(
                    original_image.shape, design_regions
                )
                logger.info(f"Found {len(design_regions)} design pack regions")
        
        # Step 3: Detect text
        logger.info("Step 3: Detecting text...")
        text_regions = self.text_detector.detect(original_image, exclusion_mask)
        logger.info(f"Detected {len(text_regions)} text regions")
        
        # Step 4: Translate text
        logger.info("Step 4: Translating text...")
        trans_config = self.config.get("translation", {})
        batch_size = trans_config.get("batch_size", 50)
        text_regions = self.translator.translate_regions(text_regions, batch_size=batch_size)
        
        # Step 5: Process image (inpaint and render)
        logger.info("Step 5: Processing image...")
        processed_image = self.image_processor.process_image(
            original_image, text_regions, exclusion_mask
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result
        result = ProcessingResult(
            original_image=original_image,
            processed_image=processed_image,
            text_regions=text_regions,
            design_pack_regions=design_regions,
            exclusion_mask=exclusion_mask,
            processing_time_seconds=processing_time,
            metadata={
                "input_path": image_path,
                "output_path": output_path,
                "config": self.config,
            }
        )
        
        # Save output
        logger.info("Saving output...")
        _ensure_writable(output_path)
        self.image_processor.save_image(processed_image, output_path)
        
        # Save intermediate results if requested
        if save_intermediate:
            self._save_intermediate_results(result, output_path)
        
        logger.info(f"Processing complete in {processing_time:.2f}s")
        logger.info(f"Output saved to: {output_path}")
        
        return result
    
    def _save_intermediate_results(
        self, 
        result: ProcessingResult, 
        output_path: str
    ):
        """Save intermediate processing results."""
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Save comparison image
        comparison = self.image_processor.create_comparison_image(
            result.original_image, result.processed_image
        )
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        _ensure_writable(comparison_path)
        self.image_processor.save_image(comparison, comparison_path)
        
        # Save detection visualization
        detection_vis = self.image_processor.visualize_detections(
            result.original_image,
            result.text_regions,
            result.design_pack_regions
        )
        detection_path = os.path.join(output_dir, f"{base_name}_detections.png")
        _ensure_writable(detection_path)
        self.image_processor.save_image(detection_vis, detection_path)
        
        # Save processing report
        report = self._generate_report(result)
        report_path = os.path.join(output_dir, f"{base_name}_report.json")
        _ensure_writable(report_path)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved intermediate results to {output_dir}")
    
    def _generate_report(self, result: ProcessingResult) -> Dict[str, Any]:
        """Generate a processing report."""
        text_summary = []
        for i, region in enumerate(result.text_regions):
            text_summary.append({
                "id": i,
                "original_text": region.text,
                "translated_text": region.translated_text,
                "text_type": region.text_type.value,
                "confidence": round(region.confidence, 3),
                "bbox": region.bbox.to_tuple(),
            })
        
        design_pack_summary = []
        for i, region in enumerate(result.design_pack_regions):
            design_pack_summary.append({
                "id": i,
                "bbox": region.bbox.to_tuple(),
                "confidence": round(region.confidence, 3),
                "detection_method": region.detection_method,
            })
        
        return {
            "summary": result.get_summary(),
            "text_regions": text_summary,
            "design_pack_regions": design_pack_summary,
            "processing_time_seconds": round(result.processing_time_seconds, 2),
        }
    
    def process_batch(
        self,
        image_paths: list,
        output_dir: str = None
    ) -> list:
        """
        Process multiple images.
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory to save outputs
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            if output_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_translated.png")
            else:
                output_path = None
            
            try:
                result = self.process(image_path, output_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append(None)
        
        return results


def create_pipeline_from_config(config_path: str) -> TechPackTranslationPipeline:
    """
    Factory function to create a pipeline from a config file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configured TechPackTranslationPipeline instance
    """
    return TechPackTranslationPipeline(config_path=config_path)
