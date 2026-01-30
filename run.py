#!/usr/bin/env python3
"""
Tech Pack Image Translation System
Crystal International - Technical Assessment

Main entry point for the translation system.

Usage:
    python run.py --input <image_path> [--output <output_path>] [--config <config_path>]
    python run.py --help

Examples:
    # Basic usage
    python run.py --input "techpack_img 1.png"
    
    # With custom output path
    python run.py --input "techpack_img 1.png" --output "translated_output.png"
    
    # With custom configuration
    python run.py --input "techpack_img 1.png" --config "config/config.yaml"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import TechPackTranslationPipeline


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tech Pack Image Translation System - Translates English text to Chinese",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py --input "techpack_img 1.png"
    python run.py --input "techpack_img 1.png" --output "outputs/translated.png"
    python run.py --input "techpack_img 1.png" --config "config/config.yaml" --verbose
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input tech pack image"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path for output translated image (default: outputs/<input>_translated.png)"
    )
    
    parser.add_argument(
        "-c", "--config",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Don't save intermediate results (comparison images, reports)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration for OCR"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Load configuration
    config_path = args.config if os.path.exists(args.config) else None
    
    logger.info("=" * 60)
    logger.info("Tech Pack Image Translation System")
    logger.info("Crystal International - Technical Assessment")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Config: {config_path or 'Using defaults'}")
    
    # Initialize pipeline
    try:
        pipeline = TechPackTranslationPipeline(config_path=config_path)
        
        # Override GPU setting if specified
        if args.gpu:
            pipeline.config["ocr"]["use_gpu"] = True
            pipeline._init_components()  # Reinitialize with GPU
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Process image
    try:
        # Ensure output path includes original image name prefix
        final_output_path = args.output
        if final_output_path:
            # If user specified output, ensure it follows naming convention
            input_basename = os.path.splitext(os.path.basename(args.input))[0]
            output_dir = os.path.dirname(final_output_path) or "outputs"
            output_ext = os.path.splitext(final_output_path)[1] or ".png"
            # Check if output already contains the input name
            if not os.path.basename(final_output_path).startswith(input_basename):
                final_output_path = os.path.join(output_dir, f"{input_basename}_translated{output_ext}")
                logger.info(f"Output path adjusted to include original image name: {final_output_path}")
        
        result = pipeline.process(
            image_path=args.input,
            output_path=final_output_path,
            save_intermediate=not args.no_intermediate
        )
        
        # Print summary
        summary = result.get_summary()
        logger.info("=" * 60)
        logger.info("Processing Summary:")
        logger.info(f"  Total text regions detected: {summary['total_text_regions']}")
        logger.info(f"  Regions translated: {summary['translated_regions']}")
        logger.info(f"  Regions preserved: {summary['preserved_regions']}")
        logger.info(f"  Design pack regions: {summary['design_pack_regions']}")
        logger.info(f"  Processing time: {summary['processing_time_seconds']:.2f}s")
        logger.info(f"  Output saved to: {result.metadata.get('output_path')}")
        logger.info("=" * 60)
        
        logger.info("Translation completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
