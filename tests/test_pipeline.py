"""
Unit tests for the Tech Pack Translation System.

Run with: pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import BoundingBox, TextRegion, TextType, DesignPackRegion


class TestBoundingBox:
    """Tests for BoundingBox data structure."""
    
    def test_bbox_creation(self):
        """Test bounding box creation."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 80
    
    def test_bbox_dimensions(self):
        """Test width and height calculation."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        assert bbox.width == 90
        assert bbox.height == 60
        assert bbox.area == 90 * 60
    
    def test_bbox_center(self):
        """Test center point calculation."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert bbox.center == (50, 50)
    
    def test_bbox_contains_point(self):
        """Test point containment check."""
        bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50)
        assert bbox.contains_point(30, 30) == True
        assert bbox.contains_point(5, 5) == False
        assert bbox.contains_point(60, 60) == False
    
    def test_bbox_iou(self):
        """Test Intersection over Union calculation."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        bbox2 = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        
        iou = bbox1.iou(bbox2)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IOU: 2500 / 17500 ≈ 0.143
        assert 0.14 < iou < 0.15
    
    def test_bbox_no_intersection(self):
        """Test IOU with no intersection."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        bbox2 = BoundingBox(x1=100, y1=100, x2=150, y2=150)
        
        assert bbox1.iou(bbox2) == 0.0


class TestTextRegion:
    """Tests for TextRegion data structure."""
    
    def test_text_region_creation(self):
        """Test text region creation."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        region = TextRegion(
            bbox=bbox,
            text="Hello World",
            confidence=0.95,
            text_type=TextType.TRANSLATABLE
        )
        
        assert region.text == "Hello World"
        assert region.confidence == 0.95
        assert region.text_type == TextType.TRANSLATABLE
    
    def test_should_translate(self):
        """Test translation decision logic."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        
        translatable = TextRegion(
            bbox=bbox,
            text="Description",
            confidence=0.9,
            text_type=TextType.TRANSLATABLE
        )
        
        preserve = TextRegion(
            bbox=bbox,
            text="DTM",
            confidence=0.9,
            text_type=TextType.PRESERVE
        )
        
        assert translatable.should_translate() == True
        assert preserve.should_translate() == False
    
    def test_get_display_text(self):
        """Test display text retrieval."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        
        # With translation
        region = TextRegion(
            bbox=bbox,
            text="Fabric",
            confidence=0.9,
            text_type=TextType.TRANSLATABLE,
            translated_text="面料"
        )
        assert region.get_display_text() == "面料"
        
        # Without translation (preserved)
        region2 = TextRegion(
            bbox=bbox,
            text="DTM",
            confidence=0.9,
            text_type=TextType.PRESERVE
        )
        assert region2.get_display_text() == "DTM"


class TestTextClassification:
    """Tests for text classification logic."""
    
    def test_preserve_patterns(self):
        """Test that preserve patterns are correctly identified."""
        from src.translator import Translator
        
        translator = Translator(service="offline")
        
        # Should be preserved
        assert translator.should_preserve("DTM") == True
        assert translator.should_preserve("CB") == True
        assert translator.should_preserve("YKK") == True
        assert translator.should_preserve("18") == True
        assert translator.should_preserve("5.5 oz") == True
        assert translator.should_preserve("N/A") == True
        assert translator.should_preserve("#FF0000") == True
        
        # Should be translated
        assert translator.should_preserve("Fabric") == False
        assert translator.should_preserve("Main Label") == False
        assert translator.should_preserve("Additional Notes") == False


class TestOfflineGlossary:
    """Tests for the offline translation glossary."""
    
    def test_glossary_translations(self):
        """Test that glossary contains expected terms."""
        from src.translator import OfflineGlossaryBackend
        
        glossary = OfflineGlossaryBackend()
        
        # Test known translations
        assert glossary.translate("fabric", "en", "zh-CN") == "面料"
        assert glossary.translate("cotton", "en", "zh-CN") == "棉"
        assert glossary.translate("zipper", "en", "zh-CN") == "拉链"
        assert glossary.translate("main label", "en", "zh-CN") == "主标"
    
    def test_case_insensitivity(self):
        """Test case-insensitive translation."""
        from src.translator import OfflineGlossaryBackend
        
        glossary = OfflineGlossaryBackend()
        
        assert glossary.translate("FABRIC", "en", "zh-CN") == "面料"
        assert glossary.translate("Fabric", "en", "zh-CN") == "面料"
        assert glossary.translate("fabric", "en", "zh-CN") == "面料"


class TestDesignPackDetection:
    """Tests for design pack detection."""
    
    def test_mask_generation(self):
        """Test exclusion mask generation."""
        from src.design_pack_detector import DesignPackDetector
        
        detector = DesignPackDetector()
        
        # Create dummy regions
        regions = [
            DesignPackRegion(
                bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
                confidence=0.9,
                detection_method="test"
            )
        ]
        
        # Generate mask
        image_shape = (500, 500, 3)
        mask = detector.generate_exclusion_mask(image_shape, regions)
        
        # Check mask properties
        assert mask.shape == (500, 500)
        assert mask[150, 150] == 255  # Inside region
        assert mask[50, 50] == 0      # Outside region


# Integration test (requires actual image and dependencies)
@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        from src.pipeline import TechPackTranslationPipeline
        
        # Should not raise
        pipeline = TechPackTranslationPipeline()
        assert pipeline is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
