"""
Translation Module

This module handles text translation from English to Chinese.
Supports multiple translation backends with fallback options.
"""

import logging
import os
import re
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from .models import TextRegion, TextType

logger = logging.getLogger(__name__)


class TranslationBackend(ABC):
    """Abstract base class for translation backends."""
    
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text string."""
        pass
    
    @abstractmethod
    def translate_batch(
        self, 
        texts: List[str], 
        source_lang: str, 
        target_lang: str
    ) -> List[str]:
        """Translate a batch of texts."""
        pass


class GoogleTranslateBackend(TranslationBackend):
    """Google Translate API backend."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Google Translate backend.
        
        Args:
            api_key: Google Cloud API key (if None, uses GOOGLE_TRANSLATE_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_TRANSLATE_API_KEY")
        
        try:
            from googletrans import Translator
            self.translator = Translator()
            self.use_free_api = True
            logger.info("Google Translate (free API) initialized")
        except ImportError:
            logger.warning("googletrans not installed, translation will use fallback")
            self.translator = None
            self.use_free_api = False
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text using Google Translate."""
        if not self.translator:
            return text
        
        try:
            # Handle language code mapping
            target = "zh-cn" if target_lang.lower() == "zh-cn" else target_lang
            source = "en" if source_lang.lower() == "en" else source_lang
            
            result = self.translator.translate(text, src=source, dest=target)
            return result.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def translate_batch(
        self, 
        texts: List[str], 
        source_lang: str, 
        target_lang: str
    ) -> List[str]:
        """Translate a batch of texts."""
        results = []
        for text in texts:
            results.append(self.translate(text, source_lang, target_lang))
        return results


class DeepLBackend(TranslationBackend):
    """DeepL API backend."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize DeepL backend.
        
        Args:
            api_key: DeepL API key (if None, uses DEEPL_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("DEEPL_API_KEY")
        
        if self.api_key:
            try:
                import deepl
                self.translator = deepl.Translator(self.api_key)
                logger.info("DeepL translator initialized")
            except ImportError:
                logger.warning("deepl package not installed")
                self.translator = None
        else:
            logger.warning("DeepL API key not provided")
            self.translator = None
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using DeepL."""
        if not self.translator:
            return text
        
        try:
            # Map language codes
            target = "ZH" if "zh" in target_lang.lower() else target_lang.upper()
            
            result = self.translator.translate_text(
                text, 
                source_lang=source_lang.upper() if source_lang else None,
                target_lang=target
            )
            return result.text
        except Exception as e:
            logger.error(f"DeepL translation error: {e}")
            return text
    
    def translate_batch(
        self, 
        texts: List[str], 
        source_lang: str, 
        target_lang: str
    ) -> List[str]:
        """Translate a batch of texts using DeepL."""
        if not self.translator:
            return texts
        
        try:
            target = "ZH" if "zh" in target_lang.lower() else target_lang.upper()
            results = self.translator.translate_text(
                texts,
                source_lang=source_lang.upper() if source_lang else None,
                target_lang=target
            )
            return [r.text for r in results]
        except Exception as e:
            logger.error(f"DeepL batch translation error: {e}")
            return texts


class OfflineGlossaryBackend(TranslationBackend):
    """
    Offline translation using a garment industry glossary.
    """
    
    # OCR Corrections (Handle common OCR misspellings)
    CORRECTIONS = {
        "topshitch": "topstitch",
        "topstiching": "topstitching",
        "topshitching": "topstitching",
        "zig zig": "zig zag",
        "zig zac": "zig zag",
        "zig-zig": "zig-zag",
        "zig-zac": "zig-zag",
    }

    # Garment industry English-Chinese glossary
    GLOSSARY = {
        # General terms
        "style name": "款式名称",
        "style number": "款号",
        "brand name": "品牌名称",
        "designer's name": "设计师姓名",
        "season": "季节",
        "release": "发布",
        "factory details": "工厂详情",
        "target volume": "目标产量",
        "size range": "尺码范围",
        "sample": "样品",
        "pack": "包",
        "image": "图像",
        "natural": "自然色",
        "natural color": "自然色",
        "contact information": "联系信息",
        "pattern file name": "版型文件名",
        "colorways": "配色方案",
        "product category": "产品类别",
        "style description": "款式描述",
        "fit block reference": "版型参考",
        
        # Fabric and materials
        "fabric": "面料",
        "cotton": "棉",
        "polyester": "聚酯纤维",
        "spandex": "氨纶",
        "poplin": "府绸",
        "hemp": "麻",
        "thread": "线",
        "piping cord": "滚边绳",
        "interfacing": "衬布",
        "fusable": "粘合衬",
        "light weight": "轻薄",
        "main fabric": "主面料",
        "lining": "里料",
        
        # Components
        "zipper": "拉链",
        "close end": "闭口",
        "auto-lock slider": "自动锁滑块",
        "button": "纽扣",
        "snap": "按扣",
        "hook": "钩扣",
        "label": "标签",
        "main label": "主标",
        "size label": "尺码标",
        "care label": "洗标",
        "content label": "成分标",
        "hang tag": "吊牌",
        "price tag": "价格标签",
        "swift tack": "速钉",
        "recycled": "再生",
        "polypropylene": "聚丙烯",
        
        # Packaging
        "packaging": "包装",
        "garment bag": "服装袋",
        "tissue paper": "薄纸",
        "desiccant": "干燥剂",
        "eco": "环保",
        "recycled paper": "再生纸",
        
        # Placement
        "placement": "位置",
        "all seams": "所有缝线",
        "armholes": "袖窿",
        "back neck": "后领",
        "backneck": "后领",
        "neck": "领子",
        "center back": "后中",
        "center front": "前中",
        "front": "前片",
        "back": "后片",
        "sleeve": "袖子",
        "collar": "领子",
        "pocket": "口袋",
        "hem": "下摆",
        "waist": "腰部",
        "cuff": "袖口",
        
        # Color
        "color": "颜色",
        "color details": "颜色详情",
        "blue light": "浅蓝",
        "mocha mousse": "摩卡慕斯",
        "ultimate gray": "极致灰",
        "beige": "米色",
        "white": "白色",
        "black": "黑色",
        "natural": "自然色",
        "clear": "透明",
        
        # Quantities and measurements
        "quantity": "数量",
        "cost": "成本",
        "unit of measure": "计量单位",
        "yards": "码",
        "each": "每个",
        "as per requirement": "按需求",
        "''": "英寸",
        "\"": "英寸",
        "inch": "英寸",
        "inches": "英寸",
        "from hem": "从下摆",
        "from cf": "从前中",
        "from cb": "从后中",
        "single": "单",
        "clay": "粘土",
        "micro-pak": "防霉片",
        "micro pak": "防霉片",
        "micro-pak clay": "防霉片(粘土)",
        
        # Descriptions
        "description": "描述",
        "printed logo": "印花标志",
        "screen printed": "丝网印刷",
        "additional notes": "附加说明",
        "comments": "备注",
        
        # Stitching
        "topstitch": "明线",
        "topstitching": "明线",
        "single needle": "单针",
        "double needle": "双针",
        "zig zag": "锯齿",
        "zig-zag": "锯齿",
        "heavy thread": "粗线",
        "use heavy thread for all topstitching": "所有明线使用粗线",
        "single needle topstitch": "单针明线",
        "double needle topstitch": "双针明线",
        "zig zag topstitch": "锯齿明线",
        "use heavy": "使用粗线",
        
        # Design pack
        "design pack image": "设计图",
        "design": "设计",
        
        # Common terms
        "from": "从",
        "all": "全部",
        "per": "每",
        "with": "带有",
        "and": "和",
        "or": "或",
        "for": "用于",
    }
    
    def __init__(self):
        """Initialize the offline glossary backend."""
        # Create case-insensitive lookup
        self.lookup = {k.lower(): v for k, v in self.GLOSSARY.items()}
        self.corrections = {k.lower(): v for k, v in self.CORRECTIONS.items()}
        logger.info("Offline glossary backend initialized")
    
    def smart_correct(self, text: str) -> str:
        """Correct OCR errors based on the CORRECTIONS map."""
        text_lower = text.lower().strip()
        for error, correction in self.corrections.items():
            if error in text_lower:
                text = re.sub(re.escape(error), correction, text, flags=re.IGNORECASE)
        return text

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using the offline glossary with fuzzy matching and measurement handling."""
        if not text:
            return text
            
        text_clean = text.strip()
        # Even more aggressive cleaning for lookup
        text_norm = re.sub(r'[^a-zA-Z0-9\'\"]+', ' ', text_clean).lower().strip()
        text_lower = text_clean.lower()
        
        # 1. Handle measurements like 4'' or 4" or 4 '
        measurement_match = re.match(r'^(\d+(?:\.\d+)?)\s*(\'\'|\'|\"|inch|inches|in)$', text_clean, re.IGNORECASE)
        if measurement_match:
            value = measurement_match.group(1)
            return f"{value}英寸"
        
        # 2. Direct lookup in lookup table (case-insensitive and normalized)
        search_key = text_norm.rstrip(':/-,. ')
        if search_key in self.lookup:
            return self.lookup[search_key]
            
        # 3. Try partial matches for complex strings
        result = text_clean
        
        # Special case for measurements: handle cases like 4'', 4", 4inch, 4 from hem, etc.
        result = re.sub(r'(\d+(?:\.\d+)?)\s*(\'\'|\'|\"|inch|inches|in\b)', r'\1英寸', result, flags=re.IGNORECASE)
        
        # Handle cases where the number is joined with text
        result = re.sub(r'^(\d+)([a-zA-Z])', r'\1 \2', result) 
        
        for eng, chi in sorted(self.lookup.items(), key=lambda x: len(x[0]), reverse=True):
            if len(eng) < 2: continue
            
            # Use regex for variations, ensure we match whole words or specific terms
            pattern = r'\b' + re.escape(eng) + r'\b'
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, chi, result, flags=re.IGNORECASE)
            elif eng in text_norm: # Fallback for cases where word boundaries fail due to OCR
                result = re.sub(re.escape(eng), chi, result, flags=re.IGNORECASE)
        
        return result
    
    def translate_batch(
        self, 
        texts: List[str], 
        source_lang: str, 
        target_lang: str
    ) -> List[str]:
        """Translate a batch of texts using the glossary."""
        return [self.translate(text, source_lang, target_lang) for text in texts]


class Translator:
    """
    Main translator class that coordinates translation operations.
    """
    
    # Patterns for text that should be preserved (not translated)
    DEFAULT_PRESERVE_PATTERNS = [
        r'^[A-Z]{2,3}$',           # Only 2-3 char all-caps like CB, YKK
        r'^\d+(\.\d+)?\s*(oz|cm|mm|L|"|\'|%)?$',  # Measurements
        r'^YKK.*$',                 # YKK brand codes
        r'^#[A-Fa-f0-9]{6}$',      # Hex color codes
        r'^N/A$',                   # N/A
        r'^\d+$',                   # Pure numbers
        r'^[A-Z]+:\s*\d+',         # Code patterns like "YKK: 316"
    ]
    
    def __init__(
        self,
        service: str = "google",
        source_language: str = "en",
        target_language: str = "zh-CN",
        preserve_patterns: List[str] = None,
        api_key: str = None,
        use_api: bool = True,
        use_local_glossary: bool = True,
        use_smart_correction: bool = True
    ):
        """
        Initialize the translator.
        """
        self.source_language = source_language
        self.target_language = target_language
        self.preserve_patterns = preserve_patterns or self.DEFAULT_PRESERVE_PATTERNS
        self.use_api = use_api
        self.use_local_glossary = use_local_glossary
        self.use_smart_correction = use_smart_correction
        
        # Compile preserve patterns
        self._compiled_patterns = [
            re.compile(p) for p in self.preserve_patterns
        ]
        
        # Initialize backends
        self.primary_backend = self._create_backend(service, api_key) if use_api else None
        self.fallback_backend = OfflineGlossaryBackend()
        
        logger.info(f"Translator initialized (API: {use_api}, Glossary: {use_local_glossary}, Correction: {use_smart_correction})")
    
    def _create_backend(self, service: str, api_key: str = None) -> TranslationBackend:
        """Create the appropriate translation backend."""
        service_lower = service.lower()
        if service_lower == "google":
            return GoogleTranslateBackend(api_key)
        elif service_lower == "deepl":
            return DeepLBackend(api_key)
        return GoogleTranslateBackend(api_key)
    
    def should_preserve(self, text: str) -> bool:
        """Check if text should be preserved."""
        text = text.strip()
        for pattern in self._compiled_patterns:
            if pattern.match(text):
                return True
        return False
    
    def translate_text(self, text: str) -> str:
        """Translate a single text string."""
        if not text or not text.strip():
            return text
        
        # 1. OCR Smart Correction
        if self.use_smart_correction:
            text = self.fallback_backend.smart_correct(text)

        # 2. Check if should preserve
        if self.should_preserve(text):
            return text
        
        # 3. Industry Glossary Check (if enabled)
        if self.use_local_glossary:
            # Check glossary first for industry terms
            glossary_trans = self.fallback_backend.translate(text, self.source_language, self.target_language)
            if glossary_trans != text:
                return glossary_trans
        
        # 4. API Translation (if enabled and no glossary match found)
        if self.use_api and self.primary_backend:
            try:
                return self.primary_backend.translate(text, self.source_language, self.target_language)
            except Exception as e:
                logger.error(f"API error: {e}")
        
        return text
    
    def translate_regions(
        self, 
        text_regions: List[TextRegion],
        batch_size: int = 10
    ) -> List[TextRegion]:
        """
        Translate all translatable text regions.
        Uses a two-pass approach:
        1. Local Glossary/Smart Correction
        2. Batch API for the remaining untranslated text
        """
        logger.info(f"Translating {len(text_regions)} text regions using two-pass approach...")
        
        # Pass 1: Local Glossary and Smart Correction
        untranslated_indices = []
        for i, region in enumerate(text_regions):
            if region.text_type != TextType.TRANSLATABLE:
                region.translated_text = region.text
                continue
            
            # 1. OCR Smart Correction
            text = region.text
            if self.use_smart_correction:
                text = self.fallback_backend.smart_correct(text)
            
            # 2. Check preservation
            if self.should_preserve(text):
                region.translated_text = text
                continue
                
            # 3. Glossary Check
            if self.use_local_glossary:
                glossary_trans = self.fallback_backend.translate(text, self.source_language, self.target_language)
                if glossary_trans != text:
                    region.translated_text = glossary_trans
                    continue
            
            # If we reach here, we need the API
            untranslated_indices.append(i)
        
        # Pass 2: Batch API for remaining
        if untranslated_indices and self.use_api and self.primary_backend:
            logger.info(f"Pass 2: Batch translating {len(untranslated_indices)} regions via API...")
            
            # Extract texts for API
            texts_to_api = [text_regions[idx].text for idx in untranslated_indices]
            
            # Batch process
            for i in range(0, len(texts_to_api), batch_size):
                chunk_texts = texts_to_api[i:i + batch_size]
                chunk_indices = untranslated_indices[i:i + batch_size]
                
                try:
                    translations = self.primary_backend.translate_batch(
                        chunk_texts,
                        self.source_language,
                        self.target_language
                    )
                    
                    for idx, translation in zip(chunk_indices, translations):
                        text_regions[idx].translated_text = translation
                        
                except Exception as e:
                    logger.error(f"Batch API error for chunk {i}: {e}")
                    # Fallback to individual translation for this chunk
                    for idx in chunk_indices:
                        try:
                            text_regions[idx].translated_text = self.primary_backend.translate(
                                text_regions[idx].text,
                                self.source_language,
                                self.target_language
                            )
                        except Exception:
                            text_regions[idx].translated_text = text_regions[idx].text
        
        # Final cleanup for anything still missing
        for region in text_regions:
            if region.translated_text is None:
                region.translated_text = region.text
        
        return text_regions
