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
            import httpx
            import asyncio
            
            # Increase timeout for slow networks (default is 5s)
            timeout = httpx.Timeout(5.0)
            self.translator = Translator(timeout=timeout)
            self.use_free_api = True
            
            # Create a persistent event loop for async calls
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            
            logger.info("Google Translate (free API) initialized")
        except ImportError:
            logger.warning("googletrans not installed, translation will use fallback")
            self.translator = None
            self.use_free_api = False
            self._loop = None

    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text using Google Translate."""
        if not self.translator:
            return text
        
        try:
            # Handle language code mapping
            target = "zh-cn" if target_lang.lower() == "zh-cn" else target_lang
            source = "en" if source_lang.lower() == "en" else source_lang
            
            # googletrans 4.0.0-rc1+ usually returns a coroutine, but check for safety
            coro = self.translator.translate(text, src=source, dest=target)
            
            # Use inspect or just check if it's awaitable
            import inspect
            if inspect.isawaitable(coro):
                result = self._loop.run_until_complete(coro)
            else:
                result = coro
            
            translated_text = result.text
            if translated_text and translated_text != text:
                logger.debug(f"Google Translate success: '{text[:20]}...' -> '{translated_text[:20]}...'")
            return translated_text
        except Exception as e:
            logger.warning(f"Google Translate API timeout/error: {e}. Falling back to DeepL API...")
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
            logger.warning(f"DeepL API error: {e}. Text will use fallback (offline glossary).")
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
            logger.warning(f"DeepL API batch error: {e}. Texts will remain untranslated.")
            return texts


class MarianMTBackend(TranslationBackend):
    """
    Local translation using Helsinki-NLP's MarianMT model.
    
    This provides fast, offline translation without requiring API keys.
    Model: Helsinki-NLP/opus-mt-en-zh (English to Chinese)
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize MarianMT backend.
        
        Args:
            use_gpu: Whether to use GPU for inference (if available)
        """
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.use_gpu = use_gpu
        self.initialized = False

    def _ensure_initialized(self):
        """Lazy load the model and tokenizer only when needed."""
        if self.initialized:
            return
            
        try:
            from transformers import MarianMTModel, MarianTokenizer
            import torch
            
            model_name = "Helsinki-NLP/opus-mt-en-zh"
            logger.info(f"Lazy loading local translation model: {model_name}")
            
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            
            # Move to GPU if requested and available
            if self.use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                self.model = self.model.to(self.device)
                logger.info("MarianMT loaded on GPU")
            else:
                logger.info("MarianMT loaded on CPU")
                
            self.initialized = True
        except ImportError as e:
            logger.warning(f"transformers/torch not installed: {e}. Local model unavailable.")
        except Exception as e:
            logger.warning(f"Failed to load MarianMT model: {e}")

    
    def _preprocess(self, text: str) -> str:
        """Minimal preprocessing to remove OCR noise."""
        if not text:
            return ""
        # COLLAPSE multiple spaces and strip
        import re
        return re.sub(r'\s+', ' ', text).strip()
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text using MarianMT."""
        # Lazy load model if not already done
        self._ensure_initialized()
        
        if not self.model or not self.tokenizer:
            return text

        
        try:
            # Minimal preprocessing
            processed_text = self._preprocess(text)
            if not processed_text:
                return text
                
            # Tokenize
            inputs = self.tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation with anti-repetition parameters
            with __import__('torch').no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,                    # Beam search for better quality
                    repetition_penalty=1.5,         # Penalize repetition
                    no_repeat_ngram_size=3,         # Prevent 3-gram repetition
                    early_stopping=True             # Stop when beams converged
                )
            
            # Decode
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Anti-repetition validation: Reject if it's just repeating same characters
            if translated and len(translated) > 4:
                # Check if a small substring makes up most of the output
                if len(set(translated)) / len(translated) < 0.2:
                    logger.warning(f"MarianMT produced highly repetitive output, keeping original: '{text[:20]}...'")
                    return text
            
            if translated and translated != text:
                logger.info(f"MarianMT success: '{text[:20]}...' -> '{translated[:20]}...'")
            
            return translated
        except Exception as e:
            logger.warning(f"MarianMT translation error: {e}")
            return text




    
    def translate_batch(
        self, 
        texts: List[str], 
        source_lang: str, 
        target_lang: str
    ) -> List[str]:
        """Translate a batch of texts using MarianMT (one at a time to avoid OOM)."""
        if not self.model or not self.tokenizer:
            return texts
        
        # Process one at a time to avoid GPU OOM issues
        translations = []
        for text in texts:
            translated = self.translate(text, source_lang, target_lang)
            translations.append(translated)
        
        logger.info(f"MarianMT batch translated {len(texts)} texts")
        return translations



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
        
        # Standalone technical terms for fragmented OCR
        "needle": "针",
        "thread": "线",
        "stitching": "缝合",
        "stitch": "缝",
        "seam": "缝",
        "seams": "缝线",
        "double": "双",
        "triple": "三",
        "quilted": "绗缝",
        "binding": "包边",
        "elastic": "松紧带",
        "elasticated": "带松紧的",
        "drawstring": "抽绳",
        "stopper": "止滑扣",
        "eyelet": "气眼",
        "grommet": "金属孔",
        "velcro": "魔术贴",
        "piping": "嵌条",
        "rib": "螺纹",
        "ribbing": "螺纹",
        "interlining": "衬布",
        "padding": "填充物",
        "down": "羽绒",
        "woven": "梭织",
        "knit": "针织",
        "jersey": "汗布",
        "fleece": "粘绒",
        "denim": "牛仔",
        "twill": "斜纹",
        "satin": "缎面",
        "canvas": "帆布",
        "mesh": "网目",
        "lace": "花边",
        "sequin": "亮片",
        "bead": "珠子",
        "embroidery": "刺绣",
        "applique": "贴补花",
        "patch": "补丁",
        "print": "印花",
        "washed": "水洗",
        "dyed": "染色",
        "garment dyed": "成衣染色",
        "specification": "规格",
        "measurements": "尺寸",
        "measurement": "测量",
        "tolerance": "公差",
        "shrinkage": "缩水",
        "weight": "重量",
        "width": "宽度",
        "length": "长度",
        "height": "高度",
        "depth": "深度",
        "diameter": "直径",
        "circumference": "周长",
        
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
        
        Translation hierarchy:
        1. OCR Smart Correction (fix typos)
        2. Check preservation patterns (skip translation for codes/numbers)
        3. Glossary lookup (instant, offline - for exact matches)
        4. Local MarianMT Model (fast, offline - main translation engine)
        5. Google Translate API (fallback if local model unavailable)
        6. DeepL API (last resort, requires DEEPL_API_KEY)
        7. Glossary post-processing (always applied to fix terminology)
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
        # Local Model: MarianMT (primary translation engine - offline, fast)
        use_gpu = True  # Use GPU if available for faster inference
        self.local_model_backend = MarianMTBackend(use_gpu=use_gpu)

        
        # API Fallbacks (used only if local model fails/unavailable)
        self.primary_backend = GoogleTranslateBackend(api_key) if use_api else None
        self.fallback_api_backend = DeepLBackend() if use_api else None
        
        # Glossary: Always used for industry-specific term lookup and post-processing
        self.glossary_backend = OfflineGlossaryBackend()
        
        # Log initialization status
        local_model_status = "ready (lazy load)"
        deepl_status = "available" if (self.fallback_api_backend and self.fallback_api_backend.translator) else "unavailable (no API key)"
        logger.info(f"Translator initialized:")
        logger.info(f"  - Glossary lookup: always first")
        logger.info(f"  - Local Model (MarianMT): {local_model_status}")
        logger.info(f"  - Fallback API 1: Google Translate")
        logger.info(f"  - Fallback API 2: DeepL ({deepl_status})")
        logger.info(f"  - Glossary post-processing: always applied")
        logger.info(f"  - Smart OCR Correction: {'enabled' if use_smart_correction else 'disabled'}")

    
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
        """
        Translate a single text string with fallback chain.
        
        Flow:
        1. OCR Smart Correction
        2. Check preservation patterns
        3. Check glossary FIRST (instant local lookup for known terms)
        4. Try Local MarianMT Model (fast, offline - main engine)
        5. If local fails, try Google Translate API
        6. If Google fails, try DeepL API (fallback)
        7. Apply glossary post-processing (always)
        """
        if not text or not text.strip():
            return text
        
        original_text = text
        
        # Step 1: OCR Smart Correction
        if self.use_smart_correction:
            text = self.glossary_backend.smart_correct(text)

        # Step 2: Check if should preserve (codes, numbers, etc.)
        if self.should_preserve(text):
            return text
        
        # Step 3: Check glossary FIRST (fast path - avoids model/API calls)
        if self.use_local_glossary:
            glossary_result = self.glossary_backend.translate(text, self.source_language, self.target_language)
            if glossary_result != text:
                # Glossary found a match - no need for model/API
                return glossary_result
        
        translated = None
        
        # Step 4: Try Local MarianMT Model (primary - fast, offline)
        if self.local_model_backend and self.local_model_backend.model:
            try:
                translated = self.local_model_backend.translate(text, self.source_language, self.target_language)
                if translated and translated != text:
                    pass  # Success - no need for API
            except Exception as e:
                logger.warning(f"Local MarianMT failed: {e}. Falling back to APIs...")
        
        # Step 5: If local failed, try Google Translate API
        if (not translated or translated == text) and self.use_api and self.primary_backend:
            try:
                translated = self.primary_backend.translate(text, self.source_language, self.target_language)
                if translated and translated != text:
                    logger.info(f"Used Google Translate fallback for: '{text[:30]}...'")
            except Exception as e:
                logger.warning(f"Google Translate API failed: {e}. Falling back to DeepL API...")
        
        # Step 6: If Google failed, try DeepL API (last resort)
        if (not translated or translated == text) and self.fallback_api_backend and self.fallback_api_backend.translator:
            try:
                translated = self.fallback_api_backend.translate(text, self.source_language, self.target_language)
                if translated and translated != text:
                    logger.info(f"Used DeepL fallback for: '{text[:30]}...'")
            except Exception as e:
                logger.warning(f"DeepL API also failed: {e}. Keeping original text.")
        
        # If still no translation, use original
        if not translated or translated == text:
            translated = text
        
        # Step 7: Apply glossary post-processing (fixes industry terminology)
        if self.use_local_glossary:
            glossary_result = self.glossary_backend.translate(translated, self.source_language, self.target_language)
            if glossary_result != translated:
                translated = glossary_result
        
        return translated

    
    def translate_regions(
        self, 
        text_regions: List[TextRegion],
        batch_size: int = 10
    ) -> List[TextRegion]:
        """
        Translate all translatable text regions.
        
        Translation flow:
        1. OCR Smart Correction
        2. Check preservation patterns
        3. Glossary lookup (fast, exact match)
        4. Local MarianMT Model (primary - fast, offline)
        5. Google Translate API fallback
        6. DeepL API fallback (last resort)
        7. Glossary post-processing (always applied)
        """
        logger.info(f"Translating {len(text_regions)} text regions...")
        
        # Prepare regions for translation
        regions_to_translate = []
        for i, region in enumerate(text_regions):
            if region.text_type != TextType.TRANSLATABLE:
                region.translated_text = region.text
                continue
            
            # Step 1: OCR Smart Correction
            text = region.text
            if self.use_smart_correction:
                text = self.glossary_backend.smart_correct(text)
            
            # Step 2: Check preservation
            if self.should_preserve(text):
                region.translated_text = text
                continue
            
            # Step 3: Try glossary FIRST (fast path - instant local lookup)
            if self.use_local_glossary:
                glossary_result = self.glossary_backend.translate(text, self.source_language, self.target_language)
                if glossary_result != text:
                    # Glossary found a match - no need for model/API
                    region.translated_text = glossary_result
                    continue
            
            # Mark for model/API translation (glossary didn't match)
            regions_to_translate.append((i, text))
        
        logger.info(f"  {len(regions_to_translate)} regions need model/API translation (others handled by glossary)")
        
        # Step 4: Try Local MarianMT Model (primary - fast, offline)
        local_failed_indices = []
        if regions_to_translate and self.local_model_backend and self.local_model_backend.model:
            logger.info(f"  Using local MarianMT model for {len(regions_to_translate)} regions...")
            
            batch_texts = [t for _, t in regions_to_translate]
            batch_indices = [i for i, _ in regions_to_translate]
            
            try:
                translations = self.local_model_backend.translate_batch(
                    batch_texts,
                    self.source_language,
                    self.target_language
                )
                
                for idx, translation in zip(batch_indices, translations):
                    if translation and translation != text_regions[idx].text:
                        text_regions[idx].translated_text = translation
                    else:
                        local_failed_indices.append(idx)
                        
            except Exception as e:
                logger.warning(f"  Local model batch failed: {e}. Will use API fallback...")
                local_failed_indices = batch_indices
        else:
            # No local model available, all regions need API fallback
            local_failed_indices = [i for i, _ in regions_to_translate]
        
        # Step 5: Google Translate API fallback for failed local translations
        google_failed_indices = []
        if local_failed_indices and self.use_api and self.primary_backend:
            logger.info(f"  Using Google Translate fallback for {len(local_failed_indices)} regions...")
            
            for batch_start in range(0, len(local_failed_indices), batch_size):
                batch_indices = local_failed_indices[batch_start:batch_start + batch_size]
                batch_texts = [text_regions[i].text for i in batch_indices]
                
                try:
                    translations = self.primary_backend.translate_batch(
                        batch_texts,
                        self.source_language,
                        self.target_language
                    )
                    
                    for idx, translation in zip(batch_indices, translations):
                        if translation and translation != text_regions[idx].text:
                            text_regions[idx].translated_text = translation
                        else:
                            google_failed_indices.append(idx)
                            
                except Exception as e:
                    logger.warning(f"  Google Translate batch failed: {e}. Will use DeepL fallback...")
                    google_failed_indices.extend(batch_indices)
        else:
            google_failed_indices = local_failed_indices
        
        # Step 6: DeepL API fallback for failed Google translations
        if google_failed_indices and self.fallback_api_backend and self.fallback_api_backend.translator:
            logger.info(f"  Using DeepL fallback for {len(google_failed_indices)} regions...")
            
            for idx in google_failed_indices:
                try:
                    original_text = text_regions[idx].text
                    translated = self.fallback_api_backend.translate(
                        original_text,
                        self.source_language,
                        self.target_language
                    )
                    if translated and translated != original_text:
                        text_regions[idx].translated_text = translated
                    else:
                        text_regions[idx].translated_text = original_text
                except Exception as e:
                    logger.warning(f"  DeepL failed for region {idx}: {e}. Keeping original text.")
                    text_regions[idx].translated_text = text_regions[idx].text
        elif google_failed_indices:
            logger.warning(f"  {len(google_failed_indices)} regions will keep original text (all translation methods failed there)")


            for idx in google_failed_indices:
                if text_regions[idx].translated_text is None:
                    text_regions[idx].translated_text = text_regions[idx].text
        
        # Step 7: Apply glossary post-processing (ALWAYS)
        if self.use_local_glossary:
            glossary_applied = 0
            for region in text_regions:
                if region.translated_text:
                    glossary_result = self.glossary_backend.translate(
                        region.translated_text, 
                        self.source_language, 
                        self.target_language
                    )
                    if glossary_result != region.translated_text:
                        region.translated_text = glossary_result
                        glossary_applied += 1
            if glossary_applied > 0:
                logger.info(f"  Glossary post-processing applied to {glossary_applied} regions")
        
        # Final cleanup
        for region in text_regions:
            if region.translated_text is None:
                region.translated_text = region.text
        
        return text_regions

