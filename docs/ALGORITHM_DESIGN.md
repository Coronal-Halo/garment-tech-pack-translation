# Tech Pack Image Translation System - Algorithm Design

## Executive Summary

This document outlines the system architecture for automatically translating text in garment industry tech pack images from English to Chinese, while preserving design pack graphics that must remain unchanged.

---

## 1. Key Challenges Identified

### 1.1 Text Detection Challenges
- **Mixed Content Types**: Tech pack images contain tables, labels, annotations, and artistic graphics
- **Variable Font Sizes**: Text ranges from large headers to small table entries
- **Text Orientation**: Some text may be rotated or curved (e.g., annotations on garment diagrams)
- **Low Contrast Areas**: Text overlaid on colored backgrounds or within table cells

### 1.2 Design Pack Image Preservation
- **Automatic Detection**: Must automatically identify the "design pack image" region
- **Variable Positioning**: The design pack image location varies across different tech packs
- **Boundary Precision**: Need accurate boundary detection to avoid partial translation

### 1.3 Translation Quality
- **Industry-Specific Terminology**: Garment industry terms require specialized translation
- **Context Preservation**: Abbreviations and codes (e.g., "DTM", "CB", "YKK") should be preserved
- **Character Limit**: Chinese characters may require different space than English text

### 1.4 Text Rendering
- **Font Matching**: Translated text should maintain visual consistency
- **Space Management**: Chinese text may need different space allocation
- **Background Inpainting**: Original text must be cleanly removed before rendering new text

---

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TECH PACK TRANSLATION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  INPUT IMAGE │
│ (Tech Pack)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           STAGE 1: PREPROCESSING                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌────────────────────────────────┐ │
│  │ Image Loading  │→ │ Color Space     │→ │ Resolution Normalization       │ │
│  │ & Validation   │  │ Conversion      │  │ (Maintain aspect ratio)        │ │
│  └────────────────┘  └─────────────────┘  └────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: DESIGN PACK IMAGE DETECTION                       │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    Multi-Strategy Detection Approach                    │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐   │  │
│  │  │ Color Histogram │  │ Edge Density    │  │ Texture Analysis     │   │  │
│  │  │ Analysis        │  │ Analysis        │  │ (High entropy areas) │   │  │
│  │  └────────┬────────┘  └────────┬────────┘  └──────────┬───────────┘   │  │
│  │           │                    │                      │               │  │
│  │           └────────────────────┼──────────────────────┘               │  │
│  │                                ▼                                       │  │
│  │                    ┌─────────────────────┐                            │  │
│  │                    │ Region Proposal &   │                            │  │
│  │                    │ Confidence Scoring  │                            │  │
│  │                    └─────────────────────┘                            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                   │                                          │
│                                   ▼                                          │
│                    ┌──────────────────────────────┐                          │
│                    │ EXCLUSION MASK GENERATION    │                          │
│                    │ (Regions to preserve)        │                          │
│                    └──────────────────────────────┘                          │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 3: TEXT DETECTION & OCR                          │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      Text Detection Engine                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │  PaddleOCR / EasyOCR / Tesseract (Configurable Backend)         │   │  │
│  │  │                                                                   │   │  │
│  │  │  • Detect text bounding boxes                                     │   │  │
│  │  │  • Extract text content with confidence scores                    │   │  │
│  │  │  • Identify text orientation                                      │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │                                │                                        │  │
│  │                                ▼                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    Filtering Pipeline                            │   │  │
│  │  │  1. Apply exclusion mask (remove design pack regions)           │   │  │
│  │  │  2. Filter by confidence threshold                               │   │  │
│  │  │  3. Merge overlapping detections                                 │   │  │
│  │  │  4. Classify: Translatable vs Non-Translatable                   │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 4: TEXT CLASSIFICATION                             │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    Classification Rules                                 │  │
│  │                                                                         │  │
│  │  ┌─────────────────────┐    ┌────────────────────────────────────────┐ │  │
│  │  │   TRANSLATE         │    │   PRESERVE (Do Not Translate)          │ │  │
│  │  │   ─────────         │    │   ──────────────────────────           │ │  │
│  │  │ • Descriptive text  │    │ • Brand codes (YKK, DTM)               │ │  │
│  │  │ • Labels & headers  │    │ • Product codes & numbers              │ │  │
│  │  │ • Instructions      │    │ • Measurements (18"L, 5.5 oz)          │ │  │
│  │  │ • Material names    │    │ • Color codes                          │ │  │
│  │  │ • Notes & comments  │    │ • Design pack image text               │ │  │
│  │  └─────────────────────┘    └────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 5: TRANSLATION                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐   │  │
│  │  │ Batch Text      │→ │ Translation API │→ │ Post-Processing      │   │  │
│  │  │ Preparation     │  │ (Google/DeepL)  │  │ & Validation         │   │  │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────────┘   │  │
│  │                                                                         │  │
│  │  Features:                                                              │  │
│  │  • Garment industry glossary support                                   │  │
│  │  • Context-aware translation                                            │  │
│  │  • Fallback to multiple translation services                           │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 6: TEXT INPAINTING & RENDERING                       │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                         │  │
│  │   ┌───────────────────┐                                                │  │
│  │   │ INPAINTING MODULE │                                                │  │
│  │   │ ─────────────────  │                                                │  │
│  │   │ 1. Create text    │     ┌───────────────────────────────────────┐  │  │
│  │   │    region masks   │     │  RENDERING MODULE                     │  │  │
│  │   │ 2. Apply OpenCV   │────▶│  ────────────────                     │  │  │
│  │   │    inpainting     │     │  1. Calculate text placement          │  │  │
│  │   │ 3. Clean          │     │  2. Select appropriate font size      │  │  │
│  │   │    background     │     │  3. Render Chinese text with Pillow   │  │  │
│  │   └───────────────────┘     │  4. Handle text wrapping if needed    │  │  │
│  │                              └───────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 7: POST-PROCESSING                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────────┐   │
│  │ Quality Check   │→ │ Format          │→ │ Output Generation          │   │
│  │ & Validation    │  │ Optimization    │  │ (PNG/JPEG + Metadata)      │   │
│  └─────────────────┘  └─────────────────┘  └────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ OUTPUT IMAGE │
│ (Translated) │
└──────────────┘
```

---

## 3. Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        main.py (Orchestrator)                      │  │
│  │                                                                    │  │
│  │  • Pipeline configuration                                          │  │
│  │  • Module coordination                                             │  │
│  │  • Error handling & logging                                        │  │
│  │  • Progress reporting                                              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PROCESSING LAYER                                  │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ design_pack_    │  │ text_detector   │  │ translator.py           │  │
│  │ detector.py     │  │ .py             │  │                         │  │
│  │                 │  │                 │  │ • Google Translate API  │  │
│  │ • Region        │  │ • PaddleOCR     │  │ • DeepL API             │  │
│  │   detection     │  │ • EasyOCR       │  │ • Offline fallback      │  │
│  │ • Texture       │  │ • Text          │  │ • Glossary support      │  │
│  │   analysis      │  │   extraction    │  │                         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
│           │                   │                       │                  │
│           └───────────────────┴───────────────────────┘                  │
│                               │                                          │
│                               ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     image_processor.py                             │  │
│  │                                                                    │  │
│  │  • Image loading/saving       • Text inpainting                   │  │
│  │  • Color space conversion     • Chinese text rendering            │  │
│  │  • Mask generation            • Font management                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         UTILITY LAYER                                    │
│  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────────┐ │
│  │ config.py         │  │ utils.py          │  │ glossary.py          │ │
│  │                   │  │                   │  │                      │ │
│  │ • YAML loading    │  │ • Logging         │  │ • Industry terms     │ │
│  │ • Validation      │  │ • File operations │  │ • Abbreviations      │ │
│  │ • Defaults        │  │ • Image utils     │  │ • Special handling   │ │
│  └───────────────────┘  └───────────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA STRUCTURES                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐
│         TextRegion              │
├─────────────────────────────────┤
│ • bbox: (x1, y1, x2, y2)       │
│ • text: str                     │
│ • confidence: float             │
│ • is_translatable: bool         │
│ • translated_text: str          │
│ • font_size: int                │
│ • text_color: tuple             │
└─────────────────────────────────┘
              │
              │  (List of TextRegions)
              ▼
┌─────────────────────────────────┐
│     TranslationResult           │
├─────────────────────────────────┤
│ • original_image: ndarray       │
│ • processed_image: ndarray      │
│ • text_regions: List[TextRegion]│
│ • exclusion_mask: ndarray       │
│ • processing_time: float        │
│ • metadata: dict                │
└─────────────────────────────────┘
```

---

## 5. Design Decisions & Rationale

### 5.1 OCR Engine Selection: PaddleOCR
**Rationale:**
- Best-in-class accuracy for mixed English/Chinese text
- Lightweight and fast inference
- Open-source with active development
- Built-in text detection + recognition pipeline

### 5.2 Translation Service: Google Translate API (with fallback)
**Rationale:**
- High-quality translations for technical content
- Support for specialized terminology
- Reliable with high uptime
- Fallback to offline dictionary for critical terms

### 5.3 Text Inpainting: OpenCV + Pillow Hybrid
**Rationale:**
- OpenCV TELEA/NS inpainting for background reconstruction
- Pillow for precise Chinese text rendering
- Support for CJK fonts with proper character spacing

### 5.4 Design Pack Detection: Multi-feature Analysis
**Rationale:**
- Color histogram analysis (design images are typically colorful/complex)
- Edge density detection (design images have irregular edge patterns)
- Texture entropy analysis (design images have high entropy)
- Confidence scoring for robust detection

---

## 6. Error Handling Strategy

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          ERROR HANDLING FLOW                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT ERROR           PROCESSING ERROR         OUTPUT ERROR               │
│   ───────────           ────────────────         ────────────               │
│   • Invalid image  ──▶  • OCR failure       ──▶  • Write failure           │
│   • Corrupt file        • Translation API        • Permission denied       │
│   • Wrong format        • Memory overflow        • Disk full               │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                   ┌─────────────────────┐                                   │
│                   │  FALLBACK STRATEGY  │                                   │
│                   ├─────────────────────┤                                   │
│                   │ • Retry with backup │                                   │
│                   │ • Graceful degrade  │                                   │
│                   │ • Log & continue    │                                   │
│                   │ • User notification │                                   │
│                   └─────────────────────┘                                   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Performance Considerations

| Component | Expected Time | Optimization Strategy |
|-----------|---------------|----------------------|
| Image Loading | <100ms | Lazy loading, memory mapping |
| Design Pack Detection | <500ms | Downsampled analysis |
| OCR Processing | 1-3s | GPU acceleration, batch processing |
| Translation | 200-500ms | Batch API calls, caching |
| Inpainting & Rendering | 500ms-1s | Optimized mask generation |
| **Total Pipeline** | **3-5s** | Parallel processing where possible |

---

## 8. Extensibility Points

1. **OCR Backend**: Pluggable architecture for different OCR engines
2. **Translation Service**: Easy to add new translation providers
3. **Output Formats**: Support for PDF, SVG, or other formats
4. **Language Pairs**: Extensible to support other language translations
5. **Custom Glossaries**: Industry-specific terminology dictionaries

---

*Document Version: 1.0*  
*Created for: Crystal International Technical Assessment*  
*Author: Yuxiang Huang*
