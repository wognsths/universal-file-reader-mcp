import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Protocol
from pathlib import Path
from dataclasses import dataclass
import json
import requests
from datetime import datetime
import logging.config

import google.generativeai as genai
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from .base_processor import BaseProcessor
from ..core.utils import with_timeout

from ..core.exceptions import (
    APIKeyError,
    ProcessingError,
    ValidationError
)

# Logger for this module (configured via ``setup_logging()`` below)
logger = logging.getLogger(__name__)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        
        return json.dumps(log_record)

def setup_logging():
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JsonFormatter
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO"
            },
            "json_file": {
                "class": "logging.FileHandler",
                "filename": "ocr_processor.log",
                "formatter": "json",
                "level": "INFO"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "json_file"],
                "level": "INFO"
            }
        }
    }
    
    logging.config.dictConfig(logging_config)

@dataclass
class OCRConfig:
    """Centralized OCR configuration"""
    MODEL_NAME: str = "gemini-2.0-flash-exp"
    MAX_DPI: int = 300
    MIN_DPI: int = 150
    MAX_FILE_SIZE_MB: int = 50
    MAX_PAGES: int = 10
    TIMEOUT_SECONDS: int = 60
    SUPPORTED_LANGUAGES: Dict[str, str] = None
    COMPRESSION_QUALITY: int = 85
    
    def __post_init__(self):
        if self.SUPPORTED_LANGUAGES is None:
            self.SUPPORTED_LANGUAGES = {
                "ko": "Korean", "en": "English", "ja": "Japanese", 
                "zh": "Chinese", "fr": "French", "de": "German",
                "es": "Spanish", "it": "Italian", "pt": "Portuguese",
                "ru": "Russian", "ar": "Arabic", "hi": "Hindi"
            }

class PageElement(BaseModel):
    """Individual page element with enhanced validation"""
    element_type: str = Field(description="Element type: text, table, graph, form, image")
    role: str = Field(description="Element role: title, paragraph, header, footer, etc") 
    content: str = Field(description="Element content")
    bounding_box: List[int] = Field(description="[x1, y1, x2, y2] coordinates")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    reading_order: int = Field(ge=1, description="Reading order")
    
    @field_validator('bounding_box')
    @classmethod
    def validate_bounding_box(cls, v):
        if len(v) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError("Invalid bounding box coordinates")
        return v
    
    @field_validator('element_type')
    @classmethod
    def validate_element_type(cls, v):
        valid_types = {"text", "table", "graph", "form", "image"}
        if v not in valid_types:
            raise ValueError(f"Element type must be one of {valid_types}")
        return v

class MultiElementResult(BaseModel):
    """Multi-element detection result with enhanced validation"""
    page_elements: List[PageElement] = Field(description="All page elements")
    layout_complexity: str = Field(description="Layout complexity")
    total_elements: int = Field(ge=0, description="Total element count")
    element_summary: Dict[str, int] = Field(description="Element type counts")
    reading_flow: List[int] = Field(description="Recommended reading order")
    detected_language: str = Field(description="Main document language")
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")
    
    @field_validator('layout_complexity')
    @classmethod
    def validate_complexity(cls, v):
        valid_complexity = {"simple", "moderate", "complex"}
        if v not in valid_complexity:
            raise ValueError(f"Layout complexity must be one of {valid_complexity}")
        return v

class OutputFormatter(Protocol):
    """Output formatter protocol"""
    def format(self, result: MultiElementResult, file_path: str) -> str:
        ...

class MarkdownFormatter:
    """Markdown output formatter"""
    
    def format(self, result: MultiElementResult, file_path: str) -> str:
        filename = Path(file_path).name
        sorted_elements = sorted(result.page_elements, key=lambda x: x.reading_order)
        
        content = f"""# Multi-Element Detection Result: {filename}

## Analysis Summary
- **Total Elements**: {result.total_elements}
- **Layout Complexity**: {result.layout_complexity}
- **Element Composition**: {', '.join([f"{k}: {v}" for k, v in result.element_summary.items()])}
- **Detected Language**: {result.detected_language}
- **Processing Time**: {result.processing_time:.2f}s

## Content (Reading Order)

"""
        
        for element in sorted_elements:
            indicators = {
                "text": "[TEXT]", "table": "[TABLE]", "graph": "[GRAPH]",
                "form": "[FORM]", "image": "[IMAGE]"
            }
            
            indicator = indicators.get(element.element_type, "[ELEMENT]")
            content += f"""### {indicator} {element.role.title()} (Order: {element.reading_order})
**Type**: {element.element_type} | **Confidence**: {element.confidence:.2%}

{element.content}

---

"""
        
        return content

class HTMLFormatter:
    """HTML output formatter"""
    
    def format(self, result: MultiElementResult, file_path: str) -> str:
        filename = Path(file_path).name
        sorted_elements = sorted(result.page_elements, key=lambda x: x.reading_order)
        
        html = f"""
<div class="ocr-result">
    <h2>Multi-Element Detection: {filename}</h2>
    <div class="summary">
        <p><strong>Elements:</strong> {result.total_elements}</p>
        <p><strong>Complexity:</strong> {result.layout_complexity}</p>
        <p><strong>Language:</strong> {result.detected_language}</p>
        <p><strong>Processing Time:</strong> {result.processing_time:.2f}s</p>
    </div>
    <div class="content">
"""
        
        for element in sorted_elements:
            html += f"""
        <div class="element">
            <h4>[{element.element_type.upper()}] {element.role.title()}</h4>
            <p><strong>Confidence:</strong> {element.confidence:.2%}</p>
            <pre>{element.content}</pre>
        </div>
"""
        
        html += """
    </div>
</div>
"""
        return html

class OutputFormatterFactory:
    """Factory for output formatters"""
    
    _formatters = {
        "markdown": MarkdownFormatter,
        "html": HTMLFormatter
    }
    
    @classmethod
    def create(cls, format_type: str) -> OutputFormatter:
        if format_type not in cls._formatters:
            raise ValueError(f"Unsupported format: {format_type}")
        return cls._formatters[format_type]()

class APIKeyValidator:
    """Enhanced API key validation utility"""
    
    @staticmethod
    def validate_google_api_key(api_key: str) -> bool:
        """
        Validate Google Gemini API key using models endpoint
        Based on search results recommendations
        """
        if not api_key or len(api_key) < 20:
            logger.error("API key too short or empty")
            return False
        
        try:
            # Use the recommended models endpoint for validation
            API_VERSION = 'v1'
            api_url = f'https://generativelanguage.googleapis.com/{API_VERSION}/models'
            
            # Use secure header method (search result [4] recommendation)
            headers = {
                'x-goog-api-key': api_key,
                'Content-Type': 'application/json'
            }
            
            # Make validation request with timeout
            response = requests.get(
                api_url, 
                headers=headers,
                timeout=10  # 10 second timeout
            )
            
            if response.status_code == 200:
                # Check if response contains models (additional validation)
                models_data = response.json()
                if 'models' in models_data and len(models_data['models']) > 0:
                    logger.info("API key validation successful")
                    return True
                else:
                    logger.error("API key valid but no models accessible")
                    return False
            
            elif response.status_code == 400:
                error_data = response.json()
                error_message = error_data.get('error', {}).get('message', 'Invalid API key')
                logger.error(f"API key validation failed: {error_message}")
                return False
            
            elif response.status_code == 403:
                logger.error("API key valid but access forbidden")
                return False
            
            else:
                logger.error(f"API validation failed with status {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Network error during API key validation: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response during API key validation: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during API key validation: {e}")
            return False
    
    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """Basic format validation before network call"""
        if not api_key:
            return False
        
        # Google API keys typically start with 'AIza' and are 39 characters
        if api_key.startswith('AIza') and len(api_key) == 39:
            return True
        
        # But also allow other formats (in case Google changes format)
        if len(api_key) >= 20:
            return True
            
        return False

class ImageProcessor:
    """Image preprocessing utility"""
    
    @staticmethod
    def optimize_image(image: Image.Image, target_size_kb: int = 500) -> Image.Image:
        """Optimize image for API processing"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Calculate compression based on current size
            original_size = len(image.tobytes())
            if original_size <= target_size_kb * 1024:
                return image
            
            # Resize if too large
            max_dimension = 2048
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            logger.warning(f"Image optimization failed: {e}")
            return image

class PromptManager:
    """Centralized prompt management"""
    
    @staticmethod
    def get_detection_prompt(language_instruction: str) -> str:
        """Get multi-element detection prompt"""
        return f"""
        Analyze this document page and detect all elements individually.
        
        Respond in the following JSON schema:
        {{
            "page_elements": [
                {{
                    "element_type": "text|table|graph|form|image",
                    "role": "title|paragraph|header|footer|sectionHeading|footnote|caption",
                    "content": "actual element content",
                    "bounding_box": [x1, y1, x2, y2],
                    "confidence": 0.95,
                    "reading_order": 1
                }}
            ],
            "layout_complexity": "simple|moderate|complex",
            "total_elements": 5,
            "element_summary": {{"text": 3, "table": 1, "graph": 1}},
            "reading_flow": [1, 2, 3, 4, 5],
            "detected_language": "ko|en|ja|zh|auto"
        }}

        Element Detection Criteria:
        1. Text: Continuous paragraphs, titles, headers, footers
        2. Table: Rows and columns (with/without borders)
        3. Graph: Charts, graphs, diagrams, figures
        4. Form: Input fields, checkboxes, forms
        5. Image: Photos, logos, illustrations

        {language_instruction}

        Important: Detect each element individually, even if multiple types exist on one page.
        """

class OCRProcessor(BaseProcessor):
    """Production-ready multi-element OCR processor"""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        super().__init__()
        self.config = config or OCRConfig()
        self._supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._setup_gemini()
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in self._supported_extensions
    
    def get_supported_extensions(self) -> List[str]:
        return self._supported_extensions
    
    @with_timeout(60)
    def process(
        self,
        file_path: str,
        output_format: str = "markdown",
        user_language: str = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for async processing.

        ``kwargs`` are ignored to maintain API compatibility with other
        processors.
        """
        try:
            logger.info(
                "Starting document processing",
                extra={
                    "file_path": file_path,
                    "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024),
                    "output_format": output_format,
                    "language": user_language,
                    "processor_id": id(self)
                }
            )

            future = self.executor.submit(
                lambda: asyncio.run(
                    self.process_async(file_path, output_format, user_language)
                )
            )
            result = future.result(timeout=self.config.TIMEOUT_SECONDS)
            
            logger.info(
                "Document processing completed",
                extra={
                    "file_path": file_path,
                    "processing_time": result.get("processing_time", 0),
                    "total_elements": result.get("total_elements", 0)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Document processing failed",
                extra={
                    "file_path": file_path,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            return self._create_error_response(str(e), file_path)

    
    def _setup_gemini(self):
        """Setup Gemini API with enhanced error handling"""
        try:
            import os
            api_key = os.getenv('GOOGLE_API_KEY')
            
            if not api_key:
                raise APIKeyError("GOOGLE_API_KEY environment variable not set")
            
            if not APIKeyValidator.validate_api_key_format(api_key):
                raise APIKeyError("Invalid API key format")
            
            if not APIKeyValidator.validate_google_api_key(api_key):
                raise APIKeyError("API key validation failed")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                self.config.MODEL_NAME,
                generation_config={
                    "response_mime_type": "application/json",
                    "max_output_tokens": 8192
                }
            )
            self.gemini_available = True
            logger.info("Gemini API configured successfully")
            
        except APIKeyError:
            raise
        except Exception as e:
            logger.error(f"Gemini setup failed: {e}")
            self.gemini_available = False
            raise ProcessingError(f"Gemini setup failed: {e}")


    async def process_async(self, file_path: str, output_format: str = "markdown", 
                           user_language: str = "auto") -> Dict[str, Any]:
        """Async processing for large documents"""
        import time
        start_time = time.time()
        
        try:
            if not self._validate_file(file_path):
                raise ProcessingError("File validation failed")
            
            if not self.gemini_available:
                raise ProcessingError("Gemini API not available")
            
            # Process document
            result = await self._detect_multiple_elements_async(file_path, user_language)
            result.processing_time = time.time() - start_time
            
            # Format output
            if output_format == "structured":
                formatted_output = result.model_dump()
            else:
                formatter = OutputFormatterFactory.create(output_format)
                formatted_output = formatter.format(result, file_path)
            
            return self._create_success_response(
                formatted_output,
                file_path,
                processing_method="Async multi-element detection",
                total_elements=result.total_elements,
                processing_time=result.processing_time
            )
            
        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"Data validation error: {e}")
            raise ValidationError(f"Data validation failed: {e}")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Network error: {e}")
            raise ProcessingError(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ProcessingError(f"Processing failed: {e}")
        
    def _validate_file(self, file_path: str) -> bool:
        """Enhanced file validation"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            if not path.is_file():
                logger.error(f"Not a file: {file_path}")
                return False
            
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.MAX_FILE_SIZE_MB:
                logger.error(f"File too large: {size_mb:.1f}MB > {self.config.MAX_FILE_SIZE_MB}MB")
                return False
            
            # Check file extension
            if not self.supports(path.suffix):
                logger.error(f"Unsupported file type: {path.suffix}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False
    
    async def _detect_multiple_elements_async(self, file_path: str, 
                                            user_language: str = "auto") -> MultiElementResult:
        """Async multi-element detection with optimizations"""
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return await self._process_pdf_async(file_path, user_language)
        else:
            return await self._process_image_async(file_path, user_language)
    
    async def _process_image_async(self, image_path: str, user_language: str) -> MultiElementResult:
        """Async image processing with optimization"""
        try:
            # Load and optimize image
            image = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._load_and_optimize_image, image_path
            )
            
            # Get language instruction
            language_instruction = self._get_language_instruction(user_language)
            
            # Get prompt
            prompt = PromptManager.get_detection_prompt(language_instruction)
            
            # Process with timeout
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    lambda: self.model.generate_content([prompt, image])
                ),
                timeout=self.config.TIMEOUT_SECONDS
            )
            
            result_dict = json.loads(response.text)
            return MultiElementResult(**result_dict)
            
        except asyncio.TimeoutError:
            raise ProcessingError("Processing timeout")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise ProcessingError(f"Image processing failed: {e}")
    
    async def _process_pdf_async(self, pdf_path: str, user_language: str) -> MultiElementResult:
        """Async PDF processing with parallel page handling"""
        try:
            from pdf2image import convert_from_path
            import fitz

            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)

            max_pages = min(total_pages, self.config.MAX_PAGES)
            semaphore = asyncio.Semaphore(3)
            page_results = []
            current_page = 1

            while current_page <= max_pages:
                end_page = min(
                    current_page + self.config.MAX_PAGE_PER_PROCESS - 1,
                    max_pages,
                )

                images = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda s=current_page, e=end_page: convert_from_path(
                        pdf_path,
                        dpi=self._calculate_optimal_dpi(pdf_path),
                        first_page=s,
                        last_page=e,
                    ),
                )

                async def process_page(page_num, image):
                    async with semaphore:
                        return await self._process_image_async_internal(
                            image,
                            user_language,
                            page_num,
                        )

                chunk_results = await asyncio.gather(
                    *[
                        process_page(page_no - 1, img)
                        for page_no, img in zip(
                            range(current_page, end_page + 1), images
                        )
                    ],
                    return_exceptions=True,
                )

                page_results.extend(chunk_results)
                current_page = end_page + 1

            return self._merge_page_results(page_results)
            
        except Exception as e:
            raise ProcessingError(f"PDF processing failed: {e}")
    
    def _load_and_optimize_image(self, image_path: str) -> Image.Image:
        """Load and optimize image"""
        image = Image.open(image_path)
        return ImageProcessor.optimize_image(image)
    
    def _calculate_optimal_dpi(self, file_path: str) -> int:
        """Calculate optimal DPI based on file size"""
        size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        
        if size_mb > 20:
            return self.config.MIN_DPI
        elif size_mb > 10:
            return 200
        else:
            return self.config.MAX_DPI
    
    def _get_language_instruction(self, user_language: str) -> str:
        """Enhanced language instruction"""
        if user_language == "auto":
            return """
            Output Language Control: 
            - Detect the main language of the document content
            - Extract and return all content in the same language as detected
            - Preserve original language exactly as it appears
            """
        elif user_language in self.config.SUPPORTED_LANGUAGES:
            target_lang = self.config.SUPPORTED_LANGUAGES[user_language]
            return f"""
            Output Language Control:
            - Extract all content and return in {target_lang}
            - Maintain technical terms in original language when appropriate
            """
        else:
            logger.warning(f"Unsupported language: {user_language}, using auto-detection")
            return self._get_language_instruction("auto")
    
    def _merge_page_results(self, page_results: List) -> MultiElementResult:
        """Merge multiple page results into a single result"""
        merged_elements = []
        element_summary = {}
        reading_flow = []
        total_elements = 0
        layout_complexity = "unknown"
        detected_language = "auto"
        processing_time = 0.0

        for result in page_results:
            if isinstance(result, Exception):
                logger.warning(f"Skipping page result due to error: {result}")
                continue
                
            merged_elements.extend(result.page_elements)
            total_elements += result.total_elements
            
            # Merge element summary
            for k, v in result.element_summary.items():
                element_summary[k] = element_summary.get(k, 0) + v
                
            # Merge reading flow (adjust to avoid duplicate order)
            max_order = max(reading_flow) if reading_flow else 0
            adjusted_flow = [order + max_order for order in result.reading_flow]
            reading_flow.extend(adjusted_flow)
            
            # For simplicity, take the highest complexity
            complexity_levels = {"simple": 1, "moderate": 2, "complex": 3, "unknown": 0}
            if complexity_levels.get(result.layout_complexity, 0) > complexity_levels.get(layout_complexity, 0):
                layout_complexity = result.layout_complexity
                
            # Take the most confident language detection
            if result.detected_language != "auto":
                detected_language = result.detected_language
                
            # Sum processing times
            processing_time += getattr(result, "processing_time", 0.0)

        # Construct merged result
        return MultiElementResult(
            page_elements=merged_elements,
            layout_complexity=layout_complexity,
            total_elements=total_elements,
            element_summary=element_summary,
            reading_flow=reading_flow,
            detected_language=detected_language,
            processing_time=processing_time
        )

    
    async def _process_image_async_internal(
        self, 
        image: Image.Image,
        user_language: str, 
        page_num: int
    ) -> MultiElementResult:
        """Process a single image asynchronously"""
        try:
            # Log processing start with structured data
            logger.info(
                f"Processing page {page_num}", 
                extra={
                    "page_num": page_num,
                    "language": user_language,
                    "image_size": f"{image.width}x{image.height}"
                }
            )
            
            # Optimize image if needed
            optimized_image = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                ImageProcessor.optimize_image, 
                image
            )
            
            # Get language instruction and prompt
            language_instruction = self._get_language_instruction(user_language)
            prompt = PromptManager.get_detection_prompt(language_instruction)
            
            # Process with API with timeout
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.model.generate_content([prompt, optimized_image])
                ),
                timeout=self.config.TIMEOUT_SECONDS
            )
            
            # Parse and validate result
            result_dict = json.loads(response.text)
            
            # Adjust element reading order for multi-page documents
            for element in result_dict.get("page_elements", []):
                element["reading_order"] += page_num * 1000  # Ensure pages are ordered
            
            result = MultiElementResult(**result_dict)
            
            # Log success with metrics
            logger.info(
                f"Successfully processed page {page_num}", 
                extra={
                    "page_num": page_num,
                    "elements_found": result.total_elements,
                    "detected_language": result.detected_language
                }
            )
            
            return result
            
        except asyncio.TimeoutError as e:
            logger.error(
                f"Timeout processing page {page_num}", 
                extra={"page_num": page_num, "timeout": self.config.TIMEOUT_SECONDS}
            )
            return Exception(f"Timeout processing page {page_num}: {str(e)}")
            
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parsing error on page {page_num}",
                extra={"page_num": page_num, "error_location": e.pos}
            )
            return Exception(f"Invalid JSON response on page {page_num}: {str(e)}")
            
        except Exception as e:
            logger.error(
                f"Error processing page {page_num}",
                extra={"page_num": page_num, "error_type": type(e).__name__}
            )
            return Exception(f"Error processing page {page_num}: {str(e)}")


# Initialize logging when the module is imported
setup_logging()
