"""Simple PDF processor for native text extraction only"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import time

import fitz
from .base_processor import BaseProcessor
from .ocr_processor import OCRProcessor
from ..core.utils import with_timeout
from ..core.config import PDFConfig

logger = logging.getLogger(__name__)

class PDFProcessor(BaseProcessor):
    """Simple PDF processor for native text extraction"""

    def __init__(self, config: Optional[PDFConfig] = None):
        super().__init__()
        self.config = config or PDFConfig()
        self._supported_extensions = [".pdf"]
        self.max_file_size_mb = self.config.MAX_FILE_SIZE_MB
        self.extract_images = self.config.EXTRACT_IMAGES
        self._ocr = OCRProcessor()

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in self._supported_extensions
    
    def get_supported_extensions(self) -> List[str]:
        return self._supported_extensions
    
    @with_timeout(60)
    def process(
        self,
        file_path: str,
        output_format: str = "markdown",
        extract_images: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process PDF with native text extraction only.

        ``kwargs`` are ignored so that callers can pass optional parameters
        (e.g. ``user_language`` used by the OCR processor) without causing a
        ``TypeError`` when this processor is selected.
        """
        start_time = time.time()

        try:
            logger.debug(
                "Starting PDF processing",
                extra={"file_path": file_path, "output_format": output_format},
            )
            # Enhanced validation
            validation_error = self._validate_pdf_file(file_path)
            if validation_error:
                return self._create_error_response(validation_error, file_path)
            
            # Extract text with robust error handling
            extraction_result = self._extract_text_safe(file_path)
            
            if not extraction_result['success']:
                return self._create_error_response(extraction_result['error'], file_path)
            
            # Determine if images should be extracted
            if extract_images is None:
                extract_images = self.extract_images

            image_results: List[Dict[str, Any]] = []
            if extract_images:
                image_results = self._extract_images_with_ocr(file_path, output_format)

            # Format output
            if output_format == "html":
                formatted_content = self._format_as_html(extraction_result, file_path)
            else:
                formatted_content = self._format_as_markdown(extraction_result, file_path)

            if extract_images and image_results:
                if output_format == "html":
                    formatted_content += self._format_image_results_html(image_results)
                else:
                    formatted_content += self._format_image_results_markdown(image_results)
            
            processing_time = time.time() - start_time
            
            logger.info(
                "PDF processing completed",
                extra={
                    "file_path": file_path,
                    "page_count": extraction_result['page_count'],
                    "word_count": extraction_result['word_count'],
                    "processing_time": processing_time
                }
            )

            logger.debug(
                "Returning PDF processing result",
                extra={"file_path": file_path, "processing_time": processing_time},
            )

            return self._create_success_response(
                formatted_content,
                file_path,
                page_count=extraction_result['page_count'],
                word_count=extraction_result['word_count'],
                processing_time=processing_time,
                extraction_method="native",
                image_count=len(image_results),
            )
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}", extra={"file_path": file_path})
            return self._create_error_response(f"PDF processing failed: {str(e)}", file_path)

    def _validate_pdf_file(self, file_path: str) -> Optional[str]:
        """Enhanced PDF file validation"""
        if not self.validate_file(file_path):
            return "File validation failed"
        
        # Check file size
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            return f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB"
        
        # Test PDF validity with fitz (PyMuPDF)
        try:
            with fitz.open(file_path) as test_doc:
                if test_doc.is_encrypted:
                    return "Password-protected PDFs not supported"
                if len(test_doc) == 0:
                    return "PDF contains no pages"
        except fitz.FileDataError:
            return "Corrupted or invalid PDF file"
        except Exception as e:
            return f"PDF validation error: {str(e)}"
        
        return None

    def _extract_text_safe(self, file_path: str) -> Dict[str, Any]:
        """Safe text extraction with comprehensive error handling"""
        try:
            logger.debug("Opening PDF", extra={"file_path": file_path})
            result = {
                'success': False,
                'text_content': '',
                'page_count': 0,
                'word_count': 0,
                'warnings': [],
                'metadata': {}
            }
            
            with fitz.open(file_path) as doc:
                result['page_count'] = len(doc)
                
                # Extract metadata
                result['metadata'] = {
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', ''),
                    'creator': doc.metadata.get('creator', ''),
                    'producer': doc.metadata.get('producer', '')
                }
                
                text_parts = []
                empty_pages = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]

                    logger.debug(
                        "Extracting page text",
                        extra={"file_path": file_path, "page": page_num + 1},
                    )

                    page_text = self._extract_page_text_safe(page, page_num)
                    
                    if page_text is None or not page_text.strip():
                        empty_pages.append(page_num + 1)
                        page_text = f"[Page {page_num + 1}: No extractable text]"
                    
                    text_parts.append(f"\n\n--- Page {page_num + 1} ---\n{page_text}")
                
                # Combine all text
                result['text_content'] = ''.join(text_parts)
                result['word_count'] = len(result['text_content'].split())
                
                # Add warnings for empty pages
                if empty_pages:
                    result['warnings'].append(
                        f"Pages with no extractable text: {', '.join(map(str, empty_pages))}"
                    )
                
                # Check if any meaningful text was extracted
                if result['word_count'] < 10:
                    result['warnings'].append(
                        "Very little text extracted - document may be image-based or scanned"
                    )
                
                result['success'] = True
                logger.debug(
                    "Completed text extraction",
                    extra={
                        "file_path": file_path,
                        "page_count": result['page_count'],
                        "word_count": result['word_count'],
                    },
                )
                return result
                
        except fitz.FileDataError as e:
            return {
                'success': False,
                'error': f"PDF file is corrupted or invalid: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Text extraction failed: {str(e)}"
            }

    def _extract_page_text_safe(self, page, page_num: int) -> Optional[str]:
        """Safe page text extraction"""
        try:
            text = page.get_text()
            
            if text is None:
                logger.warning(f"Page {page_num + 1}: get_text() returned None")
                return None
            
            return text
            
        except Exception as e:
            logger.warning(f"Page {page_num + 1} text extraction failed: {e}")
            return None

    def _format_as_markdown(self, result: Dict, file_path: str) -> str:
        """Format as markdown with metadata"""
        filename = Path(file_path).name
        metadata = result['metadata']
        
        markdown = f"""# PDF Document: {filename}

## Document Information
- **Pages**: {result['page_count']}
- **Word Count**: {result['word_count']:,}
- **Extraction Method**: Native text extraction
"""

        # Add metadata if available
        if metadata.get('title'):
            markdown += f"- **Title**: {metadata['title']}\n"
        if metadata.get('author'):
            markdown += f"- **Author**: {metadata['author']}\n"
        if metadata.get('subject'):
            markdown += f"- **Subject**: {metadata['subject']}\n"

        markdown += f"""
## Content
{result['text_content'].strip()}
"""

        # Add warnings if any
        if result['warnings']:
            markdown += "\n## Processing Notes\n"
            for warning in result['warnings']:
                markdown += f"- {warning}\n"

        return markdown

    def _format_as_html(self, result: Dict, file_path: str) -> str:
        """Format as HTML with metadata"""
        filename = Path(file_path).name
        metadata = result['metadata']
        
        html = f"""
<div class="pdf-content">
    <h2>PDF Document: {filename}</h2>
    
    <div class="document-info">
        <h3>Document Information</h3>
        <ul>
            <li><strong>Pages:</strong> {result['page_count']}</li>
            <li><strong>Word Count:</strong> {result['word_count']:,}</li>
            <li><strong>Extraction Method:</strong> Native text extraction</li>
"""

        # Add metadata if available
        if metadata.get('title'):
            html += f"            <li><strong>Title:</strong> {metadata['title']}</li>\n"
        if metadata.get('author'):
            html += f"            <li><strong>Author:</strong> {metadata['author']}</li>\n"

        html += f"""
        </ul>
    </div>
    
    <div class="document-text">
        <h3>Content</h3>
        <pre>{result['text_content'].strip()}</pre>
    </div>
"""

        # Add warnings if any
        if result['warnings']:
            html += """
    <div class="warnings">
        <h3>Processing Notes</h3>
        <ul>
"""
            for warning in result['warnings']:
                html += f"            <li>{warning}</li>\n"
            
            html += """
        </ul>
    </div>
"""

        html += """
</div>
"""
        return html

    def _extract_images_with_ocr(
        self, file_path: str, output_format: str
    ) -> List[Dict[str, Any]]:
        """Extract images from the PDF and run OCR on each."""
        ocr_results: List[Dict[str, Any]] = []
        try:
            with fitz.open(file_path) as doc:
                for page_index in range(len(doc)):
                    page = doc[page_index]
                    images = page.get_images(full=True)
                    for img_index, img in enumerate(images):
                        xref = img[0]
                        img_info = doc.extract_image(xref)
                        if not img_info:
                            continue
                        img_bytes = img_info.get("image")
                        if not img_bytes:
                            continue
                        temp_path = Path(tempfile.gettempdir()) / (
                            f"pdf_img_{page_index+1}_{img_index}.png"
                        )
                        with open(temp_path, "wb") as img_file:
                            img_file.write(img_bytes)
                        result = self._ocr.process(
                            str(temp_path), output_format=output_format
                        )
                        ocr_results.append(
                            {
                                "page": page_index + 1,
                                "index": img_index + 1,
                                "text": result.get("content", ""),
                            }
                        )
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
        return ocr_results

    def _format_image_results_markdown(
        self, results: List[Dict[str, Any]]
    ) -> str:
        md = "\n## OCR Results for Images\n"
        for item in results:
            md += (
                f"\n### Page {item['page']} Image {item['index']}\n"
                f"{item['text']}\n"
            )
        return md

    def _format_image_results_html(
        self, results: List[Dict[str, Any]]
    ) -> str:
        html = "<div class=\"image-ocr\"><h3>OCR Results for Images</h3>"
        for item in results:
            html += (
                f"<h4>Page {item['page']} Image {item['index']}</h4>"
                f"<pre>{item['text']}</pre>"
            )
        html += "</div>"
        return html
