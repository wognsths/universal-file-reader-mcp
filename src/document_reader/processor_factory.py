"""Processor factory and manager."""

from typing import Dict, Type, Optional, List, Any
from pathlib import Path
import concurrent.futures

import fitz

from .processors.base_processor import BaseProcessor
from .processors.csv_processor import CSVProcessor
from .processors.pdf_processor import PDFProcessor
from .processors.ocr_processor import OCRProcessor

from .core.config import ProcessorConfig
from .core.validators import FileValidator, SecurityValidator
from .core.exceptions import (
    FileTypeError,
    ConfigurationError,
)
from .core.logging_config import get_logger, log_processing_start, log_processing_end

logger = get_logger("processor_factory")


class ProcessorFactory:
    """Processor factory class."""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig.from_env()
        self._processors: Dict[str, BaseProcessor] = {}
        self._processor_types: Dict[str, Type[BaseProcessor]] = {
            "csv": CSVProcessor,
            "pdf": PDFProcessor,
            "ocr": OCRProcessor,
        }
        self._extension_mapping = self._build_extension_mapping()

    def _build_extension_mapping(self) -> Dict[str, str]:
        """Build mapping between file extensions and processor types."""
        mapping = {}

        # CSV files
        for ext in [".csv", ".tsv"]:
            mapping[ext] = "csv"

        # PDF files - choose between OCR and PDF processor
        mapping[".pdf"] = "auto"  # choose automatically

        # Image files
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]:
            mapping[ext] = "ocr"

        return mapping

    def get_processor(
        self, file_path: str, force_type: Optional[str] = None
    ) -> BaseProcessor:
        """Return a processor suitable for the file."""
        try:
            # Basic file validation
            validation_result = FileValidator.validate_file_basic(file_path)
            if not validation_result["is_valid"]:
                raise FileTypeError(
                    f"File validation failed: {', '.join(validation_result['errors'])}"
                )

            # Security validation
            security_result = SecurityValidator.validate_security(file_path)
            if not security_result["is_safe"]:
                raise FileTypeError(
                    f"Security validation failed: {', '.join(security_result['security_warnings'])}"
                )

            # Determine processor type
            processor_type = force_type or self._determine_processor_type(file_path)

            # Create processor if not cached
            if processor_type not in self._processors:
                self._processors[processor_type] = self._create_processor(
                    processor_type
                )

            return self._processors[processor_type]

        except Exception as e:
            logger.error(
                f"Processor creation failed: {e}", extra={"file_path": file_path}
            )
            raise

    def _determine_processor_type(self, file_path: str) -> str:
        """Determine processor type based on file path."""
        extension = Path(file_path).suffix.lower()

        if extension not in self._extension_mapping:
            raise FileTypeError(f"Unsupported file type: {extension}")

        processor_type = self._extension_mapping[extension]

        # Automatic choice for PDF files
        if processor_type == "auto" and extension == ".pdf":
            return self._choose_pdf_processor(file_path)

        return processor_type

    def _choose_pdf_processor(self, file_path: str) -> str:
        """Choose the best processor for a PDF file."""
        try:
            # Attempt native PDF text extraction first
            self._create_processor("pdf")

            # Sample multiple pages to determine if OCR is needed
            with fitz.open(file_path) as doc:
                if len(doc) > 0:
                    pages_to_check = min(3, len(doc))
                    total_words = 0
                    for i in range(pages_to_check):
                        text = doc[i].get_text()
                        total_words += len(text.split()) if text else 0

                    avg_words = total_words / pages_to_check if pages_to_check else 0
                    if avg_words >= self.config.pdf_config.MIN_TEXT_THRESHOLD:
                        logger.info(
                            "PDF processor selected: %.1f avg words across %d pages",
                            avg_words,
                            pages_to_check,
                        )
                        return "pdf"

            # Fallback to OCR processor if text is insufficient
            logger.info("OCR processor selected: OCR needed due to insufficient text")
            return "ocr"

        except Exception as e:
            logger.warning(f"Error choosing PDF processor, using OCR: {e}")
            return "ocr"

    def _create_processor(self, processor_type: str) -> BaseProcessor:
        """Instantiate a processor."""
        if processor_type not in self._processor_types:
            raise ConfigurationError(f"Unknown processor type: {processor_type}")

        processor_class = self._processor_types[processor_type]

        try:
            # Pass processor specific configuration
            if processor_type == "csv":
                return processor_class(self.config.csv_config)
            elif processor_type == "ocr":
                return processor_class(self.config.ocr_config)
            elif processor_type == "pdf":
                return processor_class()
            else:
                return processor_class()

        except Exception as e:
            logger.error(f"Failed to create {processor_type} processor: {e}")
            raise ConfigurationError(f"Processor creation failed: {e}")

    def process_file(
        self,
        file_path: str,
        output_format: str = "markdown",
        force_processor: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process a file."""
        import time

        start_time = time.time()

        try:
            # Choose processor
            processor = self.get_processor(file_path, force_processor)
            processor_type = type(processor).__name__

            # Log start of processing
            log_processing_start(
                logger,
                file_path,
                processor_type,
                output_format=output_format,
                force_processor=force_processor,
            )

            # Validate file for the processor
            max_size = self.config.get_max_file_size(
                processor_type.lower().replace("processor", "")
            )
            validation_result = FileValidator.validate_for_processor(
                file_path, processor_type.lower().replace("processor", ""), max_size
            )

            if not validation_result["is_valid"]:
                raise FileTypeError(
                    f"File validation failed: {', '.join(validation_result['errors'])}"
                )

            # Process file with optional timeout
            logger.debug(f"About to call processor.process for {file_path}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(processor.process, file_path, output_format, **kwargs)
                try:
                    result = future.result(timeout=self.config.global_config.TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "Processing timeout, falling back to OCR",
                        extra={"file_path": file_path},
                    )
                    if isinstance(processor, PDFProcessor) and self.config.pdf_config.USE_OCR_FALLBACK:
                        ocr_proc = self.get_processor(file_path, "ocr")
                        result = ocr_proc.process(file_path, output_format, **kwargs)
                        result["fallback_to_ocr"] = True
                    else:
                        raise
            logger.debug(f"processor.process completed for {file_path}")

            # Fallback to OCR if PDF processing fails or text is insufficient
            if (
                isinstance(processor, PDFProcessor)
                and self.config.pdf_config.USE_OCR_FALLBACK
                and (
                    not result.get("success", False)
                    or result.get("word_count", 0)
                    < self.config.pdf_config.MIN_TEXT_THRESHOLD
                )
            ):
                logger.info(
                    "PDF processing result insufficient, falling back to OCR",
                    extra={"file_path": file_path},
                )
                ocr_proc = self.get_processor(file_path, "ocr")
                result = ocr_proc.process(file_path, output_format, **kwargs)
                result["fallback_to_ocr"] = True

            processor_type = type(processor).__name__
            processing_time = time.time() - start_time

            # Log processing completion
            log_processing_end(
                logger,
                file_path,
                processor_type,
                True,
                processing_time,
                file_size_mb=validation_result["file_info"]["size_mb"],
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time

            # Log processing failure
            log_processing_end(
                logger,
                file_path,
                "unknown",
                False,
                processing_time,
                error_message=str(e),
            )

            raise

    def get_supported_extensions(self) -> List[str]:
        """Return all supported file extensions."""
        return list(self._extension_mapping.keys())

    def get_processor_info(self) -> Dict[str, Any]:
        """Return processor information."""
        return {
            "supported_extensions": self.get_supported_extensions(),
            "processor_types": list(self._processor_types.keys()),
            "extension_mapping": self._extension_mapping,
            "config": {
                "max_file_sizes": {
                    "csv": self.config.csv_config.MAX_FILE_SIZE_MB,
                    "pdf": self.config.pdf_config.MAX_FILE_SIZE_MB,
                    "ocr": self.config.ocr_config.MAX_FILE_SIZE_MB,
                }
            },
        }


# Global factory instance
_default_factory: Optional[ProcessorFactory] = None


def get_default_factory() -> ProcessorFactory:
    """Return the default processor factory."""
    global _default_factory
    if _default_factory is None:
        _default_factory = ProcessorFactory()
    return _default_factory


def process_file(
    file_path: str,
    output_format: str = "markdown",
    force_processor: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to process a file."""
    factory = get_default_factory()
    return factory.process_file(file_path, output_format, force_processor, **kwargs)
