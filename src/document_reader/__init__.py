"""Universal File Reader - multi format processing MCP server.

This package provides an MCP server capable of extracting text, tables and
images from PDF, CSV and image files.
"""

# ruff: noqa: E402

import os
import warnings

# NumPy 2.x 호환성 경고 억제
os.environ['NPY_DISABLE_CPU_FEATURES'] = 'AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL'
warnings.filterwarnings('ignore', message='.*NumPy 1.x.*')
warnings.filterwarnings('ignore', category=UserWarning)

__version__ = "1.0.0"
__author__ = "Document Reader Team"
__description__ = "Universal File Processing MCP server"

# Import main components
from .processor_factory import ProcessorFactory, process_file, get_default_factory
from .mcp_server import UniversalFileReaderMCP, main as run_server

# Configuration and utilities
from .core.config import ProcessorConfig, GlobalConfig, CSVConfig, OCRConfig, PDFConfig
from .core.logging_config import setup_logging, get_logger
from .core.validators import FileValidator, SecurityValidator
from .core.exceptions import DocumentProcessorError, FileError, CSVError, OCRError, PDFError

# Processors
from .processors.base_processor import BaseProcessor
from .processors.csv_processor import CSVProcessor
from .processors.pdf_processor import PDFProcessor
from .processors.ocr_processor import OCRProcessor

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    
    # Main features
    "ProcessorFactory",
    "process_file",
    "get_default_factory",
    "UniversalFileReaderMCP",
    "run_server",
    
    # Configuration
    "ProcessorConfig",
    "GlobalConfig", 
    "CSVConfig",
    "OCRConfig",
    "PDFConfig",
    
    # Utilities
    "setup_logging",
    "get_logger",
    "FileValidator",
    "SecurityValidator",
    
    # Exceptions
    "DocumentProcessorError",
    "FileError",
    "CSVError", 
    "OCRError",
    "PDFError",
    
    # Processor classes
    "BaseProcessor",
    "CSVProcessor",
    "PDFProcessor",
    "OCRProcessor"
]


def get_version() -> str:
    """Return package version."""
    return __version__


def get_supported_extensions() -> list[str]:
    """Return list of supported file extensions."""
    factory = get_default_factory()
    return factory.get_supported_extensions()


def quick_process(file_path: str, output_format: str = "markdown") -> dict:
    """Convenience helper for quick file processing."""
    return process_file(file_path, output_format)
