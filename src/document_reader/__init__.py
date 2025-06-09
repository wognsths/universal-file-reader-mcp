"""Universal File Reader - 범용 파일 처리 MCP 서버

PDF, CSV, 이미지 파일에서 텍스트, 표, 그래프 등을 추출하는 MCP 서버입니다.
"""

__version__ = "1.0.0"
__author__ = "Document Reader Team"
__description__ = "범용 파일 처리 MCP 서버"

# 메인 컴포넌트 임포트
from .processor_factory import ProcessorFactory, process_file, get_default_factory
from .mcp_server import UniversalFileReaderMCP, main as run_server

# 설정 및 유틸리티
from .core.config import ProcessorConfig, GlobalConfig, CSVConfig, OCRConfig, PDFConfig
from .core.logging_config import setup_logging, get_logger
from .core.validators import FileValidator, SecurityValidator
from .core.exceptions import DocumentProcessorError, FileError, CSVError, OCRError, PDFError

# 프로세서들
from .processors.base_processor import BaseProcessor
from .processors.csv_processor import CSVProcessor
from .processors.pdf_processor import PDFProcessor
from .processors.ocr_processor import OCRProcessor

__all__ = [
    # 버전 정보
    "__version__",
    "__author__",
    "__description__",
    
    # 메인 기능
    "ProcessorFactory",
    "process_file",
    "get_default_factory",
    "UniversalFileReaderMCP",
    "run_server",
    
    # 설정
    "ProcessorConfig",
    "GlobalConfig", 
    "CSVConfig",
    "OCRConfig",
    "PDFConfig",
    
    # 유틸리티
    "setup_logging",
    "get_logger",
    "FileValidator",
    "SecurityValidator",
    
    # 예외
    "DocumentProcessorError",
    "FileError",
    "CSVError", 
    "OCRError",
    "PDFError",
    
    # 프로세서
    "BaseProcessor",
    "CSVProcessor",
    "PDFProcessor",
    "OCRProcessor"
]


def get_version() -> str:
    """버전 정보를 반환합니다."""
    return __version__


def get_supported_extensions() -> list[str]:
    """지원하는 파일 확장자 목록을 반환합니다."""
    factory = get_default_factory()
    return factory.get_supported_extensions()


def quick_process(file_path: str, output_format: str = "markdown") -> dict:
    """빠른 파일 처리를 위한 편의 함수입니다."""
    return process_file(file_path, output_format)