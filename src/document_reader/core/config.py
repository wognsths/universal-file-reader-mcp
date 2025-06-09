"""통합 설정 관리 시스템"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import os


@dataclass
class GlobalConfig:
    """전역 설정"""
    # 공통 파일 처리 설정
    MAX_FILE_SIZE_MB: float = 100.0
    TEMP_DIR: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    ENABLE_STRUCTURED_LOGGING: bool = True
    
    # 성능 설정
    MAX_WORKERS: int = 4
    TIMEOUT_SECONDS: int = 120
    
    def __post_init__(self):
        if self.TEMP_DIR is None:
            self.TEMP_DIR = str(Path.cwd() / "temp")


@dataclass 
class CSVConfig:
    """CSV 프로세서 설정"""
    MAX_FILE_SIZE_MB: float = 30.0
    CHUNK_SIZE: int = 10_000
    SAMPLE_SIZE: int = 1_000
    MAX_COLUMNS: int = 1_000
    MAX_ROWS_PREVIEW: int = 100
    SUPPORTED_ENCODINGS: List[str] = field(default_factory=lambda: [
        "utf-8", "utf-8-sig", "latin1", "cp949", "euc-kr", "shift-jis"
    ])


@dataclass
class OCRConfig:
    """OCR 프로세서 설정"""
    MODEL_NAME: str = "gemini-2.0-flash-exp"
    MAX_FILE_SIZE_MB: float = 50.0
    MAX_DPI: int = 300
    MIN_DPI: int = 150
    MAX_PAGES: int = 10
    COMPRESSION_QUALITY: int = 85
    SUPPORTED_LANGUAGES: Dict[str, str] = field(default_factory=lambda: {
        "ko": "Korean", "en": "English", "ja": "Japanese", 
        "zh": "Chinese", "fr": "French", "de": "German",
        "es": "Spanish", "it": "Italian", "pt": "Portuguese",
        "ru": "Russian", "ar": "Arabic", "hi": "Hindi"
    })


@dataclass
class PDFConfig:
    """PDF 프로세서 설정"""
    MAX_FILE_SIZE_MB: float = 50.0
    EXTRACT_IMAGES: bool = False
    EXTRACT_TABLES: bool = True
    USE_OCR_FALLBACK: bool = True
    MIN_TEXT_THRESHOLD: int = 10  # 최소 텍스트 단어 수


@dataclass
class ProcessorConfig:
    """통합 프로세서 설정"""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    csv_config: CSVConfig = field(default_factory=CSVConfig)
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    pdf_config: PDFConfig = field(default_factory=PDFConfig)
    
    @classmethod
    def from_env(cls) -> 'ProcessorConfig':
        """환경 변수에서 설정을 로드합니다."""
        config = cls()
        
        # 환경 변수에서 설정 오버라이드
        if max_size := os.getenv('MAX_FILE_SIZE_MB'):
            config.global_config.MAX_FILE_SIZE_MB = float(max_size)
            
        if ocr_model := os.getenv('OCR_MODEL_NAME'):
            config.ocr_config.MODEL_NAME = ocr_model
            
        if temp_dir := os.getenv('TEMP_DIR'):
            config.global_config.TEMP_DIR = temp_dir
            
        return config
    
    def get_max_file_size(self, processor_type: str) -> float:
        """프로세서 타입별 최대 파일 크기를 반환합니다."""
        type_configs = {
            'csv': self.csv_config.MAX_FILE_SIZE_MB,
            'ocr': self.ocr_config.MAX_FILE_SIZE_MB,
            'pdf': self.pdf_config.MAX_FILE_SIZE_MB
        }
        return type_configs.get(processor_type, self.global_config.MAX_FILE_SIZE_MB) 