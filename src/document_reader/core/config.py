from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import os


@dataclass
class GlobalConfig:
    """Global Configuration"""
    MAX_FILE_SIZE_MB: float = 100.0
    TEMP_DIR: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    ENABLE_STRUCTURED_LOGGING: bool = True
    
    MAX_WORKERS: int = 4
    TIMEOUT_SECONDS: int = 120
    
    def __post_init__(self):
        if self.TEMP_DIR is None:
            self.TEMP_DIR = str(Path.cwd() / "temp")


@dataclass 
class CSVConfig:
    """CSV Processor Configuration"""
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
    """OCR Processor Configuration"""
    MODEL_NAME: str = "gemini-2.0-flash"
    MAX_FILE_SIZE_MB: float = 50.0
    MAX_DPI: int = 300
    MIN_DPI: int = 150
    MAX_PAGES: int = 10
    MAX_PAGE_PER_PROCESS: int = 5
    COMPRESSION_QUALITY: int = 85
    SUPPORTED_LANGUAGES: Dict[str, str] = field(default_factory=lambda: {
        "ko": "Korean", "en": "English", "ja": "Japanese", 
        "zh": "Chinese", "fr": "French", "de": "German",
        "es": "Spanish", "it": "Italian", "pt": "Portuguese",
        "ru": "Russian", "ar": "Arabic", "hi": "Hindi"
    })


@dataclass
class PDFConfig:
    """PDF Processor Configuration"""
    MAX_FILE_SIZE_MB: float = 50.0
    EXTRACT_IMAGES: bool = False
    EXTRACT_TABLES: bool = True
    USE_OCR_FALLBACK: bool = True
    MIN_TEXT_THRESHOLD: int = 10  # Minimun text word counts


@dataclass
class ProcessorConfig:
    """Overall Processor Configuration"""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    csv_config: CSVConfig = field(default_factory=CSVConfig)
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    pdf_config: PDFConfig = field(default_factory=PDFConfig)
    
    @classmethod
    def from_env(cls) -> 'ProcessorConfig':
        """Load settings from .env"""
        config = cls()
        
        # Override
        if max_size := os.getenv('MAX_FILE_SIZE_MB'):
            config.global_config.MAX_FILE_SIZE_MB = float(max_size)
            
        if ocr_model := os.getenv('OCR_MODEL_NAME'):
            config.ocr_config.MODEL_NAME = ocr_model

        if temp_dir := os.getenv('TEMP_DIR'):
            config.global_config.TEMP_DIR = temp_dir

        if max_page_per_proc := os.getenv('MAX_PAGE_PER_PROCESS'):
            config.ocr_config.MAX_PAGE_PER_PROCESS = int(max_page_per_proc)

        return config
    
    def get_max_file_size(self, processor_type: str) -> float:
        type_configs = {
            'csv': self.csv_config.MAX_FILE_SIZE_MB,
            'ocr': self.ocr_config.MAX_FILE_SIZE_MB,
            'pdf': self.pdf_config.MAX_FILE_SIZE_MB
        }
        return type_configs.get(processor_type, self.global_config.MAX_FILE_SIZE_MB) 