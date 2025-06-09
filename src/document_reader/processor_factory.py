"""프로세서 팩토리 및 관리자"""

from typing import Dict, Type, Optional, List, Any
from pathlib import Path
import logging
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
    """프로세서 팩토리 클래스"""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig.from_env()
        self._processors: Dict[str, BaseProcessor] = {}
        self._processor_types: Dict[str, Type[BaseProcessor]] = {
            'csv': CSVProcessor,
            'pdf': PDFProcessor, 
            'ocr': OCRProcessor
        }
        self._extension_mapping = self._build_extension_mapping()
        
    def _build_extension_mapping(self) -> Dict[str, str]:
        """파일 확장자와 프로세서 타입 매핑을 구축합니다."""
        mapping = {}
        
        # CSV 파일들
        for ext in ['.csv', '.tsv']:
            mapping[ext] = 'csv'
            
        # PDF 파일들 - OCR과 PDF 프로세서 중 선택
        mapping['.pdf'] = 'auto'  # 자동 선택
        
        # 이미지 파일들
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']:
            mapping[ext] = 'ocr'
            
        return mapping
    
    def get_processor(self, file_path: str, force_type: Optional[str] = None) -> BaseProcessor:
        """파일에 적합한 프로세서를 반환합니다."""
        try:
            # 파일 기본 검증
            validation_result = FileValidator.validate_file_basic(file_path)
            if not validation_result["is_valid"]:
                raise FileTypeError(f"파일 검증 실패: {', '.join(validation_result['errors'])}")
            
            # 보안 검증
            security_result = SecurityValidator.validate_security(file_path)
            if not security_result["is_safe"]:
                raise FileTypeError(f"보안 검증 실패: {', '.join(security_result['security_warnings'])}")
            
            # 프로세서 타입 결정
            processor_type = force_type or self._determine_processor_type(file_path)
            
            # 프로세서 생성 또는 반환
            if processor_type not in self._processors:
                self._processors[processor_type] = self._create_processor(processor_type)
            
            return self._processors[processor_type]
            
        except Exception as e:
            logger.error(f"프로세서 생성 실패: {e}", extra={"file_path": file_path})
            raise
    
    def _determine_processor_type(self, file_path: str) -> str:
        """파일 경로를 기반으로 프로세서 타입을 결정합니다."""
        extension = Path(file_path).suffix.lower()
        
        if extension not in self._extension_mapping:
            raise FileTypeError(f"지원하지 않는 파일 형식입니다: {extension}")
        
        processor_type = self._extension_mapping[extension]
        
        # PDF 파일의 경우 자동 선택
        if processor_type == 'auto' and extension == '.pdf':
            return self._choose_pdf_processor(file_path)
        
        return processor_type
    
    def _choose_pdf_processor(self, file_path: str) -> str:
        """PDF 파일에 대한 최적 프로세서를 선택합니다."""
        try:
            # 먼저 PDF 프로세서로 텍스트 추출 시도
            self._create_processor('pdf')
            
            # 간단한 텍스트 추출 테스트
            with fitz.open(file_path) as doc:
                if len(doc) > 0:
                    page = doc[0]
                    text = page.get_text()
                    word_count = len(text.split()) if text else 0
                    
                    # 충분한 텍스트가 있으면 PDF 프로세서 사용
                    if word_count >= self.config.pdf_config.MIN_TEXT_THRESHOLD:
                        logger.info(f"PDF 프로세서 선택: {word_count}개 단어 감지")
                        return 'pdf'
            
            # 텍스트가 부족하면 OCR 프로세서 사용
            logger.info("OCR 프로세서 선택: 텍스트 부족으로 OCR 필요")
            return 'ocr'
            
        except Exception as e:
            logger.warning(f"PDF 프로세서 선택 중 오류, OCR 사용: {e}")
            return 'ocr'
    
    def _create_processor(self, processor_type: str) -> BaseProcessor:
        """프로세서를 생성합니다."""
        if processor_type not in self._processor_types:
            raise ConfigurationError(f"알 수 없는 프로세서 타입: {processor_type}")
        
        processor_class = self._processor_types[processor_type]
        
        try:
            # 프로세서별 설정 전달
            if processor_type == 'csv':
                return processor_class(self.config.csv_config)
            elif processor_type == 'ocr':
                return processor_class(self.config.ocr_config)
            elif processor_type == 'pdf':
                return processor_class()
            else:
                return processor_class()
                
        except Exception as e:
            logger.error(f"{processor_type} 프로세서 생성 실패: {e}")
            raise ConfigurationError(f"프로세서 생성 실패: {e}")
    
    def process_file(self, file_path: str, output_format: str = "markdown", 
                    force_processor: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """파일을 처리합니다."""
        import time
        start_time = time.time()
        
        try:
            # 프로세서 선택
            processor = self.get_processor(file_path, force_processor)
            processor_type = type(processor).__name__
            
            # 처리 시작 로그
            log_processing_start(
                logger, file_path, processor_type,
                output_format=output_format,
                force_processor=force_processor
            )
            
            # 파일 검증 (프로세서별)
            max_size = self.config.get_max_file_size(processor_type.lower().replace('processor', ''))
            validation_result = FileValidator.validate_for_processor(
                file_path, processor_type.lower().replace('processor', ''), max_size
            )
            
            if not validation_result["is_valid"]:
                raise FileTypeError(f"파일 검증 실패: {', '.join(validation_result['errors'])}")
            
            # 파일 처리
            result = processor.process(file_path, output_format, **kwargs)

            # PDF 처리 실패 또는 텍스트 부족 시 OCR 폴백
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
                    "PDF 처리 결과가 부족하여 OCR 폴백을 수행합니다",
                    extra={"file_path": file_path},
                )
                ocr_proc = self.get_processor(file_path, "ocr")
                result = ocr_proc.process(
                    file_path, output_format, **kwargs
                )
                result["fallback_to_ocr"] = True
            
            processing_time = time.time() - start_time
            
            # 처리 완료 로그
            log_processing_end(
                logger, file_path, processor_type, True, processing_time,
                file_size_mb=validation_result["file_info"]["size_mb"]
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # 처리 실패 로그
            log_processing_end(
                logger, file_path, "unknown", False, processing_time,
                error_message=str(e)
            )
            
            raise
    
    def get_supported_extensions(self) -> List[str]:
        """지원하는 모든 파일 확장자를 반환합니다."""
        return list(self._extension_mapping.keys())
    
    def get_processor_info(self) -> Dict[str, Any]:
        """프로세서 정보를 반환합니다."""
        return {
            "supported_extensions": self.get_supported_extensions(),
            "processor_types": list(self._processor_types.keys()),
            "extension_mapping": self._extension_mapping,
            "config": {
                "max_file_sizes": {
                    "csv": self.config.csv_config.MAX_FILE_SIZE_MB,
                    "pdf": self.config.pdf_config.MAX_FILE_SIZE_MB,
                    "ocr": self.config.ocr_config.MAX_FILE_SIZE_MB
                }
            }
        }


# 전역 팩토리 인스턴스
_default_factory: Optional[ProcessorFactory] = None


def get_default_factory() -> ProcessorFactory:
    """기본 프로세서 팩토리를 반환합니다."""
    global _default_factory
    if _default_factory is None:
        _default_factory = ProcessorFactory()
    return _default_factory


def process_file(file_path: str, output_format: str = "markdown", 
                force_processor: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """편의 함수: 파일을 처리합니다."""
    factory = get_default_factory()
    return factory.process_file(file_path, output_format, force_processor, **kwargs) 