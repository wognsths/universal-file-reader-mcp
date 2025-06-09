"""파일 검증 유틸리티"""

import mimetypes
import magic
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from .exceptions import (
    FileSizeError,
    FileTypeError,
    FileAccessError,
    ValidationError
)

logger = logging.getLogger(__name__)


class FileValidator:
    """통합 파일 검증 클래스"""
    
    # MIME 타입과 확장자 매핑
    SUPPORTED_TYPES = {
        'application/pdf': ['.pdf'],
        'text/csv': ['.csv'],
        'text/tab-separated-values': ['.tsv'],
        'image/jpeg': ['.jpg', '.jpeg'],
        'image/png': ['.png'],
        'image/bmp': ['.bmp'],
        'image/tiff': ['.tiff', '.tif'],
        'image/webp': ['.webp']
    }
    
    @classmethod
    def validate_file_basic(cls, file_path: str) -> Dict[str, Any]:
        """기본 파일 검증을 수행합니다."""
        try:
            path = Path(file_path)
            
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "file_info": {}
            }
            
            # 파일 존재 확인
            if not path.exists():
                validation_result["is_valid"] = False
                validation_result["errors"].append("파일이 존재하지 않습니다")
                return validation_result
            
            # 파일인지 확인
            if not path.is_file():
                validation_result["is_valid"] = False
                validation_result["errors"].append("디렉토리는 처리할 수 없습니다")
                return validation_result
            
            # 파일 정보 수집
            stat = path.stat()
            validation_result["file_info"] = {
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "is_readable": os.access(file_path, os.R_OK)
            }
            
            # 읽기 권한 확인
            if not validation_result["file_info"]["is_readable"]:
                validation_result["is_valid"] = False
                validation_result["errors"].append("파일 읽기 권한이 없습니다")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"파일 검증 중 오류 발생: {e}")
            return {
                "is_valid": False,
                "errors": [f"파일 검증 중 오류 발생: {str(e)}"],
                "warnings": [],
                "file_info": {}
            }
    
    @classmethod
    def validate_file_size(cls, file_path: str, max_size_mb: float) -> bool:
        """파일 크기를 검증합니다."""
        try:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            return size_mb <= max_size_mb
        except Exception:
            return False
    
    @classmethod
    def detect_file_type(cls, file_path: str) -> Dict[str, str]:
        """파일의 실제 타입을 감지합니다."""
        try:
            # python-magic을 사용한 MIME 타입 감지
            mime_type = magic.from_file(file_path, mime=True)
            
            # mimetypes 모듈로 보조 확인
            guessed_type, _ = mimetypes.guess_type(file_path)
            
            extension = Path(file_path).suffix.lower()
            
            return {
                "detected_mime": mime_type,
                "guessed_mime": guessed_type,
                "extension": extension,
                "is_supported": mime_type in cls.SUPPORTED_TYPES
            }
        except Exception as e:
            logger.warning(f"파일 타입 감지 실패: {e}")
            extension = Path(file_path).suffix.lower()
            return {
                "detected_mime": "unknown",
                "guessed_mime": None,
                "extension": extension,
                "is_supported": False
            }
    
    @classmethod
    def validate_for_processor(cls, file_path: str, processor_type: str, 
                             max_size_mb: float) -> Dict[str, Any]:
        """특정 프로세서에 대한 파일 검증을 수행합니다."""
        # 기본 검증
        result = cls.validate_file_basic(file_path)
        
        if not result["is_valid"]:
            return result
        
        # 크기 검증
        if not cls.validate_file_size(file_path, max_size_mb):
            result["is_valid"] = False
            result["errors"].append(
                f"파일 크기가 너무 큽니다: {result['file_info']['size_mb']:.1f}MB > {max_size_mb}MB"
            )
        
        # 타입 검증
        file_type_info = cls.detect_file_type(file_path)
        result["file_info"]["type_info"] = file_type_info
        
        # 프로세서별 지원 확인
        extension = file_type_info["extension"]
        processor_extensions = cls._get_processor_extensions(processor_type)
        
        if extension not in processor_extensions:
            result["is_valid"] = False
            result["errors"].append(
                f"{processor_type} 프로세서가 지원하지 않는 파일 형식입니다: {extension}"
            )
        
        return result
    
    @classmethod
    def _get_processor_extensions(cls, processor_type: str) -> List[str]:
        """프로세서 타입별 지원 확장자를 반환합니다."""
        extensions_map = {
            'csv': ['.csv', '.tsv'],
            'pdf': ['.pdf'],
            'ocr': ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'],
            'image': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']
        }
        return extensions_map.get(processor_type, [])


class SecurityValidator:
    """보안 관련 파일 검증"""
    
    DANGEROUS_EXTENSIONS = ['.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.vbs', '.js']
    MAX_FILENAME_LENGTH = 255
    
    @classmethod
    def validate_security(cls, file_path: str) -> Dict[str, Any]:
        """보안 검증을 수행합니다."""
        path = Path(file_path)
        
        result = {
            "is_safe": True,
            "security_warnings": []
        }
        
        # 파일명 길이 검증
        if len(path.name) > cls.MAX_FILENAME_LENGTH:
            result["is_safe"] = False
            result["security_warnings"].append("파일명이 너무 깁니다")
        
        # 위험한 확장자 검증
        if path.suffix.lower() in cls.DANGEROUS_EXTENSIONS:
            result["is_safe"] = False
            result["security_warnings"].append("실행 가능한 파일은 처리할 수 없습니다")
        
        # 숨김 파일 검증 (옵션)
        if path.name.startswith('.') and len(path.name) > 1:
            result["security_warnings"].append("숨김 파일입니다")
        
        return result 