from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """기본 파일 프로세서 추상 클래스"""

    def __init__(self):
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    def process(self, file_path: str, output_format: str = "markdown") -> Dict[str, Any]:
        """파일을 처리하고 결과를 반환합니다."""
        pass
    
    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """파일 확장자를 지원하는지 확인합니다."""
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 목록을 반환합니다."""
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """기본 파일 검증을 수행합니다."""
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    def validate_file_size(self, file_path: str, max_size_mb: float) -> bool:
        """파일 크기를 검증합니다."""
        try:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            return size_mb <= max_size_mb
        except Exception:
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """파일의 기본 정보를 반환합니다."""
        try:
            path = Path(file_path)
            stat = path.stat()
            return {
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_mb": stat.st_size / (1024 * 1024),
                "size_bytes": stat.st_size,
                "exists": path.exists(),
                "is_file": path.is_file()
            }
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {}
    
    def _create_success_response(self, content: str, file_path: str, **kwargs) -> Dict[str, Any]:
        """성공 응답을 생성합니다."""
        return {
            "success": True,
            "content": content,
            "file_path": file_path,
            "processor": self.name,
            "file_info": self.get_file_info(file_path),
            **kwargs
        }
    
    def _create_error_response(self, error_msg: str, file_path: str, **kwargs) -> Dict[str, Any]:
        """에러 응답을 생성합니다."""
        logger.error(f"{self.name}: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "file_path": file_path,
            "processor": self.name,
            "file_info": self.get_file_info(file_path),
            **kwargs
        }