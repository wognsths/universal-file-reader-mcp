from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):

    def __init__(self):
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    def process(
        self, file_path: str, output_format: str = "markdown", **kwargs
    ) -> Dict[str, Any]:
        """Process file and return the result.

        ``kwargs`` are ignored by processors that do not support additional
        parameters.  This makes it safe to pass optional arguments (for
        example ``user_language`` for the OCR processor) without raising a
        ``TypeError``.
        """
        pass
    
    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """Check whether the file extension is supported"""
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Returns Supported Extensions"""
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """Validate Basic File"""
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    def validate_file_size(self, file_path: str, max_size_mb: float) -> bool:
        """Validate File Size"""
        try:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            return size_mb <= max_size_mb
        except Exception:
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Return File Information"""
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
        """Create Success Response"""
        return {
            "success": True,
            "content": content,
            "file_path": file_path,
            "processor": self.name,
            "file_info": self.get_file_info(file_path),
            **kwargs
        }
    
    def _create_error_response(self, error_msg: str, file_path: str, **kwargs) -> Dict[str, Any]:
        """Create Error Response"""
        logger.error(f"{self.name}: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "file_path": file_path,
            "processor": self.name,
            "file_info": self.get_file_info(file_path),
            **kwargs
        }