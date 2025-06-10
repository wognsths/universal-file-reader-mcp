import mimetypes
import magic
import os
from pathlib import Path
from typing import Dict, Any, List
import logging



logger = logging.getLogger(__name__)


class FileValidator:
    
    # MIME Type & Extensions Mapping
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
        try:
            path = Path(file_path)
            
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "file_info": {}
            }
            
            if not path.exists():
                validation_result["is_valid"] = False
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            if not path.is_file():
                validation_result["is_valid"] = False
                validation_result["errors"].append("Cannot process directory")
                return validation_result
            
            stat = path.stat()
            validation_result["file_info"] = {
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "is_readable": os.access(file_path, os.R_OK)
            }
            
            if not validation_result["file_info"]["is_readable"]:
                validation_result["is_valid"] = False
                validation_result["errors"].append("No authorization to read file")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error Validating File: {e}")
            return {
                "is_valid": False,
                "errors": [f"Error Validating File: {str(e)}"],
                "warnings": [],
                "file_info": {}
            }
    
    @classmethod
    def validate_file_size(cls, file_path: str, max_size_mb: float) -> bool:
        try:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            return size_mb <= max_size_mb
        except Exception:
            return False
    
    @classmethod
    def detect_file_type(cls, file_path: str) -> Dict[str, str]:
        try:
            # MIME Type Recognition via python-magic
            mime_type = magic.from_file(file_path, mime=True)
            
            # Sub-recognition via mimetypes module
            guessed_type, _ = mimetypes.guess_type(file_path)
            
            extension = Path(file_path).suffix.lower()
            
            return {
                "detected_mime": mime_type,
                "guessed_mime": guessed_type,
                "extension": extension,
                "is_supported": mime_type in cls.SUPPORTED_TYPES
            }
        except Exception as e:
            logger.warning(f"Failed to recognize file type: {e}")
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
        result = cls.validate_file_basic(file_path)
        
        if not result["is_valid"]:
            return result
        
        if not cls.validate_file_size(file_path, max_size_mb):
            result["is_valid"] = False
            result["errors"].append(
                f"파일 크기가 너무 큽니다: {result['file_info']['size_mb']:.1f}MB > {max_size_mb}MB"
            )
        
        file_type_info = cls.detect_file_type(file_path)
        result["file_info"]["type_info"] = file_type_info
        
        extension = file_type_info["extension"]
        processor_extensions = cls._get_processor_extensions(processor_type)
        
        if extension not in processor_extensions:
            result["is_valid"] = False
            result["errors"].append(
                f"Not supported processor type in {processor_type} : {extension}"
            )
        
        return result
    
    @classmethod
    def _get_processor_extensions(cls, processor_type: str) -> List[str]:
        extensions_map = {
            'csv': ['.csv', '.tsv'],
            'pdf': ['.pdf'],
            'ocr': ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'],
            'image': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']
        }
        return extensions_map.get(processor_type, [])


class SecurityValidator:
    
    DANGEROUS_EXTENSIONS = ['.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.vbs', '.js']
    MAX_FILENAME_LENGTH = 255
    
    @classmethod
    def validate_security(cls, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        
        result = {
            "is_safe": True,
            "security_warnings": []
        }
        
        if len(path.name) > cls.MAX_FILENAME_LENGTH:
            result["is_safe"] = False
            result["security_warnings"].append("Filename is too long")
        
        if path.suffix.lower() in cls.DANGEROUS_EXTENSIONS:
            result["is_safe"] = False
            result["security_warnings"].append("Executable file cannot be processed")
        
        if path.name.startswith('.') and len(path.name) > 1:
            result["security_warnings"].append("Hidden file")
        
        return result 