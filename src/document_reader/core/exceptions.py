from typing import Optional, Dict, Any


class DocumentProcessorError(Exception):
    """Base Exception of Document Processor"""
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.file_path = file_path
        self.details = details or {}
        super().__init__(self.message)


class FileError(DocumentProcessorError):
    """Base Exception of Files"""
    pass


class FileSizeError(FileError):
    """File Size Exceeded the Limit"""
    pass


class FileTypeError(FileError):
    """Not Supported File Type"""
    pass


class FileAccessError(FileError):
    """File Access Authorization Error"""
    pass


class EncodingError(FileError):
    """Failed to Encode File"""
    pass


# CSV Exceptions
class CSVError(DocumentProcessorError):
    """Base Exceptions of CSV Processor"""
    pass


class CSVParsingError(CSVError):
    """CSV Parsing Error"""
    pass


class CSVColumnError(CSVError):
    """CSV Column Error"""
    pass


# OCR Exceptions
class OCRError(DocumentProcessorError):
    """Base Exceptions of OCR Processor"""
    pass


class APIKeyError(OCRError):
    """API Key Error"""
    pass


class ProcessingError(OCRError):
    """Document Processing Error"""
    pass


class ValidationError(DocumentProcessorError):
    """Data Validation Error"""
    pass


class PDFError(DocumentProcessorError):
    """Base Exceptions of PDF Processor"""
    pass


class PDFEncryptedError(PDFError):
    """Encrypted PDF Error"""
    pass


class PDFCorruptedError(PDFError):
    """Corrupted PDF Error"""
    pass


# Network Error
class NetworkError(DocumentProcessorError):
    """Network Error"""
    pass


class TimeoutError(NetworkError):
    """Timeout Error"""
    pass


class APIError(NetworkError):
    """API Call Error"""
    pass


# Configuration Error
class ConfigurationError(DocumentProcessorError):
    """Configuration Error"""
    pass


class MissingDependencyError(DocumentProcessorError):
    """Missing Needed Dependency"""
    pass
