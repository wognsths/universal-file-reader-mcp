"""공통 예외 클래스들"""

from typing import Optional, Dict, Any


class DocumentProcessorError(Exception):
    """문서 프로세서 기본 예외"""
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.file_path = file_path
        self.details = details or {}
        super().__init__(self.message)


# 파일 관련 예외들
class FileError(DocumentProcessorError):
    """파일 관련 기본 예외"""
    pass


class FileSizeError(FileError):
    """파일 크기 제한 초과"""
    pass


class FileTypeError(FileError):
    """지원하지 않는 파일 형식"""
    pass


class FileAccessError(FileError):
    """파일 접근 권한 오류"""
    pass


class EncodingError(FileError):
    """파일 인코딩 감지/처리 실패"""
    pass


# CSV 관련 예외들
class CSVError(DocumentProcessorError):
    """CSV 처리 기본 예외"""
    pass


class CSVParsingError(CSVError):
    """CSV 파싱 오류"""
    pass


class CSVColumnError(CSVError):
    """CSV 컬럼 관련 오류"""
    pass


# OCR 관련 예외들
class OCRError(DocumentProcessorError):
    """OCR 처리 기본 예외"""
    pass


class APIKeyError(OCRError):
    """API 키 관련 오류"""
    pass


class ProcessingError(OCRError):
    """문서 처리 오류"""
    pass


class ValidationError(DocumentProcessorError):
    """데이터 검증 오류"""
    pass


# PDF 관련 예외들
class PDFError(DocumentProcessorError):
    """PDF 처리 기본 예외"""
    pass


class PDFEncryptedError(PDFError):
    """암호화된 PDF 오류"""
    pass


class PDFCorruptedError(PDFError):
    """손상된 PDF 오류"""
    pass


# 네트워크 관련 예외들
class NetworkError(DocumentProcessorError):
    """네트워크 관련 오류"""
    pass


class TimeoutError(NetworkError):
    """처리 시간 초과"""
    pass


class APIError(NetworkError):
    """외부 API 호출 오류"""
    pass


# 설정 관련 예외들
class ConfigurationError(DocumentProcessorError):
    """설정 관련 오류"""
    pass


class MissingDependencyError(DocumentProcessorError):
    """필수 의존성 누락"""
    pass
