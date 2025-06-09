"""통합 로깅 설정"""

import logging
import logging.config
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class StructuredFormatter(logging.Formatter):
    """구조화된 JSON 로그 포맷터"""
    
    def format(self, record: logging.LogRecord) -> str:
        # 기본 로그 정보
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # 추가 정보가 있다면 포함
        if hasattr(record, 'extra') and record.extra:
            log_data["extra"] = record.extra
        
        # 예외 정보가 있다면 포함
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class DocumentProcessorFilter(logging.Filter):
    """문서 프로세서 관련 로그 필터"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # 민감한 정보 제거 (API 키 등)
        if hasattr(record, 'args') and record.args:
            # API 키가 포함된 메시지 필터링
            msg = str(record.msg)
            if 'api_key' in msg.lower() or 'password' in msg.lower():
                record.msg = record.msg.replace(record.args[0] if record.args else '', '[REDACTED]')
        
        return True


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = True,
    enable_console: bool = True
) -> None:
    """로깅 시스템을 설정합니다."""
    
    # 로그 디렉토리 생성
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 포맷터 설정
    formatters = {
        "standard": {
            "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    }
    
    if enable_json:
        formatters["json"] = {
            "()": StructuredFormatter
        }
    
    # 핸들러 설정
    handlers = {}
    
    if enable_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "json" if enable_json else "standard",
            "stream": sys.stdout,
            "filters": ["processor_filter"]
        }
    
    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "json" if enable_json else "standard",
            "filename": log_file,
            "maxBytes": 10_000_000,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
            "filters": ["processor_filter"]
        }
    
    # 필터 설정
    filters = {
        "processor_filter": {
            "()": DocumentProcessorFilter
        }
    }
    
    # 로거 설정
    loggers = {
        "": {  # 루트 로거
            "level": log_level,
            "handlers": list(handlers.keys())
        },
        "document_reader": {
            "level": log_level,
            "handlers": list(handlers.keys()),
            "propagate": False
        }
    }
    
    # 로깅 설정 적용
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "filters": filters,
        "handlers": handlers,
        "loggers": loggers
    }
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """구조화된 로거를 반환합니다."""
    return logging.getLogger(f"document_reader.{name}")


def log_processing_start(logger: logging.Logger, file_path: str, 
                        processor_type: str, **kwargs) -> None:
    """처리 시작 로그를 기록합니다."""
    logger.info(
        "파일 처리 시작",
        extra={
            "event": "processing_start",
            "file_path": file_path,
            "processor_type": processor_type,
            **kwargs
        }
    )


def log_processing_end(logger: logging.Logger, file_path: str, 
                      processor_type: str, success: bool, 
                      processing_time: float, **kwargs) -> None:
    """처리 완료 로그를 기록합니다."""
    level = logging.INFO if success else logging.ERROR
    logger.log(
        level,
        "파일 처리 완료" if success else "파일 처리 실패",
        extra={
            "event": "processing_end",
            "file_path": file_path,
            "processor_type": processor_type,
            "success": success,
            "processing_time": processing_time,
            **kwargs
        }
    )


def log_error(logger: logging.Logger, error: Exception, 
              context: Dict[str, Any] = None) -> None:
    """에러 로그를 기록합니다."""
    logger.error(
        f"오류 발생: {str(error)}",
        extra={
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        },
        exc_info=True
    ) 