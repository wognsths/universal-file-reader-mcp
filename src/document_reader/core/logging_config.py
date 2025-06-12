import logging
import logging.config
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class StructuredFormatter(logging.Formatter):
    
    def format(self, record: logging.LogRecord) -> str:
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
        
        if hasattr(record, 'extra') and record.extra:
            log_data["extra"] = record.extra
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class DocumentProcessorFilter(logging.Filter):
    
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, 'args') and record.args:
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
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    handlers = {}
    
    if enable_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "json" if enable_json else "standard",
            "stream": sys.stderr,
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
    
    filters = {
        "processor_filter": {
            "()": DocumentProcessorFilter
        }
    }
    
    loggers = {
        "": {
            "level": log_level,
            "handlers": list(handlers.keys())
        },
        "document_reader": {
            "level": log_level,
            "handlers": list(handlers.keys()),
            "propagate": False
        }
    }
    
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
    return logging.getLogger(f"document_reader.{name}")


def log_processing_start(logger: logging.Logger, file_path: str, 
                        processor_type: str, **kwargs) -> None:
    logger.info(
        "File processing started",
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
    level = logging.INFO if success else logging.ERROR
    logger.log(
        level,
        "File successfully processed" if success else "Failed to process file",
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
    logger.error(
        f"Error Occured: {str(error)}",
        extra={
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        },
        exc_info=True
    ) 