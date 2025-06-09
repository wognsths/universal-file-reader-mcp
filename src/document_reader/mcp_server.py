"""MCP 서버 구현"""

import asyncio
from typing import Dict, Any, Optional, List
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .processor_factory import ProcessorFactory
from .core.config import ProcessorConfig
from .core.logging_config import setup_logging, get_logger
from .core.exceptions import DocumentProcessorError

# 로깅 설정
setup_logging(log_level="INFO", enable_json=True)
logger = get_logger("mcp_server")

# MCP 서버 인스턴스
app = Server("universal-file-reader")


class UniversalFileReaderMCP:
    """Universal File Reader MCP 서버"""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.factory = ProcessorFactory(config)
        logger.info("Universal File Reader MCP 서버 초기화 완료")
    
    async def read_file(self, file_path: str, output_format: str = "markdown", 
                       force_processor: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """파일을 읽고 처리하여 결과를 반환합니다."""
        try:
            logger.info(f"파일 처리 요청: {file_path}")
            
            result = self.factory.process_file(
                file_path=file_path,
                output_format=output_format,
                force_processor=force_processor,
                **kwargs
            )
            
            logger.info(f"파일 처리 완료: {file_path}")
            return result
            
        except DocumentProcessorError as e:
            logger.error(f"문서 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            return {
                "success": False,
                "error": f"예상치 못한 오류: {str(e)}",
                "error_type": "UnexpectedError",
                "file_path": file_path
            }
    
    def get_server_info(self) -> Dict[str, Any]:
        """서버 정보를 반환합니다."""
        return {
            "name": "Universal File Reader",
            "version": "1.0.0",
            "description": "PDF, CSV, 이미지 파일에서 텍스트, 표, 그래프를 추출하는 MCP 서버",
            "processor_info": self.factory.get_processor_info()
        }


# MCP 서버 인스턴스
mcp_server = UniversalFileReaderMCP()


@app.list_tools()
async def list_tools() -> List[Tool]:
    """사용 가능한 도구 목록을 반환합니다."""
    return [
        Tool(
            name="read_file",
            description="PDF, CSV, 이미지 파일에서 텍스트, 표, 그래프 등을 추출합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "처리할 파일의 경로"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["markdown", "html", "structured"],
                        "default": "markdown",
                        "description": "출력 형식"
                    },
                    "force_processor": {
                        "type": "string",
                        "enum": ["csv", "pdf", "ocr"],
                        "description": "강제로 사용할 프로세서 (선택사항)"
                    },
                    "user_language": {
                        "type": "string",
                        "default": "auto",
                        "description": "OCR 처리 시 언어 설정 (ko, en, ja, zh, auto 등)"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="get_supported_formats",
            description="지원하는 파일 형식과 프로세서 정보를 반환합니다.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="validate_file",
            description="파일이 처리 가능한지 검증합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "검증할 파일의 경로"
                    },
                    "processor_type": {
                        "type": "string",
                        "enum": ["csv", "pdf", "ocr", "auto"],
                        "default": "auto",
                        "description": "프로세서 타입"
                    }
                },
                "required": ["file_path"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """도구 호출을 처리합니다."""
    try:
        if name == "read_file":
            result = await mcp_server.read_file(**arguments)
            
            if result.get("success", False):
                content = result.get("content", "")
                
                # 추가 정보도 포함
                metadata = {
                    "processor": result.get("processor", "unknown"),
                    "file_info": result.get("file_info", {}),
                    "processing_time": result.get("processing_time", 0)
                }
                
                response_text = f"{content}\n\n---\n\n**처리 정보:**\n```json\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n```"
                
                return [TextContent(type="text", text=response_text)]
            else:
                error_info = {
                    "error": result.get("error", "알 수 없는 오류"),
                    "error_type": result.get("error_type", "Unknown"),
                    "file_path": result.get("file_path", "")
                }
                error_text = f"**파일 처리 실패**\n\n```json\n{json.dumps(error_info, ensure_ascii=False, indent=2)}\n```"
                return [TextContent(type="text", text=error_text)]
        
        elif name == "get_supported_formats":
            server_info = mcp_server.get_server_info()
            info_text = f"""# Universal File Reader 정보

## 지원 파일 형식
{', '.join(server_info['processor_info']['supported_extensions'])}

## 프로세서 타입
- **CSV**: CSV, TSV 파일 처리 (구조 분석, 통계, 청킹)
- **PDF**: PDF 네이티브 텍스트 추출 (메타데이터 포함)
- **OCR**: 이미지 및 스캔된 문서의 OCR 처리 (다중 요소 감지)

## 설정
```json
{json.dumps(server_info['processor_info']['config'], ensure_ascii=False, indent=2)}
```
"""
            return [TextContent(type="text", text=info_text)]
        
        elif name == "validate_file":
            file_path = arguments["file_path"]
            processor_type = arguments.get("processor_type", "auto")
            
            from .core.validators import FileValidator, SecurityValidator
            
            # 기본 검증
            basic_result = FileValidator.validate_file_basic(file_path)
            security_result = SecurityValidator.validate_security(file_path)
            
            # 프로세서별 검증 (auto가 아닌 경우)
            processor_result = None
            if processor_type != "auto":
                max_size = mcp_server.factory.config.get_max_file_size(processor_type)
                processor_result = FileValidator.validate_for_processor(
                    file_path, processor_type, max_size
                )
            
            validation_info = {
                "file_path": file_path,
                "basic_validation": basic_result,
                "security_validation": security_result,
                "processor_validation": processor_result,
                "overall_valid": (
                    basic_result.get("is_valid", False) and 
                    security_result.get("is_safe", False) and
                    (processor_result is None or processor_result.get("is_valid", False))
                )
            }
            
            validation_text = f"""# 파일 검증 결과

## 전체 검증 결과
{'✅ 처리 가능' if validation_info['overall_valid'] else '❌ 처리 불가능'}

## 상세 정보
```json
{json.dumps(validation_info, ensure_ascii=False, indent=2)}
```
"""
            return [TextContent(type="text", text=validation_text)]
        
        else:
            return [TextContent(type="text", text=f"알 수 없는 도구: {name}")]
    
    except Exception as e:
        logger.error(f"도구 호출 오류: {e}")
        error_text = f"**도구 호출 오류**\n\n오류: {str(e)}\n도구: {name}\n인수: {json.dumps(arguments, ensure_ascii=False, indent=2)}"
        return [TextContent(type="text", text=error_text)]


async def main():
    """MCP 서버를 시작합니다."""
    logger.info("Universal File Reader MCP 서버 시작")
    
    async with stdio_server() as streams:
        await app.run(
            streams[0], streams[1],
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main()) 