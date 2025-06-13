"""MCP server implementation"""

import asyncio
from typing import Dict, Any, Optional, List
import json
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except Exception:  # noqa: BLE001
    pass
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .processor_factory import ProcessorFactory
from .core.config import ProcessorConfig
from .core.logging_config import setup_logging, get_logger
from .core.exceptions import DocumentProcessorError

# Configure logging
setup_logging(log_level="INFO", enable_json=True)
logger = get_logger("mcp_server")

# MCP server instance
app = Server("universal-file-reader")


class UniversalFileReaderMCP:
    """Universal File Reader MCP server."""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.factory = ProcessorFactory(config)
        self.initialized = False
        logger.info("Universal File Reader MCP server initialised")

    async def initialize(self) -> None:
        """Perform initialization tasks before serving requests."""
        # Pre-instantiate processors so that first request is handled promptly
        for proc_type in ["csv", "pdf", "ocr"]:
            try:
                self.factory._create_processor(proc_type)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to pre-initialize %s processor: %s", proc_type, exc
                )
        self.initialized = True
        logger.info("Universal File Reader MCP server ready")

    async def read_file(
        self,
        file_path: str,
        output_format: str = "markdown",
        force_processor: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Read a file and return the processed result."""
        try:
            logger.info(f"File processing requested: {file_path}")

            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.factory.process_file,
                    file_path=file_path,
                    output_format=output_format,
                    force_processor=force_processor,
                    **kwargs,
                ),
                timeout=30.0,
            )

            logger.info(f"File processing finished: {file_path}")
            return result

        except asyncio.TimeoutError:
            logger.error(f"Processing timeout for: {file_path}")
            return {
                "success": False,
                "error": "Processing timeout (30 seconds)",
                "error_type": "TimeoutError",
                "file_path": file_path,
            }

        except DocumentProcessorError as e:
            logger.error(f"Document processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "file_path": file_path,
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                "success": False,
                "error": f"Unexpected Error: {str(e)}",
                "error_type": "UnexpectedError",
                "file_path": file_path,
            }

    def get_server_info(self) -> Dict[str, Any]:
        """Return basic server information."""
        return {
            "name": "Universal File Reader",
            "version": "1.0.0",
            "description": "MCP server that extracts text, tables and graphics from PDFs, CSVs and images",
            "processor_info": self.factory.get_processor_info(),
        }


# MCP server instance
mcp_server = UniversalFileReaderMCP()


@app.list_tools()
async def list_tools() -> List[Tool]:
    """Return the list of available tools."""
    return [
        Tool(
            name="read_file",
            description="Extracts text, tables, and graphics from PDF, CSV, and image files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to process",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["markdown", "html", "structured"],
                        "default": "markdown",
                        "description": "Output format",
                    },
                    "force_processor": {
                        "type": "string",
                        "enum": ["csv", "pdf", "ocr"],
                        "description": "Force use of specific processor (optional)",
                    },
                    "user_language": {
                        "type": "string",
                        "default": "auto",
                        "description": "Language setting for OCR processing (ko, en, ja, zh, auto, etc.)",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="get_supported_formats",
            description="Returns supported file formats and processor information.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="validate_file",
            description="Validate whether a file can be processed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to validate",
                    },
                    "processor_type": {
                        "type": "string",
                        "enum": ["csv", "pdf", "ocr", "auto"],
                        "default": "auto",
                        "description": "Processor type",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool invocation."""
    try:
        if name == "read_file":
            result = await mcp_server.read_file(**arguments)

            if result.get("success", False):
                content = result.get("content", "")

                # Include processing metadata
                metadata = {
                    "processor": result.get("processor", "unknown"),
                    "file_info": result.get("file_info", {}),
                    "processing_time": result.get("processing_time", 0),
                }

                response_text = f"{content}\n\n---\n\n**Processing Information:**\n```json\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n```"

                return [TextContent(type="text", text=response_text)]
            else:
                error_info = {
                    "error": result.get("error", "Unknown error"),
                    "error_type": result.get("error_type", "Unknown"),
                    "file_path": result.get("file_path", ""),
                }
                error_text = f"**File Processing Failed**\n\n```json\n{json.dumps(error_info, ensure_ascii=False, indent=2)}\n```"
                return [TextContent(type="text", text=error_text)]

        elif name == "get_supported_formats":
            server_info = mcp_server.get_server_info()
            info_text = f"""# Universal File Reader Information

## Supported File Formats
{', '.join(server_info['processor_info']['supported_extensions'])}

## Processor Types
- **CSV**: CSV, TSV file processing (structure analysis, statistics, chunking)
- **PDF**: PDF native text extraction (with metadata)
- **OCR**: OCR processing for images and scanned documents (multi-element detection)

## Configuration
```json
{json.dumps(server_info['processor_info']['config'], ensure_ascii=False, indent=2)}
```
"""
            return [TextContent(type="text", text=info_text)]

        elif name == "validate_file":
            file_path = arguments["file_path"]
            processor_type = arguments.get("processor_type", "auto")

            from .core.validators import FileValidator, SecurityValidator

            # Basic validation
            basic_result = FileValidator.validate_file_basic(file_path)
            security_result = SecurityValidator.validate_security(file_path)

            # Processor specific validation (when not auto)
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
                    basic_result.get("is_valid", False)
                    and security_result.get("is_safe", False)
                    and (
                        processor_result is None
                        or processor_result.get("is_valid", False)
                    )
                ),
            }

            validation_text = f"""# File validation result

## Overall Validation Result
{'✅ Can be processed' if validation_info['overall_valid'] else '❌ Cannot be processed'}

## Detailed Information
```json
{json.dumps(validation_info, ensure_ascii=False, indent=2)}
```
"""
            return [TextContent(type="text", text=validation_text)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool call error: {e}")
        error_text = f"**Tool Call Error**\n\nError: {str(e)}\nTool: {name}\nArguments: {json.dumps(arguments, ensure_ascii=False, indent=2)}"
        return [TextContent(type="text", text=error_text)]


async def _main() -> None:
    """Asynchronous entry point for starting the MCP server."""
    logger.info("Universal File Reader MCP server starting")

    await mcp_server.initialize()

    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


def main() -> None:
    """Run the asynchronous server using ``asyncio.run``."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
