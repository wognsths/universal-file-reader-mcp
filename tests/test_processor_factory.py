import sys
import types
import pytest

# Stub mcp package modules required by package imports
mcp = types.ModuleType('mcp')
server_mod = types.ModuleType('mcp.server')
stdio_mod = types.ModuleType('mcp.server.stdio')
types_mod = types.ModuleType('mcp.types')
class DummyServer:
    def __init__(self, *args, **kwargs):
        pass

    def list_tools(self):
        def decorator(func):
            return func
        return decorator

    def call_tool(self):
        def decorator(func):
            return func
        return decorator

# Provide dummy API objects for mcp.types
for name in ['Tool', 'TextContent', 'ImageContent', 'EmbeddedResource']:
    setattr(types_mod, name, object)


def dummy_stdio_server():
    class DummyContext:
        async def __aenter__(self):
            return None
        async def __aexit__(self, exc_type, exc, tb):
            pass
    return DummyContext()

server_mod.Server = DummyServer
server_mod.stdio = stdio_mod
stdio_mod.stdio_server = dummy_stdio_server
mcp.server = server_mod
mcp.types = types_mod
sys.modules.setdefault('mcp', mcp)
sys.modules.setdefault('mcp.server', server_mod)
sys.modules.setdefault('mcp.server.stdio', stdio_mod)
sys.modules.setdefault('mcp.types', types_mod)

from src.document_reader import processor_factory
from src.document_reader.processors import csv_processor, ocr_processor, pdf_processor


@pytest.fixture(autouse=True)
def mock_validators(monkeypatch):
    """Return valid results for basic and security validation."""
    from src.document_reader.core import validators

    monkeypatch.setattr(
        validators.FileValidator,
        'validate_file_basic',
        lambda path: {"is_valid": True, "errors": [], "warnings": [], "file_info": {"is_readable": True}},
    )
    monkeypatch.setattr(
        validators.SecurityValidator,
        'validate_security',
        lambda path: {"is_safe": True, "security_warnings": []},
    )

    class DummyOCRProcessor:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(ocr_processor, 'OCRProcessor', DummyOCRProcessor)
    monkeypatch.setattr(processor_factory, 'OCRProcessor', DummyOCRProcessor)


def test_get_processor_csv(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b\n1,2")
    factory = processor_factory.ProcessorFactory()
    processor = factory.get_processor(str(file_path))
    assert isinstance(processor, csv_processor.CSVProcessor)


def test_get_processor_image(tmp_path):
    file_path = tmp_path / "img.png"
    file_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    factory = processor_factory.ProcessorFactory()
    processor = factory.get_processor(str(file_path))
    assert isinstance(processor, ocr_processor.OCRProcessor)


def test_get_processor_pdf_with_patch(tmp_path, monkeypatch):
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")
    factory = processor_factory.ProcessorFactory()
    monkeypatch.setattr(factory, "_choose_pdf_processor", lambda p: "pdf")
    processor = factory.get_processor(str(file_path))
    assert isinstance(processor, pdf_processor.PDFProcessor)


def test_process_file_pdf_fallback(tmp_path, monkeypatch):
    """PDF 프로세서 실패 시 OCR로 폴백되는지 확인"""
    file_path = tmp_path / "fail.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    class DummyPDFProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, *args, **kwargs):
            return {"success": False, "error": "fail"}

    DummyPDFProcessor.__name__ = "PDFProcessor"

    class DummyOCRProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, *args, **kwargs):
            return {"success": True, "content": "ocr", "processor": "OCRProcessor"}

    DummyOCRProcessor.__name__ = "OCRProcessor"

    factory = processor_factory.ProcessorFactory()
    monkeypatch.setattr(factory, "_choose_pdf_processor", lambda p: "pdf")
    monkeypatch.setattr(processor_factory, "PDFProcessor", DummyPDFProcessor)
    monkeypatch.setattr(processor_factory, "OCRProcessor", DummyOCRProcessor)
    monkeypatch.setattr(
        processor_factory.FileValidator,
        "validate_for_processor",
        lambda *a, **k: {"is_valid": True, "errors": [], "warnings": [], "file_info": {"size_mb": 1}},
    )
    factory._processor_types["pdf"] = DummyPDFProcessor
    factory._processor_types["ocr"] = DummyOCRProcessor

    result = factory.process_file(str(file_path))
    assert result["success"] is True
    assert result.get("fallback_to_ocr") is True
