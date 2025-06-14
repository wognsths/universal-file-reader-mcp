# ruff: noqa: E402

import sys
import types
import pytest

# Stub mcp package modules required by package imports
mcp = types.ModuleType("mcp")
server_mod = types.ModuleType("mcp.server")
stdio_mod = types.ModuleType("mcp.server.stdio")
types_mod = types.ModuleType("mcp.types")


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
for name in ["Tool", "TextContent", "ImageContent", "EmbeddedResource"]:
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
sys.modules.setdefault("mcp", mcp)
sys.modules.setdefault("mcp.server", server_mod)
sys.modules.setdefault("mcp.server.stdio", stdio_mod)
sys.modules.setdefault("mcp.types", types_mod)

from src.document_reader import processor_factory
from src.document_reader.processors import csv_processor, ocr_processor, pdf_processor


@pytest.fixture(autouse=True)
def mock_validators(monkeypatch):
    """Return valid results for basic and security validation."""
    from src.document_reader.core import validators

    monkeypatch.setattr(
        validators.FileValidator,
        "validate_file_basic",
        lambda path: {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {"is_readable": True},
        },
    )
    monkeypatch.setattr(
        validators.SecurityValidator,
        "validate_security",
        lambda path: {"is_safe": True, "security_warnings": []},
    )

    class DummyOCRProcessor:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(ocr_processor, "OCRProcessor", DummyOCRProcessor)
    monkeypatch.setattr(processor_factory, "OCRProcessor", DummyOCRProcessor)
    monkeypatch.setattr(pdf_processor, "OCRProcessor", DummyOCRProcessor)


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
    """Verify OCR fallback when PDF processor fails"""
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
        lambda *a, **k: {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {"size_mb": 1},
        },
    )
    factory._processor_types["pdf"] = DummyPDFProcessor
    factory._processor_types["ocr"] = DummyOCRProcessor

    result = factory.process_file(str(file_path))
    assert result["success"] is True
    assert result.get("fallback_to_ocr") is True


def test_choose_pdf_processor_average(monkeypatch, tmp_path):
    file_path = tmp_path / "math.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    class DummyPage:
        def __init__(self, text):
            self._text = text

        def get_text(self):
            return self._text

    class DummyDoc:
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return DummyPage(self.texts[idx])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    dummy_doc = DummyDoc(["few", "word " * 15, "text " * 20])

    factory = processor_factory.ProcessorFactory()
    monkeypatch.setattr(factory, "_create_processor", lambda t: None)
    monkeypatch.setattr(
        processor_factory, "fitz", types.SimpleNamespace(open=lambda p: dummy_doc)
    )

    assert factory._choose_pdf_processor(str(file_path)) == "pdf"


def test_process_file_pdf_timeout(tmp_path, monkeypatch):
    file_path = tmp_path / "slow.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    class SlowPDFProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, *args, **kwargs):
            import time

            time.sleep(2)
            return {"success": True, "content": "pdf", "word_count": 20}

    SlowPDFProcessor.__name__ = "PDFProcessor"

    class DummyOCRProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, *args, **kwargs):
            return {"success": True, "content": "ocr"}

    DummyOCRProcessor.__name__ = "OCRProcessor"

    config = processor_factory.ProcessorConfig()
    config.global_config.TIMEOUT_SECONDS = 1
    factory = processor_factory.ProcessorFactory(config)

    monkeypatch.setattr(factory, "_choose_pdf_processor", lambda p: "pdf")
    monkeypatch.setattr(processor_factory, "PDFProcessor", SlowPDFProcessor)
    monkeypatch.setattr(processor_factory, "OCRProcessor", DummyOCRProcessor)
    monkeypatch.setattr(
        processor_factory.FileValidator,
        "validate_for_processor",
        lambda *a, **k: {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {"size_mb": 1},
        },
    )

    factory._processor_types["pdf"] = SlowPDFProcessor
    factory._processor_types["ocr"] = DummyOCRProcessor

    result = factory.process_file(str(file_path))
    assert result["success"] is True
    assert result.get("fallback_to_ocr") is True


def test_pdf_processor_extract_images(monkeypatch, tmp_path):
    file_path = tmp_path / "img.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    class DummyPDFProcessor(pdf_processor.PDFProcessor):
        def __init__(self, *a, **k):
            config = processor_factory.ProcessorConfig().pdf_config
            config.EXTRACT_IMAGES = True
            config.USE_OCR_FALLBACK = False
            super().__init__(config)

        def _validate_pdf_file(self, file_path):
            return None

        def _extract_text_safe(self, file_path):
            return {
                "success": True,
                "text_content": "text",
                "page_count": 1,
                "word_count": 20,
                "warnings": [],
                "metadata": {},
            }

        def _extract_images_with_ocr(self, file_path, output_format):
            return [{"page": 1, "index": 1, "text": "img-text"}]

    DummyPDFProcessor.__name__ = "PDFProcessor"

    factory = processor_factory.ProcessorFactory(processor_factory.ProcessorConfig())
    monkeypatch.setattr(factory, "_choose_pdf_processor", lambda p: "pdf")

    monkeypatch.setattr(processor_factory, "PDFProcessor", DummyPDFProcessor)
    monkeypatch.setattr(
        processor_factory.FileValidator,
        "validate_for_processor",
        lambda *a, **k: {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {"size_mb": 1},
        },
    )

    factory._processor_types["pdf"] = DummyPDFProcessor

    class DummyOCRProc:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, *args, **kwargs):
            return {"success": True, "content": "image"}

    monkeypatch.setattr(processor_factory, "OCRProcessor", DummyOCRProc)
    monkeypatch.setattr(pdf_processor, "OCRProcessor", DummyOCRProc)

    factory._processor_types["ocr"] = DummyOCRProc

    result = factory.process_file(str(file_path))
    assert "img-text" in result["content"]
