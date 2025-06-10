# Universal File Reader MCP

Universal File Reader is an MCP server for extracting text and structural information from PDF, CSV and image files. The server automatically selects the appropriate processor and can fall back to OCR when needed.

## Installation

```bash
pip install -e .
```

## Running the server

```bash
universal-file-reader
```

The server exposes three tools: `read_file`, `get_supported_formats` and `validate_file`. See `src/document_reader/mcp_server.py` for detailed schemas.

## Environment variables

- `GOOGLE_API_KEY` – API key used for Gemini based OCR processing.

## Running tests

Install dependencies and run the test suite:

```bash
pip install -r requirements.txt
pytest
```

## Development

```bash
pip install -e .[development]
ruff check
```
