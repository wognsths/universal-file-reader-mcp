# Universal File Reader MCP

This project provides an MCP server capable of extracting information from PDF, CSV and image files. The server automatically selects the best processor for each file and can fall back to OCR when needed.

## Installation

```bash
pip install -e .
```

## Usage

Run the MCP server using the provided console script:

```bash
universal-file-reader
```

The server exposes tools for reading files and validating them. See `src/document_reader/mcp_server.py` for details.

## Environment

Some processors require additional configuration:

- `GOOGLE_API_KEY` – API key for Gemini based OCR.

## Development

Install development dependencies and run lint checks:

```bash
pip install -e .[development]
ruff check
```
