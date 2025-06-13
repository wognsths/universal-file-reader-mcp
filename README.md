# Universal File Reader MCP

<p align="right">

  <a href="README_KO.md">Korean</a>

</p>

Universal File Reader is an MCP server for extracting text and structural information from PDF, CSV and image files. The server automatically selects the appropriate processor and can fall back to OCR when needed.

## Installation

```bash
pip install -e .
```

## Running the server

```bash
universal-file-reader
```

### Running over SSH

You can also run the server on a remote machine via SSH:

```bash
ssh user@remote-host universal-file-reader
```

In Google’s ADK this command can be used with `ToolSubprocess` to
communicate with the remote server.

The server exposes three tools: `read_file`, `get_supported_formats` and `validate_file`. See `src/document_reader/mcp_server.py` for detailed schemas.

## Running the API server

You can also start a REST API that wraps the same functionality. The service exposes two endpoints:

- `POST /mcp` – accept MCP JSON messages
- `POST /upload` – simple file upload

Start the server locally:

```bash
universal-file-reader-api
```

Using Docker Compose:

```bash
docker-compose up
```

The API will be available on <http://localhost:8000>.

### MCP message format

`POST /mcp` expects a JSON body with the following structure:

```json
{
  "tool": "read_file",
  "arguments": {
    "file_path": "/path/to/file.pdf",
    "output_format": "markdown"
  }
}
```

The `tool` field corresponds to one of the MCP tools (`read_file`,
`get_supported_formats`, or `validate_file`). `arguments` contains the
parameters for that tool.

## Environment variables

- `GOOGLE_API_KEY` – API key used for Gemini based OCR processing.
- `MAX_PAGE_PER_PROCESS` – Maximum number of PDF pages processed in one OCR batch.
- `TIMEOUT_SECONDS` – Processing timeout in seconds for PDF tasks.

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
