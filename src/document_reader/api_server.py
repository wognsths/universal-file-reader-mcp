from fastapi import FastAPI, UploadFile, File
from typing import Any, Dict
import os
import uvicorn
import logging

app = FastAPI()


@app.post("/mcp")
async def receive_mcp(message: Dict[str, Any]):
    """Receive MCP JSON messages."""
    logging.info("Received MCP message: %s", message)
    return {"status": "received"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Simple file upload endpoint."""
    contents = await file.read()  # noqa: WPS110
    logging.info("Uploaded file %s with %d bytes", file.filename, len(contents))
    return {"filename": file.filename}


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("document_reader.api_server:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
