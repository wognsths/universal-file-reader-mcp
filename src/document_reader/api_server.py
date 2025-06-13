
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Any, Dict, List
import os
import uvicorn
import logging
from pydantic import BaseModel, Field

from .mcp_server import call_tool, mcp_server


app = FastAPI()

class MCPRequest(BaseModel):
    """Schema for MCP JSON requests."""

    tool: str = Field(..., description="Name of the MCP tool to invoke")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the selected tool",
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the underlying MCP server once."""
    if not mcp_server.initialized:
        await mcp_server.initialize()


@app.post("/mcp")
async def receive_mcp(request: MCPRequest) -> Dict[str, Any]:
    """Handle MCP JSON messages by invoking an MCP tool."""
    logging.info("Received MCP request: %s", request.model_dump())
    try:
        result = await call_tool(request.tool, request.arguments)
    except Exception as exc:  # noqa: BLE001
        logging.error("MCP processing error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"results": [item.model_dump() for item in result]}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a file and process it using the MCP server."""
    contents = await file.read()  # noqa: WPS110
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as tmp:
        tmp.write(contents)

    logging.info("Uploaded file %s with %d bytes", file.filename, len(contents))

    try:
        result = await call_tool(
            "read_file",
            {"file_path": temp_path, "output_format": "markdown"},
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return {"results": [item.model_dump() for item in result]}



def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("document_reader.api_server:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
