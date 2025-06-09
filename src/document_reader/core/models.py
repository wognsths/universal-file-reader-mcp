from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ColumnInfo(BaseModel):
    """Information about a CSV column"""

    name: str = Field(description="Column name")
    dtype: str = Field(description="Data type")
    non_null_count: int = Field(ge=0, description="Non‑null values count")
    null_count: int = Field(ge=0, description="Null values count")
    unique_count: int = Field(ge=0, description="Unique values count")
    sample_values: List[str] = Field(default_factory=list, description="Sample values (≤10)")

    # cap length defensively
    @field_validator("sample_values")
    @classmethod
    def _cap_samples(cls, v: List[str]) -> List[str]:  # noqa: D401, N805
        return v[:10]


class CSVAnalysis(BaseModel):
    """Light‑weight schema + stats of the CSV file"""

    file_path: str
    file_size_mb: float = Field(ge=0)
    encoding: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    column_info: List[ColumnInfo]
    data_types_summary: Dict[str, int]
    memory_usage_mb: float = Field(ge=0)
    has_header: bool
    delimiter: str
    processing_time: float = Field(ge=0)

    @field_validator("delimiter")
    @classmethod
    def _warn_uncommon_delimiter(cls, v: str) -> str:  # noqa: D401, N805
        if v not in {",", ";", "\t", "|", ":"}:
            logger.warning("Uncommon delimiter detected: '%s'", v)
        return v


class CSVChunk(BaseModel):
    chunk_id: int = Field(ge=0)
    start_row: int = Field(ge=0)
    end_row: int = Field(ge=0)
    row_count: int = Field(ge=0)
    preview_data: str


class CSVResult(BaseModel):
    analysis: CSVAnalysis
    preview_data: str
    chunks: List[CSVChunk] = Field(default_factory=list)
    summary_stats: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)