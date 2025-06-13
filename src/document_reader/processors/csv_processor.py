"""CSV processor with chunking, size limits, and structured output
Fixed version – addresses formatting bugs, HTML generation, and minor robustness tweaks."""

import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_processor import BaseProcessor
from ..core.utils import with_timeout

from ..core.exceptions import (
    CSVError,
    FileSizeError,
)
from ..core.models import (
    CSVAnalysis,
    CSVChunk,
    CSVResult,
    ColumnInfo,
)


logger = logging.getLogger(__name__)


@dataclass
class CSVConfig:
    """CSV processor configuration"""

    MAX_FILE_SIZE_MB: int = 30  # Hard‑limit on file size
    CHUNK_SIZE: int = 10_000  # Rows per chunk for large files
    SAMPLE_SIZE: int = 1_000  # Rows used for structure analysis
    MAX_COLUMNS: int = 1_000  # Abort if column count exceeds this
    MAX_ROWS_PREVIEW: int = 100  # Rows shown in preview output
    TIMEOUT_SECONDS: int = 30  # Per‑op timeout (future‑use)
    SUPPORTED_ENCODINGS: List[str] | None = None

    def __post_init__(self) -> None:  # noqa: D401
        if self.SUPPORTED_ENCODINGS is None:
            self.SUPPORTED_ENCODINGS = [
                "utf-8",
                "utf-8-sig",
                "latin1",
                "cp949",
                "euc-kr",
            ]

# ───────────────────────── Processor implementation ───────────────────────── #
class CSVProcessor(BaseProcessor):
    """CSV file processor with size limits and chunking"""

    _supported_extensions = {".csv", ".tsv"}

    def __init__(self, config: Optional[CSVConfig] = None):
        super().__init__()
        self.config = config or CSVConfig()
        # ThreadPool keeps CPU‑bound pandas work off the event‑loop
        self._executor = ThreadPoolExecutor(max_workers=2)

    # ——— Public API ——— #
    def supports(self, file_extension: str) -> bool:  # noqa: D401
        return file_extension.lower() in self._supported_extensions

    def get_supported_extensions(self) -> List[str]:  # noqa: D401
        return list(self._supported_extensions)

    @with_timeout(30)
    def process(
        self, file_path: str, output_format: str = "markdown", **kwargs
    ) -> Dict[str, Any]:  # noqa: D401
        """Process a CSV file.

        Extra keyword arguments are accepted for API compatibility but ignored.
        """

        start_time = time.time()

        try:
            if not self._validate_file(file_path):
                raise CSVError("File validation failed")

            result: CSVResult = self._process_csv(file_path)
            result.analysis.processing_time = time.time() - start_time

            # Render
            if output_format == "structured":
                formatted_output: Any = result.model_dump()
            elif output_format == "markdown":
                formatted_output = self._format_as_markdown(result, file_path)
            else:
                formatted_output = self._format_as_html(result, file_path)

            return self._create_success_response(
                formatted_output,
                file_path,
                processing_method="CSV analysis with chunking",
                total_rows=result.analysis.total_rows,
                total_columns=result.analysis.total_columns,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("CSV processing failed: %s", exc)
            return self._create_error_response(str(exc), file_path)

    # ——— Internal helpers ——— #
    def _validate_file(self, file_path: str) -> bool:  # noqa: D401
        path = Path(file_path)

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(file_path)

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.MAX_FILE_SIZE_MB:
            raise FileSizeError(f"File too large: {size_mb:.1f} MB (limit {self.config.MAX_FILE_SIZE_MB} MB)")

        if path.suffix.lower() not in self._supported_extensions:
            raise CSVError(f"Unsupported extension: {path.suffix}")

        return True

    def _process_csv(self, file_path: str) -> CSVResult:  # noqa: D401
        """Process CSV file synchronously."""
        encoding = self._detect_encoding(file_path)
        analysis = self._analyze_csv_structure(file_path, encoding)

        if analysis.file_size_mb > 10:
            return self._process_large_csv(file_path, encoding, analysis)
        return self._process_small_csv(file_path, encoding, analysis)

    # ——— Encoding detection ——— #
    def _detect_encoding(self, file_path: str) -> str:  # noqa: D401
        import chardet

        def _detect() -> str | None:
            with open(file_path, "rb") as fp:
                sample = fp.read(10_240)
                result = chardet.detect(sample)
                return result.get("encoding") if result else None

        future = self._executor.submit(_detect)
        encoding = future.result(timeout=self.config.TIMEOUT_SECONDS)
        if encoding and encoding.lower() in self.config.SUPPORTED_ENCODINGS:
            return encoding
        logger.warning(
            "Unsupported or undetected encoding '%s'. Falling back to utf‑8.",
            encoding,
        )
        return "utf-8"

    # ——— Structure analysis ——— #
    def _analyze_csv_structure(self, file_path: str, encoding: str) -> CSVAnalysis:  # noqa: D401
        def _analyze() -> CSVAnalysis:  # noqa: D401
            sample_df = pd.read_csv(
                file_path,
                encoding=encoding,
                nrows=self.config.SAMPLE_SIZE,
                low_memory=False,
            )

            with open(file_path, "r", encoding=encoding) as fp:
                first_line = fp.readline()
                total_rows = sum(1 for _ in fp)
            delimiter = self._detect_delimiter(first_line)

            column_info: List[ColumnInfo] = []
            type_summary: Dict[str, int] = {}
            for col in sample_df.columns[: self.config.MAX_COLUMNS]:
                col_data = sample_df[col]
                dtype_str = str(col_data.dtype)
                type_summary[dtype_str] = type_summary.get(dtype_str, 0) + 1
                column_info.append(
                    ColumnInfo(
                        name=col,
                        dtype=dtype_str,
                        non_null_count=int(col_data.count()),
                        null_count=int(col_data.isnull().sum()),
                        unique_count=int(col_data.nunique()),
                        sample_values=col_data.dropna().astype(str).unique()[:5].tolist(),
                    )
                )

            mem_mb = sample_df.memory_usage(deep=True).sum() / (1024 * 1024)

            return CSVAnalysis(
                file_path=file_path,
                file_size_mb=Path(file_path).stat().st_size / (1024 * 1024),
                encoding=encoding,
                total_rows=total_rows,
                total_columns=len(sample_df.columns),
                column_info=column_info,
                data_types_summary=type_summary,
                memory_usage_mb=mem_mb,
                has_header=True,
                delimiter=delimiter,
                processing_time=0.0,
            )

        future = self._executor.submit(_analyze)
        return future.result(timeout=self.config.TIMEOUT_SECONDS)

    # ——— Chunk helpers ——— #
    def _detect_delimiter(self, first_line: str) -> str:  # noqa: D401
        delimiters = [",", ";", "\t", "|", ":"]
        counts = {d: first_line.count(d) for d in delimiters}
        best = max(counts, key=counts.get)
        return best if counts[best] else ","

    def _process_small_csv(self, file_path: str, encoding: str, analysis: CSVAnalysis) -> CSVResult:  # noqa: D401
        def _process() -> CSVResult:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            preview_df = df.head(self.config.MAX_ROWS_PREVIEW)
            numeric_cols = df.select_dtypes(include=[np.number])
            summary_stats = (
                numeric_cols.describe().to_dict() if not numeric_cols.empty else {}
            )
            return CSVResult(
                analysis=analysis,
                preview_data=preview_df.to_markdown(index=False),
                summary_stats=summary_stats,
            )

        future = self._executor.submit(_process)
        return future.result(timeout=self.config.TIMEOUT_SECONDS)

    def _process_large_csv(self, file_path: str, encoding: str, analysis: CSVAnalysis) -> CSVResult:  # noqa: D401
        def _process() -> CSVResult:
            chunks: List[CSVChunk] = []
            for chunk_id, chunk_df in enumerate(
                pd.read_csv(
                    file_path,
                    encoding=encoding,
                    chunksize=self.config.CHUNK_SIZE,
                    low_memory=False,
                )
            ):
                start_row = chunk_id * self.config.CHUNK_SIZE + 1  # header counted separately
                end_row = start_row + len(chunk_df) - 1
                chunks.append(
                    CSVChunk(
                        chunk_id=chunk_id,
                        start_row=start_row,
                        end_row=end_row,
                        row_count=len(chunk_df),
                        preview_data=chunk_df.head(10).to_markdown(index=False),
                    )
                )
                if chunk_id >= 9:  # Stop after 10 chunks (0‑indexed)
                    remaining = analysis.total_rows - end_row
                    if remaining > 0:
                        chunks.append(
                            CSVChunk(
                                chunk_id=chunk_id + 1,
                                start_row=end_row + 1,
                                end_row=analysis.total_rows,
                                row_count=remaining,
                                preview_data="… (remaining data not processed due to size limit)",
                            )
                        )
                    break

            preview_df = pd.read_csv(
                file_path,
                encoding=encoding,
                nrows=self.config.MAX_ROWS_PREVIEW,
                low_memory=False,
            )
            warnings = [
                f"Large file detected ({analysis.file_size_mb:.1f} MB) — processed in {len(chunks)} chunk(s).",
                "Only preview and partial chunk information is shown to avoid memory issues.",
            ]
            return CSVResult(
                analysis=analysis,
                preview_data=preview_df.to_markdown(index=False),
                chunks=chunks,
                warnings=warnings,
            )

        future = self._executor.submit(_process)
        return future.result(timeout=self.config.TIMEOUT_SECONDS)

    # ——— Output formatting ——— #
    def _format_as_markdown(self, result: CSVResult, file_path: str) -> str:  # noqa: D401
        analysis = result.analysis
        parts: List[str] = [
            f"# CSV Analysis Result: {Path(file_path).name}\n",
            "## File Information",
            f"- **File Size**: {analysis.file_size_mb:.2f} MB",
            f"- **Encoding**: {analysis.encoding}",
            f"- **Total Rows**: {analysis.total_rows:,}",
            f"- **Total Columns**: {analysis.total_columns}",
            f"- **Delimiter**: `{analysis.delimiter}`",
            f"- **Processing Time**: {analysis.processing_time:.2f}s\n",
            "## Column Information",
        ]
        for col in analysis.column_info:
            parts += [
                f"### {col.name}",
                f"- **Type**: {col.dtype}",
                f"- **Non‑null**: {col.non_null_count:,} | **Null**: {col.null_count:,}",
                f"- **Unique values**: {col.unique_count:,}",
                f"- **Sample values**: {', '.join(col.sample_values[:3])}\n",
            ]

        type_summary = ", ".join(f"{t}: {c}" for t, c in analysis.data_types_summary.items()) or "(none)"
        parts += ["## Data Types Summary", type_summary, "", "## Data Preview", result.preview_data]

        if result.chunks:
            parts += [f"\n## Chunk Information (first {min(5, len(result.chunks))} shown)"]
            for chunk in result.chunks[:5]:
                parts += [
                    f"### Chunk {chunk.chunk_id + 1}",
                    f"- **Rows**: {chunk.start_row:,}‑{chunk.end_row:,} ({chunk.row_count:,} rows)",
                ]

        if result.warnings:
            parts += ["## Warnings", *[f"- {w}" for w in result.warnings]]

        if result.summary_stats:
            parts += ["## Summary Statistics", "```json", json.dumps(result.summary_stats, indent=2, ensure_ascii=False), "```"]

        return "\n".join(parts)

    def _format_as_html(self, result: CSVResult, file_path: str) -> str:  # noqa: D401
        analysis = result.analysis
        html_parts: List[str] = [
            '<div class="csv-analysis-result">',
            f'<h2>CSV Analysis: {Path(file_path).name}</h2>',
            '<div class="file-info">',
            '<h3>File Information</h3>',
            '<ul>',
            f'<li><strong>File Size:</strong> {analysis.file_size_mb:.2f} MB</li>',
            f'<li><strong>Encoding:</strong> {analysis.encoding}</li>',
            f'<li><strong>Total Rows:</strong> {analysis.total_rows:,}</li>',
            f'<li><strong>Total Columns:</strong> {analysis.total_columns}</li>',
            f'<li><strong>Delimiter:</strong> <code>{analysis.delimiter}</code></li>',
            f'<li><strong>Processing Time:</strong> {analysis.processing_time:.2f}s</li>',
            '</ul>',
            '</div>',  # file-info
            '<div class="column-info">',
            '<h3>Column Information</h3>',
        ]
        for col in analysis.column_info:
            html_parts += [
                '<div class="column">',
                f'<h4>{col.name}</h4>',
                f'<p><strong>Type:</strong> {col.dtype}</p>',
                f'<p><strong>Non-null:</strong> {col.non_null_count:,} | <strong>Null:</strong> {col.null_count:,}</p>',
                f'<p><strong>Unique values:</strong> {col.unique_count:,}</p>',
                f'<p><strong>Sample values:</strong> {", ".join(col.sample_values[:3])}</p>',
                '</div>',
            ]
        html_parts += ['</div>']  # column-info

        html_parts += [
            '<div class="data-preview">',
            '<h3>Data Preview</h3>',
            f'<pre>{result.preview_data}</pre>',
            '</div>',
        ]

        if result.warnings:
            html_parts += ['<div class="warnings">', '<h3>Warnings</h3>', '<ul>']
            html_parts += [f'<li>{w}</li>' for w in result.warnings]
            html_parts += ['</ul>', '</div>']

        html_parts += ['</div>']  # csv-analysis-result
        return "\n".join(html_parts)
