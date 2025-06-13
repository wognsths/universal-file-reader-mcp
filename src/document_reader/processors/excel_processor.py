from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from .csv_processor import CSVProcessor, CSVConfig
from ..core.models import ColumnInfo, CSVAnalysis, CSVResult
from ..core.exceptions import CSVError, FileSizeError
from ..core.utils import with_timeout

logger = logging.getLogger(__name__)


class ExcelProcessor(CSVProcessor):
    """Processor for Excel files using pandas."""

    _supported_extensions = {".xlsx", ".xls"}

    def __init__(self, config: Optional[CSVConfig] = None) -> None:
        super().__init__(config)

    def supports(self, file_extension: str) -> bool:  # noqa: D401
        return file_extension.lower() in self._supported_extensions

    def get_supported_extensions(self) -> List[str]:  # noqa: D401
        return list(self._supported_extensions)

    def _validate_file(self, file_path: str) -> bool:  # noqa: D401
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(file_path)

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.MAX_FILE_SIZE_MB:
            raise FileSizeError(
                f"File too large: {size_mb:.1f} MB (limit {self.config.MAX_FILE_SIZE_MB} MB)"
            )

        if path.suffix.lower() not in self._supported_extensions:
            raise CSVError(f"Unsupported extension: {path.suffix}")

        return True

    @with_timeout(30)
    def process(
        self, file_path: str, output_format: str = "markdown", **kwargs: Any
    ) -> Dict[str, Any]:
        """Process an Excel file and return results."""
        import time

        start_time = time.time()
        try:
            if not self._validate_file(file_path):
                raise CSVError("File validation failed")

            df = pd.read_excel(file_path)
            preview_df = df.head(self.config.MAX_ROWS_PREVIEW)
            column_info: List[ColumnInfo] = []
            type_summary: Dict[str, int] = {}
            for col in df.columns[: self.config.MAX_COLUMNS]:
                col_data = df[col]
                dtype_str = str(col_data.dtype)
                type_summary[dtype_str] = type_summary.get(dtype_str, 0) + 1
                column_info.append(
                    ColumnInfo(
                        name=col,
                        dtype=dtype_str,
                        non_null_count=int(col_data.count()),
                        null_count=int(col_data.isnull().sum()),
                        unique_count=int(col_data.nunique()),
                        sample_values=col_data.dropna()
                        .astype(str)
                        .unique()[:5]
                        .tolist(),
                    )
                )

            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            analysis = CSVAnalysis(
                file_path=file_path,
                file_size_mb=Path(file_path).stat().st_size / (1024 * 1024),
                encoding="binary",
                total_rows=len(df),
                total_columns=len(df.columns),
                column_info=column_info,
                data_types_summary=type_summary,
                memory_usage_mb=mem_mb,
                has_header=True,
                delimiter=",",
                processing_time=time.time() - start_time,
            )

            numeric_cols = df.select_dtypes(include=[np.number])
            summary_stats = (
                numeric_cols.describe().to_dict() if not numeric_cols.empty else {}
            )
            result = CSVResult(
                analysis=analysis,
                preview_data=preview_df.to_markdown(index=False),
                summary_stats=summary_stats,
            )

            if output_format == "structured":
                formatted_output: Any = result.model_dump()
            elif output_format == "markdown":
                formatted_output = self._format_as_markdown(result, file_path)
            else:
                formatted_output = self._format_as_html(result, file_path)

            return self._create_success_response(
                formatted_output,
                file_path,
                processing_method="Excel analysis",
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Excel processing failed: %s", exc)
            return self._create_error_response(str(exc), file_path)
