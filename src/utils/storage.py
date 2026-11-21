"""Cloud-ready storage & tabular I/O abstraction (Local Phase 1 -> S3/GCS Phase 2)."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StorageConfig:
    """Unified storage configuration driven by environment variables."""

    backend: str = "local"  # "local" | "s3" | "gcs"
    data_root: str = "./data"
    s3_bucket_uri: str = ""
    gcs_bucket_uri: str = ""

    def resolve_uri(self, relative_path: str) -> str:
        """Resolve a dataset relative path (e.g. 'processed/ml/intent_train.parquet') to URI."""
        clean_rel = relative_path.lstrip("/\\").replace("\\", "/")
        if self.backend == "s3" and self.s3_bucket_uri:
            return f"{self.s3_bucket_uri.rstrip('/')}/{clean_rel}"
        if self.backend == "gcs" and self.gcs_bucket_uri:
            return f"{self.gcs_bucket_uri.rstrip('/')}/{clean_rel}"
        return str(Path(self.data_root) / clean_rel)


def get_storage_config() -> StorageConfig:
    """Build StorageConfig from current environment variables."""
    return StorageConfig(
        backend=os.getenv("STORAGE_BACKEND", "local").lower(),
        data_root=os.getenv("DATA_ROOT", "./data"),
        s3_bucket_uri=os.getenv("S3_BUCKET_URI", ""),
        gcs_bucket_uri=os.getenv("GCS_BUCKET_URI", ""),
    )


def write_tabular_records(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write tabular records to Parquet (when pandas+pyarrow available) with JSONL sidecar fallback.

    In Cloud/Docker/Colab environments with pyarrow installed, writes native Apache Parquet.
    In lightweight local environments without pyarrow, writes structured JSON-lines records
    into the target file (with magic header marker) so downstream readers & tests remain lossless.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd  # type: ignore[import-untyped]

        df = pd.DataFrame(records)
        df.to_parquet(target, index=False)
        return target
    except ImportError:
        with target.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("# FORMAT:JSONL_FALLBACK\n")
            for row in records:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return target


def read_tabular_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Read tabular records from Parquet (via pandas) or JSONL fallback / CSV."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Tabular file not found: {source}")

    with source.open("rb") as fb:
        head = fb.read(24)

    if head.startswith(b"# FORMAT:JSONL_FALLBACK"):
        rows: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if idx == 0:
                    continue
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    import pandas as pd  # type: ignore[import-untyped]

    df = pd.read_parquet(source)
    return df.to_dict(orient="records")
