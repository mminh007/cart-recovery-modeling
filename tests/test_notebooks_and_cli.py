"""Verify all Jupyter notebooks (.ipynb) and CLI entrypoints work cleanly."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


def test_all_notebooks_valid_json_and_python_syntax() -> None:
    nb_dir = Path("notebooks")
    notebooks = sorted(nb_dir.glob("*.ipynb"))
    assert len(notebooks) >= 3

    for nb_path in notebooks:
        payload = json.loads(nb_path.read_text(encoding="utf-8"))
        assert payload.get("nbformat") == 4
        cells = payload.get("cells", [])
        assert len(cells) >= 5

        for idx, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            raw_lines = cell.get("source", [])
            cleaned = "\n".join(
                line for line in ("".join(raw_lines)).splitlines()
                if not line.lstrip().startswith(("!", "%"))
            )
            ast.parse(cleaned, filename=f"{nb_path.name}#cell-{idx}")


def test_cli_pipeline_entrypoints_subprocess(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    ml_dir = tmp_path / "processed" / "ml"
    llm_dir = tmp_path / "processed" / "llm"
    report_path = tmp_path / "processed" / "quality_report.json"

    subprocess.run(
        [sys.executable, "scripts/download_datasets.py", "--synthetic-demo", "--raw-dir", str(raw_dir)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data_prep.quality_filter",
            "--data-root",
            str(raw_dir),
            "--report-path",
            str(report_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data_prep.feature_engineering",
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(ml_dir),
            "--llm-output-dir",
            str(llm_dir),
            "--schema-path",
            "configs/feature_schema.yaml",
        ],
        check=True,
    )

    assert report_path.exists()
    assert (ml_dir / "intent_train.parquet").exists()
    assert (ml_dir / "uplift_train.parquet").exists()
    assert (llm_dir / "vietnamese_catalog.jsonl").exists()
