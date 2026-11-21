.PHONY: help setup download-demo download-kaggle process-data init-mlflow test docker-up docker-down clean

PYTHON ?= python

help:
	@echo "Cart Abandonment Modeling & Fine-tune — Commands"
	@echo "  make setup            Install core + dev dependencies"
	@echo "  make download-demo    Generate schema-accurate synthetic raw data for local testing"
	@echo "  make download-kaggle  Download full datasets from Kaggle (requires KAGGLE credentials)"
	@echo "  make process-data     Run data validation + REES46/Hillstrom/Catalog processors"
	@echo "  make init-mlflow      Initialize MLflow experiments"
	@echo "  make test             Run unit & integration test suite"
	@echo "  make docker-up        Start Airflow + MLflow + Postgres + Jupyter via Docker Compose"
	@echo "  make docker-down      Stop Docker Compose stack"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

download-demo:
	$(PYTHON) scripts/download_datasets.py --synthetic-demo

download-kaggle:
	$(PYTHON) scripts/download_datasets.py --dataset all

process-data:
	$(PYTHON) -m src.data_prep.quality_filter --data-root data/raw --report-path data/processed/quality_report.json
	$(PYTHON) -m src.data_prep.feature_engineering --raw-dir data/raw --output-dir data/processed/ml --schema-path configs/feature_schema.yaml

init-mlflow:
	$(PYTHON) -m src.utils.mlflow_config --init

test:
	$(PYTHON) -m pytest tests/

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
