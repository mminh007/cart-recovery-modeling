# 🐛 [BUG-XXX] Concise Bug Summary Title

- **Date Discovered:** YYYY-MM-DD
- **Author:** @author
- **Sprint / Task:** Sprint X — Task X.Y
- **Component:** `data_prep` | `ml_models` | `llm_finetune` | `airflow` | `mlflow` | `docker` | `serving`
- **Severity:** 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low
- **Status:** `Open` | `Investigating` | `Fixed` | `Verified`

---

## 1. 📝 Summary

Provide a concise (2–4 sentences) description of the unexpected behavior and its impact on pipeline execution, data integrity, or model metrics.

## 2. 🖥️ Environment

- **Runtime:** Local Windows / Docker Compose / Google Colab / vast.ai GPU
- **Python / Container Image:** (e.g., Python 3.11, `apache/airflow:2.8.1-python3.11`)
- **Dataset / Slice:** (e.g., `2019-Nov.csv` chunk #4, or `hillstrom.csv`)

## 3. 🔁 Steps to Reproduce

1. Run command:
   ```bash
   python -m src.data_prep.feature_engineering ...
   ```
2. Observed error / anomalous output:
   ```text
   Paste stack trace or anomalous metric output here
   ```

## 4. 🔍 Root Cause Analysis

- **Direct Trigger:** (Which specific file, function, or config line triggered the failure?)
- **Underlying Root Cause (Why did it happen?):** (e.g., unhandled nullable column, temporal leakage across observation boundary, database lock contention...)

## 5. 🛠️ Fix Applied

- Modified files:
  - `src/...`
- Explanation of the fix and why this approach was selected over alternatives.

## 6. ✅ Verification & Regression Prevention

- [ ] Added automated regression test in `tests/test_...py`
- [ ] Full `pytest` suite passes (`100%`)
- [ ] Verified compatibility with `configs/feature_schema.yaml` contract (if shared with Repo 2)
