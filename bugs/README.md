# 🐞 Bug Reports & Root Cause Analysis (`/bugs`)

This directory stores structured post-mortem **Root Cause Analysis (RCA)** documents for bugs encountered across the Data Pipeline, Tabular ML Training (Google Colab), LLM Fine-tuning (vast.ai), and MLOps Infrastructure (Airflow / MLflow / Docker).

---

## 📋 File Naming Convention

Create one Markdown file per bug using the format:

```text
BUG-<number>-<short-kebab-case-summary>.md
```

**Examples:**
- `BUG-001-rees46-chunk-memory-spike.md`
- `BUG-002-temporal-leakage-intent-labels.md`
- `BUG-003-airflow-sqlite-concurrency-lock.md`
- `BUG-004-qlora-bf16-oom-vastai.md`

---

## 🛠️ Bug Investigation & Resolution Workflow

1. **Create a report** by copying [BUG_TEMPLATE.md](./BUG_TEMPLATE.md).
2. **Reproduce deterministically:** Record the exact CLI command, input slice, environment (Local / Docker / Colab / vast.ai), and full traceback.
3. **Perform Root Cause Analysis (5 Whys):** Identify the underlying code defect, schema mismatch, or infrastructure misconfiguration rather than applying surface workarounds.
4. **Add a Regression Test:** Write an automated test in `tests/` that fails before the fix and passes after the fix.
5. **Update the Bug Index** table below.

---

## 📊 Recorded Bugs Index

| ID | Summary | Component | Severity | Status | Report |
|---|---|---|---|---|---|
| — | *No bugs recorded yet (Sprint 1 initialized)* | — | — | — | [Template](./BUG_TEMPLATE.md) |
