# ExoQML Backend

FastAPI backend for ExoQML MVP:

- Target lookup (`TIC`, `KIC`, or name)
- Light curve acquisition (`lightkurve`, with resilient fallback)
- Standard preprocessing
- BLS-like classical baseline
- Classical inference + temporal relevance map
- Analysis history in SQLite
- JSON/CSV export

## Local run

```powershell
uv venv --python 3.11 .venv
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\uvicorn exoqml.main:app --reload --port 8000
```

Optional science stack:

```powershell
.\.venv\Scripts\python -m pip install -e .[science]
```

## Aggressive training

```powershell
cd ..
.\scripts\train_max.ps1 -Epochs 40 -ReserveFreeGb 60 -DiskUtilization 0.9
```

Auto-resume artifacts:

- `data/train_max/manifest.csv` (ingestion progress)
- `data/train_max/current_run.json` (active run pointer)
- `data/train_max/runs/<run_id>/latest_model.pt` (resume checkpoint)
- `data/train_max/runs/<run_id>/best_model.pt` (best validation checkpoint)

## Environment variables

`EXOQML_DATABASE_URL`
`EXOQML_CORS_ORIGINS`
`EXOQML_MAX_POINTS`
`EXOQML_ENABLE_QML`
`EXOQML_ALLOW_SYNTHETIC_FALLBACK`
`EXOQML_MODEL_PATH`
`EXOQML_DEVICE`
