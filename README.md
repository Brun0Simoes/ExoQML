# ExoQML MVP (adaptado ao ambiente local)

Projeto montado na raiz `E:\Estrela` com arquitetura recomendada no PRD:

- Frontend: React + Vite (`/frontend`)
- Backend: FastAPI + SQLite (`/backend`)
- Pipeline: aquisicao/preprocessamento/BLS/inferencia/XAI em Python

## O que esta implementado

- RF01/RF02: entrada `TIC`, `KIC` e nome com validacao
- RF03/RF04: aquisicao por `lightkurve` + fallback resiliente
- RF05: preprocessamento padronizado (NaN, outlier, normalize, detrend, resize)
- RF06: inferencia classica (Torch quando disponivel, fallback heuristico)
- RF07: baseline BLS-like com periodos e top picos
- RF08: mapa temporal de relevancia (Grad-CAM 1D no caminho Torch)
- RF09: proveniencia exibida no resultado
- RF10: historico em SQLite
- RF11: export JSON/CSV por analise
- RF12: flag de modo experimental QML

## Setup (PowerShell)

```powershell
.\scripts\setup.ps1
```

## Rodar em dev

Terminal 1:

```powershell
.\scripts\run_backend.ps1
```

Terminal 2:

```powershell
.\scripts\run_frontend.ps1
```

Abra:

- Frontend: `http://127.0.0.1:5173`
- API: `http://127.0.0.1:8000/api/v1/health`

## Treino agressivo (max hardware)

Pipeline de treino com:

- ingestao DR24 TCE (Kepler) com cache em disco;
- split por estrela (evita leakage);
- ajuste automatico de `batch_size`, `num_workers` e device;
- checkpoints e retomada automatica apos desligamento/queda.

Executar:

```powershell
.\scripts\train_max.ps1 -Epochs 40 -ReserveFreeGb 60 -DiskUtilization 0.9 -Device auto
```

Para limitar ingestao em uma rodada piloto:

```powershell
.\scripts\train_max.ps1 -MaxStars 800 -Epochs 12
```

Retomada:

- O progresso fica em `backend/data/train_max/manifest.csv` e `backend/data/train_max/current_run.json`.
- O treino salva `latest_model.pt` e `best_model.pt` em `backend/data/train_max/runs/<run_id>/`.
- Ao executar novamente `train_max.ps1`, a retomada e automatica (use `-NoResume` para iniciar um run novo).

## Variaveis uteis

Copie `backend/.env.example` para `backend/.env` e ajuste se necessario.

Campos principais:

- `EXOQML_DATABASE_URL`
- `EXOQML_CORS_ORIGINS`
- `EXOQML_MAX_POINTS`
- `EXOQML_ENABLE_QML`
- `EXOQML_ALLOW_SYNTHETIC_FALLBACK`
- `EXOQML_DEVICE`
- `EXOQML_MODEL_PATH`

## Notas de compatibilidade

- Python padrao do host e `3.14.0`, mas o backend usa `3.11` para estabilidade do ecossistema cientifico.
- O projeto funciona sem `torch`/`lightkurve`; nesses casos entra modo fallback com avisos explicitos.
- Para uso cientifico real, instale stack `science` e forneca pesos treinados em `EXOQML_MODEL_PATH`.
