# Diagnostico do Ambiente (capturado em 2026-03-07)

## Hardware

- SO: Windows 10 Pro 64-bit (10.0.19045)
- CPU: Intel Xeon E5-2640 v3 (8 cores / 16 threads, 2.60 GHz)
- RAM: 34.26 GB
- GPU: NVIDIA GeForce RTX 3070 (8 GB VRAM)
- Driver/CUDA (`nvidia-smi`): 595.71 / CUDA 13.2

## Software detectado

- Python principal: 3.14.0
- Python adicional disponivel: 3.11.14 (via `uv`)
- pip: 25.2
- Node.js: v24.11.0
- npm: 11.6.1
- uv: 0.10.4
- git: 2.52.0
- pnpm/yarn/poetry/conda: nao detectados neste workspace

## Adaptacoes aplicadas no projeto

- `uv` adotado como base de setup Python.
- Backend fixado em Python `>=3.11,<3.13` para compatibilidade do stack cientifico.
- Execucao CPU-first por padrao (GPU opcional via `EXOQML_DEVICE=auto|cuda`).
- Frontend baseado em `npm` (sem dependencias de `pnpm`/`yarn`).
- Fallback sintetico resiliente ativado para continuar fluxo mesmo sem `lightkurve`.
