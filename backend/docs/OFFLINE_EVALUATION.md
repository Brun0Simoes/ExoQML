# Avaliação Offline do Modelo

Fontes principais:

- [report.json](/E:/Estrela/backend/data/train_max/runs/20260311_053035/report.json)
- [best_model_calibrated_report.json](/E:/Estrela/backend/data/train_max/runs/20260311_053035/best_model_calibrated_report.json)

## Checkpoint de produção atual

- checkpoint: [best_model_calibrated.pt](/E:/Estrela/backend/data/train_max/runs/20260311_053035/best_model_calibrated.pt)
- arquitetura: `transit-multiview-tce`
- melhor época: `21`
- threshold calibrado na validação: `0.45`

## Dataset usado

- missão principal de treino: `Kepler`
- total de TCEs: `15737`
- TCEs positivos: `3600`
- TCEs negativos: `12137`
- estrelas totais: `9865`
- estrelas positivas: `2716`
- estrelas negativas: `7149`
- split por estrela:
  - treino: `12558` TCEs
  - validação: `1588` TCEs
  - teste: `1591` TCEs
- tamanho em disco: `64.84 GB`

## Entradas do modelo

- view global phase-folded: `401` bins
- view local phase-folded: `121` bins
- features escalares:
  - período
  - duração
  - profundidade
  - model SNR

## Comparação de runs

Leitura principal: `test_metrics_threshold_best_on_val`

| Run | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline antigo `20260311_000539` | 0.5279 | 0.3482 | 0.8931 | 0.5011 | 0.6919 | 0.3915 |
| Campeão `20260311_053035` | 0.9164 | 0.7995 | 0.8575 | 0.8275 | 0.9563 | 0.8068 |
| Mining tunado `20260311_072919` | 0.9089 | 0.7709 | 0.8683 | 0.8167 | 0.9539 | 0.8106 |

## Calibração do score

Checkpoint calibrado por `Platt scaling` no split de validação.

### Validação

| Métrica | Antes | Depois |
|---|---:|---:|
| F1 | 0.8114 | 0.8099 |
| Brier | 0.1009 | 0.0722 |
| ECE | 0.1312 | 0.0221 |

### Teste

| Métrica | Antes | Depois |
|---|---:|---:|
| F1 | 0.8286 | 0.8281 |
| Brier | 0.0950 | 0.0666 |
| ECE | 0.1289 | 0.0209 |

## Interpretação

- O salto relevante veio da mudança para classificação por `TCE` com views `phase-folded`.
- O checkpoint calibrado é o melhor checkpoint de produção porque preserva ranking e melhora fortemente a calibração do score.
- O run com `hard-negative mining` ganhou um pouco em ranking e recall, mas perdeu em `F1` e `precision`.

## Limitações atuais

- calibração formal do score concluída com `Platt scaling`
- avaliação separada por missão ainda não fechada
- treino principal ainda concentrado em `Kepler`
- modo QML continua só como feature flag experimental

## Status frente ao PRD

- Fase 1: concluída
- Fase 2: concluída
- parte offline do aceite do MVP: atendida com ressalva apenas de avaliação por missão
