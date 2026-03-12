# Benchmark Operacional do MVP

Fonte principal: [latest_analysis_benchmark.json](/E:/Estrela/backend/data/benchmarks/latest_analysis_benchmark.json)

## Metodologia

- data: `2026-03-11`
- dispositivo: `CPU`
- alvo usado: `KIC 10000490`
- missão retornada: `Kepler Quarter 02`
- checkpoint usado: [best_model_calibrated.pt](/E:/Estrela/backend/data/train_max/runs/20260311_053035/best_model_calibrated.pt)
- cenários medidos:
  - `compute_only`: preprocessamento + BLS + inferência com a curva já em memória
  - `warm_full_pipeline`: aquisição com cache quente + preprocessamento + BLS + inferência
  - `cold_full_pipeline`: aquisição fria em cache vazio + preprocessamento + BLS + inferência

## Resultados

| Cenário | Mediana total | Aquisição | Preprocess | BLS | Inferência |
|---|---:|---:|---:|---:|---:|
| compute_only | 0.7418 s | 0.0000 s | 0.0239 s | 0.5865 s | 0.1292 s |
| warm_full_pipeline | 0.8179 s | 0.0108 s | 0.0207 s | 0.6441 s | 0.1394 s |
| cold_full_pipeline | 30.6404 s | 29.9118 s | 0.0184 s | 0.5843 s | 0.1260 s |

Observação extra:

- primeira execução de aquecimento: `1.5771 s`

## Leitura frente ao PRD

PRD alvo para experiência no MVP:

- típico em CPU com dados acessíveis: `<= 8 s`
- caso pesado: `<= 15 s`

Status atual:

- `compute_only`: atende com ampla margem
- `warm_full_pipeline`: atende com ampla margem
- `cold_full_pipeline`: não atende

## Conclusão operacional

O gargalo atual do produto não está no modelo nem no pipeline local. Ele está na aquisição fria de dados.

Hoje o MVP já entrega a meta de latência quando os dados estão acessíveis em cache local. O ponto que ainda impede fechamento integral do aceite operacional do PRD é:

- reduzir a latência de aquisição fria
- melhorar política de cache
- eventualmente pré-aquecer metadados ou curvas mais usadas

## Próximos passos recomendados

- cache mais agressivo de aquisição
- política de reuse por alvo e missão
- benchmark adicional com alvo TESS
- benchmark por rota HTTP para saúde, histórico e análise
