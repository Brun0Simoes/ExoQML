# Trilha Experimental QML

Artefatos principais:

- QML hibrido inicial: [report.json](/E:/Estrela/backend/data/qml_experiment/runs/20260311_163657/report.json)
- QML residual inicial: [report.json](/E:/Estrela/backend/data/qml_residual/runs/20260311_170513/report.json)
- QML residual tunado: [report.json](/E:/Estrela/backend/data/qml_residual_tuning/runs/20260311_172904/report.json)
- Resumo do tuning: [latest_tuning_summary.json](/E:/Estrela/backend/data/qml_residual_tuning/latest_tuning_summary.json)
- Benchmark atual: [latest_qml_compare.json](/E:/Estrela/backend/data/benchmarks/latest_qml_compare.json)

## O que esta implementado

- cabeca hibrida real com `PennyLane`
- variante `residual QML` que corrige o logit classico em vez de substituir o backbone
- gate por faixa ambigua para rodar QML so quando o score classico cai na zona dificil
- comparacao classico vs QML no backend e no frontend
- checkpoint QML calibrado utilizavel pela rota de analise

Arquivos principais:

- [model.py](/E:/Estrela/backend/exoqml/training/model.py)
- [inference.py](/E:/Estrela/backend/exoqml/services/inference.py)
- [train_qml_experimental.py](/E:/Estrela/scripts/train_qml_experimental.py)
- [train_qml_residual.py](/E:/Estrela/scripts/train_qml_residual.py)
- [benchmark_qml_compare.py](/E:/Estrela/scripts/benchmark_qml_compare.py)

## Configuracao do residual QML

- checkpoint base: [best_model_calibrated.pt](/E:/Estrela/backend/data/train_max/runs/20260311_053035/best_model_calibrated.pt)
- estrategia: congelar encoders e cabeca classica; treinar apenas a correcao residual
- qubits: `4`
- camadas quanticas: `2`
- device PennyLane: `default.qubit`
- melhor faixa ambigua encontrada: `[0.35, 0.65]`
- melhor `residual_alpha_init`: `0.15`
- amostras de foco no treino: `1783`
- amostras que acionam o gate:
  - validacao: `121`
  - teste: `132`

## Resultado offline

Leitura principal: `test_metrics_threshold_best_on_val`

| Modelo | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classico calibrado | 0.7770 | 0.8898 | 0.8296 | 0.9563 | 0.8069 | 0.0666 | 0.0209 |
| QML hibrido inicial | 0.7829 | 0.8629 | 0.8210 | 0.9478 | 0.7666 | 0.0728 | 0.0347 |
| QML residual inicial | 0.7844 | 0.8898 | 0.8338 | 0.9569 | 0.8085 | 0.0660 | 0.0269 |
| QML residual tunado | 0.7845 | 0.9005 | 0.8385 | 0.9569 | 0.8082 | 0.0664 | 0.0255 |

Leitura complementar: `test_metrics_best_on_test`

| Modelo | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Classico calibrado | 0.7770 | 0.8898 | 0.8296 | 0.9563 | 0.8069 |
| QML residual inicial | 0.7995 | 0.8790 | 0.8374 | 0.9569 | 0.8085 |
| QML residual tunado | 0.7845 | 0.9005 | 0.8385 | 0.9569 | 0.8082 |

## Latencia

Benchmark local no mesmo alvo:

| Caminho | Mediana |
|---|---:|
| Classico direto | 0.1332 s |
| QML residual direto | 0.1782 s |
| Segunda etapa completa | 0.3286 s |

## Decisao tecnica

- o QML residual melhorou o QML inicial de forma clara
- o tuning de banda e `residual_alpha` melhorou o residual inicial
- o QML residual tunado superou levemente o classico em `F1` e `ROC-AUC`
- o ganho veio sem trocar o classico como caminho principal
- o custo adicional continua pequeno porque o gate roda QML so na faixa ambigua

Conclusao:

- o classico continua sendo o modelo principal do produto
- o QML agora faz sentido como segunda etapa experimental para casos ambiguos
- esta e a primeira variante QML do projeto que agrega valor pratico ao classico
