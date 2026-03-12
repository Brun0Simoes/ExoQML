# Plano de Fechamento do PRD

Base de comparação: [prd_exoqml.md](./prd_exoqml.md)

## Status atual

- Fase 0: concluída
- Fase 1: concluída
- Fase 2: concluída
- Fase 3: concluída para o MVP
- Fase 4: concluída como trilha experimental

## Bloco 1 — fechar MVP operacional

- concluído

## Bloco 2 — fechar aceite do MVP

- concluído para cenário típico e cache quente
- documentado com benchmark e avaliação offline
- ressalva restante: aquisição fria continua limitada pela fonte externa

## Bloco 3 — endurecimento de produto

- checkpoint endurecido
- cache local de curva de luz implementado
- histórico continua global no MVP
- Postgres e autenticação seguem opcionais para produção

## Bloco 4 — trilha experimental QML

- head híbrida com PennyLane implementada
- benchmark clássico vs QML executado
- custo, latência e métricas documentados
- decisão tomada: QML permanece no produto como modo experimental, não como caminho principal
