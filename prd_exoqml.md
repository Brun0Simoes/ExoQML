# PRD — ExoQML

**Versão:** 2.0  
**Status:** Planejamento orientado a MVP  
**Data:** 2026-03-07  
**Owner:** Produto + Engenharia + IA

---

## 1) Resumo executivo

O **ExoQML** será uma aplicação web para buscar estrelas por identificadores públicos (ex.: **TIC**, **KIC**, nome Kepler/TESS), baixar suas **curvas de luz** em fontes oficiais, executar um pipeline de inferência e exibir:

1. **probabilidade de assinatura de trânsito**;
2. **curva de luz processada**;
3. **explicação visual do modelo** (Grad-CAM 1D ou mapa de relevância equivalente);
4. **baseline astronômico clássico** para comparação (BLS / Box Least Squares);
5. **metadados e histórico de análises**.

### Decisão principal deste PRD

Para tornar o projeto **produtivo, funcional e demonstrável**, a versão inicial **não** deve depender de QML como requisito central. O melhor caminho é:

- **V1 (recomendado):** modelo clássico em Python + XAI + produto web utilizável.
- **V1.1:** expansão para múltiplas missões e melhor calibração.
- **Trilha P&D opcional:** **QML experimental**, atrás de feature flag, sem bloquear entrega.

Em outras palavras: **manter o nome ExoQML é aceitável, mas o módulo quântico deve ser tratado como diferencial experimental, não como fundação do MVP.**

---

## 2) Por que esta adaptação é a melhor

A ideia original é forte, mas mistura três níveis de ambição ao mesmo tempo:

- produto web utilizável;
- pipeline científico de séries temporais;
- pesquisa exploratória com QML.

Se tudo entrar no MVP, o risco técnico sobe demais. O caminho mais seguro é separar o produto em **camadas de valor**:

### Camada 1 — valor imediato
- busca de alvo;
- download de curva de luz;
- preprocessamento robusto;
- inferência clássica;
- XAI;
- histórico.

### Camada 2 — valor científico/engenharia
- baseline BLS comparável ao modelo;
- suporte a TESS + Kepler/K2;
- avaliação séria com métricas corretas para classes desbalanceadas.

### Camada 3 — diferencial de pesquisa
- classificador híbrido PyTorch + PennyLane;
- comparação formal entre baseline clássico vs. híbrido;
- publicação técnica / benchmark interno.

### Recomendação objetiva

**Não vender o MVP como “detector quântico de exoplanetas”.**  
Vender como:

> “plataforma web de análise de curvas de luz com IA explicável, preparada para benchmark híbrido clássico-quântico.”

Isso reduz risco de execução, melhora credibilidade e aumenta a chance de entrega real.

---

## 3) Verificação de viabilidade e disponibilidade das fontes

### Fontes de dados validadas

#### MAST / STScI
Fonte principal para **light curves**, **target pixel files**, **data validation products** e holdings de missões como **Kepler, K2 e TESS**.

**Uso no produto:** aquisição dos sinais e metadados observacionais.

#### NASA Exoplanet Archive
Fonte principal para **catálogos de planetas confirmados**, **candidatos**, tabelas da missão Kepler/KOI e **TESS Project Candidates**.

**Uso no produto:** rótulos, enriquecimento de metadados, benchmarking e validação.

### Bibliotecas validadas

#### lightkurve
Biblioteca Python adequada para:
- buscar curvas de luz em MAST;
- baixar coleções de observações;
- fazer `stitch`, `flatten`, `remove_nans`, `remove_outliers`, `fold`, `bin`;
- executar análises clássicas como BLS;
- converter dados para formatos úteis ao pipeline de ML.

#### PyTorch
Adequado para encoder 1D, treinamento, persistência de pesos e serving.

#### PennyLane
Viável para integrar um classificador quântico/híbrido ao PyTorch, mas deve ser usado como trilha experimental.

---

## 4) Problema do produto

Hoje, desenvolvedores e estudantes conseguem montar notebooks que classificam curvas de luz, mas raramente entregam uma solução que:

- consulte dados públicos de forma reproduzível;
- gere inferência utilizável por terceiros;
- explique visualmente a decisão do modelo;
- compare IA com um baseline astronômico clássico;
- preserve histórico e rastreabilidade da análise.

O ExoQML resolve isso transformando experimentos dispersos em um **sistema de análise end-to-end**.

---

## 5) Objetivo do produto

Desenvolver uma aplicação web que permita ao usuário consultar uma estrela em bases públicas, baixar sua curva de luz, processá-la com um pipeline de IA para detecção de sinais de trânsito e visualizar tanto a **predição** quanto a **evidência temporal** usada para a decisão.

### Objetivos de negócio / portfólio

- demonstrar maturidade em produto de IA aplicado a dados científicos;
- provar capacidade full-stack + ML serving;
- mostrar explicabilidade real e não apenas acurácia;
- manter espaço para uma trilha de P&D com QML.

### Objetivos do MVP

- suportar busca por IDs válidos;
- processar curva de luz com latência aceitável em CPU;
- mostrar score, gráfico e explicação;
- registrar histórico;
- expor proveniência dos dados.

### Não objetivos do MVP

- anunciar “nova descoberta científica”;
- substituir vetting astronômico profissional;
- treinar modelo em tempo real no navegador;
- depender de hardware quântico real;
- suportar varredura massiva de catálogos inteiros na interface pública.

---

## 6) Público-alvo

### Primário
- recrutadores e avaliadores técnicos de portfólio;
- desenvolvedores/engenheiros de IA;
- estudantes de astrofísica de dados.

### Secundário
- pesquisadores que queiram um demonstrador leve;
- comunidade maker/citizen science interessada em trânsito planetário.

---

## 7) Proposta de valor

O produto deve permitir que alguém informe um ID astronômico e receba, em poucos passos:

- os dados observacionais correspondentes;
- uma análise reproduzível;
- um score probabilístico;
- um gráfico interpretável;
- um comparativo entre heurística astronômica clássica e deep learning.

### Frase de posicionamento

> “Uma plataforma web para inspeção de curvas de luz com IA explicável, usando dados públicos da NASA/STScI e pronta para benchmark com módulos híbridos clássico-quânticos.”

---

## 8) Melhor caminho de adaptação da arquitetura

## Recomendação principal: backend Python-first

### Arquitetura recomendada para V1
- **Frontend:** React + Vite
- **Backend principal:** FastAPI (Python)
- **Motor de dados/IA:** Python (lightkurve + PyTorch)
- **Banco:** SQLite em dev / Postgres em produção
- **Fila assíncrona (opcional V1.1):** Celery/RQ + Redis

### Motivo

A maior parte do domínio crítico já é Python-native:
- ingestão em `lightkurve`;
- preprocessamento;
- treinamento/inferência em PyTorch;
- Grad-CAM 1D;
- integração opcional com PennyLane.

Adicionar **Node.js no MVP** cria uma fronteira extra entre frontend e o stack científico sem aumentar muito o valor do produto. Para portfólio, um **backend Python bem organizado** entrega melhor relação entre complexidade e funcionalidade.

### Quando manter Node.js

Manter **Node.js + Python** faz sentido apenas se o objetivo explícito for demonstrar:
- arquitetura polyglot;
- orquestração de serviços;
- experiência forte em ecossistema JS no backend.

### Conclusão arquitetural

- **Recomendado:** React + FastAPI + Python ML service.
- **Aceitável:** React + Node gateway + Python worker.
- **Não recomendado para MVP:** React + Node + Python + QML obrigatório.

---

## 9) Estratégia de dados

## 9.1 Fontes

### Para sinais observacionais
- **MAST / STScI**: Kepler, K2, TESS, DV products, cutouts e holdings associados.

### Para rótulos e metadados
- **NASA Exoplanet Archive**:
  - **PS / PSCompPars** para sistemas/planetas confirmados;
  - **KOI Cumulative** para candidatos Kepler;
  - **TESS Project Candidates** para candidatos TESS;
  - **K2 Planets and Candidates** quando necessário.

## 9.2 Estratégia recomendada de treino

### Fase A — dataset base com Kepler
Usar **Kepler/KOI** como primeiro conjunto principal de treinamento e validação por ser uma base madura e amplamente utilizada para trânsito.

### Fase B — generalização para TESS
Adicionar **TESS** como conjunto de validação externa e, depois, como suporte de inferência no produto.

### Fase C — fallback operacional
Se não houver light curve pré-processada disponível para um alvo, usar:
- `search_lightcurve()` primeiro;
- `search_tesscut()` como fallback para FFIs, quando aplicável.

## 9.3 Política de rótulos

Definir rótulos com clareza para evitar vazamento de semântica:

- **positivo (confirmado):** planetas confirmados / candidatos validados segundo política escolhida;
- **negativo:** falso positivo, alvo sem trânsito confirmado, ou amostras de controle com cuidado para não introduzir viés;
- **ambíguo:** candidatos não confirmados podem ser usados em trilha separada de ranking, não necessariamente como verdade absoluta.

## 9.4 Regras de split

Para evitar leakage:
- dividir por **alvo estelar** e não apenas por janelas da série temporal;
- separar treino/validação/teste por alvo e missão quando possível;
- evitar que segmentos do mesmo objeto apareçam em treino e teste.

---

## 10) Pipeline de dados e ML

## 10.1 Pipeline online de inferência

1. Usuário informa `TIC`, `KIC` ou nome suportado.  
2. Backend valida o identificador.  
3. Sistema busca produtos observacionais em MAST.  
4. Curva de luz é baixada e preprocessada.  
5. Pipeline executa:
   - baseline astronômico (BLS);
   - encoder 1D + classificador clássico;
   - módulo de explicabilidade.  
6. Resultado é salvo e devolvido ao frontend.

## 10.2 Preprocessamento mínimo obrigatório

- remoção de NaNs;
- remoção de outliers;
- normalização do fluxo;
- flatten/detrending;
- stitching entre setores/quarters quando necessário;
- binning ou windowing para entrada de tamanho fixo;
- preservação de máscaras/índices para reconstrução visual no frontend.

## 10.3 Baseline recomendado

Antes de qualquer QML, a aplicação deve possuir dois modos de análise:

### Baseline científico clássico
- **BLS periodogram** para sinal periódico compatível com trânsito.

### Baseline de deep learning
- **1D CNN / ResNet1D / TCN** para classificação da série temporal ou janelas candidatas.

### Justificativa

Sem baseline clássico, o produto vira apenas uma “caixa preta bonita”. O BLS aumenta credibilidade, ajuda depuração e melhora a narrativa do portfólio.

## 10.4 Explicabilidade

Implementar **Grad-CAM 1D** ou técnica equivalente aplicada à última camada convolucional do encoder clássico.

**Saída esperada:** mapa temporal de relevância sobreposto à curva de luz processada.

## 10.5 QML (opcional, experimental)

O módulo quântico deve entrar somente depois do baseline clássico consolidado.

### Estratégia recomendada

Não alimentar o circuito quântico com a curva bruta. Em vez disso:
- usar o encoder clássico para comprimir a entrada;
- enviar um vetor pequeno de features ao circuito parametrizado;
- comparar custo, latência e qualidade contra a cabeça clássica densa.

### Critério de permanência

O modo QML só continua no produto se demonstrar ao menos um destes ganhos:
- melhor PR-AUC / Recall em casos difíceis;
- melhor robustez a ruído;
- redução relevante de parâmetros com desempenho comparável;
- valor demonstrável de P&D para portfólio/publicação.

Caso contrário, permanece como branch experimental.

---

## 11) Requisitos funcionais

### RF01 — Busca de alvo
O sistema deve aceitar pelo menos um dos seguintes identificadores:
- TIC ID;
- KIC ID;
- nome de alvo suportado pelas buscas do MAST/lightkurve.

### RF02 — Validação do identificador
O sistema deve informar quando o identificador for inválido, ambíguo ou sem dados observacionais compatíveis.

### RF03 — Coleta de dados
O backend deve buscar e baixar a curva de luz associada ao alvo em fontes públicas suportadas.

### RF04 — Fallback de aquisição
Se não houver light curve pronta, o sistema deve tentar estratégia de fallback compatível (ex.: TESScut para TESS/FFI quando suportado).

### RF05 — Preprocessamento
O sistema deve aplicar pipeline padronizado e reprodutível antes da inferência.

### RF06 — Inferência clássica
O sistema deve retornar:
- classe prevista;
- score/probabilidade;
- versão/modelo utilizado;
- status da análise.

### RF07 — Baseline BLS
O sistema deve exibir um resultado clássico de referência, como pico(s) relevantes do BLS e período estimado quando aplicável.

### RF08 — Explicabilidade
O sistema deve gerar um mapa temporal de relevância e sobrepor esse mapa à curva de luz visualizada.

### RF09 — Proveniência
O sistema deve exibir origem da análise:
- missão;
- setor/quarter/campaign, quando disponível;
- fonte do dado;
- timestamp da análise.

### RF10 — Histórico
O sistema deve registrar consultas e permitir listar análises recentes.

### RF11 — Download/exportação leve
O sistema deve permitir exportar, no mínimo, um JSON ou CSV com metadados e resultados da análise.

### RF12 — Modo experimental
O sistema deve permitir habilitar/desabilitar o modo QML por feature flag sem impactar o fluxo padrão.

---

## 12) Requisitos não funcionais

### RNF01 — Modularidade
Treino, inferência e frontend devem ser desacoplados.

### RNF02 — Reprodutibilidade
Cada inferência deve registrar:
- versão do modelo;
- parâmetros do preprocessamento;
- origem dos dados;
- timestamp.

### RNF03 — Performance
Meta de experiência no MVP:
- até **8 s** para análise típica em CPU, quando os dados já forem acessíveis;
- até **15 s** em casos com aquisição/preprocessamento mais pesados.

### RNF04 — Observabilidade
Logs estruturados devem registrar falhas de busca, falhas de preprocessamento e falhas de inferência.

### RNF05 — Resiliência
A aplicação deve degradar graciosamente quando:
- a fonte externa estiver indisponível;
- não houver curva utilizável;
- o sinal estiver muito corrompido.

### RNF06 — Segurança
Não carregar pesos arbitrários de fonte não confiável e não executar artefatos externos sem validação.

### RNF07 — Transparência científica
A interface deve deixar claro que a análise é uma **triagem assistida por IA**, não uma confirmação observacional oficial.

---

## 13) Métricas de sucesso

## Produto
- taxa de sucesso na busca e processamento de IDs suportados ≥ **90%** em conjunto de teste interno;
- tempo mediano de resposta ≤ **8 s**;
- taxa de erro de pipeline ≤ **5%** nas rotas principais.

## ML
Evitar usar apenas acurácia.

Métricas mínimas:
- **PR-AUC**;
- **ROC-AUC**;
- **F1**;
- **Recall** em classe positiva;
- **calibração** do score;
- avaliação separada por missão, quando aplicável.

## UX
- usuário consegue analisar um alvo em até **3 passos**;
- gráfico e explicação devem aparecer na mesma tela do resultado.

---

## 14) Arquitetura lógica

```text
[React/Vite Frontend]
        |
        v
[FastAPI Backend]
   |        |        \
   |        |         -> [SQLite/Postgres]
   |        -> [Model Registry / arquivos .pt/.pth]
   -> [Data & ML Service: lightkurve + PyTorch + XAI]
                    \
                     -> [PennyLane experimental]
```

### Entidades mínimas de dados

#### AnalysisLog
- id
- target_id
- target_type
- mission
- data_source
- model_name
- model_version
- prediction_label
- prediction_score
- bls_period
- status
- created_at

#### CachedLightCurve (opcional V1.1)
- id
- target_id
- sector_or_quarter
- cache_path
- checksum
- created_at

---

## 15) Fluxos principais do usuário

## Fluxo A — análise simples
1. usuário digita TIC/KIC;
2. sistema valida o alvo;
3. sistema busca e processa a curva;
4. tela retorna score, gráfico, XAI e baseline BLS;
5. resultado fica salvo no histórico.

## Fluxo B — análise com erro
1. usuário informa alvo inválido ou sem dados adequados;
2. sistema informa causa provável;
3. sistema sugere ação: trocar ID, missão, ou tentar alvo suportado.

## Fluxo C — comparação experimental
1. usuário ativa “modo experimental QML”; 
2. sistema roda cabeça híbrida sobre features comprimidas;
3. interface compara resultado clássico vs. QML.

---

## 16) Roadmap prático

## Fase 0 — prova de viabilidade de dados
- conectar `lightkurve`;
- buscar 10–20 alvos reais;
- validar download, stitch, flatten e serialização;
- validar mapeamento com rótulos do Archive.

**Saída:** notebook/CLI confiável de ingestão.

## Fase 1 — MVP científico local
- pipeline de preprocessamento;
- baseline BLS;
- modelo 1D CNN/ResNet1D;
- Grad-CAM 1D;
- avaliação offline com split correto.

**Saída:** serviço local reproduzível com métricas.

## Fase 2 — MVP web funcional
- API FastAPI;
- frontend React/Vite;
- histórico de análises;
- tela única com score + curva + XAI + BLS.

**Saída:** produto demonstrável ponta a ponta.

## Fase 3 — robustez e produto
- cache de curvas;
- tratamento melhor de erros;
- exportação de resultados;
- autenticação opcional;
- migração SQLite -> Postgres, se necessário.

## Fase 4 — trilha experimental QML
- head híbrida com PennyLane;
- benchmark contra baseline clássico;
- relatório comparativo de latência e performance;
- decisão de continuidade baseada em evidência.

---

## 17) Riscos e mitigação

### Risco 1 — QML aumentar custo e latência sem ganho claro
**Mitigação:** feature flag; benchmark obrigatório; QML fora do caminho crítico.

### Risco 2 — qualidade dos sinais variar muito entre missões
**Mitigação:** começar com Kepler; adicionar TESS depois; avaliar por missão.

### Risco 3 — vazamento de dados no split
**Mitigação:** split por alvo/sistema e auditoria de pipeline.

### Risco 4 — dependência excessiva de fontes externas em tempo real
**Mitigação:** cache local, retries, mensagens claras de indisponibilidade.

### Risco 5 — UX parecer “mágica” demais
**Mitigação:** mostrar proveniência, BLS, parâmetros e explicação visual.

### Risco 6 — promessa científica exagerada
**Mitigação:** linguagem cuidadosa, disclaimers e foco em triagem assistida.

---

## 18) Critérios de aceite do MVP

O MVP será considerado pronto quando:

- aceitar ao menos `TIC` e `KIC` ou nomes suportados;
- conseguir buscar e processar alvos válidos de ponta a ponta;
- exibir score + curva + explicação + baseline BLS;
- registrar histórico;
- manter latência e estabilidade dentro das metas;
- possuir documentação de avaliação offline;
- não depender do modo QML para funcionar.

---

## 19) Decisões fechadas

- **QML não é requisito de entrega da V1.**
- **FastAPI é o backend recomendado para o MVP.**
- **Node.js torna-se opcional e arquitetural, não obrigatório.**
- **BLS entra como baseline obrigatório.**
- **Kepler/KOI é a trilha inicial de treino mais segura.**
- **TESS entra primeiro como inferência e validação externa, depois como expansão de treino.**
- **SQLite serve para dev/demo; Postgres fica previsto para produção.**

---

## 20) Questões em aberto

- quais classes exatas compõem o negativo do treinamento?
- o score final será binário, ranking, ou ambos?
- o produto mostrará apenas curvas prontas ou também suportará cutouts/TESScut no MVP?
- haverá autenticação ou tudo será público?
- o histórico será por sessão ou por usuário?
- qual estratégia de calibração do score será adotada?

---

## 21) Anexo — fontes oficiais validadas

### Dados e arquivos
- [MAST / STScI](https://archive.stsci.edu/home) — holdings e interfaces para Kepler, K2, TESS e exo.MAST
- [TESS mission page (MAST)](https://archive.stsci.edu/missions-and-data/tess) — light curves, target pixel files e data validation files
- [Search Interfaces / exo.MAST](https://archive.stsci.edu/search-interfaces) — busca por exoplanetas, trânsito e curvas dobradas
- [TESS-SPOC HLSP](https://archive.stsci.edu/hlsp/tess-spoc) — DV products e produtos derivados de TCEs

### Catálogos e tabelas
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — portal principal
- [PS / PSCompPars docs](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html) — tabelas atuais para sistemas confirmados e parâmetros compostos
- [Kepler KOI docs](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html) — tabela cumulativa de candidatos Kepler
- [Purpose of KOI Cumulative Table](https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html) — racional da tabela cumulativa
- [TESS Project Candidates](https://exoplanetarchive.ipac.caltech.edu/docs/TESSMission.html) — candidatos TOI/TESS
- [Programmatic Interfaces](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html) — TAP/API do Archive

### Ferramentas Python
- [Lightkurve `search_lightcurve`](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html)
- [Lightkurve `search_tesscut`](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_tesscut.html)
- [Lightkurve `flatten`](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.LightCurve.flatten.html)
- [Lightkurve `remove_nans`](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.LightCurve.remove_nans.html)
- [Lightkurve `remove_outliers`](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.LightCurve.remove_outliers.html)
- [Lightkurve `stitch`](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.LightCurveCollection.stitch.html)
- [Lightkurve exoplanet tutorial / BLS](https://lightkurve.github.io/lightkurve/tutorials/3-science-examples/exoplanets-identifying-transiting-planet-signals.html)
- [PyTorch save/load](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch `torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [PennyLane PyTorch interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html)
- [PennyLane `TorchLayer`](https://docs.pennylane.ai/en/stable/_modules/pennylane/qnn/torch.html)

### Referências técnicas de explicabilidade / QML
- [Grad-CAM (paper original)](https://arxiv.org/pdf/1610.02391)
- [XCM: explainable CNN for time series](https://arxiv.org/pdf/2009.04796)
- [Review de QML](https://arxiv.org/pdf/2401.11351)
- [Noise-induced barren plateaus in VQAs](https://www.nature.com/articles/s41467-021-27045-6)

---

## 22) Resumo final de direcionamento

Se o objetivo é montar um projeto que realmente impressione em portfólio e ainda tenha profundidade científica, o posicionamento correto é:

### O que entregar primeiro
- produto web utilizável;
- pipeline robusto de curvas de luz;
- baseline clássico + deep learning;
- XAI útil.

### O que deixar como diferencial de pesquisa
- QML.

### Em uma frase

**O ExoQML deve nascer como uma plataforma confiável de análise de curvas de luz com IA explicável — e evoluir para um laboratório comparativo de QML, não o contrário.**
