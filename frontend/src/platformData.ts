export const PLATFORM_STATS = [
  {
    label: "Catalogo local",
    value: "9.867 alvos",
    detail: "Consulta local pronta para KIC, TIC e nomes guiados.",
  },
  {
    label: "Dataset",
    value: "15.737 TCEs",
    detail: "9.865 estrelas processadas e 64,8 GB de dados prontos.",
  },
  {
    label: "Modelo principal",
    value: "F1 82,8%",
    detail: "PR-AUC 80,7% no pipeline multiview TCE em producao.",
  },
  {
    label: "Latencia quente",
    value: "0,82 s",
    detail: "Pipeline local com cache quente em CPU, mediana validada.",
  },
];

export const PRODUCTION_STACK = [
  {
    title: "Classificador principal",
    value: "Transit multiview TCE",
    detail: "Modelo classico calibrado que decide a triagem principal.",
  },
  {
    title: "Segunda etapa QML",
    value: "Residual on-demand",
    detail: "Atua so em casos ambiguos para corrigir o logit classico.",
  },
  {
    title: "Baseline cientifico",
    value: "BLS periodogram",
    detail: "Comparativo classico exibido junto do score da IA.",
  },
  {
    title: "Explicabilidade",
    value: "XAI temporal",
    detail: "Relevancia projetada de volta para a curva de luz processada.",
  },
];

export const WORKFLOW_OVERVIEW = [
  {
    title: "Escolha um alvo",
    detail: "Use o catalogo local, um nome conhecido ou um codigo KIC/TIC.",
  },
  {
    title: "Rode a triagem",
    detail: "A plataforma baixa, prepara a curva e envia para o modelo certo.",
  },
  {
    title: "Leia o resultado",
    detail: "Score, BLS, curva, XAI, mapa do alvo e export ficam na mesma tela.",
  },
];
