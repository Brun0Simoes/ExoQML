import type { TargetType } from "./types";

export type GuideTarget = {
  id: string;
  type: TargetType;
  title: string;
  mission: string;
  summary: string;
  whyItMatters: string;
  howToType: string;
  skyRegion: string;
  coordinates: {
    ra: number;
    dec: number;
  };
  references: string[];
};

export type SkyReferencePoint = {
  label: string;
  ra: number;
  dec: number;
  kind: "reference" | "region";
};

export const GUIDE_TARGETS: GuideTarget[] = [
  {
    id: "KIC 10000490",
    type: "kic",
    title: "Exemplo numerico do catalogo Kepler",
    mission: "Kepler Quarter 02",
    summary: "Bom exemplo para quem quer entender um codigo KIC puro, sem nome popular.",
    whyItMatters: "Mostra como funcionam os alvos do catalogo da missao Kepler usados no treino principal do projeto.",
    howToType: "Digite exatamente: KIC 10000490",
    skyRegion: "Campo principal da missao Kepler, no hemisferio norte celeste.",
    coordinates: {
      ra: 286.5560,
      dec: 46.9573,
    },
    references: ["Campo Kepler", "regiao de Cisne e Lira", "norte celeste"],
  },
  {
    id: "TIC 25155310",
    type: "tic",
    title: "Exemplo TESS no hemisferio sul",
    mission: "TESS Sector 01",
    summary: "Bom exemplo para quem quer testar um alvo do catalogo TIC da missao TESS.",
    whyItMatters: "Ajuda a ver a diferenca entre um codigo TIC e um KIC, alem de mostrar uma observacao no ceu do sul.",
    howToType: "Digite exatamente: TIC 25155310",
    skyRegion: "Regiao proxima da Grande Nuvem de Magalhaes, no ceu austral.",
    coordinates: {
      ra: 63.3739,
      dec: -69.2268,
    },
    references: ["Grande Nuvem de Magalhaes", "sul celeste", "primeiros setores do TESS"],
  },
  {
    id: "Kepler-10",
    type: "name",
    title: "Exemplo por nome conhecido",
    mission: "Kepler",
    summary: "Ideal para usuario leigo que nao sabe um codigo de catalogo de cabeca.",
    whyItMatters: "Mostra que a aplicacao tambem aceita nomes conhecidos de alvos observados em missoes publicas.",
    howToType: "Digite exatamente: Kepler-10",
    skyRegion: "Mesmo campo observado pela missao Kepler, em regiao alta do norte celeste.",
    coordinates: {
      ra: 285.6794,
      dec: 50.2413,
    },
    references: ["Campo Kepler", "alvo com nome popular", "norte celeste"],
  },
];

export const SKY_REFERENCE_POINTS: SkyReferencePoint[] = [
  { label: "Polaris", ra: 37.95, dec: 89.26, kind: "reference" },
  { label: "Sirius", ra: 101.29, dec: -16.72, kind: "reference" },
  { label: "Grande Nuvem de Magalhaes", ra: 80.89, dec: -69.76, kind: "region" },
  { label: "Campo Kepler", ra: 290.0, dec: 44.5, kind: "region" },
];

export const IDENTIFIER_GUIDE = [
  {
    title: "Nome conhecido",
    value: "Kepler-10",
    detail: "Melhor opcao para quem nao sabe um codigo de catalogo.",
  },
  {
    title: "TIC",
    value: "TIC 25155310",
    detail: "Catalogo principal usado pela missao TESS.",
  },
  {
    title: "KIC",
    value: "KIC 10000490",
    detail: "Catalogo principal usado pela missao Kepler.",
  },
];

export const SCENARIO_CARDS = [
  {
    title: "O que e uma curva de luz",
    body: "E uma serie temporal do brilho de uma estrela. Se o brilho cai de forma repetitiva, isso pode indicar que algo passou na frente dela.",
  },
  {
    title: "O que e um transito",
    body: "E a passagem de um planeta pela frente da estrela do ponto de vista do observador. Isso costuma gerar uma pequena queda no brilho.",
  },
  {
    title: "O que a aplicacao realmente faz",
    body: "Ela nao confirma um planeta. Ela faz triagem assistida, procurando sinais que merecem inspecao mais cuidadosa.",
  },
];

export const PIPELINE_STEPS = [
  {
    title: "1. Localiza o alvo",
    body: "A aplicacao interpreta o nome ou codigo informado e resolve a busca em catalogos publicos suportados.",
  },
  {
    title: "2. Baixa os dados publicos",
    body: "A curva de luz e buscada em fontes oficiais da NASA/STScI, como holdings das missoes Kepler e TESS.",
  },
  {
    title: "3. Limpa e resume o sinal",
    body: "O pipeline remove outliers, normaliza o fluxo, calcula o baseline BLS e prepara a entrada em formato fixo para o modelo.",
  },
  {
    title: "4. Mostra score e evidencia",
    body: "A IA devolve um score, um mapa de relevancia temporal, o comparativo BLS e a proveniencia da analise.",
  },
];

export const RESULT_GUIDE = [
  {
    title: "Probabilidade",
    body: "Resume o quanto o modelo acha que existe um padrao parecido com transito. Nao e uma confirmacao astronomica.",
  },
  {
    title: "BLS",
    body: "Mostra um metodo classico da astronomia para procurar sinais periodicos em forma de caixa na curva de luz.",
  },
  {
    title: "Mapa de relevancia",
    body: "Indica em quais trechos da curva o modelo concentrou mais evidencia para chegar ao resultado.",
  },
  {
    title: "Proveniencia",
    body: "Mostra de qual missao vieram os dados, quando a analise foi feita e qual checkpoint do modelo foi usado.",
  },
];

export const BEGINNER_FAQ = [
  {
    question: "Eu preciso saber um TIC ou KIC de memoria?",
    answer:
      "Nao. Voce pode usar um exemplo guiado, digitar um nome conhecido como Kepler-10 ou colar um codigo encontrado em catalogos publicos.",
  },
  {
    question: "Se eu digitar so numeros, o que acontece?",
    answer:
      "Hoje o backend tenta interpretar numeros puros como TIC. Para evitar duvida, o ideal e escrever TIC 123... ou KIC 123....",
  },
  {
    question: "O que significa um score alto?",
    answer:
      "Significa que o padrao visto pelo modelo se parece com um transito planetario. Ainda assim, o resultado precisa ser lido junto com o BLS e a curva.",
  },
  {
    question: "O que significa um score baixo?",
    answer:
      "Significa que o modelo nao viu um padrao claro de transito naquela curva processada. Nao e prova absoluta de ausencia de planeta.",
  },
  {
    question: "Qual e a diferenca entre classico e QML?",
    answer:
      "O modo classico faz a triagem principal. O modo QML experimental entra apenas em casos ambiguos para tentar corrigir a decisao classica.",
  },
];
