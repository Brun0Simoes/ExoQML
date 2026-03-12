import {
  Aperture,
  Cpu,
  Database,
  Download,
  Gauge,
  History,
  Layers3,
  LoaderCircle,
  MapPinned,
  Orbit,
  Radar,
  Search,
  ShieldAlert,
  Telescope,
  Zap,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, analyzeTarget, exportUrl, getAnalysis, listHistory, listTargetCatalog } from "./api";
import { GUIDE_TARGETS, SKY_REFERENCE_POINTS, type GuideTarget } from "./guideData";
import { PLATFORM_STATS, PRODUCTION_STACK, WORKFLOW_OVERVIEW } from "./platformData";
import type { AnalysisResponse, HistoryItem, SeriesPoint, SkyCoordinates, TargetCatalogItem, TargetType } from "./types";

type InputType = TargetType | "auto";
type CatalogScope = "all" | "guide" | "candidate";
type MissionFilter = "all" | "Kepler" | "TESS";
type SkyMapTarget = {
  id: string;
  label: string;
  mission: string;
  coordinates: SkyCoordinates;
  markerTone: "preview" | "analysis";
};

const TARGET_OPTIONS: Array<{ value: InputType; label: string }> = [
  { value: "auto", label: "Detectar" },
  { value: "tic", label: "TIC" },
  { value: "kic", label: "KIC" },
  { value: "name", label: "Nome" },
];

const QUICK_TARGETS = GUIDE_TARGETS.map((target) => ({
  id: target.id,
  type: target.type,
  label: target.title,
}));

const PAGE_SIZE = 60;

function sample(points: SeriesPoint[], maxPoints = 420): SeriesPoint[] {
  if (points.length <= maxPoints) {
    return points;
  }
  const step = Math.ceil(points.length / maxPoints);
  return points.filter((_, index) => index % step === 0);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) {
    return "n/a";
  }
  return new Date(value).toLocaleString("pt-BR");
}

function normalizeSearch(value: string): string {
  return value.trim().toLowerCase();
}

function humanizePredictionLabel(label: string): string {
  const normalized = label.toLowerCase();
  if (normalized === "non_transit") {
    return "Sem indicio de transito";
  }
  if (normalized === "transit" || normalized === "transit_candidate") {
    return "Sinal de transito";
  }
  return label.replace(/_/g, " ");
}

function predictionTone(label: string): string {
  return label.toLowerCase().startsWith("non")
    ? "border-lime-300/25 bg-lime-300/10 text-lime-300"
    : "border-amber-400/30 bg-amber-400/12 text-amber-300";
}

function targetTypeLabel(value: TargetType): string {
  if (value === "name") {
    return "Nome";
  }
  return value.toUpperCase();
}

function humanizeStatus(value: string): string {
  const normalized = value.toLowerCase();
  if (normalized === "success" || normalized === "completed") {
    return "Concluida";
  }
  if (normalized === "running" || normalized === "processing") {
    return "Em execucao";
  }
  if (normalized === "failed" || normalized === "error") {
    return "Falha";
  }
  return value;
}

function missionBucket(value: string): MissionFilter | "other" {
  const normalized = value.toLowerCase();
  if (normalized.includes("kepler")) {
    return "Kepler";
  }
  if (normalized.includes("tess")) {
    return "TESS";
  }
  return "other";
}

function coordinatesLabel(coordinates: SkyCoordinates | null | undefined): string {
  if (!coordinates) {
    return "Coordenadas indisponiveis";
  }
  return `RA ${coordinates.ra.toFixed(3)} deg  DEC ${coordinates.dec.toFixed(3)} deg`;
}

function catalogMatchScore(item: TargetCatalogItem, query: string): number {
  if (!query) {
    return 0;
  }

  const haystacks = [
    item.query,
    item.target_id,
    item.display_name,
    item.summary,
    item.mission,
    `${targetTypeLabel(item.target_type)} ${item.target_id}`,
  ].map((value) => value.toLowerCase());

  if (haystacks[0] === query || haystacks[1] === query || haystacks[2] === query) {
    return 0;
  }
  if (haystacks.some((value) => value.startsWith(query))) {
    return 1;
  }
  if (haystacks.some((value) => value.includes(query))) {
    return 2;
  }
  return 99;
}

function isGuideCatalogItem(item: TargetCatalogItem): boolean {
  return GUIDE_TARGETS.some((target) => normalizeSearch(target.id) === normalizeSearch(item.query));
}

function MetricCard(props: { label: string; value: string; detail: string }) {
  return (
    <div className="metric-card animate-rise-in">
      <p className="text-[0.72rem] uppercase tracking-[0.22em] text-starlight-300">{props.label}</p>
      <p className="safe-wrap mt-3 text-xl font-semibold tracking-[-0.04em] text-white sm:text-2xl">{props.value}</p>
      <p className="safe-wrap mt-2 text-sm leading-6 text-starlight-300">{props.detail}</p>
    </div>
  );
}

function SeriesChart(props: { series: SeriesPoint[]; relevance: SeriesPoint[] }) {
  const series = useMemo(() => sample(props.series), [props.series]);
  const relevance = useMemo(() => sample(props.relevance), [props.relevance]);

  if (series.length < 2) {
    return (
      <div className="flex h-[22rem] items-center justify-center rounded-[22px] border border-dashed border-white/10 bg-white/[0.02] text-sm text-starlight-300">
        Nao ha pontos suficientes para exibir a curva.
      </div>
    );
  }

  const width = 1120;
  const height = 420;
  const pad = { top: 28, right: 24, bottom: 36, left: 26 };
  const xMin = Math.min(...series.map((point) => point.x));
  const xMax = Math.max(...series.map((point) => point.x));
  const yMin = Math.min(...series.map((point) => point.y));
  const yMax = Math.max(...series.map((point) => point.y));
  const relevanceMax = Math.max(...relevance.map((point) => point.y), 1e-9);

  const scaleX = (x: number) => {
    const normalized = (x - xMin) / (xMax - xMin || 1);
    return pad.left + normalized * (width - pad.left - pad.right);
  };

  const scaleY = (y: number) => {
    const normalized = (y - yMin) / (yMax - yMin || 1);
    return height - pad.bottom - normalized * (height - pad.top - pad.bottom);
  };

  const linePoints = series.map((point) => `${scaleX(point.x)},${scaleY(point.y)}`).join(" ");
  const horizontalGrid = Array.from({ length: 5 }, (_, index) => pad.top + (index / 4) * (height - pad.top - pad.bottom));
  const verticalGrid = Array.from({ length: 6 }, (_, index) => pad.left + (index / 5) * (width - pad.left - pad.right));

  return (
    <div className="chart-shell">
      <svg className="h-auto w-full" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Curva de luz e relevancia">
        <defs>
          <linearGradient id="signalLine" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#86f0ff" />
            <stop offset="100%" stopColor="#4fd8f2" />
          </linearGradient>
          <linearGradient id="heatColumns" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="rgba(255,171,92,0)" />
            <stop offset="100%" stopColor="rgba(255,171,92,0.68)" />
          </linearGradient>
        </defs>

        <rect x="0" y="0" width={width} height={height} rx="26" fill="rgba(3, 10, 24, 0.75)" />

        {horizontalGrid.map((y) => (
          <line key={`h-${y}`} x1={pad.left} y1={y} x2={width - pad.right} y2={y} stroke="rgba(255,255,255,0.08)" strokeDasharray="4 8" />
        ))}

        {verticalGrid.map((x) => (
          <line key={`v-${x}`} x1={x} y1={pad.top} x2={x} y2={height - pad.bottom} stroke="rgba(255,255,255,0.06)" strokeDasharray="4 10" />
        ))}

        {relevance.map((point) => {
          const x = scaleX(point.x);
          const columnHeight = (point.y / relevanceMax) * (height - pad.top - pad.bottom) * 0.42;
          return (
            <line
              key={`rel-${point.x}`}
              x1={x}
              y1={height - pad.bottom}
              x2={x}
              y2={height - pad.bottom - columnHeight}
              stroke="url(#heatColumns)"
              strokeOpacity="0.82"
              strokeWidth="2.6"
            />
          );
        })}

        <polyline fill="none" stroke="url(#signalLine)" strokeWidth="2.3" points={linePoints} />

        <text x={pad.left} y="18" fill="rgba(148,171,200,0.88)" fontSize="12" letterSpacing="2.6">
          FLUXO NORMALIZADO
        </text>
        <text x={width - pad.right - 84} y={height - 10} fill="rgba(148,171,200,0.88)" fontSize="12" letterSpacing="2.6">
          TEMPO
        </text>
      </svg>
    </div>
  );
}

function SkyReferenceMap(props: {
  targets: GuideTarget[];
  selectedTargetId: string;
  previewTarget?: SkyMapTarget | null;
  analysisTarget?: SkyMapTarget | null;
}) {
  const width = 860;
  const height = 420;
  const pad = { top: 36, right: 24, bottom: 34, left: 40 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  const xForRa = (ra: number) => pad.left + (1 - ra / 360) * plotWidth;
  const yForDec = (dec: number) => pad.top + ((90 - dec) / 180) * plotHeight;

  const selected = props.targets.find((target) => target.id === props.selectedTargetId) ?? props.targets[0];
  const latitudeBands = [-60, -30, 0, 30, 60];
  const hourLines = Array.from({ length: 7 }, (_, index) => index * 60);
  const previewTarget = props.previewTarget ?? null;
  const analysisTarget = props.analysisTarget ?? null;

  return (
    <div className="atlas-map-shell">
      <svg
        className="h-auto w-full"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Mapa simplificado do ceu com referencias e alvo selecionado"
      >
        <defs>
          <linearGradient id="atlasGlow" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="rgba(79,216,242,0.16)" />
            <stop offset="100%" stopColor="rgba(73,212,155,0.08)" />
          </linearGradient>
        </defs>

        <rect x="0" y="0" width={width} height={height} rx="28" fill="rgba(3, 10, 24, 0.86)" />
        <rect x="0" y="0" width={width} height={height} rx="28" fill="url(#atlasGlow)" />

        {latitudeBands.map((dec) => (
          <g key={`lat-${dec}`}>
            <line
              x1={pad.left}
              y1={yForDec(dec)}
              x2={width - pad.right}
              y2={yForDec(dec)}
              stroke="rgba(255,255,255,0.10)"
              strokeDasharray="4 8"
            />
            <text x="8" y={yForDec(dec) + 4} fill="rgba(148,171,200,0.8)" fontSize="11">
              {`${dec} deg`}
            </text>
          </g>
        ))}

        {hourLines.map((ra) => (
          <g key={`ra-${ra}`}>
            <line
              x1={xForRa(ra)}
              y1={pad.top}
              x2={xForRa(ra)}
              y2={height - pad.bottom}
              stroke="rgba(255,255,255,0.08)"
              strokeDasharray="4 8"
            />
            <text x={xForRa(ra) - 12} y={height - 10} fill="rgba(148,171,200,0.8)" fontSize="11">
              {`${Math.round(ra / 15)}h`}
            </text>
          </g>
        ))}

        {SKY_REFERENCE_POINTS.map((point) => (
          <g key={point.label}>
            <circle
              cx={xForRa(point.ra)}
              cy={yForDec(point.dec)}
              r={point.kind === "region" ? 5 : 3.4}
              fill={point.kind === "region" ? "rgba(255,171,92,0.9)" : "rgba(134,240,255,0.9)"}
            />
            <text x={xForRa(point.ra) + 8} y={yForDec(point.dec) - 8} fill="rgba(212,229,255,0.92)" fontSize="11">
              {point.label}
            </text>
          </g>
        ))}

        {props.targets.map((target) => {
          const active = target.id === selected.id;
          return (
            <g key={target.id}>
              <circle
                cx={xForRa(target.coordinates.ra)}
                cy={yForDec(target.coordinates.dec)}
                r={active ? 6.8 : 5}
                fill={active ? "#4fd8f2" : "rgba(134,240,255,0.72)"}
                stroke={active ? "rgba(255,255,255,0.72)" : "rgba(255,255,255,0.24)"}
              />
              <text
                x={xForRa(target.coordinates.ra) + 10}
                y={yForDec(target.coordinates.dec) + (active ? -12 : 16)}
                fill={active ? "rgba(255,255,255,0.96)" : "rgba(212,229,255,0.76)"}
                fontSize={active ? "12" : "11"}
              >
                {target.id}
              </text>
            </g>
          );
        })}

        {previewTarget && (
          <g key={previewTarget.id}>
            <circle
              cx={xForRa(previewTarget.coordinates.ra)}
              cy={yForDec(previewTarget.coordinates.dec)}
              r={18}
              fill="rgba(255,171,92,0.10)"
              stroke="rgba(255,171,92,0.42)"
            />
            <circle
              cx={xForRa(previewTarget.coordinates.ra)}
              cy={yForDec(previewTarget.coordinates.dec)}
              r={7.5}
              fill="#ffab5c"
              stroke="rgba(255,255,255,0.72)"
            />
            <text
              x={xForRa(previewTarget.coordinates.ra) + 10}
              y={yForDec(previewTarget.coordinates.dec) - 12}
              fill="rgba(255,255,255,0.96)"
              fontSize="12"
            >
              {previewTarget.label}
            </text>
          </g>
        )}

        {analysisTarget && (
          <g key={analysisTarget.id}>
            <circle
              cx={xForRa(analysisTarget.coordinates.ra)}
              cy={yForDec(analysisTarget.coordinates.dec)}
              r={19}
              fill="rgba(73,212,155,0.12)"
              stroke="rgba(73,212,155,0.42)"
            />
            <circle
              cx={xForRa(analysisTarget.coordinates.ra)}
              cy={yForDec(analysisTarget.coordinates.dec)}
              r={8}
              fill="#49d49b"
              stroke="rgba(255,255,255,0.74)"
            />
            <text
              x={xForRa(analysisTarget.coordinates.ra) + 10}
              y={yForDec(analysisTarget.coordinates.dec) + 18}
              fill="rgba(255,255,255,0.98)"
              fontSize="12"
            >
              {analysisTarget.label}
            </text>
          </g>
        )}
      </svg>

      <div className="mt-3 flex flex-wrap gap-2 text-xs text-starlight-300">
        <span className="orbit-chip border-signal-400/30 bg-signal-400/10 text-white">Azul: referencias guiadas</span>
        {previewTarget && <span className="orbit-chip border-amber-400/30 bg-amber-400/10 text-white">Laranja: alvo selecionado</span>}
        {analysisTarget && <span className="orbit-chip border-aurora-400/30 bg-aurora-400/10 text-white">Verde: alvo analisado</span>}
      </div>
    </div>
  );
}

function App() {
  const [targetId, setTargetId] = useState("");
  const [targetType, setTargetType] = useState<InputType>("auto");
  const [selectedGuideTargetId, setSelectedGuideTargetId] = useState(GUIDE_TARGETS[0]?.id ?? "");
  const [catalogScope, setCatalogScope] = useState<CatalogScope>("all");
  const [missionFilter, setMissionFilter] = useState<MissionFilter>("all");
  const [visibleCatalogCount, setVisibleCatalogCount] = useState(PAGE_SIZE);
  const [catalog, setCatalog] = useState<TargetCatalogItem[]>([]);
  const [isCatalogLoading, setIsCatalogLoading] = useState(false);
  const [catalogLoadError, setCatalogLoadError] = useState<string | null>(null);
  const [experimentalQml, setExperimentalQml] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<{ message: string; suggestion?: string; code?: string; stage?: string } | null>(null);
  const resultSectionRef = useRef<HTMLElement | null>(null);

  const normalizedTargetInput = useMemo(() => normalizeSearch(targetId), [targetId]);
  const preprocessEntries = useMemo(() => (analysis ? Object.entries(analysis.preprocess_params) : []), [analysis]);
  const topPeak = analysis?.bls_peaks[0] ?? null;
  const experimentalComparison = analysis?.experimental_comparison ?? null;
  const selectedGuideTarget = useMemo(
    () => GUIDE_TARGETS.find((target) => target.id === selectedGuideTargetId) ?? GUIDE_TARGETS[0],
    [selectedGuideTargetId],
  );
  const totalCandidateTargets = useMemo(
    () => catalog.filter((item) => item.positive_tce_count > 0).length,
    [catalog],
  );

  const filteredCatalog = useMemo(() => {
    const ranked = catalog
      .filter((item) => {
        if (catalogScope === "guide" && !isGuideCatalogItem(item)) {
          return false;
        }
        if (catalogScope === "candidate" && item.positive_tce_count <= 0) {
          return false;
        }
        if (missionFilter !== "all" && missionBucket(item.mission) !== missionFilter) {
          return false;
        }
        return true;
      })
      .map((item) => ({
        item,
        score: catalogMatchScore(item, normalizedTargetInput),
      }))
      .filter((entry) => entry.score < 99)
      .sort((left, right) => {
        if (left.score !== right.score) {
          return left.score - right.score;
        }
        if (isGuideCatalogItem(left.item) !== isGuideCatalogItem(right.item)) {
          return isGuideCatalogItem(left.item) ? -1 : 1;
        }
        if (left.item.positive_tce_count !== right.item.positive_tce_count) {
          return right.item.positive_tce_count - left.item.positive_tce_count;
        }
        if (left.item.tce_count !== right.item.tce_count) {
          return right.item.tce_count - left.item.tce_count;
        }
        return left.item.display_name.localeCompare(right.item.display_name, "pt-BR");
      });

    return ranked.map((entry) => entry.item);
  }, [catalog, catalogScope, missionFilter, normalizedTargetInput]);

  const visibleCatalogItems = useMemo(
    () => filteredCatalog.slice(0, visibleCatalogCount),
    [filteredCatalog, visibleCatalogCount],
  );

  const focusedCatalogItem = useMemo(() => {
    if (filteredCatalog.length === 0) {
      return null;
    }

    const exact = filteredCatalog.find((item) => {
      const normalizedQuery = normalizeSearch(item.query);
      const normalizedName = normalizeSearch(item.display_name);
      const normalizedIdentifier = normalizeSearch(`${targetTypeLabel(item.target_type)} ${item.target_id}`);
      return (
        normalizedQuery === normalizedTargetInput ||
        normalizedName === normalizedTargetInput ||
        normalizedIdentifier === normalizedTargetInput
      );
    });

    return exact ?? filteredCatalog[0];
  }, [filteredCatalog, normalizedTargetInput]);

  const previewSkyTarget = useMemo<SkyMapTarget | null>(() => {
    if (focusedCatalogItem?.sky_coordinates) {
      return {
        id: `preview-${focusedCatalogItem.query}`,
        label: focusedCatalogItem.query,
        mission: focusedCatalogItem.mission,
        coordinates: focusedCatalogItem.sky_coordinates,
        markerTone: "preview",
      };
    }

    if (selectedGuideTarget?.coordinates) {
      return {
        id: `guide-${selectedGuideTarget.id}`,
        label: selectedGuideTarget.id,
        mission: selectedGuideTarget.mission,
        coordinates: selectedGuideTarget.coordinates,
        markerTone: "preview",
      };
    }

    return null;
  }, [focusedCatalogItem, selectedGuideTarget]);

  const analysisSkyTarget = useMemo<SkyMapTarget | null>(() => {
    const coordinates = analysis?.provenance.sky_coordinates;
    if (!analysis || !coordinates) {
      return null;
    }
    return {
      id: `analysis-${analysis.target_type}-${analysis.target_id}`,
      label: `${targetTypeLabel(analysis.target_type)} ${analysis.target_id}`,
      mission: analysis.provenance.mission,
      coordinates,
      markerTone: "analysis",
    };
  }, [analysis]);

  useEffect(() => {
    void loadCatalog();
    void loadHistory();
  }, []);

  useEffect(() => {
    setVisibleCatalogCount(PAGE_SIZE);
  }, [normalizedTargetInput, catalogScope, missionFilter]);

  useEffect(() => {
    if (!analysis) {
      return;
    }
    resultSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [analysis]);

  async function loadCatalog() {
    setIsCatalogLoading(true);
    setCatalogLoadError(null);
    try {
      const items = await listTargetCatalog(10000);
      setCatalog(items);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Falha ao carregar o catalogo local.";
      setCatalogLoadError(message);
    } finally {
      setIsCatalogLoading(false);
    }
  }

  async function loadHistory() {
    try {
      const items = await listHistory(12);
      setHistory(items);
    } catch {
      setHistory([]);
    }
  }

  function syncGuideSelection(value: string) {
    const guideTarget = GUIDE_TARGETS.find((target) => normalizeSearch(target.id) === normalizeSearch(value));
    if (guideTarget) {
      setSelectedGuideTargetId(guideTarget.id);
    }
  }

  function pickGuideTarget(target: GuideTarget) {
    setTargetId(target.id);
    setTargetType(target.type);
    setSelectedGuideTargetId(target.id);
    setError(null);
  }

  function pickCatalogItem(item: TargetCatalogItem) {
    setTargetId(item.query);
    setTargetType(item.target_type);
    syncGuideSelection(item.query);
    setError(null);
  }

  async function submitAnalysis(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const cleaned = targetId.trim();
    if (!cleaned) {
      setError({
        message: "Escolha um alvo no catalogo ou digite um identificador valido antes de iniciar.",
        suggestion: "Use a lista abaixo ou digite um nome conhecido, um TIC ou um KIC.",
      });
      return;
    }

    setIsBusy(true);
    setError(null);

    try {
      const response = await analyzeTarget({
        target_id: cleaned,
        target_type: targetType === "auto" ? undefined : targetType,
        experimental_qml: experimentalQml,
      });
      setAnalysis(response);
      await loadHistory();
    } catch (caught) {
      if (caught instanceof ApiError) {
        setError({
          message: caught.message,
          suggestion: caught.suggestion,
          code: caught.code,
          stage: caught.stage,
        });
      } else {
        setError({
          message: "A analise nao foi concluida.",
          suggestion: "Tente outro alvo ou rode novamente em alguns segundos.",
        });
      }
    } finally {
      setIsBusy(false);
    }
  }

  async function openHistoryItem(id: number) {
    setIsBusy(true);
    setError(null);

    try {
      const response = await getAnalysis(id);
      setAnalysis(response);
      setTargetId(response.target_id);
      setTargetType(response.target_type);
      syncGuideSelection(response.target_id);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Nao foi possivel abrir a analise.";
      setError({
        message,
        suggestion: "Atualize a pagina ou selecione outra analise do historico.",
      });
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <main className="min-h-screen overflow-x-hidden pb-10">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
        <section className="grid gap-6 xl:grid-cols-[minmax(0,1.28fr)_minmax(320px,0.72fr)]">
          <article className="panel p-5 sm:p-8">
            <div className="flex flex-col gap-6">
              <div className="flex flex-col gap-4">
                <div className="max-w-3xl">
                  <div className="eyebrow">
                    <Telescope className="h-3.5 w-3.5" />
                    Plataforma de triagem em producao
                  </div>
                  <h1 className="mt-4 text-3xl font-semibold tracking-[-0.05em] text-white sm:text-4xl xl:text-[3.35rem]">
                    Escolha um alvo, rode a analise e veja o modelo funcionando de ponta a ponta.
                  </h1>
                  <p className="mt-4 max-w-2xl text-sm leading-7 text-starlight-300 sm:text-base">
                    Catalogo local amplo, pipeline cientifico com BLS, classificador principal calibrado e segunda etapa QML pronta para casos ambiguos.
                  </p>
                </div>

                <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-4">
                  {PLATFORM_STATS.map((item) => (
                    <MetricCard key={item.label} label={item.label} value={item.value} detail={item.detail} />
                  ))}
                </div>
              </div>

              <form className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_190px_auto]" onSubmit={submitAnalysis}>
                <div className="min-w-0">
                  <label htmlFor="target-id" className="field-label">
                    Identificador do alvo
                  </label>
                  <div className="relative">
                    <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-starlight-300" />
                    <input
                      id="target-id"
                      className="field-control pl-11"
                      list="target-catalog-options"
                      value={targetId}
                      onChange={(event) => {
                        const nextValue = event.target.value;
                        setTargetId(nextValue);
                        syncGuideSelection(nextValue);
                      }}
                      placeholder="Ex.: Kepler-10, TIC 25155310, KIC 10000490"
                      autoComplete="off"
                    />
                    <datalist id="target-catalog-options">
                      {catalog.map((item) => (
                        <option key={`${item.target_type}-${item.target_id}-${item.query}`} value={item.query}>
                          {item.display_name}
                        </option>
                      ))}
                    </datalist>
                  </div>
                  <p className="mt-2 text-sm text-starlight-300">
                    O campo aceita nome, TIC e KIC. A lista abaixo mostra o catalogo local disponivel para selecao direta.
                  </p>
                </div>

                <div className="min-w-0">
                  <label htmlFor="target-type" className="field-label">
                    Tipo de entrada
                  </label>
                  <select
                    id="target-type"
                    className="field-control"
                    value={targetType}
                    onChange={(event) => setTargetType(event.target.value as InputType)}
                  >
                    {TARGET_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="min-w-0">
                  <label className="field-label">Execucao</label>
                  <button type="submit" className="cta-button w-full" disabled={isBusy}>
                    {isBusy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Radar className="h-4 w-4" />}
                    {isBusy ? "Processando" : "Analisar alvo"}
                  </button>
                </div>
              </form>

              <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.72fr)]">
                <div className="panel-hover rounded-[24px] border border-white/10 bg-ink-900/65 p-4">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <p className="text-[0.72rem] uppercase tracking-[0.22em] text-starlight-300">Alvo em foco</p>
                      <p className="safe-wrap mt-3 text-2xl font-semibold tracking-[-0.04em] text-white">
                        {focusedCatalogItem?.display_name ?? (targetId.trim() || "Digite ou selecione um alvo")}
                      </p>
                    </div>
                    {focusedCatalogItem && (
                      <div className="flex flex-wrap gap-2 text-xs text-starlight-200">
                        <span className="orbit-chip border-white/10 bg-white/[0.04]">{targetTypeLabel(focusedCatalogItem.target_type)}</span>
                        <span className="orbit-chip border-white/10 bg-white/[0.04]">{focusedCatalogItem.mission}</span>
                        {focusedCatalogItem.positive_tce_count > 0 && (
                          <span className="orbit-chip border-amber-400/30 bg-amber-400/10 text-white">
                            {focusedCatalogItem.positive_tce_count} TCE PC
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    <div className="rounded-[22px] border border-white/10 bg-white/[0.03] p-4">
                      <p className="text-[0.72rem] uppercase tracking-[0.22em] text-starlight-300">Resumo</p>
                      <p className="safe-wrap mt-3 text-sm leading-6 text-starlight-200">
                        {focusedCatalogItem?.summary ?? "O catalogo local ainda esta carregando ou nao ha correspondencia para o termo digitado."}
                      </p>
                    </div>

                    <div className="rounded-[22px] border border-white/10 bg-white/[0.03] p-4">
                      <p className="text-[0.72rem] uppercase tracking-[0.22em] text-starlight-300">Dados prontos</p>
                      <div className="mt-3 grid gap-3 sm:grid-cols-2">
                        <div className="data-knot">
                          <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Consulta</p>
                          <p className="safe-wrap mt-2 font-mono text-white">{focusedCatalogItem?.query ?? "n/a"}</p>
                        </div>
                        <div className="data-knot">
                          <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Coordenadas</p>
                          <p className="safe-wrap mt-2 text-sm text-starlight-200">
                            {coordinatesLabel(focusedCatalogItem?.sky_coordinates ?? null)}
                          </p>
                        </div>
                        <div className="data-knot">
                          <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">TCEs</p>
                          <p className="mt-2 font-mono text-white">{focusedCatalogItem?.tce_count ?? "n/a"}</p>
                        </div>
                        <div className="data-knot">
                          <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">QML experimental</p>
                          <label className="mt-2 inline-flex cursor-pointer items-center gap-3 text-sm text-starlight-200">
                            <input
                              type="checkbox"
                              className="h-4 w-4 rounded border-white/20 bg-transparent"
                              checked={experimentalQml}
                              onChange={(event) => setExperimentalQml(event.target.checked)}
                            />
                            Revisar casos ambiguos com a segunda etapa QML
                          </label>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="panel-hover rounded-[24px] border border-white/10 bg-ink-900/65 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-[0.72rem] uppercase tracking-[0.22em] text-starlight-300">Atalhos de inicio</p>
                      <p className="mt-2 text-sm text-starlight-300">Um clique para preencher alvos que ja validam bem a plataforma.</p>
                    </div>
                    <span className="orbit-chip border-white/10 bg-white/[0.04]">{QUICK_TARGETS.length} exemplos</span>
                  </div>

                  <div className="mt-4 flex flex-wrap gap-2">
                    {QUICK_TARGETS.map((target) => (
                      <button key={target.id} type="button" className="quick-target" onClick={() => pickGuideTarget(GUIDE_TARGETS.find((item) => item.id === target.id) ?? GUIDE_TARGETS[0])}>
                        {target.id}
                      </button>
                    ))}
                  </div>

                  <div className="mt-4 rounded-[22px] border border-white/10 bg-white/[0.03] p-4">
                    <p className="text-[0.72rem] uppercase tracking-[0.22em] text-starlight-300">Exemplo ativo</p>
                    <p className="mt-2 text-lg font-semibold text-white">{selectedGuideTarget.title}</p>
                    <p className="mt-2 text-sm leading-6 text-starlight-300">{selectedGuideTarget.whyItMatters}</p>
                  </div>
                </div>
              </div>
            </div>
          </article>

          <aside className="space-y-6">
            <article className="panel p-5 sm:p-7">
              <div className="eyebrow">
                <Cpu className="h-3.5 w-3.5" />
                Stack em producao
              </div>
              <div className="mt-5 grid gap-3">
                {PRODUCTION_STACK.map((item) => (
                  <div key={item.title} className="stack-card">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-[0.72rem] uppercase tracking-[0.2em] text-starlight-300">{item.title}</p>
                        <p className="mt-2 text-lg font-semibold text-white">{item.value}</p>
                      </div>
                      <span className="orbit-chip border-aurora-400/25 bg-aurora-400/10 text-white">Ativo</span>
                    </div>
                    <p className="mt-2 text-sm leading-6 text-starlight-300">{item.detail}</p>
                  </div>
                ))}
              </div>
            </article>

            <article className="panel p-5 sm:p-7">
              <div className="eyebrow">
                <MapPinned className="h-3.5 w-3.5" />
                Mapa do alvo
              </div>
              <div className="mt-5">
                <SkyReferenceMap
                  targets={GUIDE_TARGETS}
                  selectedTargetId={selectedGuideTarget.id}
                  previewTarget={previewSkyTarget}
                  analysisTarget={analysisSkyTarget}
                />
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div className="data-knot">
                  <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Selecionado</p>
                  <p className="safe-wrap mt-2 text-sm text-starlight-200">{coordinatesLabel(previewSkyTarget?.coordinates ?? null)}</p>
                </div>
                <div className="data-knot">
                  <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Analisado</p>
                  <p className="safe-wrap mt-2 text-sm text-starlight-200">{coordinatesLabel(analysisSkyTarget?.coordinates ?? null)}</p>
                </div>
              </div>
            </article>

            <article className="panel p-5 sm:p-7">
              <div className="eyebrow">
                <Layers3 className="h-3.5 w-3.5" />
                Fluxo da plataforma
              </div>
              <div className="mt-5 space-y-3">
                {WORKFLOW_OVERVIEW.map((step, index) => (
                  <div key={step.title} className="workflow-row">
                    <div className="workflow-index">{index + 1}</div>
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-white">{step.title}</p>
                      <p className="mt-1 text-sm leading-6 text-starlight-300">{step.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            </article>
          </aside>
        </section>

        <section className="panel p-5 sm:p-8">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <div className="eyebrow">
                <Database className="h-3.5 w-3.5" />
                Catalogo local de alvos
              </div>
              <h2 className="mt-4 text-2xl font-semibold tracking-[-0.04em] text-white">Selecione pelo nome ou pelo catalogo completo.</h2>
              <p className="mt-3 text-sm leading-7 text-starlight-300">
                O campo acima aceita digitacao livre, mas a lista abaixo permite navegar por todos os alvos carregados localmente antes de rodar a analise.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <MetricCard label="Catalogo carregado" value={`${catalog.length || 0}`} detail="Entradas locais disponiveis agora." />
              <MetricCard label="Com candidato" value={`${totalCandidateTargets}`} detail="Alvos com pelo menos um TCE positivo no dataset." />
              <MetricCard label="Historico" value={`${history.length}`} detail="Analises recentes prontas para reabrir." />
            </div>
          </div>

          <div className="catalog-toolbar mt-6">
            <div className="flex flex-wrap gap-2">
              {[
                { value: "all", label: "Tudo" },
                { value: "candidate", label: "Com candidato" },
                { value: "guide", label: "Exemplos guiados" },
              ].map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className={`filter-chip ${catalogScope === option.value ? "is-active" : ""}`}
                  onClick={() => setCatalogScope(option.value as CatalogScope)}
                >
                  {option.label}
                </button>
              ))}
            </div>

            <div className="flex flex-wrap gap-2">
              {[
                { value: "all", label: "Todas as missoes" },
                { value: "Kepler", label: "Kepler" },
                { value: "TESS", label: "TESS" },
              ].map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className={`filter-chip ${missionFilter === option.value ? "is-active" : ""}`}
                  onClick={() => setMissionFilter(option.value as MissionFilter)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3 text-sm text-starlight-300">
            <span className="orbit-chip border-white/10 bg-white/[0.04]">
              Mostrando {visibleCatalogItems.length} de {filteredCatalog.length}
            </span>
            <span className="orbit-chip border-white/10 bg-white/[0.04]">
              Busca atual: {targetId.trim() ? `"${targetId.trim()}"` : "sem filtro"}
            </span>
            {catalogLoadError && <span className="orbit-chip border-amber-400/25 bg-amber-400/10 text-white">{catalogLoadError}</span>}
          </div>

          {error && (
            <div className="mt-6 rounded-[24px] border border-rose-300/20 bg-rose-300/8 p-4 text-sm text-rose-300">
              <p className="font-medium text-white">{error.message}</p>
              {error.suggestion && <p className="mt-2 text-rose-300">{error.suggestion}</p>}
              {(error.code || error.stage) && (
                <p className="mt-2 font-mono text-xs uppercase tracking-[0.18em] text-rose-300/90">
                  {error.code ?? "erro"} {error.stage ? `- ${error.stage}` : ""}
                </p>
              )}
            </div>
          )}

          {isCatalogLoading ? (
            <div className="mt-6 flex items-center gap-3 rounded-[24px] border border-white/10 bg-white/[0.03] p-5 text-sm text-starlight-200">
              <LoaderCircle className="h-4 w-4 animate-spin" />
              Carregando catalogo local.
            </div>
          ) : (
            <div className="mt-6 grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
              {visibleCatalogItems.map((item) => {
                const isSelected = item.query === focusedCatalogItem?.query;
                return (
                  <button
                    type="button"
                    key={`${item.target_type}-${item.target_id}-${item.query}`}
                    className={`catalog-entry text-left ${isSelected ? "border-signal-400/35 bg-signal-400/10" : ""}`}
                    onClick={() => pickCatalogItem(item)}
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="safe-wrap text-lg font-semibold text-white">{item.display_name}</p>
                        <p className="safe-wrap mt-1 font-mono text-sm text-starlight-300">{item.query}</p>
                      </div>
                      <div className="flex flex-wrap gap-2 text-xs">
                        <span className="orbit-chip border-white/10 bg-white/[0.04]">{targetTypeLabel(item.target_type)}</span>
                        <span className="orbit-chip border-white/10 bg-white/[0.04]">{missionBucket(item.mission)}</span>
                        {item.positive_tce_count > 0 && (
                          <span className="orbit-chip border-amber-400/30 bg-amber-400/10 text-white">
                            {item.positive_tce_count} PC
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="safe-wrap mt-3 text-sm leading-6 text-starlight-300">{item.summary}</p>
                    <div className="mt-4 grid gap-2 sm:grid-cols-2">
                      <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2 text-sm text-starlight-200">
                        <span className="text-starlight-300">Missao:</span> {item.mission}
                      </div>
                      <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2 text-sm text-starlight-200">
                        <span className="text-starlight-300">TCEs:</span> {item.tce_count}
                      </div>
                      <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2 text-sm text-starlight-200 sm:col-span-2">
                        <span className="text-starlight-300">Posicao:</span> {coordinatesLabel(item.sky_coordinates)}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {!isCatalogLoading && filteredCatalog.length === 0 && (
            <div className="mt-6 rounded-[24px] border border-dashed border-white/10 bg-white/[0.02] p-5 text-sm text-starlight-300">
              Nenhum alvo foi encontrado com esse filtro. Tente digitar menos texto ou troque a missao.
            </div>
          )}

          {!isCatalogLoading && visibleCatalogItems.length < filteredCatalog.length && (
            <div className="mt-6 flex justify-center">
              <button type="button" className="cta-button" onClick={() => setVisibleCatalogCount((value) => value + PAGE_SIZE)}>
                Mostrar mais {Math.min(PAGE_SIZE, filteredCatalog.length - visibleCatalogItems.length)} itens
              </button>
            </div>
          )}
        </section>

        {analysis && (
          <section ref={resultSectionRef} className="grid gap-6 2xl:grid-cols-[minmax(0,1.25fr)_minmax(320px,0.75fr)]">
            <div className="space-y-6">
              <article className="panel p-5 sm:p-8">
                <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
                  <div className="min-w-0">
                    <div className="eyebrow">
                      <Radar className="h-3.5 w-3.5" />
                      Resultado atual
                    </div>
                    <h2 className="safe-wrap mt-4 text-3xl font-semibold tracking-[-0.05em] text-white">
                      {targetTypeLabel(analysis.target_type)} {analysis.target_id}
                    </h2>
                    <div className="mt-3 flex flex-wrap items-center gap-2 text-sm text-starlight-300">
                      <span className="orbit-chip border-white/10 bg-white/[0.04]">{analysis.provenance.mission}</span>
                      <span className="orbit-chip border-white/10 bg-white/[0.04]">{analysis.model_name}</span>
                      <span className={`orbit-chip ${predictionTone(analysis.prediction_label)}`}>
                        {humanizePredictionLabel(analysis.prediction_label)}
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-3">
                    <a href={exportUrl(analysis.id, "json")} target="_blank" rel="noreferrer" className="cta-button">
                      <Download className="h-4 w-4" />
                      Exportar JSON
                    </a>
                    <a href={exportUrl(analysis.id, "csv")} target="_blank" rel="noreferrer" className="cta-button">
                      <Download className="h-4 w-4" />
                      Exportar CSV
                    </a>
                  </div>
                </div>

                <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  <MetricCard label="Score" value={formatPercent(analysis.prediction_score)} detail="Probabilidade estimada pelo modelo principal." />
                  <MetricCard
                    label="Melhor BLS"
                    value={analysis.bls_period == null ? "n/a" : `${formatNumber(analysis.bls_period)} d`}
                    detail="Periodo classico mais forte encontrado pelo baseline BLS."
                  />
                  <MetricCard label="Modelo" value={analysis.model_version} detail="Checkpoint em uso para esta execucao." />
                  <MetricCard label="Status" value={humanizeStatus(analysis.status)} detail={formatTimestamp(analysis.provenance.analysis_timestamp)} />
                </div>
              </article>

              <article className="panel p-5 sm:p-8">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <div className="eyebrow">
                      <Aperture className="h-3.5 w-3.5" />
                      Curva e relevancia
                    </div>
                    <p className="mt-3 text-sm leading-6 text-starlight-300">
                      Linha azul: curva processada. Barras amarelas: onde o modelo concentrou mais evidencia.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.2em] text-starlight-300">
                    <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-signal-300" /> Curva</span>
                    <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-amber-400" /> XAI</span>
                  </div>
                </div>
                <div className="mt-6">
                  <SeriesChart series={analysis.lightcurve_points} relevance={analysis.xai_points} />
                </div>
              </article>

              <div className="grid gap-6 xl:grid-cols-2">
                <article className="panel p-5 sm:p-7">
                  <div className="eyebrow">
                    <Orbit className="h-3.5 w-3.5" />
                    Baseline BLS
                  </div>
                  {analysis.bls_peaks.length === 0 ? (
                    <div className="mt-6 rounded-[24px] border border-dashed border-white/10 bg-white/[0.02] p-5 text-sm text-starlight-300">
                      Nenhum pico BLS relevante foi retornado para este alvo.
                    </div>
                  ) : (
                    <div className="mt-6 space-y-4">
                      {topPeak && (
                        <div className="rounded-[24px] border border-white/10 bg-ink-900/65 p-4">
                          <p className="text-[0.72rem] uppercase tracking-[0.22em] text-starlight-300">Melhor pico</p>
                          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
                            <div>
                              <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Periodo</p>
                              <p className="mt-2 font-mono text-white">{formatNumber(topPeak.period)} d</p>
                            </div>
                            <div>
                              <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Potencia</p>
                              <p className="mt-2 font-mono text-white">{formatNumber(topPeak.power)}</p>
                            </div>
                            <div>
                              <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Profundidade</p>
                              <p className="mt-2 font-mono text-white">{topPeak.depth.toExponential(2)}</p>
                            </div>
                          </div>
                        </div>
                      )}

                      <div className="table-shell">
                        <table className="min-w-[38rem] text-left text-sm sm:min-w-full">
                          <thead className="border-b border-white/10 bg-white/[0.03] text-starlight-300">
                            <tr>
                              <th className="px-4 py-3 font-medium uppercase tracking-[0.18em]">Periodo (d)</th>
                              <th className="px-4 py-3 font-medium uppercase tracking-[0.18em]">Potencia</th>
                              <th className="px-4 py-3 font-medium uppercase tracking-[0.18em]">Profundidade</th>
                            </tr>
                          </thead>
                          <tbody>
                            {analysis.bls_peaks.map((peak) => (
                              <tr key={`${peak.period}-${peak.power}`} className="border-b border-white/8 last:border-b-0">
                                <td className="px-4 py-3 font-mono text-starlight-200">{formatNumber(peak.period)}</td>
                                <td className="px-4 py-3 font-mono text-starlight-200">{formatNumber(peak.power)}</td>
                                <td className="px-4 py-3 font-mono text-starlight-200">{peak.depth.toExponential(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </article>

                <article className="panel p-5 sm:p-7">
                  <div className="eyebrow">
                    <ShieldAlert className="h-3.5 w-3.5" />
                    Integridade da execucao
                  </div>
                  <div className="mt-6 space-y-4">
                    <div className="rounded-[24px] border border-white/10 bg-ink-900/65 p-4">
                      <p className="text-[0.72rem] uppercase tracking-[0.22em] text-starlight-300">Avisos</p>
                      {analysis.warnings.length === 0 ? (
                        <p className="mt-3 text-sm text-starlight-200">Nenhum aviso registrado.</p>
                      ) : (
                        <ul className="mt-3 space-y-2 text-sm leading-6 text-starlight-200">
                          {analysis.warnings.map((warning) => (
                            <li key={warning} className="rounded-2xl border border-amber-400/15 bg-amber-400/8 px-3 py-2">
                              {warning}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>

                    <div className="rounded-[24px] border border-white/10 bg-ink-900/65 p-4">
                      <p className="text-[0.72rem] uppercase tracking-[0.22em] text-starlight-300">Parametros</p>
                      <div className="mt-4 flex flex-wrap gap-2">
                        {preprocessEntries.map(([key, value]) => (
                          <span key={key} className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-starlight-200">
                            <span className="text-starlight-300">{key.replace(/_/g, " ")}:</span> {String(value)}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </article>
              </div>
            </div>

            <aside className="space-y-6">
              <article className="panel p-5 sm:p-7">
                <div className="eyebrow">
                  <Gauge className="h-3.5 w-3.5" />
                  Pipeline que rodou
                </div>
                <div className="mt-6 space-y-4 text-sm">
                  <div className="rounded-[22px] border border-white/10 bg-ink-900/70 p-4">
                    <p className="text-[0.72rem] uppercase tracking-[0.2em] text-starlight-300">Fonte</p>
                    <p className="safe-wrap mt-2 text-base font-medium text-white">{analysis.provenance.data_source}</p>
                  </div>
                  <div className="rounded-[22px] border border-white/10 bg-ink-900/70 p-4">
                    <p className="text-[0.72rem] uppercase tracking-[0.2em] text-starlight-300">Missao</p>
                    <p className="safe-wrap mt-2 text-base font-medium text-white">
                      {analysis.provenance.mission}
                      {analysis.provenance.sector_or_quarter ? ` - ${analysis.provenance.sector_or_quarter}` : ""}
                    </p>
                  </div>
                  <div className="rounded-[22px] border border-white/10 bg-ink-900/70 p-4">
                    <p className="text-[0.72rem] uppercase tracking-[0.2em] text-starlight-300">Coordenadas</p>
                    <p className="safe-wrap mt-2 font-mono text-starlight-200">{coordinatesLabel(analysis.provenance.sky_coordinates)}</p>
                  </div>
                  <div className="rounded-[22px] border border-white/10 bg-ink-900/70 p-4">
                    <p className="text-[0.72rem] uppercase tracking-[0.2em] text-starlight-300">Checkpoint</p>
                    <p className="safe-wrap mt-2 font-mono text-starlight-200">{analysis.model_version}</p>
                  </div>
                </div>
              </article>

              {experimentalComparison && (
                <article className="panel p-5 sm:p-7">
                  <div className="eyebrow">
                    <Zap className="h-3.5 w-3.5" />
                    Revisao QML
                  </div>
                  <div className="mt-6 space-y-4">
                    <div className="rounded-[22px] border border-white/10 bg-ink-900/70 p-4">
                      <p className="text-[0.72rem] uppercase tracking-[0.2em] text-starlight-300">Modo final</p>
                      <p className="mt-2 text-lg font-semibold text-white">
                        {experimentalComparison.selected_mode === "qml" ? "QML residual acionado" : "Classico mantido"}
                      </p>
                      <p className="mt-2 text-sm leading-6 text-starlight-300">
                        {experimentalComparison.activation_reason ?? "Sem motivo informado pelo backend."}
                      </p>
                    </div>

                    <div className="grid gap-3">
                      <div className="data-knot">
                        <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Score classico</p>
                        <p className="mt-2 font-mono text-white">
                          {experimentalComparison.classical ? formatPercent(experimentalComparison.classical.prediction_score) : "n/a"}
                        </p>
                      </div>
                      <div className="data-knot">
                        <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Score QML</p>
                        <p className="mt-2 font-mono text-white">
                          {experimentalComparison.qml ? formatPercent(experimentalComparison.qml.prediction_score) : "n/a"}
                        </p>
                      </div>
                      <div className="data-knot">
                        <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Delta aplicado</p>
                        <p className="mt-2 font-mono text-white">
                          {experimentalComparison.score_delta == null ? "n/a" : `${experimentalComparison.score_delta.toFixed(4)}`}
                        </p>
                      </div>
                      <div className="data-knot">
                        <p className="text-xs uppercase tracking-[0.18em] text-starlight-300">Faixa ambigua</p>
                        <p className="mt-2 font-mono text-white">
                          {experimentalComparison.ambiguity_lower == null || experimentalComparison.ambiguity_upper == null
                            ? "n/a"
                            : `${experimentalComparison.ambiguity_lower.toFixed(2)} - ${experimentalComparison.ambiguity_upper.toFixed(2)}`}
                        </p>
                      </div>
                    </div>
                  </div>
                </article>
              )}

              {analysisSkyTarget && (
                <article className="panel p-5 sm:p-7">
                  <div className="eyebrow">
                    <MapPinned className="h-3.5 w-3.5" />
                    Alvo no ceu
                  </div>
                  <div className="mt-5">
                    <SkyReferenceMap
                      targets={GUIDE_TARGETS}
                      selectedTargetId={selectedGuideTarget.id}
                      previewTarget={previewSkyTarget}
                      analysisTarget={analysisSkyTarget}
                    />
                  </div>
                </article>
              )}
            </aside>
          </section>
        )}

        <section className="panel p-5 sm:p-8">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="eyebrow">
                <History className="h-3.5 w-3.5" />
                Historico recente
              </div>
              <p className="mt-3 text-sm leading-6 text-starlight-300">
                Reabra analises para comparar score, BLS, proveniencia e saida QML sem refazer o processo inteiro.
              </p>
            </div>
            <span className="orbit-chip border-white/10 bg-white/[0.04]">{history.length} itens</span>
          </div>

          {history.length === 0 ? (
            <div className="mt-6 rounded-[24px] border border-dashed border-white/10 bg-white/[0.02] p-5 text-sm text-starlight-300">
              Nenhuma analise foi registrada ainda.
            </div>
          ) : (
            <div className="mt-6 grid gap-3">
              {history.map((item) => (
                <button type="button" key={item.id} className="history-entry" onClick={() => void openHistoryItem(item.id)}>
                  <div className="grid gap-3 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,140px)_minmax(0,180px)_minmax(0,180px)]">
                    <div className="min-w-0">
                      <p className="text-xs uppercase tracking-[0.22em] text-starlight-300">Alvo</p>
                      <p className="safe-wrap mt-2 text-lg font-medium text-white">
                        #{item.id} {targetTypeLabel(item.target_type)} {item.target_id}
                      </p>
                      <p className="mt-1 text-sm text-starlight-300">{item.mission}</p>
                    </div>
                    <div className="min-w-0">
                      <p className="text-xs uppercase tracking-[0.22em] text-starlight-300">Score</p>
                      <p className="mt-2 font-mono text-white">{formatPercent(item.prediction_score)}</p>
                    </div>
                    <div className="min-w-0">
                      <p className="text-xs uppercase tracking-[0.22em] text-starlight-300">Melhor BLS</p>
                      <p className="mt-2 font-mono text-white">{item.bls_period == null ? "n/a" : `${formatNumber(item.bls_period)} d`}</p>
                    </div>
                    <div className="min-w-0">
                      <p className="text-xs uppercase tracking-[0.22em] text-starlight-300">Criado em</p>
                      <p className="safe-wrap mt-2 font-mono text-starlight-200">{formatTimestamp(item.created_at)}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

export default App;
