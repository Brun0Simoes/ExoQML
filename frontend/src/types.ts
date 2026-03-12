export type TargetType = "tic" | "kic" | "name";

export type SeriesPoint = {
  x: number;
  y: number;
};

export type BLSPeak = {
  period: number;
  power: number;
  depth: number;
};

export type SkyCoordinates = {
  ra: number;
  dec: number;
};

export type TargetCatalogItem = {
  query: string;
  target_id: string;
  target_type: TargetType;
  display_name: string;
  mission: string;
  source: string;
  summary: string;
  tce_count: number;
  positive_tce_count: number;
  sky_coordinates: SkyCoordinates | null;
};

export type AnalysisResponse = {
  id: number;
  status: string;
  target_id: string;
  target_type: TargetType;
  prediction_label: string;
  prediction_score: number;
  bls_period: number | null;
  model_name: string;
  model_version: string;
  warnings: string[];
  preprocess_params: Record<string, string | number>;
  provenance: {
    mission: string;
    data_source: string;
    sector_or_quarter: string | null;
    analysis_timestamp: string;
    sky_coordinates: SkyCoordinates | null;
  };
  lightcurve_points: SeriesPoint[];
  xai_points: SeriesPoint[];
  bls_peaks: BLSPeak[];
  experimental_comparison?: {
    requested: boolean;
    available: boolean;
    activated: boolean;
    activation_reason: string | null;
    selected_mode: "classical" | "qml" | null;
    ambiguity_lower: number | null;
    ambiguity_upper: number | null;
    score_delta: number | null;
    absolute_score_delta: number | null;
    classical: {
      mode: "classical";
      prediction_label: string;
      prediction_score: number;
      model_name: string;
      model_version: string;
      score_delta_vs_classical?: number | null;
    } | null;
    qml: {
      mode: "qml";
      prediction_label: string;
      prediction_score: number;
      model_name: string;
      model_version: string;
      score_delta_vs_classical?: number | null;
    } | null;
  } | null;
};

export type HistoryItem = {
  id: number;
  target_id: string;
  target_type: TargetType;
  mission: string;
  prediction_label: string;
  prediction_score: number;
  bls_period: number | null;
  status: string;
  created_at: string;
};
