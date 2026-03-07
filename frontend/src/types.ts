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
  };
  lightcurve_points: SeriesPoint[];
  xai_points: SeriesPoint[];
  bls_peaks: BLSPeak[];
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
