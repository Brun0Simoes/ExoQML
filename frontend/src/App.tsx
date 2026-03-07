import { FormEvent, useEffect, useMemo, useState } from "react";
import { analyzeTarget, exportUrl, getAnalysis, listHistory } from "./api";
import type { AnalysisResponse, HistoryItem, SeriesPoint, TargetType } from "./types";

type InputType = TargetType | "auto";

const TARGET_OPTIONS: Array<{ value: InputType; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "tic", label: "TIC" },
  { value: "kic", label: "KIC" },
  { value: "name", label: "Name" },
];

function sample(points: SeriesPoint[], maxPoints = 320): SeriesPoint[] {
  if (points.length <= maxPoints) {
    return points;
  }
  const step = Math.ceil(points.length / maxPoints);
  return points.filter((_, index) => index % step === 0);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function SeriesChart(props: { series: SeriesPoint[]; relevance: SeriesPoint[] }) {
  const series = useMemo(() => sample(props.series), [props.series]);
  const relevance = useMemo(() => sample(props.relevance), [props.relevance]);

  if (series.length < 2) {
    return <div className="empty-chart">No chart points available.</div>;
  }

  const width = 860;
  const height = 280;
  const pad = 18;
  const xMin = Math.min(...series.map((p) => p.x));
  const xMax = Math.max(...series.map((p) => p.x));
  const yMin = Math.min(...series.map((p) => p.y));
  const yMax = Math.max(...series.map((p) => p.y));
  const relMax = Math.max(...relevance.map((p) => p.y), 1e-9);

  const scaleX = (x: number) => {
    const norm = (x - xMin) / (xMax - xMin || 1);
    return pad + norm * (width - pad * 2);
  };
  const scaleY = (y: number) => {
    const norm = (y - yMin) / (yMax - yMin || 1);
    return height - pad - norm * (height - pad * 2);
  };

  const linePoints = series.map((p) => `${scaleX(p.x)},${scaleY(p.y)}`).join(" ");

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Light curve with relevance map">
      <rect x="0" y="0" width={width} height={height} fill="var(--panel-2)" rx="14" />
      {relevance.map((point) => {
        const x = scaleX(point.x);
        const h = (point.y / relMax) * 88;
        return (
          <line
            key={`rel-${point.x}`}
            x1={x}
            y1={height - pad}
            x2={x}
            y2={height - pad - h}
            stroke="var(--heat)"
            strokeOpacity="0.28"
            strokeWidth="2"
          />
        );
      })}
      <polyline fill="none" stroke="var(--signal)" strokeWidth="2.1" points={linePoints} />
    </svg>
  );
}

function App() {
  const [targetId, setTargetId] = useState("TIC 25155310");
  const [targetType, setTargetType] = useState<InputType>("auto");
  const [experimentalQml, setExperimentalQml] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  async function refreshHistory() {
    try {
      const data = await listHistory();
      setHistory(data);
    } catch (err) {
      console.error(err);
    }
  }

  useEffect(() => {
    void refreshHistory();
  }, []);

  async function onAnalyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setIsBusy(true);
    try {
      const payload = {
        target_id: targetId.trim(),
        experimental_qml: experimentalQml,
        ...(targetType !== "auto" ? { target_type: targetType } : {}),
      };
      const data = await analyzeTarget(payload);
      setAnalysis(data);
      await refreshHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error while analyzing target.");
    } finally {
      setIsBusy(false);
    }
  }

  async function openHistoryItem(id: number) {
    setError(null);
    try {
      const item = await getAnalysis(id);
      setAnalysis(item);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load analysis from history.");
    }
  }

  return (
    <main className="page">
      <section className="headline card">
        <h1>ExoQML</h1>
        <p>Explainable light-curve triage with classical BLS baseline and optional experimental QML flag.</p>
        <p className="disclaimer">
          This is an AI-assisted screening workflow. It is not an official observational confirmation.
        </p>
      </section>

      <section className="card">
        <form className="controls" onSubmit={onAnalyze}>
          <label>
            Target ID
            <input
              value={targetId}
              onChange={(event) => setTargetId(event.target.value)}
              placeholder="TIC 123, KIC 456 or mission name"
              required
            />
          </label>
          <label>
            Type
            <select value={targetType} onChange={(event) => setTargetType(event.target.value as InputType)}>
              {TARGET_OPTIONS.map((option) => (
                <option value={option.value} key={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={experimentalQml}
              onChange={(event) => setExperimentalQml(event.target.checked)}
            />
            Enable experimental QML mode
          </label>
          <button type="submit" disabled={isBusy}>
            {isBusy ? "Running..." : "Analyze"}
          </button>
        </form>
        {error && <p className="error">{error}</p>}
      </section>

      {analysis && (
        <section className="result-grid">
          <article className="card metrics">
            <h2>Result</h2>
            <div className="kpi">
              <span>Prediction</span>
              <strong>{analysis.prediction_label}</strong>
            </div>
            <div className="kpi">
              <span>Probability</span>
              <strong>{formatPercent(analysis.prediction_score)}</strong>
            </div>
            <div className="kpi">
              <span>BLS best period</span>
              <strong>{analysis.bls_period ? `${analysis.bls_period.toFixed(3)} d` : "n/a"}</strong>
            </div>
            <div className="kpi">
              <span>Model</span>
              <strong>
                {analysis.model_name} ({analysis.model_version})
              </strong>
            </div>
            <div className="kpi">
              <span>Mission / Source</span>
              <strong>
                {analysis.provenance.mission} / {analysis.provenance.data_source}
              </strong>
            </div>
            <div className="export-actions">
              <a href={exportUrl(analysis.id, "json")} target="_blank" rel="noreferrer">
                Export JSON
              </a>
              <a href={exportUrl(analysis.id, "csv")} target="_blank" rel="noreferrer">
                Export CSV
              </a>
            </div>
          </article>

          <article className="card chart-card">
            <h2>Light Curve + Relevance</h2>
            <SeriesChart series={analysis.lightcurve_points} relevance={analysis.xai_points} />
          </article>

          <article className="card">
            <h2>BLS Peaks</h2>
            {analysis.bls_peaks.length === 0 && <p>No BLS peaks for this run.</p>}
            {analysis.bls_peaks.length > 0 && (
              <table>
                <thead>
                  <tr>
                    <th>Period (d)</th>
                    <th>Power</th>
                    <th>Depth</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.bls_peaks.map((peak) => (
                    <tr key={`${peak.period}-${peak.power}`}>
                      <td>{peak.period.toFixed(3)}</td>
                      <td>{peak.power.toFixed(3)}</td>
                      <td>{peak.depth.toExponential(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </article>

          <article className="card">
            <h2>Warnings</h2>
            {analysis.warnings.length === 0 && <p>No warnings.</p>}
            {analysis.warnings.length > 0 && (
              <ul>
                {analysis.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            )}
          </article>
        </section>
      )}

      <section className="card">
        <h2>Recent Analyses</h2>
        {history.length === 0 && <p>No analyses yet.</p>}
        {history.length > 0 && (
          <div className="history">
            {history.map((item) => (
              <button type="button" key={item.id} className="history-item" onClick={() => void openHistoryItem(item.id)}>
                <span>
                  #{item.id} {item.target_type.toUpperCase()} {item.target_id}
                </span>
                <span>{formatPercent(item.prediction_score)}</span>
                <span>{new Date(item.created_at).toLocaleString()}</span>
              </button>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

export default App;
