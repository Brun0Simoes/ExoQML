import type { AnalysisResponse, HistoryItem, TargetType } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      // Ignore JSON parse errors and keep default message.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export async function analyzeTarget(payload: {
  target_id: string;
  target_type?: TargetType;
  experimental_qml: boolean;
}): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<AnalysisResponse>(response);
}

export async function listHistory(limit = 12): Promise<HistoryItem[]> {
  const response = await fetch(`${API_BASE}/history?limit=${limit}`);
  return handleResponse<HistoryItem[]>(response);
}

export async function getAnalysis(id: number): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE}/history/${id}`);
  return handleResponse<AnalysisResponse>(response);
}

export function exportUrl(id: number, format: "json" | "csv"): string {
  return `${API_BASE}/history/${id}/export?format=${format}`;
}
