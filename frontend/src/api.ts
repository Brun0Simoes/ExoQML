import type { AnalysisResponse, HistoryItem, TargetCatalogItem, TargetType } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

type ApiErrorDetail = {
  code?: string;
  message?: string;
  suggestion?: string;
  stage?: string;
};

export class ApiError extends Error {
  code?: string;
  suggestion?: string;
  stage?: string;

  constructor(message: string, detail?: ApiErrorDetail) {
    super(message);
    this.name = "ApiError";
    this.code = detail?.code;
    this.suggestion = detail?.suggestion;
    this.stage = detail?.stage;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    let detail: ApiErrorDetail | undefined;
    try {
      const payload = (await response.json()) as { detail?: string | ApiErrorDetail };
      if (typeof payload.detail === "string") {
        message = payload.detail;
      } else if (payload.detail) {
        detail = payload.detail;
        message = payload.detail.message ?? message;
      }
    } catch {
      // Ignore JSON parse errors and keep default message.
    }
    throw new ApiError(message, detail);
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

export async function listTargetCatalog(limit = 10000): Promise<TargetCatalogItem[]> {
  try {
    const response = await fetch(`${API_BASE}/targets/catalog?limit=${limit}`);
    return await handleResponse<TargetCatalogItem[]>(response);
  } catch {
    const fallback = await import("./catalogFallback.generated.json");
    return (fallback.default as TargetCatalogItem[]).slice(0, limit);
  }
}

export function exportUrl(id: number, format: "json" | "csv"): string {
  return `${API_BASE}/history/${id}/export?format=${format}`;
}
