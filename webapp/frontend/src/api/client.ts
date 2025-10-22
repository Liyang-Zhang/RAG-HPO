import axios from "axios";

import type { AnalyzeResponse } from "../types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
  timeout: 120_000,
});

export async function analyzeClinicalNote(text: string): Promise<AnalyzeResponse> {
  const { data } = await api.post<AnalyzeResponse>("/analyze", { text });
  return data;
}
