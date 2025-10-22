export interface PhenotypeRow {
  phrase: string;
  category?: string;
  translation?: string;
  hpo_id: string;
  keep: boolean;
}

export interface AnalyzeResponse {
  patient_id: number;
  phenotypes: PhenotypeRow[];
  runtime_seconds: number;
  raw_entries?: unknown;
}
