export interface SequenceSummary {
  id: string;
  rank: number;
  length: number | null;
  digestibility_score: number | null;
  aa_essential_frac: number | null;
}

export interface FeatureEntry {
  metric: string;
  value: string;
}

export interface SequenceDetail {
  id: string;
  rank: number;
  sequence: string | null;
  digestibility_score: number | null;
  metrics: Record<string, number | boolean>;
  features: Record<string, FeatureEntry[]>;
  structure_available: boolean;
  blastp_url: string | null;
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  input_type: string;
  organisms: string[];
  sequence_count: number;
  ranked?: SequenceSummary[];
  download_url?: string;
  error?: string;
}

export interface ExampleInfo {
  name: string;
  description: string;
  sequences: string | null;
  file_path: string | null;
  input_type: string;
}

export interface ProgressUpdate {
  job_id: string;
  status: string;
  percent: number;
  description: string;
}
