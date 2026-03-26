import type { ExampleInfo, JobStatus, SequenceDetail } from "../types";

const BASE = "";

export async function fetchExamples(): Promise<ExampleInfo[]> {
  const res = await fetch(`${BASE}/api/examples`);
  if (!res.ok) throw new Error("Failed to fetch examples");
  return res.json();
}

export async function startAnalysis(params: {
  inputType: string;
  organism?: string;
  organisms?: string[];
  sequences?: string;
  file?: File;
  options?: Record<string, unknown>;
}): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("input_type", params.inputType);
  if (params.organism) form.append("organism", params.organism);
  form.append("organisms", JSON.stringify(params.organisms ?? []));
  if (params.sequences) form.append("sequences", params.sequences);
  if (params.file) form.append("file", params.file);
  form.append("options", JSON.stringify(params.options ?? {}));

  const res = await fetch(`${BASE}/api/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Analysis request failed" }));
    throw new Error(err.detail || "Analysis request failed");
  }
  return res.json();
}

export async function fetchJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${BASE}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error("Failed to fetch job status");
  return res.json();
}

export async function fetchSequenceDetail(
  jobId: string,
  seqId: string
): Promise<SequenceDetail> {
  const res = await fetch(`${BASE}/api/jobs/${jobId}/sequence/${encodeURIComponent(seqId)}`);
  if (!res.ok) throw new Error("Failed to fetch sequence detail");
  return res.json();
}

export async function fetchStructurePdb(
  jobId: string,
  seqId: string
): Promise<string | null> {
  const res = await fetch(
    `${BASE}/api/jobs/${jobId}/sequence/${encodeURIComponent(seqId)}/structure`
  );
  if (!res.ok) return null;
  const data = await res.json();
  return data.pdb ?? null;
}

export function downloadUrl(jobId: string): string {
  return `${BASE}/api/jobs/${jobId}/download`;
}
