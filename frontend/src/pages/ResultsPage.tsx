import { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft,
  Download,
  Loader2,
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  ChevronRight,
  Dna,
  Target,
  Microscope,
  ShieldCheck,
} from "lucide-react";
import clsx from "clsx";
import { useJobPoller } from "../hooks/useJobPoller";
import { fetchSequenceDetail, fetchStructurePdb, downloadUrl } from "../api/client";
import type { SequenceDetail, SequenceSummary } from "../types";
import RadarChart from "../components/RadarChart";
import FeatureTable from "../components/FeatureTable";
import StructureViewer from "../components/StructureViewer";
import ProgressBar from "../components/ProgressBar";
import ScoreGauge from "../components/ScoreGauge";

const FEATURE_TABS = [
  { key: "aa", label: "Amino Acid Composition", icon: Dna },
  { key: "regsite", label: "Protease Recognition", icon: Target },
  { key: "structure", label: "Structural Context", icon: Microscope },
  { key: "function", label: "Functional Annotation", icon: ShieldCheck },
];

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { job, error: pollError } = useJobPoller(jobId ?? null);
  const [selectedSeqId, setSelectedSeqId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SequenceDetail | null>(null);
  const [pdbText, setPdbText] = useState<string | null>(null);
  const [activeFeatureTab, setActiveFeatureTab] = useState("aa");
  const [loadingDetail, setLoadingDetail] = useState(false);

  const isRunning = job?.status === "pending" || job?.status === "running";
  const isComplete = job?.status === "completed";
  const isFailed = job?.status === "failed";

  // Auto-select first sequence when results arrive
  useEffect(() => {
    if (isComplete && job?.ranked && job.ranked.length > 0 && !selectedSeqId) {
      setSelectedSeqId(job.ranked[0].id);
    }
  }, [isComplete, job?.ranked, selectedSeqId]);

  // Fetch detail when selection changes
  useEffect(() => {
    if (!jobId || !selectedSeqId || !isComplete) return;
    setLoadingDetail(true);
    setPdbText(null);
    Promise.all([
      fetchSequenceDetail(jobId, selectedSeqId),
      fetchStructurePdb(jobId, selectedSeqId),
    ])
      .then(([d, pdb]) => {
        setDetail(d);
        setPdbText(pdb);
      })
      .catch(() => {})
      .finally(() => setLoadingDetail(false));
  }, [jobId, selectedSeqId, isComplete]);

  const progressPercent = isRunning
    ? job?.status === "pending"
      ? 15
      : 55
    : isComplete
    ? 100
    : 0;

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Link to="/predict" className="btn-ghost p-2">
            <ArrowLeft size={18} />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-brand-900 tracking-tight">
              Analysis Results
            </h1>
            {job && (
              <p className="text-sm text-brand-400 mt-0.5">
                {job.sequence_count} sequence{job.sequence_count !== 1 ? "s" : ""}{" "}
                &middot; {job.input_type}
                {job.organisms.length > 0 &&
                  ` \u00b7 ${job.organisms.join(", ")}`}
              </p>
            )}
          </div>
        </div>
        {isComplete && job?.download_url && (
          <a
            href={downloadUrl(jobId!)}
            className="btn-secondary text-sm"
            download
          >
            <Download size={16} />
            Download CSV
          </a>
        )}
      </div>

      {/* Progress / Error states */}
      {isRunning && (
        <div className="card p-8 text-center mb-8">
          <Loader2 size={40} className="mx-auto text-brand-400 animate-spin mb-4" />
          <h2 className="text-lg font-semibold text-brand-800 mb-2">
            Analyzing your sequences...
          </h2>
          <p className="text-sm text-brand-400 mb-6">
            {job?.status === "pending"
              ? "Preparing pipeline..."
              : "Running feature extraction and ML ranking..."}
          </p>
          <div className="max-w-md mx-auto">
            <ProgressBar percent={progressPercent} />
          </div>
        </div>
      )}

      {isFailed && (
        <div className="card p-8 border-red-200 bg-red-50/50 mb-8">
          <div className="flex items-start gap-3">
            <AlertCircle size={24} className="text-red-500 flex-shrink-0" />
            <div>
              <h2 className="text-lg font-semibold text-red-800 mb-1">
                Analysis Failed
              </h2>
              <p className="text-sm text-red-600">
                {pollError || job?.error || "An unknown error occurred."}
              </p>
              <Link to="/predict" className="btn-secondary text-sm mt-4 inline-flex">
                Try Again
              </Link>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {isComplete && job?.ranked && (
        <div className="grid lg:grid-cols-[340px_1fr] gap-6">
          {/* Left: Ranked list */}
          <div className="card p-4 max-h-[calc(100vh-200px)] overflow-y-auto">
            <h3 className="text-sm font-semibold text-brand-400 uppercase tracking-wider px-2 mb-3">
              Ranked Proteins ({job.ranked.length})
            </h3>
            <div className="space-y-1">
              {job.ranked.map((seq: SequenceSummary) => (
                <button
                  key={seq.id}
                  onClick={() => setSelectedSeqId(seq.id)}
                  className={clsx(
                    "w-full text-left p-3 rounded-xl transition-all",
                    selectedSeqId === seq.id
                      ? "bg-brand-50 border border-brand-200 shadow-sm"
                      : "hover:bg-brand-50/60 border border-transparent"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2.5 min-w-0">
                      <span className="w-7 h-7 rounded-lg bg-brand-100 text-brand-600 text-xs font-bold flex items-center justify-center flex-shrink-0">
                        {seq.rank}
                      </span>
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-brand-800 truncate">
                          {seq.id}
                        </p>
                        {seq.length && (
                          <p className="text-xs text-brand-400">
                            {seq.length} aa
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {seq.digestibility_score != null && (
                        <span className="text-sm font-semibold text-brand-600">
                          {seq.digestibility_score.toFixed(3)}
                        </span>
                      )}
                      <ChevronRight
                        size={14}
                        className="text-brand-300"
                      />
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Right: Detail */}
          <div className="space-y-6">
            {loadingDetail ? (
              <div className="card p-12 text-center">
                <Loader2 size={24} className="mx-auto text-brand-400 animate-spin" />
              </div>
            ) : detail ? (
              <>
                {/* Score overview */}
                <div className="card p-6">
                  <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
                    <div>
                      <h2 className="text-lg font-bold text-brand-900">
                        {detail.id}
                      </h2>
                      <p className="text-sm text-brand-400">
                        Rank #{detail.rank}
                        {detail.sequence &&
                          ` \u00b7 ${detail.sequence.length} residues`}
                      </p>
                    </div>
                    <div className="flex items-center gap-3">
                      {detail.blastp_url && (
                        <a
                          href={detail.blastp_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="btn-ghost text-xs border border-brand-100 rounded-full px-3 py-1.5"
                        >
                          BLASTp
                          <ExternalLink size={12} />
                        </a>
                      )}
                      {detail.metrics.red_flag && (
                        <span className="badge-red">Red Flag</span>
                      )}
                      {detail.metrics.green_flag && (
                        <span className="badge-green">Green Flag</span>
                      )}
                    </div>
                  </div>

                  <div className="grid sm:grid-cols-2 gap-6">
                    {/* Score gauge */}
                    <div className="flex flex-col items-center justify-center p-4">
                      <ScoreGauge
                        score={detail.digestibility_score ?? 0}
                        label="Digestibility Score"
                      />
                    </div>

                    {/* Radar chart */}
                    <div className="min-h-[280px]">
                      <RadarChart metrics={detail.metrics} />
                    </div>
                  </div>
                </div>

                {/* Key metrics cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    {
                      label: "Essential AA",
                      value: detail.metrics.aa_essential_frac,
                      format: (v: number) => `${(v * 100).toFixed(1)}%`,
                    },
                    {
                      label: "Protease Sites",
                      value: detail.metrics.protease_total_sites,
                      format: (v: number) => String(Math.round(v)),
                    },
                    {
                      label: "Accessibility",
                      value: detail.metrics.cleavage_site_accessible_fraction,
                      format: (v: number) => `${(v * 100).toFixed(1)}%`,
                    },
                    {
                      label: "pLDDT Mean",
                      value: detail.metrics.plddt_mean,
                      format: (v: number) => v.toFixed(1),
                    },
                  ].map(({ label, value, format }) => (
                    <div key={label} className="card p-4 text-center">
                      <p className="text-xs text-brand-400 font-medium mb-1">
                        {label}
                      </p>
                      <p className="text-xl font-bold text-brand-800">
                        {value != null ? format(value as number) : "\u2014"}
                      </p>
                    </div>
                  ))}
                </div>

                {/* Structure viewer */}
                {detail.structure_available && pdbText && (
                  <div className="card p-6">
                    <h3 className="text-sm font-semibold text-brand-400 uppercase tracking-wider mb-4">
                      Predicted Structure
                    </h3>
                    <StructureViewer pdbText={pdbText} seqId={detail.id} />
                  </div>
                )}

                {/* Feature tabs */}
                <div className="card overflow-hidden">
                  <div className="flex border-b border-brand-100 overflow-x-auto">
                    {FEATURE_TABS.map(({ key, label, icon: Icon }) => {
                      const features = detail.features[key] ?? [];
                      return (
                        <button
                          key={key}
                          onClick={() => setActiveFeatureTab(key)}
                          className={clsx(
                            "flex items-center gap-2 px-5 py-3.5 text-sm font-medium whitespace-nowrap border-b-2 transition-all",
                            activeFeatureTab === key
                              ? "border-brand-400 text-brand-700 bg-brand-50/50"
                              : "border-transparent text-brand-400 hover:text-brand-600 hover:bg-brand-50/30"
                          )}
                        >
                          <Icon size={15} />
                          {label}
                          {features.length > 0 && (
                            <span className="ml-1 text-xs text-brand-300">
                              ({features.length})
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                  <div className="p-5">
                    <FeatureTable
                      features={detail.features[activeFeatureTab] ?? []}
                    />
                  </div>
                </div>

                {/* Sequence display */}
                {detail.sequence && (
                  <div className="card p-6">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-semibold text-brand-400 uppercase tracking-wider">
                        Sequence
                      </h3>
                      <button
                        onClick={() =>
                          navigator.clipboard.writeText(
                            `>${detail.id}\n${detail.sequence}`
                          )
                        }
                        className="btn-ghost text-xs"
                      >
                        Copy FASTA
                      </button>
                    </div>
                    <pre className="text-xs font-mono text-brand-600 bg-brand-50/50 rounded-xl p-4 overflow-x-auto leading-relaxed whitespace-pre-wrap break-all">
                      &gt;{detail.id}
                      {"\n"}
                      {detail.sequence.match(/.{1,60}/g)?.join("\n")}
                    </pre>
                  </div>
                )}
              </>
            ) : (
              <div className="card p-12 text-center">
                <p className="text-brand-400">
                  Select a sequence from the list to view details.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
