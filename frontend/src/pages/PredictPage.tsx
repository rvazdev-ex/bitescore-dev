import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  FileText,
  Play,
  Loader2,
  AlertCircle,
  ChevronDown,
  Sparkles,
  X,
} from "lucide-react";
import clsx from "clsx";
import { startAnalysis, fetchExamples } from "../api/client";
import type { ExampleInfo } from "../types";

const INPUT_TYPES = [
  { value: "proteome", label: "Proteomic", description: "Protein FASTA (.faa)" },
  { value: "genome", label: "Genomic", description: "Genome FASTA (.fna)" },
  { value: "metagenome", label: "Metagenomic", description: "Metagenome assembly (.fna)" },
  { value: "sequences", label: "Plain Sequences", description: "One sequence per line" },
];

export default function PredictPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [inputType, setInputType] = useState("proteome");
  const [organism, setOrganism] = useState<string>("prok");
  const [organisms, setOrganisms] = useState<string[]>([]);
  const [organismInput, setOrganismInput] = useState("");
  const [sequences, setSequences] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [examples, setExamples] = useState<ExampleInfo[]>([]);
  const [structureEnabled, setStructureEnabled] = useState(true);
  const [alphafoldEnabled, setAlphafoldEnabled] = useState(false);

  const isGenome = ["genome", "metagenome"].includes(inputType);

  useEffect(() => {
    fetchExamples().then(setExamples).catch(() => {});
  }, []);

  const addOrganism = () => {
    const name = organismInput.trim();
    if (name && !organisms.includes(name)) {
      setOrganisms((prev) => [...prev, name]);
    }
    setOrganismInput("");
  };

  const removeOrganism = (name: string) => {
    setOrganisms((prev) => prev.filter((o) => o !== name));
  };

  const loadExample = (ex: ExampleInfo) => {
    if (ex.sequences) {
      setSequences(ex.sequences);
      setFile(null);
    }
    setInputType(ex.input_type);
    if (ex.input_type === "genome") {
      setOrganism("prok");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setSequences("");
    }
  };

  const handleSubmit = async () => {
    setError(null);
    if (!file && !sequences.trim()) {
      setError("Please upload a FASTA file or paste sequences.");
      return;
    }
    if (isGenome && !organism) {
      setError("Please select the organism type for genome inputs.");
      return;
    }

    setLoading(true);
    try {
      const result = await startAnalysis({
        inputType,
        organism: isGenome ? organism : organisms[0] || undefined,
        organisms,
        sequences: sequences || undefined,
        file: file || undefined,
        options: {
          no_structure: !structureEnabled,
          alphafold: structureEnabled && alphafoldEnabled,
        },
      });
      navigate(`/results/${result.job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start analysis");
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto animate-fade-in">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-brand-900 tracking-tight">
          Predict Digestibility
        </h1>
        <p className="text-brand-500 mt-2">
          Upload sequences, configure your analysis, and get per-protein
          digestibility scores.
        </p>
      </div>

      <div className="space-y-6">
        {/* Step 1: Input Type */}
        <div className="card p-6">
          <h2 className="text-sm font-semibold text-brand-400 uppercase tracking-wider mb-4">
            Step 1 &mdash; Input Type
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {INPUT_TYPES.map(({ value, label, description }) => (
              <button
                key={value}
                onClick={() => setInputType(value)}
                className={clsx(
                  "p-3 rounded-xl border text-left transition-all",
                  inputType === value
                    ? "border-brand-400 bg-brand-50 shadow-glow"
                    : "border-brand-100 bg-white hover:border-brand-200 hover:bg-brand-50/50"
                )}
              >
                <div className="text-sm font-semibold text-brand-800">
                  {label}
                </div>
                <div className="text-xs text-brand-400 mt-0.5">
                  {description}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Step 2: Organism */}
        <div className="card p-6">
          <h2 className="text-sm font-semibold text-brand-400 uppercase tracking-wider mb-4">
            Step 2 &mdash; Organism
          </h2>
          {isGenome ? (
            <div>
              <p className="text-sm text-brand-500 mb-3">
                Select the organism type for gene calling:
              </p>
              <div className="flex gap-3">
                {[
                  { value: "prok", label: "Prokaryotic" },
                  { value: "euk", label: "Eukaryotic" },
                ].map(({ value, label }) => (
                  <button
                    key={value}
                    onClick={() => setOrganism(value)}
                    className={clsx(
                      "px-5 py-2.5 rounded-xl border text-sm font-medium transition-all",
                      organism === value
                        ? "border-brand-400 bg-brand-50 text-brand-700 shadow-glow"
                        : "border-brand-100 text-brand-500 hover:border-brand-200"
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div>
              <div className="flex gap-2 mb-3">
                <input
                  type="text"
                  className="input-field flex-1"
                  placeholder="e.g. Escherichia coli, Homo sapiens"
                  value={organismInput}
                  onChange={(e) => setOrganismInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && addOrganism()}
                />
                <button onClick={addOrganism} className="btn-secondary text-sm">
                  Add
                </button>
              </div>
              {organisms.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {organisms.map((org) => (
                    <span
                      key={org}
                      className="badge-blue flex items-center gap-1.5 pr-2"
                    >
                      {org}
                      <button
                        onClick={() => removeOrganism(org)}
                        className="hover:text-brand-900 transition-colors"
                      >
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                </div>
              )}
              {organisms.length === 0 && (
                <p className="text-xs text-brand-300 italic">
                  Optional. Add organisms for annotation context.
                </p>
              )}
            </div>
          )}
        </div>

        {/* Step 3: Sequences */}
        <div className="card p-6">
          <h2 className="text-sm font-semibold text-brand-400 uppercase tracking-wider mb-4">
            Step 3 &mdash; Sequences
          </h2>

          {/* File upload */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className={clsx(
              "border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all",
              file
                ? "border-brand-400 bg-brand-50"
                : "border-brand-200 hover:border-brand-300 hover:bg-brand-50/50"
            )}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".faa,.fna,.fasta,.fa,.txt"
              onChange={handleFileChange}
              className="hidden"
            />
            {file ? (
              <div className="flex items-center justify-center gap-3">
                <FileText size={24} className="text-brand-500" />
                <div className="text-left">
                  <p className="text-sm font-semibold text-brand-700">
                    {file.name}
                  </p>
                  <p className="text-xs text-brand-400">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                  }}
                  className="ml-2 p-1 rounded-full hover:bg-brand-100"
                >
                  <X size={16} className="text-brand-400" />
                </button>
              </div>
            ) : (
              <>
                <Upload size={32} className="mx-auto text-brand-300 mb-3" />
                <p className="text-sm text-brand-600 font-medium">
                  Click to upload or drag & drop
                </p>
                <p className="text-xs text-brand-400 mt-1">
                  FASTA format (.faa, .fna, .fasta)
                </p>
              </>
            )}
          </div>

          <div className="flex items-center gap-3 my-4">
            <div className="flex-1 h-px bg-brand-100" />
            <span className="text-xs text-brand-300 font-medium">OR</span>
            <div className="flex-1 h-px bg-brand-100" />
          </div>

          {/* Sequence textarea */}
          <textarea
            className="input-field font-mono text-sm min-h-[160px] resize-y"
            placeholder=">protein_1 Example protein&#10;MSTNPKPQRITKRRVVYAAFVVLLVLTALLASSSKRRRYYYAA&#10;&#10;Paste in FASTA format or plain sequences (one per line)"
            value={sequences}
            onChange={(e) => {
              setSequences(e.target.value);
              if (e.target.value) setFile(null);
            }}
          />

          {/* Examples */}
          {examples.length > 0 && (
            <div className="mt-4">
              <p className="text-xs text-brand-400 font-medium mb-2">
                Quick start with examples:
              </p>
              <div className="flex flex-wrap gap-2">
                {examples.map((ex) => (
                  <button
                    key={ex.name}
                    onClick={() => loadExample(ex)}
                    className="btn-ghost text-xs gap-1.5 px-3 py-1.5 border border-brand-100 rounded-full hover:border-brand-200"
                  >
                    <Sparkles size={12} />
                    {ex.name}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-3 p-4 rounded-xl bg-red-50 border border-red-200 animate-fade-in">
            <AlertCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {/* Step 4: Structural Context */}
        <div className="card p-6">
          <h2 className="text-sm font-semibold text-brand-400 uppercase tracking-wider mb-4">
            Step 4 &mdash; Structural Context
          </h2>
          <div className="space-y-4">
            <label className="flex items-start gap-3">
              <input
                type="checkbox"
                className="mt-1"
                checked={structureEnabled}
                onChange={(e) => {
                  const enabled = e.target.checked;
                  setStructureEnabled(enabled);
                  if (!enabled) setAlphafoldEnabled(false);
                }}
              />
              <div>
                <p className="text-sm font-medium text-brand-700">
                  Enable structure features
                </p>
                <p className="text-xs text-brand-400">
                  Computes structure-related features from sequence and optional 3D structure-derived metrics.
                </p>
              </div>
            </label>

            <label className="flex items-start gap-3">
              <input
                type="checkbox"
                className="mt-1"
                checked={alphafoldEnabled}
                disabled={!structureEnabled}
                onChange={(e) => setAlphafoldEnabled(e.target.checked)}
              />
              <div>
                <p className={clsx("text-sm font-medium", structureEnabled ? "text-brand-700" : "text-brand-300")}>
                  Enable AlphaFold lookup
                </p>
                <p className={clsx("text-xs", structureEnabled ? "text-brand-400" : "text-brand-300")}>
                  Uses UniProt-like FASTA IDs (e.g. sp|P12345|...) to fetch AlphaFold metadata.
                </p>
              </div>
            </label>
          </div>
        </div>

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="btn-primary w-full text-base py-4"
        >
          {loading ? (
            <>
              <Loader2 size={20} className="animate-spin" />
              Starting Analysis...
            </>
          ) : (
            <>
              <Play size={18} />
              Run Analysis
            </>
          )}
        </button>
      </div>
    </div>
  );
}
