import { Link } from "react-router-dom";
import {
  FlaskConical,
  Dna,
  BarChart3,
  ShieldCheck,
  Zap,
  ArrowRight,
  Microscope,
  Target,
} from "lucide-react";

const FEATURES = [
  {
    icon: Dna,
    title: "Amino Acid Profiling",
    description:
      "Comprehensive essential amino acid analysis with FAO/WHO nutritional scoring and limiting amino acid identification.",
  },
  {
    icon: Target,
    title: "Protease Accessibility",
    description:
      "Maps recognition sites for 8 gastrointestinal proteases (trypsin, chymotrypsin, pepsin, etc.) with heuristic accessibility scoring.",
  },
  {
    icon: Microscope,
    title: "Structure-Aware Analysis",
    description:
      "Sequence-derived structural proxies plus optional AlphaFold/ColabFold 3D predictions for cleavage site exposure assessment.",
  },
  {
    icon: ShieldCheck,
    title: "Functional Screening",
    description:
      "Detects red flags (protease inhibitors, toxins) and green flags (secreted/extracellular) via GO terms and multi-source annotation.",
  },
  {
    icon: BarChart3,
    title: "ML Ranking Engine",
    description:
      "Random Forest model synthesizes all features into a single digestibility score, ranking proteins from most to least digestible.",
  },
  {
    icon: Zap,
    title: "Graceful Degradation",
    description:
      "Pipeline adapts to available tools. Full analysis with all bioinformatics databases, or lightweight mode with sequence-only proxies.",
  },
];

export default function HomePage() {
  return (
    <div className="animate-fade-in">
      {/* Hero */}
      <section className="relative text-center py-16 sm:py-24">
        <div className="absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-brand-100/40 blur-3xl" />
        </div>
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="inline-flex items-center gap-2 badge-blue text-sm px-4 py-1.5 mb-2">
            <FlaskConical size={14} />
            Protein Digestibility Intelligence
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-brand-900 leading-tight">
            Predict protein
            <br />
            <span className="gradient-text">digestibility from sequence</span>
          </h1>
          <p className="text-lg text-brand-500 max-w-2xl mx-auto leading-relaxed">
            BiteScore screens proteins for nutritional quality and predicts
            their digestibility. Analyze essential-amino-acid balance and
            enzyme-cleavage accessibility to deliver an explainable,
            per-protein digestibility score.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
            <Link to="/predict" className="btn-primary text-base px-8 py-3.5">
              Start Analysis
              <ArrowRight size={18} />
            </Link>
            <Link to="/about" className="btn-secondary text-base">
              Learn More
            </Link>
          </div>
        </div>
      </section>

      {/* Pipeline overview */}
      <section className="py-12">
        <div className="text-center mb-12">
          <h2 className="section-title">How It Works</h2>
          <p className="section-subtitle max-w-xl mx-auto">
            A modular pipeline that extracts four complementary feature sets and
            combines them into a single digestibility prediction.
          </p>
        </div>

        {/* Pipeline steps */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 mb-16">
          {[
            "Input FASTA",
            "Gene Calling",
            "Feature Extraction",
            "ML Ranking",
            "Results",
          ].map((step, i) => (
            <div key={step} className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="w-8 h-8 rounded-full bg-brand-400 text-white text-sm font-bold flex items-center justify-center shadow-sm">
                  {i + 1}
                </span>
                <span className="text-sm font-medium text-brand-700 whitespace-nowrap">
                  {step}
                </span>
              </div>
              {i < 4 && (
                <ArrowRight
                  size={16}
                  className="text-brand-300 hidden sm:block"
                />
              )}
            </div>
          ))}
        </div>

        {/* Feature cards */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map(({ icon: Icon, title, description }) => (
            <div key={title} className="card-hover p-6 group">
              <div className="w-11 h-11 rounded-xl bg-brand-50 flex items-center justify-center mb-4 group-hover:bg-brand-100 transition-colors">
                <Icon size={22} className="text-brand-500" />
              </div>
              <h3 className="text-base font-semibold text-brand-800 mb-2">
                {title}
              </h3>
              <p className="text-sm text-brand-500 leading-relaxed">
                {description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="py-12">
        <div className="card p-8 sm:p-12 text-center bg-gradient-to-br from-brand-50 to-white">
          <h2 className="text-2xl font-bold text-brand-900 mb-3">
            Ready to analyze your proteins?
          </h2>
          <p className="text-brand-500 mb-6 max-w-lg mx-auto">
            Upload a FASTA file or paste sequences to get a full digestibility
            analysis with explainable, per-protein scores.
          </p>
          <Link to="/predict" className="btn-primary text-base px-8 py-3.5">
            Start Prediction
            <ArrowRight size={18} />
          </Link>
        </div>
      </section>
    </div>
  );
}
