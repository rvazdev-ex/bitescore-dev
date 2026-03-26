import { Link } from "react-router-dom";
import {
  ArrowRight,
  Dna,
  Target,
  Microscope,
  ShieldCheck,
  BarChart3,
  BookOpen,
  Users,
} from "lucide-react";

const SCIENCE_SECTIONS = [
  {
    icon: Dna,
    title: "Amino Acid Composition",
    points: [
      "Essential amino acid profiling (HILKMFWTV) against FAO/WHO reference requirements",
      "Per-residue counts & fractions for all 20 standard amino acids",
      "Limiting amino acid identification for nutritional quality assessment",
      "Physicochemical proxies: aromatic content, charge balance, glycosylation sites",
    ],
  },
  {
    icon: Target,
    title: "Protease Recognition Sites",
    points: [
      "Cleavage site mapping for 8 GI proteases: trypsin, chymotrypsin, pepsin, Lys-C, Arg-C, Glu-C, Asp-N, Lys-N",
      "Heuristic accessibility scoring using exposure propensity and flexibility windows",
      "Cleavage accessibility proxy = 0.5 \u00d7 surface exposure + 0.5 \u00d7 local flexibility",
      "Position-tracking with ProteaseRule data structures",
    ],
  },
  {
    icon: Microscope,
    title: "Structural Features",
    points: [
      "Layer 1: Sequence-only proxies \u2014 hydrophobicity, disorder propensity, secondary structure, surface accessibility, Shannon entropy",
      "Layer 2: AlphaFold DB lookup \u2014 pre-computed pLDDT statistics via UniProt/EBI API",
      "Layer 3: LocalColabFold prediction \u2014 full 3D structure with per-residue confidence, contact numbers, radius of gyration",
      "Graceful fallback: each layer activates only when the required data or tools are available",
    ],
  },
  {
    icon: ShieldCheck,
    title: "Functional Annotation",
    points: [
      "Hierarchical annotation: UniProt \u2192 InterPro patterns \u2192 BLAST reference \u2192 external hooks",
      "Red flags: protease inhibitors, toxins (harmful for digestibility)",
      "Green flags: extracellular/secreted proteins (beneficial for accessibility)",
      "Pluggable hook system for DIAMOND, NCBI BLAST, Pfam (hmmscan), and InterProScan",
    ],
  },
  {
    icon: BarChart3,
    title: "ML Ranking",
    points: [
      "Random Forest regressor (200 estimators) with median imputation and standard scaling",
      "Features: essential AA fraction, protease sites, trypsin K/R sites, cleavage accessibility, disorder, pLDDT, red/green flags",
      "Demo mode: self-trained on heuristic targets; production mode: user-provided empirical labels",
      "Output: normalized digestibility_score per protein, ranked descending",
    ],
  },
];

const TEAM_MEMBERS = [
  { name: "Ricardo", role: "Computational Biology, Bioinformatics" },
  { name: "Miranda", role: "Computer Science, Biochemical Engineering" },
  { name: "Jimmy", role: "Biomedical Engineering, Machine Learning" },
  { name: "Alexia", role: "Molecular Biology, Data Visualization" },
];

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      {/* Hero */}
      <section className="text-center py-12">
        <div className="inline-flex items-center gap-2 badge-blue text-sm px-4 py-1.5 mb-4">
          <BookOpen size={14} />
          About BiteScore
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-brand-900 tracking-tight mb-4">
          The Science Behind
          <br />
          <span className="gradient-text">Digestibility Prediction</span>
        </h1>
        <p className="text-brand-500 max-w-2xl mx-auto leading-relaxed">
          BiteScore screens individual proteins for nutritional quality and
          predicts their digestibility from sequence. It analyzes
          essential-amino-acid balance and enzyme-cleavage accessibility to
          deliver an explainable, per-protein digestibility score.
        </p>
      </section>

      {/* What & Why */}
      <section className="grid sm:grid-cols-2 gap-5 mb-12">
        <div className="card p-6">
          <h2 className="text-lg font-bold text-brand-900 mb-3">What it does</h2>
          <p className="text-sm text-brand-600 leading-relaxed">
            BiteScore extracts four complementary feature sets from protein
            sequences &mdash; amino acid composition, protease recognition sites,
            structural context, and functional annotations &mdash; then combines
            them through a Random Forest model to produce a single digestibility
            score per protein.
          </p>
        </div>
        <div className="card p-6">
          <h2 className="text-lg font-bold text-brand-900 mb-3">Why it matters</h2>
          <p className="text-sm text-brand-600 leading-relaxed">
            By pinpointing proteins with strong essential-amino-acid profiles and
            high predicted digestibility, BiteScore helps teams choose candidates
            to overexpress in a host organism for production. Single,
            well-characterized proteins can often follow clearer, simpler
            regulatory paths than complex multi-ingredient foods.
          </p>
        </div>
      </section>

      {/* Science details */}
      <section className="mb-12">
        <h2 className="section-title text-center mb-8">Feature Pipeline</h2>
        <div className="space-y-5">
          {SCIENCE_SECTIONS.map(({ icon: Icon, title, points }) => (
            <div key={title} className="card p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-brand-50 flex items-center justify-center">
                  <Icon size={20} className="text-brand-500" />
                </div>
                <h3 className="text-base font-semibold text-brand-800">
                  {title}
                </h3>
              </div>
              <ul className="space-y-2">
                {points.map((point, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-brand-600 leading-relaxed"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-brand-300 mt-2 flex-shrink-0" />
                    {point}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* Team */}
      <section className="mb-12">
        <h2 className="section-title text-center mb-8">
          <Users size={24} className="inline mr-2" />
          Our Team
        </h2>
        <div className="card p-6 mb-6 overflow-hidden">
          <img
            src="/api/assets/team.jpg"
            alt="BiteScore team"
            className="w-full max-w-xl mx-auto rounded-2xl shadow-card object-cover"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {TEAM_MEMBERS.map(({ name, role }) => (
            <div key={name} className="card p-5 text-center">
              <div className="w-12 h-12 rounded-full bg-brand-100 flex items-center justify-center mx-auto mb-3">
                <span className="text-lg font-bold text-brand-600">
                  {name[0]}
                </span>
              </div>
              <h4 className="text-sm font-semibold text-brand-800">{name}</h4>
              <p className="text-xs text-brand-400 mt-1">{role}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="text-center py-8">
        <Link to="/predict" className="btn-primary text-base px-8 py-3.5">
          Try BiteScore
          <ArrowRight size={18} />
        </Link>
      </section>
    </div>
  );
}
