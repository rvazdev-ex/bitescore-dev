# bitescore

A pip-installable Python package and CLI to predict and rank **digestibility of protein sequences** using an ML-powered recommender algorithm.

## Highlights

- **Flexible inputs**: genome/proteome FASTA, single sequences, groups of genomes, metagenomic assemblies
- **Gene calling**: built-in prokaryotic (Prodigal) and eukaryotic (Augustus) gene callers via `--organism prok|euk`
- **Modular CLI**: run the full pipeline end-to-end or execute individual steps
- **Optional annotation layers**: DIAMOND/BLAST, GO term mapping, HMMER (Pfam), InterProScan, CD-HIT clustering, low-complexity masking
- **AlphaFold integration**: optional structure lookup by UniProt accession
- **ESM-2 embeddings**: optional protein language model features via `--esm`
- **Multiple Instance Learning**: train a MIL model on food-level experimental digestibility data
- **DIAAS calibration**: calibrate scores against Digestible Indispensable Amino Acid Score reference values
- **Web application**: FastAPI + React dashboard via `bitescore-web` with job tracking, interactive charts, and structure visualization

## Installation

### With Conda/Mamba (recommended)

This installs Python dependencies **and** external bioinformatics tools (Prodigal, Augustus, DIAMOND, BLAST, HMMER, CD-HIT):

```bash
mamba env create -f requirements.yml
mamba activate bitescore
pip install -e .
```

### With pip only

If you already have the external tools installed or only need proteome-level analysis:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[go]"   # Gene Ontology support (goatools)
pip install -e ".[dev]"  # Development tools (pytest, ruff)
```

## Quickstart

Run the full pipeline on an example proteome:

```bash
bitescore pipeline \
  --input data/examples/example_proteome.faa \
  --input-type proteome \
  --organism prok \
  --out results/run1 \
  --train
```

## CLI Reference

### End-to-end pipeline

```bash
bitescore pipeline --input <FILE> --input-type <TYPE> --organism <prok|euk> --out <DIR> [OPTIONS]
```

### Modular steps

| Command | Description |
|---|---|
| `bitescore load` | Load and normalise input sequences |
| `bitescore call-genes` | Call genes from genome/metagenome inputs |
| `bitescore features-aa` | Amino acid composition features |
| `bitescore features-regsite` | Protease cleavage site features |
| `bitescore features-structure` | Structural proxy features (with optional AlphaFold) |
| `bitescore features-function` | Functional annotation features (DIAMOND, BLAST, Pfam, GO) |
| `bitescore features-esm` | ESM-2 protein language model embeddings |
| `bitescore rank` | Train or apply the ranking model |
| `bitescore train-mil` | Train a Multiple Instance Learning model on experimental digestibility data |
| `bitescore report` | Generate a summary report |

`bitescore call-genes` accepts the same `--input`/`--input-type` pairing as the pipeline when you want to run loading and gene-calling together.

### Input types

| Type | Description |
|---|---|
| `genome` | Single FASTA genome file |
| `genomes` | Directory of genome FASTA files |
| `metagenome` | Metagenomic assembly |
| `proteome` | Protein FASTA file |
| `sequences` | Raw sequence strings (one per line) |

### Common options

```
--organism prok|euk           Gene calling mode (required for genome inputs)
--threads N                   Parallel processing threads
--config FILE                 YAML configuration file
--train                       Train a demo model on heuristic targets
--model FILE                  Path to a pre-trained .joblib model
--alphafold                   Fetch AlphaFold2 structures from UniProt
--no-structure                Skip structure features (faster)
--cluster-cdhit               Pre-cluster sequences with CD-HIT
--cdhit-threshold FLOAT       Clustering identity threshold (default 0.95)
--low-complexity              Mask low-complexity regions
--diamond-db PATH             DIAMOND database for similarity searches
--blast-db PATH               BLAST database (alternative to DIAMOND)
--pfam-hmms PATH              Pfam HMM database for domain scans
--interpro                    Run InterProScan
--go-map PATH                 Gene Ontology mapping file (id2go.tsv)
--pfam2go PATH                Pfam2GO mapping file for domain → GO annotation
--interpro2go PATH            InterPro2GO mapping file
--diamond-evalue FLOAT        DIAMOND E-value cutoff (default 1e-5)
--blast-evalue FLOAT          BLAST E-value cutoff (default 1e-5)
--pfam-evalue FLOAT           Pfam hmmscan E-value cutoff (default 1e-5)
--esm                         Enable ESM-2 protein language model embeddings
--esm-model NAME              ESM-2 model name (default: esm2_t6_8M_UR50D)
--no-calibrate                Disable DIAAS calibration (on by default)
--calibration-method TYPE     Calibration method: isotonic or linear (default: isotonic)
--mil-train                   Train a MIL model on reference digestibility data
--mil-model PATH              Pre-trained MIL model (.pt) path
--digestibility-ref PATH      CSV with food-level experimental digestibility values
--food-composition PATH       CSV with protein abundance per food
```

### LocalColabFold (optional, for 3D structure prediction)

`bitescore` can run local structure prediction through `localcolabfold` when available.

1. Install and verify LocalColabFold:
   ```bash
   localcolabfold --help
   ```
2. If the binary is not on your `PATH`, set:
   ```bash
   export LOCALCOLABFOLD_BIN=/absolute/path/to/localcolabfold
   ```
3. Run with structure + AlphaFold enabled:
   ```bash
   bitescore pipeline \
     --input data/examples/example_proteome.faa \
     --input-type proteome \
     --out results/af_run \
     --alphafold
   ```

Notes:
- AlphaFold DB lookup uses UniProt-like FASTA IDs (for example: `sp|P12345|...`).
- LocalColabFold results are cached under `<outdir>/cache/localcolabfold/`.
- Confirm outputs in `features_structure.csv` (`structure_source`, `predicted_structure_path`, `plddt_*`).

## Web Application

Launch the FastAPI backend with the React frontend:

```bash
bitescore-web   # opens at http://localhost:8000
```

The web app provides:

- **Predict page** -- upload FASTA files or paste sequences and run the pipeline from the browser
- **Results page** -- interactive charts (Chart.js) and a sortable feature table for ranked proteins
- **About page** -- background on the scoring methodology

On the **Predict** page, use **Structural Context** to:
- enable/disable structure features
- enable AlphaFold lookup (sent as analysis options to backend)

### Frontend development

```bash
cd frontend
npm install
npm run dev       # Vite dev server with hot reload
npm run build     # Production build
```

## Output files

The pipeline writes the following to the output directory:

| File | Description |
|---|---|
| `loaded.faa` / `loaded.fna` | Normalised input sequences |
| `called.faa` | Gene-called proteins (genome inputs) |
| `clustered.faa` | Deduplicated sequences (if CD-HIT enabled) |
| `masked.faa` | Low-complexity masked sequences (if enabled) |
| `features_aa.csv` | Amino acid composition features |
| `features_regsite.csv` | Protease cleavage site features |
| `features_structure.csv` | Structural features |
| `features_function.csv` | Functional annotation features |
| `features_esm.csv` | ESM-2 embedding features (if `--esm` enabled) |
| `features.csv` | Combined feature matrix |
| `ranked.csv` | Final ranking with digestibility scores |
| `model.joblib` | Trained ML model (if `--train` used) |
| `mil_model.pt` | Trained MIL model (if `--mil-train` used) |
| `log.txt` | Execution log |

## Optional databases

For full annotation support, download and prepare:

- **`uniprot.dmnd`** -- DIAMOND-formatted UniProt database
- **`Pfam-A.hmm`** -- Pfam HMM profiles (run `hmmpress` after downloading)
- **`id2go.tsv`** -- UniProt-to-GO term mapping

See `resources/README.txt` for details.

## Biological background

Proteins vary widely in how readily gastrointestinal or secreted proteases can liberate amino acids. Nutritional quality, feed digestibility, and the utility of engineered enzymes all depend on how quickly enzymes such as trypsin, chymotrypsin, and acidic proteases cleave exposed peptide bonds. Accessibility of Lys/Arg-rich loops for trypsin or aromatic residues for chymotrypsin, the presence of flexible linkers, and the avoidance of heavily cross-linked or glycosylated motifs all contribute to the overall digestibility of a sequence. When working from genomes or metagenomes, identifying accurate coding sequences is therefore an essential precursor to reasoning about downstream protein-level traits.

## Biology-aware feature set

- **Sequence composition.** Essential amino acid content and residue frequencies, enabling high-value nutritional profiles to be surfaced from large proteomes or metagenomic assemblies.
- **Physicochemical proxies.** Aromatic content, charge balance, and glycosylation site proxies provide context about protease resistance and post-translational modifications.
- **Cleavage accessibility.** Counts of Lys/Arg (trypsin) and Phe/Trp/Tyr (chymotrypsin) sites paired with heuristic exposure and flexibility scores that approximate how easily enzymes can reach their preferred motifs.
- **Functional annotation hooks.** Optional DIAMOND/BLAST searches, GO term mapping, Pfam domain scans, and InterProScan integration connect digestibility predictions with biological roles and enzyme families.
- **Structural context.** Lightweight structure-aware proxies cached for every sequence, with optional AlphaFold summary statistics (e.g., mean pLDDT) for interpreting disorder or structural confidence.
- **Protein language model embeddings.** Optional ESM-2 embeddings capture deep evolutionary and structural patterns that complement hand-crafted features.
- **Genome-aware preprocessing.** Built-in gene calling, low-complexity masking, and CD-HIT clustering ensure only representative, biologically plausible protein sequences enter the feature matrix.

## Interpreting digestibility scores

The default ranking model is a random forest regressor that operates on the feature matrix described above. In demonstration mode it self-trains on heuristics emphasizing essential amino acid abundance and trypsin-accessible sites, but in production settings you should retrain the model with empirical digestibility measurements that match your organism, protease cocktail, or processing environment.

For more accurate predictions, use the **Multiple Instance Learning (MIL)** pathway (`--mil-train`) with experimental digestibility data. The MIL model learns protein-level digestibility from food-level measurements by treating each food as a bag of its constituent proteins. Pair this with **DIAAS calibration** (enabled by default) to align scores with the Digestible Indispensable Amino Acid Score framework used in nutritional science.

Predictions are best treated as a prioritization aid: experimental assays (in vitro digestion, animal trials, or proteomics) remain the gold standard for validating protein digestibility. Use the optional annotation layers to connect high-scoring candidates with their biological context before investing in laboratory follow-up.

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src/bitescore --cov-report=html
```

## Project structure

```
src/bitescore/
  cli.py              CLI entry point (Click)
  pipeline.py          Core pipeline orchestration
  report.py            Report generation
  api/                 FastAPI REST backend + static assets
  features/            Feature extraction (aa, cleavage, structure, function, ESM-2)
  gene_callers/        Gene calling (Prodigal, Augustus, ORF fallback)
  ml/                  Ranking model, MIL model, DIAAS calibration
  tools/               External tool wrappers (BLAST, HMMER, CD-HIT, etc.)
  io/                  FASTA loaders
  utils/               Configuration and logging
  data/                Reference digestibility dataset

frontend/              React + TypeScript web dashboard (Vite, Tailwind CSS)
```

## License

MIT
