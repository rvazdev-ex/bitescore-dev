# bitescore

A pip-installable Python package and CLI to predict and rank **digestibility of protein sequences** using an ML-powered recommender algorithm.

## Highlights
- Inputs: genome/proteome FASTA, single sequences, groups of genomes, metagenomic sets.
- User specifies gene-calling mode for genome-like inputs: `--organism prok|euk`.
- Modular steps with clear CLI:
  - `bitescore pipeline` (end-to-end)
  - `bitescore load`, `bitescore call-genes`, `bitescore features`, `bitescore rank`, `bitescore report`
  - `bitescore call-genes` accepts the same `--input`/`--input-type` pairing as the pipeline when you want to run loading and gene-calling together.
- Optional tools: DIAMOND/BLAST → GO TSV mapping, HMMER (Pfam), InterProScan, CD-HIT clustering, low-complexity masking.
- Optional AlphaFold lookup by UniProt accession.
- Gradio **chatbot UI**: `bitescore-chat`.

## Quickstart
```bash
mamba env create -f requirements.yml
mamba activate bitescore-chat
pip install -e .
bitescore pipeline --input data/examples/example_proteome.faa --input-type proteome --organism prok --out results/run1 --train
```

## Chat UI
```bash
bitescore-chat   # http://localhost:7860
```

See `resources/README.txt` for optional DBs.

## Biological background

Proteins vary widely in how readily gastrointestinal or secreted proteases can liberate amino acids. Nutritional quality, feed digestibility, and the utility of engineered enzymes all depend on how quickly enzymes such as trypsin, chymotrypsin, and acidic proteases cleave exposed peptide bonds. Accessibility of Lys/Arg-rich loops for trypsin or aromatic residues for chymotrypsin, the presence of flexible linkers, and the avoidance of heavily cross-linked or glycosylated motifs all contribute to the overall digestibility of a sequence. When working from genomes or metagenomes, identifying accurate coding sequences is therefore an essential precursor to reasoning about downstream protein-level traits.

## Biology-aware feature set

- **Sequence composition.** bitescore quantifies essential amino acid content and the relative frequency of each residue so that high-value nutritional profiles can be surfaced from large proteomes or metagenomic assemblies.
- **Physicochemical proxies.** Aromatic content, charge balance, and simple glycosylation site proxies provide context about how resistant a protein might be to specific proteases or post-translational modifications.
- **Cleavage accessibility.** Counts of Lys/Arg (trypsin) and Phe/Trp/Tyr (chymotrypsin) sites are paired with heuristic exposure and flexibility scores, giving the ML model features that approximate how easily enzymes can reach their preferred motifs.
- **Functional annotation hooks.** Optional DIAMOND/BLAST searches followed by GO term mapping, Pfam domain scans, and InterProScan integration allow downstream analyses to connect digestibility predictions with biological roles, secretion signals, or enzyme families of interest.
- **Structural context.** A lightweight structure-aware proxy is cached for every sequence, and when UniProt accessions are detected the pipeline can pull AlphaFold summary statistics (e.g., mean pLDDT) to better interpret disorder or confidence in structural models.
- **Genome-aware preprocessing.** Built-in gene calling, low-complexity masking, and CD-HIT clustering ensure that only representative, biologically plausible protein sequences contribute to the feature matrix and subsequent ranking.

## Interpreting digestibility scores

The default ranking model is a random forest regressor that operates on the feature matrix described above. In demonstration mode it self-trains on heuristics emphasizing essential amino acid abundance and trypsin-accessible sites, but in production settings you should retrain the model with empirical digestibility measurements that match your organism, protease cocktail, or processing environment. Predictions are best treated as a prioritization aid: experimental assays (in vitro digestion, animal trials, or proteomics) remain the gold standard for validating protein digestibility. Use the optional annotation layers to connect high-scoring candidates with their biological context before investing in laboratory follow-up.
