import click
from pathlib import Path
from .utils.config import load_config
from .pipeline import (
    run_pipeline,
    step_load,
    step_call_genes,
    step_features_aa,
    step_features_regsite,
    step_features_structure,
    step_features_function,
    step_features_esm,
    step_train_mil,
    step_rank,
)
from .report import make_report

@click.group()
def main():
    """bitescore: Predict and rank digestibility of proteins."""

def common_opts(f):
    f = click.option("--config", type=click.Path(path_type=Path), default=None, help="YAML config.")(f)
    f = click.option("--out", "outdir", required=True, type=click.Path(path_type=Path), help="Output directory.")(f)
    f = click.option("--threads", type=int, default=None, help="Parallel threads.")(f)
    return f

@main.command(name="pipeline")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path), help="Input file/folder.")
@click.option("--input-type", required=True, type=click.Choice(["genome","proteome","sequences","genomes","metagenome"]))
@click.option("--organism", required=False, type=click.Choice(["prok","euk"]), default=None, help="Required for genome-like inputs.")
@click.option("--no-structure", is_flag=True, help="Disable structure retrieval features.")
@click.option("--alphafold", is_flag=True, help="Enable AlphaFold DB retrieval (requires internet).")
@click.option("--model", type=click.Path(path_type=Path), default=None, help="Pretrained model .joblib path.")
@click.option("--train", is_flag=True, help="Train a simple model on-the-fly (demo).")
@click.option("--go-map", type=click.Path(exists=False, path_type=Path), default=None, help="TSV mapping accession->GO list.")
@click.option("--diamond-db", type=click.Path(path_type=Path), default=None, help="DIAMOND database path.")
@click.option("--blast-db", type=click.Path(path_type=Path), default=None, help="BLAST database path.")
@click.option("--pfam-hmms", type=click.Path(path_type=Path), default=None, help="Pfam-A HMMs file.")
@click.option("--interpro", is_flag=True, help="Run InterProScan if available.")
@click.option("--pfam2go", type=click.Path(path_type=Path), default=None, help="Pfam2GO mapping file for Pfam domain -> GO term annotation.")
@click.option("--interpro2go", type=click.Path(path_type=Path), default=None, help="InterPro2GO mapping file.")
@click.option("--diamond-evalue", type=float, default=None, help="DIAMOND E-value cutoff (default 1e-5).")
@click.option("--blast-evalue", type=float, default=None, help="BLAST E-value cutoff (default 1e-5).")
@click.option("--pfam-evalue", type=float, default=None, help="Pfam hmmscan E-value cutoff (default 1e-5).")
@click.option("--cluster-cdhit", is_flag=True, help="Cluster sequences with CD-HIT before features.")
@click.option("--cdhit-threshold", type=float, default=None, help="CD-HIT identity threshold (default 0.95).")
@click.option("--low-complexity", is_flag=True, help="Mask low-complexity regions (segmasker) before features.")
@click.option("--esm", is_flag=True, help="Enable ESM-2 protein language model embeddings (requires torch + fair-esm).")
@click.option("--esm-model", type=str, default=None, help="ESM-2 model name (default: esm2_t6_8M_UR50D).")
@click.option("--no-calibrate", is_flag=True, help="Disable DIAAS calibration (calibration is on by default).")
@click.option("--calibration-method", type=click.Choice(["isotonic", "linear"]), default=None, help="Calibration method (default: isotonic).")
@click.option("--mil-model", type=click.Path(path_type=Path), default=None, help="Pre-trained MIL model (.pt) path.")
@click.option("--mil-train", is_flag=True, help="Train a MIL model on reference digestibility data.")
@click.option("--digestibility-ref", type=click.Path(exists=True, path_type=Path), default=None, help="CSV with food-level experimental digestibility values.")
@click.option("--food-composition", type=click.Path(exists=True, path_type=Path), default=None, help="CSV with protein abundance per food.")
@common_opts
def pipeline(input_path, input_type, organism, no_structure, alphafold, model, train, go_map, diamond_db, blast_db, pfam_hmms, interpro, pfam2go, interpro2go, diamond_evalue, blast_evalue, pfam_evalue, cluster_cdhit, cdhit_threshold, low_complexity, esm, esm_model, no_calibrate, calibration_method, mil_model, mil_train, digestibility_ref, food_composition, config, outdir, threads):
    overrides = dict(
        input_path=str(input_path),
        input_type=input_type,
        organism=organism,
        structure_enabled=not no_structure,
        alphafold_enabled=bool(alphafold),
        outdir=str(outdir),
        model_path=str(model) if model else None,
        train_demo=bool(train),
        go_map=str(go_map) if go_map else None,
        diamond_db=str(diamond_db) if diamond_db else None,
        blast_db=str(blast_db) if blast_db else None,
        pfam_hmms=str(pfam_hmms) if pfam_hmms else None,
        interpro=bool(interpro),
        pfam2go=str(pfam2go) if pfam2go else None,
        interpro2go=str(interpro2go) if interpro2go else None,
        diamond_evalue=diamond_evalue,
        blast_evalue=blast_evalue,
        pfam_evalue=pfam_evalue,
        cluster_cdhit=bool(cluster_cdhit),
        cdhit_threshold=cdhit_threshold,
        low_complexity=bool(low_complexity),
        esm_enabled=bool(esm),
        esm_model=esm_model,
        calibrate=not bool(no_calibrate),
        calibration_method=calibration_method,
        mil_model_path=str(mil_model) if mil_model else None,
        mil_train=bool(mil_train),
        digestibility_ref=str(digestibility_ref) if digestibility_ref else None,
        food_composition=str(food_composition) if food_composition else None,
        threads=threads,
    )
    cfg = load_config(str(config) if config else None, overrides)
    if cfg["input_type"] in {"genome","genomes","metagenome"} and cfg.get("organism") is None:
        raise click.UsageError("For genome-like inputs you must pass --organism prok|euk for gene-calling.")
    run_pipeline(cfg)

@main.command(name="load")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path), help="Input file/folder.")
@click.option("--input-type", required=True, type=click.Choice(["genome","proteome","sequences","genomes","metagenome"]))
@common_opts
def cmd_load(input_path, input_type, config, outdir, threads):
    cfg = load_config(str(config) if config else None, dict(input_path=str(input_path), input_type=input_type, outdir=str(outdir), threads=threads))
    step_load(cfg)

@main.command(name="call-genes")
@click.option("--input", "input_path", required=False, type=click.Path(exists=True, path_type=Path), help="Input file/folder. If provided, runs the load step first.")
@click.option("--input-type", required=False, type=click.Choice(["genome","proteome","sequences","genomes","metagenome"]))
@click.option("--organism", required=True, type=click.Choice(["prok","euk"]))
@click.option("--loaded", "loaded_path", required=False, type=click.Path(exists=True, path_type=Path), help="Existing loaded sequences FASTA to use instead of the default location.")
@common_opts
def cmd_call_genes(input_path, input_type, organism, loaded_path, config, outdir, threads):
    overrides = dict(
        organism=organism,
        outdir=str(outdir),
        threads=threads,
        input_type=input_type,
        loaded_path=str(loaded_path) if loaded_path else None,
    )
    if input_path is not None:
        overrides["input_path"] = str(input_path)
    cfg = load_config(str(config) if config else None, overrides)
    if input_path is not None:
        if cfg.get("input_type") is None:
            raise click.UsageError("--input-type must be provided when --input is set.")
        step_load(cfg)
    step_call_genes(cfg)

@main.command(name="features-aa")
@click.option("--sequences", required=False, type=click.Path(exists=True, path_type=Path), help="Override FASTA input for feature extraction.")
@common_opts
def cmd_features_aa(sequences, config, outdir, threads):
    cfg = load_config(
        str(config) if config else None,
        dict(outdir=str(outdir), threads=threads, feature_sequences=str(sequences) if sequences else None)
    )
    step_features_aa(cfg)


@main.command(name="features-regsite")
@click.option("--sequences", required=False, type=click.Path(exists=True, path_type=Path), help="Override FASTA input for feature extraction.")
@common_opts
def cmd_features_regsite(sequences, config, outdir, threads):
    cfg = load_config(
        str(config) if config else None,
        dict(outdir=str(outdir), threads=threads, feature_sequences=str(sequences) if sequences else None)
    )
    step_features_regsite(cfg)


@main.command(name="features-structure")
@click.option("--no-structure", is_flag=True)
@click.option("--alphafold", is_flag=True)
@click.option("--sequences", required=False, type=click.Path(exists=True, path_type=Path), help="Override FASTA input for feature extraction.")
@common_opts
def cmd_features_structure(no_structure, alphafold, sequences, config, outdir, threads):
    overrides = dict(
        structure_enabled=not no_structure,
        alphafold_enabled=bool(alphafold),
        outdir=str(outdir),
        threads=threads,
        feature_sequences=str(sequences) if sequences else None,
    )
    cfg = load_config(str(config) if config else None, overrides)
    step_features_structure(cfg)


@main.command(name="features-function")
@click.option("--go-map", type=click.Path(exists=False, path_type=Path), default=None)
@click.option("--diamond-db", type=click.Path(path_type=Path), default=None)
@click.option("--blast-db", type=click.Path(path_type=Path), default=None)
@click.option("--pfam-hmms", type=click.Path(path_type=Path), default=None)
@click.option("--interpro", is_flag=True)
@click.option("--pfam2go", type=click.Path(path_type=Path), default=None, help="Pfam2GO mapping file.")
@click.option("--interpro2go", type=click.Path(path_type=Path), default=None, help="InterPro2GO mapping file.")
@click.option("--diamond-evalue", type=float, default=None, help="DIAMOND E-value cutoff.")
@click.option("--blast-evalue", type=float, default=None, help="BLAST E-value cutoff.")
@click.option("--pfam-evalue", type=float, default=None, help="Pfam hmmscan E-value cutoff.")
@click.option("--sequences", required=False, type=click.Path(exists=True, path_type=Path), help="Override FASTA input for feature extraction.")
@common_opts
def cmd_features_function(go_map, diamond_db, blast_db, pfam_hmms, interpro, pfam2go, interpro2go, diamond_evalue, blast_evalue, pfam_evalue, sequences, config, outdir, threads):
    overrides = dict(
        go_map=str(go_map) if go_map else None,
        diamond_db=str(diamond_db) if diamond_db else None,
        blast_db=str(blast_db) if blast_db else None,
        pfam_hmms=str(pfam_hmms) if pfam_hmms else None,
        interpro=bool(interpro),
        pfam2go=str(pfam2go) if pfam2go else None,
        interpro2go=str(interpro2go) if interpro2go else None,
        diamond_evalue=diamond_evalue,
        blast_evalue=blast_evalue,
        pfam_evalue=pfam_evalue,
        outdir=str(outdir),
        threads=threads,
        feature_sequences=str(sequences) if sequences else None,
    )
    cfg = load_config(str(config) if config else None, overrides)
    step_features_function(cfg)

@main.command(name="features-esm")
@click.option("--sequences", required=False, type=click.Path(exists=True, path_type=Path), help="Override FASTA input for feature extraction.")
@click.option("--esm-model", type=str, default=None, help="ESM-2 model name (default: esm2_t6_8M_UR50D).")
@common_opts
def cmd_features_esm(sequences, esm_model, config, outdir, threads):
    overrides = dict(
        outdir=str(outdir),
        threads=threads,
        feature_sequences=str(sequences) if sequences else None,
        esm_enabled=True,
        esm_model=esm_model,
    )
    cfg = load_config(str(config) if config else None, overrides)
    step_features_esm(cfg)


@main.command(name="train-mil")
@click.option("--esm", is_flag=True, help="Include ESM embeddings in MIL training features.")
@click.option("--esm-model", type=str, default=None, help="ESM-2 model name.")
@click.option("--digestibility-ref", type=click.Path(exists=True, path_type=Path), default=None, help="CSV with food-level digestibility values.")
@click.option("--food-composition", type=click.Path(exists=True, path_type=Path), default=None, help="CSV with protein abundance per food.")
@click.option("--mil-epochs", type=int, default=None, help="MIL training epochs (default: 200).")
@click.option("--mil-lr", type=float, default=None, help="MIL learning rate (default: 1e-3).")
@common_opts
def cmd_train_mil(esm, esm_model, digestibility_ref, food_composition, mil_epochs, mil_lr, config, outdir, threads):
    overrides = dict(
        outdir=str(outdir),
        threads=threads,
        esm_enabled=bool(esm),
        esm_model=esm_model,
        mil_train=True,
        digestibility_ref=str(digestibility_ref) if digestibility_ref else None,
        food_composition=str(food_composition) if food_composition else None,
        mil_epochs=mil_epochs,
        mil_lr=mil_lr,
    )
    cfg = load_config(str(config) if config else None, overrides)
    step_train_mil(cfg)


@main.command(name="rank")
@click.option("--model", type=click.Path(path_type=Path), default=None)
@click.option("--train", is_flag=True)
@click.option("--no-calibrate", is_flag=True, help="Disable DIAAS calibration.")
@click.option("--mil-model", type=click.Path(path_type=Path), default=None, help="Pre-trained MIL model (.pt) path.")
@click.option("--features-aa", "features_aa", type=click.Path(exists=True, path_type=Path), default=None, help="Path to a precomputed amino acid feature table.")
@click.option("--features-regsite", "features_regsite", type=click.Path(exists=True, path_type=Path), default=None, help="Path to a precomputed protease recognition site feature table.")
@click.option("--features-structure", "features_structure", type=click.Path(exists=True, path_type=Path), default=None, help="Path to a precomputed structure accessibility feature table.")
@click.option("--features-function", "features_function", type=click.Path(exists=True, path_type=Path), default=None, help="Path to a precomputed function feature table.")
@click.option("--features-esm", "features_esm", type=click.Path(exists=True, path_type=Path), default=None, help="Path to a precomputed ESM embedding feature table.")
@common_opts
def cmd_rank(model, train, no_calibrate, mil_model, features_aa, features_regsite, features_structure, features_function, features_esm, config, outdir, threads):
    cfg = load_config(
        str(config) if config else None,
        dict(
            model_path=str(model) if model else None,
            train_demo=bool(train),
            calibrate=not bool(no_calibrate),
            mil_model_path=str(mil_model) if mil_model else None,
            outdir=str(outdir),
            threads=threads,
            features_aa_path=str(features_aa) if features_aa else None,
            features_regsite_path=str(features_regsite) if features_regsite else None,
            features_structure_path=str(features_structure) if features_structure else None,
            features_function_path=str(features_function) if features_function else None,
            features_esm_path=str(features_esm) if features_esm else None,
        ),
    )
    step_rank(cfg)

@main.command(name="report")
@common_opts
def cmd_report(config, outdir, threads):
    out = make_report(Path(outdir))
    click.echo(f"Report: {out}")
