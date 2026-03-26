from typing import Callable, List, Optional
from Bio.SeqRecord import SeqRecord
from .orf import simple_orf_caller
from .external import call_prodigal, call_augustus

LogFn = Optional[Callable[[str], None]]


def call_genes_if_needed(
    records: List[SeqRecord],
    input_type: str,
    organism: str | None,
    logger: LogFn = None,
):
    if input_type in {"proteome","sequences"}:
        return records
    if input_type in {"genome","genomes","metagenome"}:
        if organism not in {"prok","euk"}:
            raise ValueError("For genome-like inputs, --organism must be 'prok' or 'euk'.")
        if organism == "prok":
            aa = call_prodigal(records, logger=logger)
            if aa: return aa
        else:
            aa = call_augustus(records)
            if aa: return aa
        return simple_orf_caller(records)
    raise ValueError(f"Unsupported input_type: {input_type}")
