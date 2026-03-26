"""ESM-2 protein language model embeddings for digestibility prediction.

Extracts per-protein embeddings from Meta's ESM-2 pretrained protein language
model.  The embeddings encode evolutionary, structural, and functional
information learned from millions of protein sequences — significantly richer
than handcrafted features alone.

Requires the optional ``torch`` and ``fair-esm`` packages::

    pip install torch fair-esm

When these are not installed the module gracefully returns empty DataFrames so
that the rest of the pipeline continues to work.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maximum sequence length ESM-2 can handle in one pass (1022 tokens + BOS/EOS)
_ESM_MAX_SEQ_LEN = 1022

# Available ESM-2 model tiers (name -> embedding dim)
ESM_MODELS: Dict[str, int] = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D": 2560,
}

DEFAULT_MODEL = "esm2_t6_8M_UR50D"


def _check_dependencies() -> bool:
    """Return True if torch and esm are importable."""
    try:
        import torch  # noqa: F401
        import esm  # noqa: F401
        return True
    except ImportError:
        return False


def _load_esm_model(model_name: str):
    """Load an ESM-2 model and its alphabet.

    Returns (model, alphabet, batch_converter, embedding_dim).
    """
    import torch
    import esm

    loader = getattr(esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(
            f"Unknown ESM model: {model_name}. "
            f"Available: {list(ESM_MODELS.keys())}"
        )
    model, alphabet = loader()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    embedding_dim = ESM_MODELS.get(model_name, 320)
    return model, alphabet, batch_converter, embedding_dim


def _truncate_sequence(seq: str, max_len: int = _ESM_MAX_SEQ_LEN) -> str:
    """Truncate a sequence to fit ESM-2's maximum input length."""
    if len(seq) > max_len:
        logger.warning(
            "Sequence length %d exceeds ESM max %d; truncating.",
            len(seq), max_len,
        )
        return seq[:max_len]
    return seq


def _embed_batch(
    batch_data: List[tuple],
    model,
    batch_converter,
    repr_layer: int,
) -> Dict[str, np.ndarray]:
    """Run a batch through ESM-2 and return mean-pooled embeddings keyed by ID."""
    import torch

    _, _, batch_tokens = batch_converter(batch_data)
    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

    representations = results["representations"][repr_layer]  # (B, L, D)

    embeddings: Dict[str, np.ndarray] = {}
    for i, (label, _seq) in enumerate(batch_data):
        seq_len = len(_seq)
        # Skip BOS token (index 0), take seq_len residue tokens
        token_repr = representations[i, 1: seq_len + 1]  # (seq_len, D)
        mean_repr = token_repr.mean(dim=0).cpu().numpy()
        embeddings[label] = mean_repr
    return embeddings


def compute_esm_embeddings(
    sequences: Sequence[tuple],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
    cache_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Compute ESM-2 mean-pooled embeddings for a list of (id, sequence) tuples.

    Parameters
    ----------
    sequences : list of (id, sequence) tuples
    model_name : ESM-2 model name
    batch_size : number of sequences per forward pass
    cache_dir : optional directory for caching embeddings

    Returns
    -------
    dict mapping sequence id -> embedding numpy array of shape (embedding_dim,)
    """
    if not _check_dependencies():
        logger.warning("torch/esm not installed; returning empty embeddings.")
        return {}

    model, alphabet, batch_converter, embedding_dim = _load_esm_model(model_name)
    repr_layer = model.num_layers

    # Prepare sequences (truncate if needed)
    prepared = [
        (sid, _truncate_sequence(seq))
        for sid, seq in sequences
        if len(seq) > 0
    ]

    # Check cache
    all_embeddings: Dict[str, np.ndarray] = {}
    to_compute: List[tuple] = []

    if cache_dir is not None:
        cache_dir = Path(cache_dir) / "esm_embeddings" / model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        for sid, seq in prepared:
            cache_key = hashlib.sha256(seq.encode()).hexdigest()[:16]
            cache_file = cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                all_embeddings[sid] = np.load(cache_file)
            else:
                to_compute.append((sid, seq))
    else:
        to_compute = prepared

    # Batch inference
    for i in range(0, len(to_compute), batch_size):
        batch = to_compute[i: i + batch_size]
        batch_embs = _embed_batch(batch, model, batch_converter, repr_layer)
        for sid, emb in batch_embs.items():
            all_embeddings[sid] = emb
            if cache_dir is not None:
                seq = dict(to_compute)[sid]
                cache_key = hashlib.sha256(seq.encode()).hexdigest()[:16]
                np.save(cache_dir / f"{cache_key}.npy", emb)

    return all_embeddings


def esm_embeddings_to_dataframe(
    embeddings: Dict[str, np.ndarray],
    prefix: str = "esm_",
) -> pd.DataFrame:
    """Convert a dict of embeddings to a DataFrame with prefixed column names.

    Each embedding dimension becomes a column: esm_0, esm_1, ..., esm_N.
    """
    if not embeddings:
        return pd.DataFrame(columns=["id"])

    ids = list(embeddings.keys())
    matrix = np.stack([embeddings[sid] for sid in ids])
    dim = matrix.shape[1]
    columns = [f"{prefix}{i}" for i in range(dim)]
    df = pd.DataFrame(matrix, columns=columns)
    df.insert(0, "id", ids)
    return df


def compute_esm_feature_table(
    records,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute ESM-2 embedding features for a list of SeqRecords.

    Returns a DataFrame with columns: id, esm_0, esm_1, ..., esm_N.
    If torch/esm are not installed, returns a DataFrame with only the id column.
    """
    if not _check_dependencies():
        logger.warning(
            "ESM dependencies (torch, fair-esm) not installed. "
            "Skipping ESM embedding features. Install with: "
            "pip install torch fair-esm"
        )
        return pd.DataFrame({"id": [rec.id for rec in records]})

    sequences = [(rec.id, str(rec.seq)) for rec in records]
    embeddings = compute_esm_embeddings(
        sequences,
        model_name=model_name,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )

    df = esm_embeddings_to_dataframe(embeddings)

    # Ensure all input records appear in the output (fill missing with NaN)
    all_ids = pd.DataFrame({"id": [rec.id for rec in records]})
    df = all_ids.merge(df, on="id", how="left")
    return df
