"""Utilities for protease recognition site and cleavage counting features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ProteaseRule:
    """Definition of a protease cleavage preference.

    Parameters
    ----------
    name:
        Short handle used to build feature names.
    cleavage_side:
        "C" when the protease cleaves C-terminal to the specificity residues
        (i.e. after the residue), or "N" when the protease cleaves N-terminal
        to the specificity residues (i.e. before the residue).
    residues:
        A set of residues that define the specificity. When provided, cleavage
        is counted whenever the relevant residue is encountered and the
        ``blocked_by`` filter does not apply.
    bond_pairs:
        Explicit residue pairs (left, right) that define cleavage bonds. When
        supplied it overrides ``residues`` for those bonds, enabling enzymes
        such as pepsin that favour particular dipeptides.
    blocked_by:
        Residues that inhibit cleavage when directly adjacent to the cleavage
        bond on the opposite side of ``cleavage_side`` (e.g. Proline often
        blocks serine proteases when it follows the recognition residue).
    """

    name: str
    cleavage_side: str
    residues: frozenset[str] | None = None
    bond_pairs: frozenset[tuple[str, str]] | None = None
    blocked_by: frozenset[str] = frozenset()

    def _matches(self, left: str, right: str) -> bool:
        bond_pairs = self.bond_pairs or frozenset()
        if bond_pairs and (left, right) in bond_pairs:
            return True

        residues = self.residues or frozenset()
        if not residues:
            return False

        side = self.cleavage_side.upper()
        if side == "C":
            return left in residues and right not in self.blocked_by
        if side == "N":
            return right in residues and left not in self.blocked_by
        return False

    def cleavage_positions(self, seq: str) -> list[int]:
        """Return zero-based residue positions for recognised cleavage bonds."""

        if len(seq) < 2:
            return []

        positions: list[int] = []
        side = self.cleavage_side.upper()

        for index, (left, right) in enumerate(zip(seq, seq[1:])):
            if not self._matches(left, right):
                continue

            if side == "N":
                positions.append(index + 1)
            else:
                positions.append(index)

        return positions

    def count_sites(self, seq: str) -> int:
        """Count recognition/cleavage bonds in *seq* for this protease."""

        return len(self.cleavage_positions(seq))


DEFAULT_PROTEASES: tuple[ProteaseRule, ...] = (
    ProteaseRule(
        name="trypsin",
        cleavage_side="C",
        residues=frozenset("KR"),
        blocked_by=frozenset("P"),
    ),
    ProteaseRule(
        name="chymotrypsin",
        cleavage_side="C",
        residues=frozenset("FWY"),
        blocked_by=frozenset("P"),
    ),
    ProteaseRule(
        name="lys_c",
        cleavage_side="C",
        residues=frozenset("K"),
        blocked_by=frozenset("P"),
    ),
    ProteaseRule(
        name="arg_c",
        cleavage_side="C",
        residues=frozenset("R"),
        blocked_by=frozenset("P"),
    ),
    ProteaseRule(
        name="glu_c",
        cleavage_side="C",
        residues=frozenset("E"),
        blocked_by=frozenset("P"),
    ),
    ProteaseRule(
        name="asp_n",
        cleavage_side="N",
        residues=frozenset("D"),
        blocked_by=frozenset("P"),
    ),
    ProteaseRule(
        name="lys_n",
        cleavage_side="N",
        residues=frozenset("K"),
        blocked_by=frozenset("P"),
    ),
    ProteaseRule(
        name="pepsin",
        cleavage_side="C",
        bond_pairs=frozenset(
            {
                ("F", "L"),
                ("F", "W"),
                ("F", "Y"),
                ("W", "L"),
                ("Y", "L"),
                ("Y", "W"),
            }
        ),
    ),
)


EXPOSURE_PROP = {
    "A": 0.6,
    "C": 0.4,
    "D": 0.8,
    "E": 0.8,
    "F": 0.2,
    "G": 0.9,
    "H": 0.5,
    "I": 0.2,
    "K": 0.9,
    "L": 0.3,
    "M": 0.3,
    "N": 0.8,
    "P": 0.9,
    "Q": 0.7,
    "R": 0.8,
    "S": 0.8,
    "T": 0.7,
    "V": 0.3,
    "W": 0.1,
    "Y": 0.3,
}


def _avg_exposure(seq: str) -> float:
    if not seq:
        return 0.0
    vals = [EXPOSURE_PROP.get(a, 0.5) for a in seq]
    return sum(vals) / len(vals)


def _flexibility(seq: str, k: int = 7) -> float:
    if not seq:
        return 0.0
    if len(seq) <= k:
        return _avg_exposure(seq)
    vals = []
    for i in range(len(seq) - k + 1):
        window = seq[i : i + k]
        vals.append(_avg_exposure(window))
    return sum(vals) / len(vals)


def protease_cleavage_counts(
    seq: str,
    proteases: Iterable[ProteaseRule] | None = None,
) -> Mapping[str, int]:
    """Return the number of predicted cleavage bonds for each protease."""

    proteases = tuple(proteases or DEFAULT_PROTEASES)
    counts: dict[str, int] = {}
    total = 0
    for protease in proteases:
        count = protease.count_sites(seq)
        counts[f"protease_{protease.name}_sites"] = count
        total += count
    counts["protease_total_sites"] = total
    return counts


def cleavage_site_positions(
    seq: str, proteases: Iterable[ProteaseRule] | None = None
) -> list[int]:
    """Return sorted unique residue indices predicted as cleavage sites."""

    proteases = tuple(proteases or DEFAULT_PROTEASES)
    positions: set[int] = set()
    for protease in proteases:
        positions.update(protease.cleavage_positions(seq))
    return sorted(positions)


def _trypsin_specific_counts(seq: str) -> tuple[int, int]:
    """Return counts for Lys- and Arg-mediated trypsin bonds."""

    k_count = 0
    r_count = 0
    for left, right in zip(seq, seq[1:]):
        if right == "P":
            continue
        if left == "K":
            k_count += 1
        if left == "R":
            r_count += 1
    return k_count, r_count


def _chymotrypsin_count(seq: str) -> int:
    total = 0
    for left, right in zip(seq, seq[1:]):
        if left in {"F", "W", "Y"} and right != "P":
            total += 1
    return total


def cleavage_accessibility_scores(seq: str) -> dict:
    trypsin_k, trypsin_r = _trypsin_specific_counts(seq)
    chymo = _chymotrypsin_count(seq)
    acidic = seq.count("D") + seq.count("E")
    exposure = _avg_exposure(seq)
    flex = _flexibility(seq, k=7)
    protease_counts = dict(protease_cleavage_counts(seq))

    return {
        **protease_counts,
        "trypsin_K_sites": trypsin_k,
        "trypsin_R_sites": trypsin_r,
        "chymotrypsin_sites": chymo,
        "acidic_residues": acidic,
        "exposure_avg": exposure,
        "flexibility_win7": flex,
        "cleavage_accessibility_proxy": 0.5 * exposure + 0.5 * flex,
    }


__all__ = [
    "ProteaseRule",
    "DEFAULT_PROTEASES",
    "protease_cleavage_counts",
    "cleavage_site_positions",
    "cleavage_accessibility_scores",
]
