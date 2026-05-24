"""
Generate all in-silico peptide pairs from a FASTA protein database.

Mirrors the prediction-mode behaviour of the Java carafe-rank utility:
for every protein, digest the sequence in silico, then emit every pair of
(peptide, charge) forms as a tab-separated row.

Output columns:  protein | peptide_pair | peptide_a | peptide_b
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from typing import Iterator


# ---------------------------------------------------------------------------
# FASTA parser
# ---------------------------------------------------------------------------


def _parse_protein_id(header: str) -> str:
    token = header.split()[0] if header.split() else header
    # Handle UniProt format: sp|ACCESSION|NAME or tr|ACCESSION|NAME
    if token.startswith(("sp|", "tr|")):
        parts = token.split("|")
        return parts[1] if len(parts) >= 2 else token
    return token


def parse_fasta(path: str) -> Iterator[tuple[str, str]]:
    """Yield (protein_id, sequence) pairs from a FASTA file."""
    protein_id: str | None = None
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if protein_id is not None:
                    yield protein_id, "".join(buf)
                protein_id = _parse_protein_id(line[1:])
                buf = []
            else:
                buf.append(line)
    if protein_id is not None:
        yield protein_id, "".join(buf)


# ---------------------------------------------------------------------------
# Enzyme digestion
# ---------------------------------------------------------------------------

# Each entry: (regex for cleavage residue, cut_after_match, apply_p_rule)
# cut_after=True  → cut right after the matched residue (most enzymes)
# cut_after=False → cut before the matched residue (Arg-N)
# p_rule=True     → suppress cut when the following residue is P
_ENZYME_DEF: dict[int, tuple[str, bool, bool] | None] = {
    0: None,  # non-enzyme  → whole protein
    1: ("[KR]", True, True),  # Trypsin
    2: ("[KR]", True, False),  # Trypsin (no P rule)
    3: ("R", True, True),  # Arg-C
    4: ("R", True, False),  # Arg-C (no P rule)
    5: ("R", False, False),  # Arg-N (cut before R)
    6: ("[ED]", True, False),  # Glu-C
    7: ("K", True, False),  # Lys-C
}

ENZYME_NAMES = {
    0: "non-enzyme",
    1: "trypsin",
    2: "trypsin-nop",
    3: "argc",
    4: "argc-nop",
    5: "argn",
    6: "gluc",
    7: "lysc",
}


def _cleavage_sites(seq: str, enzyme_id: int) -> list[int]:
    """
    Return sorted cut positions (index of the first residue in the next
    segment, i.e. the split point in seq[a:b] notation).
    """
    defn = _ENZYME_DEF[enzyme_id]
    if defn is None:
        return []

    pattern, cut_after, p_rule = defn
    sites: list[int] = []

    for m in re.finditer(pattern, seq):
        if cut_after:
            site = m.end()  # position right after the matched residue
            if p_rule and site < len(seq) and seq[site] == "P":
                continue
            if site < len(seq):
                sites.append(site)
        else:
            # Arg-N: cut before the matched residue
            site = m.start()
            if site > 0:
                sites.append(site)

    return sorted(set(sites))


def _add_peptides(
    seq: str,
    enzyme_id: int,
    max_missed: int,
    min_len: int,
    max_len: int,
    out: set[str],
) -> None:
    boundaries = [0] + _cleavage_sites(seq, enzyme_id) + [len(seq)]
    n = len(boundaries)
    for i in range(n - 1):
        # j spans from i+1 (0 missed cleavages) up to i+max_missed+1
        for j in range(i + 1, min(i + max_missed + 2, n)):
            pep = seq[boundaries[i] : boundaries[j]]
            if min_len <= len(pep) <= max_len:
                out.add(pep)


def digest(
    seq: str,
    enzyme_id: int,
    max_missed: int,
    min_len: int,
    max_len: int,
    clip_n_term_m: bool = True,
) -> set[str]:
    """Return the set of peptides from in-silico digestion of *seq*."""
    seq = seq.upper().rstrip("*")
    peptides: set[str] = set()
    _add_peptides(seq, enzyme_id, max_missed, min_len, max_len, peptides)
    # N-terminal methionine clipping: also digest the sequence without leading M
    if clip_n_term_m and seq.startswith("M") and len(seq) > 1:
        _add_peptides(seq[1:], enzyme_id, max_missed, min_len, max_len, peptides)
    return peptides


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------


def generate_pairs(
    fasta_path: str,
    enzyme_id: int = 1,
    max_missed: int = 0,
    min_len: int = 7,
    max_len: int = 35,
    min_charge: int = 2,
    max_charge: int = 4,
    clip_n_term_m: bool = True,
    i2l: bool = False,
    output_path: str | None = None,
    n_proteins: int | None = None,
    random_seed: int = 42,
) -> None:
    """
    Read *fasta_path*, digest every protein, and write all unordered pairs of
    (peptide, charge) forms to *output_path* (stdout when None).
    """
    charges = list(range(min_charge, max_charge + 1))

    all_proteins = list(parse_fasta(fasta_path))
    if n_proteins is not None:
        import random

        rng = random.Random(random_seed)
        all_proteins = rng.sample(all_proteins, min(n_proteins, len(all_proteins)))

    fh = open(output_path, "w") if output_path else sys.stdout
    try:
        fh.write("protein\tpeptide_pair\tpeptide_a\tpeptide_b\n")
        for protein_id, seq in all_proteins:
            if i2l:
                seq = seq.replace("I", "L")
            peptides = digest(
                seq, enzyme_id, max_missed, min_len, max_len, clip_n_term_m
            )
            # All (peptide, charge) forms, sorted for reproducible output
            forms = [(pep, ch) for pep in sorted(peptides) for ch in charges]
            for (pep_a, ch_a), (pep_b, ch_b) in itertools.permutations(forms, 2):
                form_a = f"{pep_a}|{ch_a}"
                form_b = f"{pep_b}|{ch_b}"
                fh.write(f"{protein_id}\t{form_a}:{form_b}\t{form_a}\t{form_b}\n")
    finally:
        if output_path:
            fh.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bromo-pairs",
        description=(
            "Generate all in-silico peptide pairs from a FASTA protein database.\n"
            "Output is a TSV with columns: protein, peptide_pair, peptide_a, peptide_b."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Enzyme IDs:\n"
            "  0  non-enzyme (whole protein)\n"
            "  1  Trypsin (default)\n"
            "  2  Trypsin (no P rule)\n"
            "  3  Arg-C\n"
            "  4  Arg-C (no P rule)\n"
            "  5  Arg-N\n"
            "  6  Glu-C\n"
            "  7  Lys-C"
        ),
    )
    parser.add_argument(
        "-db", required=True, metavar="<fasta>", help="Input FASTA file"
    )
    parser.add_argument(
        "-o", metavar="<file>", help="Output TSV file (default: stdout)"
    )
    parser.add_argument(
        "-enzyme",
        type=int,
        default=1,
        metavar="<int>",
        choices=list(_ENZYME_DEF),
        help="Enzyme ID (default: 1 = Trypsin)",
    )
    parser.add_argument(
        "-miss_c",
        type=int,
        default=0,
        metavar="<int>",
        help="Max missed cleavages (default: 0)",
    )
    parser.add_argument(
        "-min_pep_length",
        type=int,
        default=7,
        metavar="<int>",
        help="Min peptide length (default: 7)",
    )
    parser.add_argument(
        "-max_pep_length",
        type=int,
        default=35,
        metavar="<int>",
        help="Max peptide length (default: 35)",
    )
    parser.add_argument(
        "-min_pep_charge",
        type=int,
        default=2,
        metavar="<int>",
        help="Min precursor charge (default: 2)",
    )
    parser.add_argument(
        "-max_pep_charge",
        type=int,
        default=4,
        metavar="<int>",
        help="Max precursor charge (default: 4)",
    )
    parser.add_argument(
        "--i2l", action="store_true", help="Convert I → L before digestion"
    )
    parser.add_argument(
        "--no-clip-m",
        dest="clip_n_term_m",
        action="store_false",
        help="Disable N-terminal Met clipping (enabled by default)",
    )
    parser.add_argument(
        "-n_proteins",
        type=int,
        default=None,
        metavar="<int>",
        help="Randomly subsample this many proteins (default: all)",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=42,
        metavar="<int>",
        help="Random seed for protein subsampling (default: 42)",
    )
    args = parser.parse_args()

    generate_pairs(
        fasta_path=args.db,
        enzyme_id=args.enzyme,
        max_missed=args.miss_c,
        min_len=args.min_pep_length,
        max_len=args.max_pep_length,
        min_charge=args.min_pep_charge,
        max_charge=args.max_pep_charge,
        clip_n_term_m=args.clip_n_term_m,
        i2l=args.i2l,
        output_path=args.o,
        n_proteins=args.n_proteins,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
