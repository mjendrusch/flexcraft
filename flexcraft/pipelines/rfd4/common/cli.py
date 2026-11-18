"""CLI argument parsing and output directory helpers for RFD4 pipelines."""

from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

import flexcraft.sequence.aa_codes as aas
from flexcraft.files.csv import ScoreCSV

_CANONICAL_AA = frozenset(aas.AF2_CODE)


def parse_bool(value: str | bool) -> bool:
    """Parse a boolean CLI flag accepting both bare flags and True/False strings."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ("true", "t", "yes", "y", "1"):
        return True
    if normalized in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def add_gen_only_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--gen_only`` / ``--no-gen_only`` flags to an RFD4 pipeline parser."""
    parser.add_argument(
        "--gen_only",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="{True,False}",
        help="only generate RFD4 structures into attempts/, skipping structure prediction and scores.csv",
    )
    parser.add_argument(
        "--no-gen_only",
        dest="gen_only",
        action="store_false",
        help="enable structure prediction and score writing (default)",
    )


def add_forbidden_args(
    parser: argparse.ArgumentParser,
    *,
    default: str = "C",
    help_text: str | None = None,
) -> None:
    """Add ``--forbidden`` single-letter amino acid exclusion flag to an RFD4 parser."""
    default_help = (
        "single-letter amino acid codes to forbid on designed residues via C_SEQ_PROB_<AA>=0.0 "
        f"(default: {default!r}; pass '' or 'none' to allow all)"
    )
    parser.add_argument(
        "--forbidden",
        type=str,
        default=default,
        help=default_help if help_text is None else help_text,
    )


def parse_forbidden_aas(forbidden: str) -> tuple[str, ...]:
    """Validate and deduplicate single-letter canonical amino acid codes to forbid."""
    cleaned = forbidden.strip().upper()
    if cleaned in ("", "NONE"):
        return ()
    invalid = sorted(set(cleaned) - _CANONICAL_AA)
    if invalid:
        raise ValueError(
            f"--forbidden contains non-canonical amino acid code(s): {invalid!r}."
        )
    ordered: list[str] = []
    for code in cleaned:
        if code not in ordered:
            ordered.append(code)
    if len(ordered) >= len(_CANONICAL_AA):
        raise ValueError("Cannot forbid all 20 canonical amino acids on designable residues.")
    return tuple(ordered)


def discover_pdb_paths(in_path: pathlib.Path) -> list[pathlib.Path]:
    """Resolve ``in_path`` to a sorted list of existing ``.pdb`` file paths."""
    if not in_path.exists():
        raise ValueError(f"--in_path does not exist: {in_path}")
    if in_path.is_file():
        return [in_path]
    if in_path.is_dir():
        paths = sorted(p for p in in_path.glob("*.pdb") if p.is_file())
        if not paths:
            raise ValueError(f"No .pdb files found in directory: {in_path}")
        return paths
    raise ValueError(f"--in_path is neither a file nor a directory: {in_path}")


def prepare_output_dir(
    out_path: pathlib.Path,
    *,
    overwrite: bool = False,
    subdirs: Sequence[str] = (),
) -> pathlib.Path:
    """Validate and create ``out_path`` plus requested subdirectories."""
    if out_path.exists():
        if not out_path.is_dir():
            raise FileExistsError(f"--out_path exists and is not a directory: {out_path}")
        if any(out_path.iterdir()) and not overwrite:
            raise FileExistsError(
                f"--out_path is non-empty ({out_path}); pass --overwrite to allow reuse"
            )
    out_path.mkdir(parents=True, exist_ok=True)
    for sub in subdirs:
        (out_path / sub).mkdir(parents=True, exist_ok=True)
    return out_path


def setup_rfd4_output_dirs(
    out_path: pathlib.Path,
    *,
    gen_only: bool = False,
    score_keys: Sequence[str] = (),
) -> tuple[pathlib.Path, pathlib.Path | None, ScoreCSV | None]:
    """Create RFD4 output directories and optional ``ScoreCSV`` writer."""
    attempts_dir = out_path / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    if gen_only:
        return attempts_dir, None, None
    predictions_dir = out_path / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = ScoreCSV(out_path / "scores.csv", keys=list(score_keys)) if score_keys else None
    return attempts_dir, predictions_dir, scores_csv
