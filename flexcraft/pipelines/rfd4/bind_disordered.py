"""De-novo binder design against a disordered target sequence with RFD4 and AF3Like filtering.

Adapts ``rfd4fx/scripts/binder_disordered.py`` with an MSA-free, template-free
``AF3Like`` structure prediction, scoring, and selection step mirroring
``flexcraft.pipelines.rfd4.bind``.

Example:
    python -m flexcraft.pipelines.rfd4.bind_disordered \\
        --target_sequence MERRYCHRISTMAS --num_designs 5 --num_aa 72
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

import jax
import numpy as np
from rfd4fx.flow_matching import RFD4Result, RFD4Sampler, THINKING_HIGH
from rfd4fx.model import RFD4
from rfd4fx.rfd4_input import RFD4Input

from flexcraft.pipelines.rfd4.bind import (
    SCORE_KEYS,
    evaluate_binder_prediction,
)
from flexcraft.pipelines.rfd4.common import (
    add_forbidden_args,
    add_gen_only_args,
    extract_binder_sequence,
    forbid_amino_acids,
    parse_bool,
    parse_forbidden_aas,
    setup_rfd4_output_dirs,
)
from flexcraft.structure.af3_like import AF3Like, AF3LikeSpec

_KIND_TO_POLYMER: dict[str, str] = {
    "aa": "protein",
    "dna": "dna",
    "rna": "rna",
}

_INTERACTION_SPECS: tuple[tuple[str, str, str], ...] = (
    ("h_bond", "C_HB", "C_HB donor/acceptor roles (1/2/3)"),
    ("salt_bridge", "C_SALTBRIDGE", "C_SALTBRIDGE charge sign (1/2)"),
    ("pi_stack", "C_PISTACK", "C_PISTACK on aromatic-ring atoms"),
    ("pi_cation", "C_PICATION", "C_PICATION aromatic (1) / cation (2) roles"),
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the disordered-target binder pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_sequence",
        required=True,
        help="disordered target sequence in one-letter codes",
    )
    parser.add_argument(
        "--kind",
        default="aa",
        choices=["aa", "dna", "rna"],
        help="target polymer vocabulary (default: aa)",
    )
    parser.add_argument(
        "--hotspots",
        default="",
        help="comma-separated 1-indexed target residues (e.g. '1,5,6-10'; C_HOT on sidechains)",
    )
    parser.add_argument(
        "--auto_condition",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="{True,False}",
        help="C_HB donor/acceptor conditioning on all target sidechain donors/acceptors "
        "(mutually exclusive with explicit interaction flags)",
    )
    parser.add_argument(
        "--no-auto_condition",
        dest="auto_condition",
        action="store_false",
        help="disable automatic C_HB donor/acceptor conditioning",
    )
    for key, _, description in _INTERACTION_SPECS:
        parser.add_argument(
            f"--{key}",
            default=None,
            help=f"{description} on the eligible subset of selected target residues/atoms "
            "(mutually exclusive with --auto_condition)",
        )
    parser.add_argument(
        "--model",
        default="joltz",
        help="AF3Like structure predictor identifier ('joltz', 'protenix_*', 'esmfold2_*', 'af2_*')",
    )
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=None,
        help="optional parameter cache directory passed to AF3Like",
    )
    parser.add_argument(
        "--num_designs",
        type=int,
        default=5,
        help="number of passing binder designs to generate before stopping",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=200,
        help="maximum number of design attempts before stopping",
    )
    parser.add_argument(
        "--num_aa",
        type=int,
        default=80,
        help="length of the designable protein binder chain",
    )
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        default=pathlib.Path("runs/rfd4-binder-disordered"),
    )
    parser.add_argument(
        "--f_min_ipae",
        type=float,
        default=2.0,
        help="maximum sample-mean min ipAE (in Angstroms, unnormalized) for a passing design",
    )
    parser.add_argument(
        "--f_plddt",
        type=float,
        default=0.8,
        help="minimum sample-mean binder pLDDT (in [0, 1]) for a passing design",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="number of structure predictor diffusion samples per design",
    )
    parser.add_argument(
        "--num_recycle",
        type=int,
        default=3,
        help="number of structure predictor recycling iterations",
    )
    add_forbidden_args(parser)
    add_gen_only_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def prepare_af3_spec(
    binder_sequence: str,
    target_sequence: str,
    *,
    kind: str = "aa",
    num_residue_atoms: int = 24,
) -> AF3LikeSpec:
    """Build an atom-padded, MSA-free, template-free ``AF3LikeSpec``.

    Args:
        binder_sequence: Designed 1-letter protein binder sequence.
        target_sequence: Disordered 1-letter target sequence.
        kind: Target polymer kind (``"aa"``, ``"dna"``, or ``"rna"``).
        num_residue_atoms: Per-residue atom budget for static atom padding.

    Returns:
        Configured ``AF3LikeSpec`` with ``use_msa=False`` on all chains and no templates.
    """
    if kind not in _KIND_TO_POLYMER:
        raise ValueError(f"Unsupported target kind {kind!r}; expected one of {sorted(_KIND_TO_POLYMER)}.")
    spec = (
        AF3LikeSpec()
        .add_protein(binder_sequence, use_msa=False)
        .add_polymer(target_sequence, kind=_KIND_TO_POLYMER[kind], use_msa=False)
    )
    if num_residue_atoms > 0:
        spec = spec.pad_atoms(num_residue_atoms=num_residue_atoms)
    return spec


def main(argv: Sequence[str] | None = None) -> None:
    """Run disordered-target RFD4 binder design with MSA-free AF3Like filtering."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        parse_forbidden_aas(args.forbidden)
    except ValueError as exc:
        parser.error(str(exc))

    interactions = {
        key: getattr(args, key)
        for key, _, _ in _INTERACTION_SPECS
        if getattr(args, key) is not None
    }
    if args.auto_condition and interactions:
        parser.error(
            "--auto_condition is mutually exclusive with the explicit interaction keys "
            "(--h_bond/--salt_bridge/--pi_stack/--pi_cation)"
        )
    if not args.gen_only and args.model.startswith("af2") and args.kind != "aa":
        raise ValueError(
            f"AF2 model {args.model!r} only supports protein targets (--kind aa), got {args.kind!r}."
        )

    attempts_dir, predictions_dir, scores_csv = setup_rfd4_output_dirs(
        args.out_path,
        gen_only=args.gen_only,
        score_keys=SCORE_KEYS,
    )

    target: dict[str, object] = {"kind": args.kind, "sequence": args.target_sequence}
    if args.auto_condition:
        target["auto_condition"] = True
    if args.hotspots:
        target["hotspot"] = args.hotspots
    target.update(interactions)

    inp = RFD4Input.from_chains(args.num_aa, target)
    inp = forbid_amino_acids(inp, args.forbidden, designable_mask=np.ones(args.num_aa, dtype=bool))
    n_target = int(np.asarray(inp.padding_mask).sum()) - args.num_aa
    channels = ["C_HOT"] if args.hotspots else []
    if args.auto_condition:
        channels.append("C_HB")
    channel_map = {key: tag for key, tag, _ in _INTERACTION_SPECS}
    channels += [channel_map[key] for key in interactions]
    print(
        f"compiled: {args.num_aa} design + {n_target} target tokens "
        f"({args.kind} sequence-only conditioning: {' + '.join(['C_SEQ'] + channels)}; "
        f"no coordinates)",
        flush=True,
    )

    jax_model = RFD4.from_pretrained()
    sampler = RFD4Sampler(jax_model.step(), jax_model)
    batch = inp.pad_to_nearest(64).to_batch()

    predictor = None
    if not args.gen_only:
        af3_kwargs = {"cache": str(args.cache)} if args.cache is not None else {}
        predictor = AF3Like(model=args.model, **af3_kwargs).predictor(
            num_recycle=args.num_recycle,
            num_samples=args.num_samples,
        )

    base_key = jax.random.key(args.seed)
    success_count = 0
    attempt = 0

    while success_count < args.num_designs and attempt < args.max_attempts:
        attempt_key = jax.random.fold_in(base_key, attempt)
        rfd4_key, pred_key = jax.random.split(attempt_key)

        rfd4_result: RFD4Result = sampler.sample(
            batch,
            key=rfd4_key,
            recycle_schedule=THINKING_HIGH,
        )
        attempt_pdb = attempts_dir / f"design_{attempt}.pdb"
        rfd4_result.save_pdb(attempt_pdb)

        if args.gen_only:
            success_count += 1
            print(
                f"design_{attempt}.pdb written (gen_only, {success_count}/{args.num_designs})",
                flush=True,
            )
            attempt += 1
            continue

        assert predictor is not None and predictions_dir is not None and scores_csv is not None
        rfd4_atom24 = rfd4_result.to_atom24()
        binder_sequence = extract_binder_sequence(
            rfd4_result,
            args.num_aa,
            rfd4_restype=rfd4_atom24.restype,
        )

        spec = prepare_af3_spec(
            binder_sequence,
            args.target_sequence,
            kind=args.kind,
        )
        prediction = predictor(pred_key, spec)
        summary, best_sample_idx = evaluate_binder_prediction(
            prediction.result,
            rfd4_atom24.coords,
            rfd4_atom24.mask,
            f_min_ipae=args.f_min_ipae,
            f_plddt=args.f_plddt,
        )

        prediction_pdb = predictions_dir / f"design_{attempt}.pdb"
        prediction.save_pdb(prediction_pdb, sample_index=best_sample_idx)

        row: dict[str, object] = {
            "attempt": attempt,
            "sequence": binder_sequence,
            **summary,
        }
        scores_csv.write_line(row)

        if summary["success"]:
            success_count += 1
        print(
            f"design_{attempt}.pdb written (min_ipae={summary['min_ipae']:.2f}, "
            f"binder_plddt={summary['binder_plddt']:.3f}, success={summary['success']}, "
            f"{success_count}/{args.num_designs})",
            flush=True,
        )
        attempt += 1


if __name__ == "__main__":
    main()
