"""De-novo binder design with RFD4 and Boltz2/Joltz2 structure filtering.

Adapts ``rfd4fx/scripts/binder_minimal.py`` with a Boltz2/Joltz2 structure
prediction and interface scoring step.

Example:
    python -m flexcraft.pipelines.rfd4.bind --in_path target.pdb \\
        --target_selection "A" --hotspots "A45;A47@N,CA,C" \\
        --num_designs 5 --num_aa 80 --out_path out_binder/
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from rfd4fx.flow_matching import RFD4Result, RFD4Sampler, THINKING_HIGH
from rfd4fx.model import RFD4
from rfd4fx.rfd4_input import RFD4Input

from flexcraft.pipelines.rfd4.common import (
    BACKBONE_RMSD as _BACKBONE_RMSD,
    add_forbidden_args,
    add_gen_only_args,
    decode_rfd4_tokens as _decode_rfd4_tokens,
    extract_atom24_samples,
    extract_binder_sequence,
    extract_target_sequences,
    forbid_amino_acids,
    parse_bool as _parse_bool,
    parse_forbidden_aas,
    setup_rfd4_output_dirs,
)
from flexcraft.structure.boltz import Joltz2, Joltz2Writer, JoltzInput, JoltzResult, JoltzSpec

SCORE_KEYS: tuple[str, ...] = (
    "attempt",
    "sequence",
    "binder_plddt",
    "min_ipae",
    "min_ipae_min",
    "mean_ipae",
    "mean_ipae_min",
    "mean_ipsae",
    "min_ipsae",
    "mean_iptm",
    "min_iptm",
    "rmsd",
    "min_rmsd",
    "success",
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the RFD4 binder pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_path", type=pathlib.Path, required=True)
    parser.add_argument(
        "--target_selection",
        default="",
        help="semicolon-separated selection entries (chain/residue/atom) for the target",
    )
    parser.add_argument(
        "--hotspots",
        default="",
        help="semicolon-separated hotspot selection entries (C_HOT; + C_HB with --auto_condition)",
    )
    parser.add_argument(
        "--auto_condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hbond donor/acceptor conditioning on the hotspot residues",
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
    parser.add_argument("--num_aa", type=int, default=80)
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        default=pathlib.Path("runs/rfd4-test-binder"),
    )
    parser.add_argument(
        "--use_template",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="use RFD4 target structure as a backbone-only template instead of target MSA",
    )
    parser.add_argument(
        "--no-use_template",
        dest="use_template",
        action="store_false",
        help="disable target template and use target MSA",
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
        help="number of Boltz2/Joltz2 diffusion samples per design",
    )
    parser.add_argument(
        "--num_recycle",
        type=int,
        default=3,
        help="number of Boltz2/Joltz2 recycling iterations",
    )
    parser.add_argument(
        "--boltz_path",
        type=pathlib.Path,
        default=pathlib.Path("./params/boltz/"),
        help="directory containing Boltz2 parameters and CCD cache",
    )
    add_forbidden_args(parser)
    add_gen_only_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def prepare_joltz_input(
    binder_sequence: str,
    target_sequences: Sequence[str],
    *,
    use_template: bool,
    target_msa_input: JoltzInput | None,
    rfd4_coords24: jax.Array | np.ndarray | None,
    rfd4_mask24: jax.Array | np.ndarray | None,
    boltz_path: pathlib.Path,
    num_residue_atoms: int = 24,
) -> tuple[JoltzInput, Joltz2Writer]:
    """Build an atom-padded ``JoltzInput`` and ``Joltz2Writer`` with inherited MSA or template.

    Args:
        binder_sequence: Designed 1-letter binder sequence.
        target_sequences: Per-chain 1-letter target sequences.
        use_template: If True, set a backbone-only target template from ``rfd4_coords24``.
            If False, inherit the precomputed target MSA from ``target_msa_input``.
        target_msa_input: Precomputed target-only ``JoltzInput`` carrying the target MSA.
        rfd4_coords24: RFD4 complex coordinates in ``atom24`` layout ``(L, 24, 3)``.
        rfd4_mask24: RFD4 complex atom mask in ``atom24`` layout ``(L, 24)``.
        boltz_path: Boltz2 parameter cache path.
        num_residue_atoms: Per-residue atom budget for static atom padding.

    Returns:
        Tuple ``(joltz_input, joltz_writer)``.
    """
    spec = (
        JoltzSpec()
        .add_protein(binder_sequence, use_msa=False)
        .add_protein(*target_sequences, use_msa=False)
    )
    joltz_input, joltz_writer = spec.to_input(pad=False, cache=boltz_path)
    binder_length = len(binder_sequence)

    if use_template:
        if rfd4_coords24 is None or rfd4_mask24 is None:
            raise ValueError("rfd4_coords24 and rfd4_mask24 are required when use_template=True.")
        target_bb_coords = jnp.asarray(rfd4_coords24[binder_length:, :4], dtype=jnp.float32)
        target_bb_mask = jnp.asarray(
            rfd4_mask24[binder_length:, :3].all(axis=-1), dtype=jnp.bool_
        )
        if joltz_input.features["template_mask"].shape[1] == 0:
            joltz_input = joltz_input.pad_to(None, num_templates=1)
        joltz_input = joltz_input.set_template(
            coords=target_bb_coords,
            start=binder_length,
            mask=target_bb_mask,
        )
    else:
        if target_msa_input is None:
            raise ValueError("target_msa_input is required when use_template=False.")
        joltz_input = joltz_input.inherit_msa(target_msa_input, start=binder_length)

    if num_residue_atoms > 0:
        joltz_input = joltz_input.pad_atoms(num_residue_atoms=num_residue_atoms)
    return joltz_input, joltz_writer


@jax.jit
def score_prediction_samples(
    joltz_result: JoltzResult,
    rfd4_coords24: jax.Array,
    rfd4_mask24: jax.Array,
) -> dict[str, jax.Array]:
    """Compute per-sample binder-target confidence metrics and full-complex backbone RMSD.

    All target chains (``chain_index != 0``) are collapsed into a single target group
    (``1``) against the binder chain (``0``) so intra-target interfaces are excluded.

    Args:
        joltz_result: ``JoltzResult`` across ``S`` diffusion samples with valid complex tokens.
        rfd4_coords24: Reference RFD4 ``atom24`` coordinates of shape ``(L, 24, 3)``.
        rfd4_mask24: Reference RFD4 ``atom24`` mask of shape ``(L, 24)``.

    Returns:
        Dictionary of 1D ``(S,)`` per-sample metric arrays: ``min_ipae`` (Å),
        ``mean_ipae`` (Å), ``ipsae``, ``iptm``, ``binder_plddt``, and ``rmsd`` (Å).
    """
    group_chain = (jnp.asarray(joltz_result.chain_index, dtype=jnp.int32) != 0).astype(
        jnp.int32
    )
    min_ipae = jnp.atleast_1d(joltz_result.max_pae * joltz_result.index_min_ipae(group_chain))
    mean_ipae = jnp.atleast_1d(joltz_result.max_pae * joltz_result.index_ipae(group_chain))
    ipsae = jnp.atleast_1d(joltz_result.index_ipsae(group_chain))
    iptm = jnp.atleast_1d(joltz_result.index_iptm(group_chain))
    binder_plddt = jnp.atleast_1d(joltz_result.index_plddt(group_chain == 0))

    pred_atom24, pred_mask24 = extract_atom24_samples(joltz_result)

    valid_bb_mask = jnp.asarray(pred_mask24[:, :4], dtype=jnp.bool_).all(axis=-1) & jnp.asarray(
        rfd4_mask24[:, :4], dtype=jnp.bool_
    ).all(axis=-1)
    rfd4_pos = jnp.asarray(rfd4_coords24, dtype=jnp.float32)
    rmsd = jax.vmap(lambda coords: _BACKBONE_RMSD(coords, rfd4_pos, mask=valid_bb_mask))(
        pred_atom24
    )
    return {
        "min_ipae": min_ipae,
        "mean_ipae": mean_ipae,
        "ipsae": ipsae,
        "iptm": iptm,
        "binder_plddt": binder_plddt,
        "rmsd": rmsd,
    }


def evaluate_binder_prediction(
    joltz_result: JoltzResult,
    rfd4_coords24: jax.Array | np.ndarray,
    rfd4_mask24: jax.Array | np.ndarray,
    *,
    f_min_ipae: float = 2.0,
    f_plddt: float = 0.8,
) -> tuple[dict[str, float | bool], int]:
    """Evaluate a ``JoltzResult`` against an RFD4 design and aggregate across samples.

    Args:
        joltz_result: ``JoltzResult`` from ``Joltz2``.
        rfd4_coords24: RFD4 ``atom24`` coordinates of shape ``(L, 24, 3)``.
        rfd4_mask24: RFD4 ``atom24`` mask of shape ``(L, 24)``.
        f_min_ipae: Success cutoff for sample-mean ``min_ipae`` in Angstroms.
        f_plddt: Success cutoff for sample-mean ``binder_plddt`` in ``[0, 1]``.

    Returns:
        Tuple ``(summary_dict, best_sample_index)``.
    """
    sample_metrics = score_prediction_samples(
        joltz_result,
        jnp.asarray(rfd4_coords24, dtype=jnp.float32),
        jnp.asarray(rfd4_mask24, dtype=jnp.bool_),
    )

    min_ipae_arr = np.asarray(sample_metrics["min_ipae"], dtype=np.float64)
    mean_ipae_arr = np.asarray(sample_metrics["mean_ipae"], dtype=np.float64)
    ipsae_arr = np.asarray(sample_metrics["ipsae"], dtype=np.float64)
    iptm_arr = np.asarray(sample_metrics["iptm"], dtype=np.float64)
    binder_plddt_arr = np.asarray(sample_metrics["binder_plddt"], dtype=np.float64)
    rmsd_arr = np.asarray(sample_metrics["rmsd"], dtype=np.float64)

    min_ipae_mean = float(min_ipae_arr.mean())
    binder_plddt_mean = float(binder_plddt_arr.mean())
    is_success = bool((min_ipae_mean < f_min_ipae) and (binder_plddt_mean > f_plddt))
    best_sample_index = int(np.argmin(min_ipae_arr))

    summary: dict[str, float | bool] = {
        "binder_plddt": binder_plddt_mean,
        "min_ipae": min_ipae_mean,
        "min_ipae_min": float(min_ipae_arr.min()),
        "mean_ipae": float(mean_ipae_arr.mean()),
        "mean_ipae_min": float(mean_ipae_arr.min()),
        "mean_ipsae": float(ipsae_arr.mean()),
        "min_ipsae": float(ipsae_arr.min()),
        "mean_iptm": float(iptm_arr.mean()),
        "min_iptm": float(iptm_arr.min()),
        "rmsd": float(rmsd_arr.mean()),
        "min_rmsd": float(rmsd_arr.min()),
        "success": is_success,
    }
    return summary, best_sample_index


def main(argv: Sequence[str] | None = None) -> None:
    """Run the RFD4 binder design and Boltz2/Joltz2 filtering pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        parse_forbidden_aas(args.forbidden)
    except ValueError as exc:
        parser.error(str(exc))

    attempts_dir, predictions_dir, scores_csv = setup_rfd4_output_dirs(
        args.out_path,
        gen_only=args.gen_only,
        score_keys=SCORE_KEYS,
    )

    target: dict[str, object] = {"pdb": str(args.in_path)}
    if args.target_selection:
        target["select"] = args.target_selection.split(";")
    if args.hotspots:
        target["hotspot"] = args.hotspots
    target["auto_condition"] = args.auto_condition

    inp = RFD4Input.from_chains(args.num_aa, target)
    inp = forbid_amino_acids(inp, args.forbidden, designable_mask=np.ones(args.num_aa, dtype=bool))
    target_sequences = extract_target_sequences(inp, args.num_aa)
    print(
        f"compiled: {int(np.asarray(inp.padding_mask).sum())} tokens",
        flush=True,
    )

    jax_model = RFD4.from_pretrained()
    sampler = RFD4Sampler(jax_model.step(), jax_model)
    batch = inp.pad_to_nearest(64).to_batch()

    joltz_predictor = None
    target_msa_input: JoltzInput | None = None
    if not args.gen_only:
        joltz_model = Joltz2(cache=args.boltz_path)
        joltz_predictor = joltz_model.predictor(
            num_recycle=args.num_recycle,
            num_samples=args.num_samples,
        )
        if not args.use_template:
            target_msa_input, _ = (
                JoltzSpec()
                .add_protein(*target_sequences, use_msa=True)
                .to_input(pad=False, cache=args.boltz_path)
            )

    base_key = jax.random.key(args.seed)
    success_count = 0
    attempt = 0

    while success_count < args.num_designs and attempt < args.max_attempts:
        attempt_key = jax.random.fold_in(base_key, attempt)
        rfd4_key, joltz_key = jax.random.split(attempt_key)

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

        assert joltz_predictor is not None and predictions_dir is not None and scores_csv is not None
        rfd4_atom24 = rfd4_result.to_atom24()
        binder_sequence = extract_binder_sequence(
            rfd4_result,
            args.num_aa,
            rfd4_restype=rfd4_atom24.restype,
        )

        joltz_input, joltz_writer = prepare_joltz_input(
            binder_sequence,
            target_sequences,
            use_template=args.use_template,
            target_msa_input=target_msa_input,
            rfd4_coords24=rfd4_atom24.coords,
            rfd4_mask24=rfd4_atom24.mask,
            boltz_path=args.boltz_path,
        )
        prediction = joltz_predictor(joltz_key, joltz_input, writer=joltz_writer)
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
