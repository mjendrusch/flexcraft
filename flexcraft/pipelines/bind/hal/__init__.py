"""Shared masked metrics and reporting helpers for binder hallucination pipelines."""

import jax
import jax.numpy as jnp
import numpy as np

import flexcraft.sequence.aa_codes as aas
from flexcraft.files.csv import ScoreCSV
from flexcraft.sequence.mpnn import batch_mpnn


def _parse_chunk(spec: str) -> np.ndarray:
    if "-" not in spec:
        return np.array([int(spec) - 1], dtype=np.int32)
    start, end = [int(c) for c in spec.split("-")]
    return np.arange(start - 1, end, dtype=np.int32)


def _parse_hotspots(spec: str, binder_length: int = 100) -> np.ndarray | None:
    if spec == "none":
        return None
    return binder_length + np.concatenate(
        [_parse_chunk(c) for c in spec.split(",")], axis=0
    )


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Compute the mean of `values` over entries where `mask` is True."""
    mask_f = jnp.asarray(mask, dtype=values.dtype)
    return (values * mask_f).sum() / jnp.maximum(1.0, mask_f.sum())


def binder_hallucination_metrics(
    sequence: jax.Array,
    key: jax.Array,
    result,
    *,
    binder_length: int,
    pmpnn,
    mpnn_params,
    inter_contact_distance: float = 20.0,
    intra_contact_distance: float = 14.0,
    hotspot_index: np.ndarray | None = None,
) -> dict:
    """Compute padding-safe binder hallucination losses and auxiliary metrics."""
    pad_mask = jnp.asarray(result.token_pad_mask, dtype=jnp.bool_)
    num_tokens = pad_mask.shape[0]
    token_idx = jnp.arange(num_tokens)
    is_binder = (token_idx < binder_length) & pad_mask
    is_target = (token_idx >= binder_length) & pad_mask

    res_data = result.to_data(return_samples=True)
    binder_data = res_data[:binder_length]
    dssp = binder_data.p_dssp
    pmpnn_input = res_data.update(
        aa=aas.translate(res_data["aa"], aas.AF2_CODE, aas.PMPNN_CODE)
    )
    logits_full = batch_mpnn(pmpnn)(
        mpnn_params, key, pmpnn_input.drop_aa(where=is_binder)
    )["logits"]
    logits_full = aas.translate_onehot(logits_full, aas.PMPNN_CODE, aas.AF2_CODE)[
        ..., :20
    ]
    if not result.is_single_sample:
        logits_full = logits_full.mean(axis=0)
    logits = logits_full[:binder_length]
    binder_valid = is_binder[:binder_length]
    binder_count = jnp.maximum(1.0, binder_valid.astype(logits.dtype).sum())
    center = (logits * binder_valid[:, None]).sum(axis=0) / binder_count
    logits = logits - center
    prob = jax.lax.stop_gradient(
        jax.nn.softmax(logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1)
    )
    entropy = masked_mean(
        -(jax.nn.log_softmax(logits, axis=-1) * prob).sum(axis=-1), binder_valid
    )
    recovery = masked_mean((sequence * prob).sum(axis=-1), binder_valid)

    plddt = result.plddt
    pae = result.max_pae * result.pae
    if not result.is_single_sample:
        plddt = plddt.mean(axis=0)
        pae = pae.mean(axis=0)

    binder_selector = is_binder
    if hotspot_index is not None:
        hotspot_selector = (
            token_idx[:, None] == jnp.asarray(hotspot_index)[None, :]
        ).any(axis=-1) & pad_mask
    else:
        hotspot_selector = is_target

    hotspot_contacts = 0.5 * (
        result.index_contact_score(
            binder_selector,
            hotspot_selector,
            num_contacts=3,
            contact_distance=inter_contact_distance,
        )
        + result.index_contact_score(
            hotspot_selector,
            binder_selector,
            num_contacts=3,
            contact_distance=inter_contact_distance,
        )
    )
    distance_map = result.distogram_mean
    hotspot_distance_mask = hotspot_selector[:, None] & binder_selector[None, :]
    hotspot_distance = masked_mean(distance_map, hotspot_distance_mask)

    if hotspot_index is not None:
        binder_target_contacts = hotspot_contacts
    else:
        binder_target_contacts = result.index_contact_score(
            is_target,
            binder_selector,
            num_contacts=3,
            contact_distance=inter_contact_distance,
        )
    within_binder_contacts = result.index_contact_score(
        binder_selector,
        binder_selector,
        num_contacts=25,
        min_resi_distance=10,
        contact_distance=intra_contact_distance,
    )

    binary_chain = jnp.where(pad_mask, jnp.where(is_binder, 0, 1), -1)
    return dict(
        logits=logits,
        entropy=entropy,
        plddt=masked_mean(plddt, is_binder),
        recovery=recovery,
        binder_target_contacts=binder_target_contacts,
        within_binder_contacts=within_binder_contacts,
        hotspot_contacts=hotspot_contacts,
        hotspot_distance=hotspot_distance,
        within_binder_pae=masked_mean(pae, is_binder[:, None] & is_binder[None, :]),
        binder_target_pae=masked_mean(pae, is_binder[:, None] & is_target[None, :]),
        target_binder_pae=masked_mean(pae, is_target[:, None] & is_binder[None, :]),
        iptm=result.index_iptm(binary_chain).max(),
        eptm=result.index_ptm_score(binary_chain).mean(),
        ipsae=result.index_ipsae(binary_chain).max(),
        average_num_nonzero=masked_mean((sequence > 0.01).sum(axis=-1), binder_valid),
        L=dssp["L"],
        H=dssp["H"],
        E=dssp["E"],
    )


def off_target_hallucination_metrics(
    sequence: jax.Array,
    key: jax.Array,
    target_logits: jax.Array,
    off_target_result,
    *,
    binder_length: int,
    pmpnn,
    mpnn_params,
) -> dict:
    """Compute padding-safe off-target binder hallucination metrics."""
    pad_mask = jnp.asarray(off_target_result.token_pad_mask, dtype=jnp.bool_)
    token_idx = jnp.arange(pad_mask.shape[0])
    is_binder = (token_idx < binder_length) & pad_mask
    is_target = (token_idx >= binder_length) & pad_mask

    res_data = off_target_result.to_data(return_samples=True)
    pmpnn_input = res_data.update(
        aa=aas.translate(res_data["aa"], aas.AF2_CODE, aas.PMPNN_CODE)
    )
    logits_full = batch_mpnn(pmpnn)(
        mpnn_params, key, pmpnn_input.drop_aa(where=is_binder)
    )["logits"]
    logits_full = aas.translate_onehot(logits_full, aas.PMPNN_CODE, aas.AF2_CODE)[
        ..., :20
    ]
    if not off_target_result.is_single_sample:
        logits_full = logits_full.mean(axis=0)
    logits = logits_full[:binder_length]
    binder_valid = is_binder[:binder_length]
    binder_count = jnp.maximum(1.0, binder_valid.astype(logits.dtype).sum())
    center = (logits * binder_valid[:, None]).sum(axis=0) / binder_count
    logits = logits - center
    diff_mask = (target_logits.argmax(axis=1) != logits.argmax(axis=1)) & binder_valid
    prob = jax.lax.stop_gradient(
        jax.nn.softmax(logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1)
    )
    pae = off_target_result.max_pae * off_target_result.pae
    if not off_target_result.is_single_sample:
        pae = pae.mean(axis=0)
    binary_chain = jnp.where(pad_mask, jnp.where(is_binder, 0, 1), -1)
    return dict(
        off_target_recovery=masked_mean((sequence * prob).sum(axis=-1), diff_mask),
        off_target_contacts=off_target_result.index_contact_score(
            is_target, is_binder, num_contacts=3, contact_distance=20.0
        ),
        binder_off_target_pae=masked_mean(pae, is_binder[:, None] & is_target[None, :]),
        off_target_binder_pae=masked_mean(pae, is_target[:, None] & is_binder[None, :]),
        off_target_iptm=off_target_result.index_iptm(binary_chain).max(),
        off_target_ipsae=off_target_result.index_ipsae(binary_chain).max(),
    )


def _report_trajectory(index: int, writer: ScoreCSV, start_step: int = 0):
    def _report(step: int, loss: float, aux: dict):
        line_info = dict(
            attempt=index, step=start_step + step, total=float(loss.mean())
        )
        for key in writer.keys:
            if key in aux:
                line_info[key] = float(aux[key].mean())
        writer.write_line(line_info)

    return _report


def _report_result(
    attempt: int,
    stage: str,
    key,
    writer: ScoreCSV,
    sequence,
    result,
    metrics_fn,
    aux=None,
):
    if aux is None:
        aux = metrics_fn(sequence, key, result)
    line_info = dict(
        attempt=attempt,
        stage=stage,
        sequence=aas.decode(jnp.argmax(sequence, axis=-1), aas.AF2_CODE),
    )
    for field_name in writer.keys:
        if field_name in aux:
            line_info[field_name] = float(aux[field_name].mean())
    writer.write_line(line_info)
    return aux


def _passing(aux: dict) -> bool:
    return bool(
        aux["ipsae"] > 0.75
        and aux["plddt"] > 0.8
        and aux["binder_target_pae"] < 5.0
        and aux["target_binder_pae"] < 5.0
        and aux["recovery"] > 0.6
    )
