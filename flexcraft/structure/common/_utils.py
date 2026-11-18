import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from salad.modules.utils.geometry import compute_pseudo_cb, positions_to_ncacocb


def align_up(val: int, multiple: int) -> int:
    """Round non-negative integer `val` up to the nearest multiple of `multiple`."""
    val = int(val)
    multiple = int(multiple)
    if multiple <= 1:
        return val
    return ((val + multiple - 1) // multiple) * multiple


def pad_axis(array, target_size: int, axis: int = 0, pad_value: Any = 0):
    """Right-pad `array` along `axis` to `target_size` with `pad_value`."""
    current_size = array.shape[axis]
    remaining = int(target_size) - current_size
    if remaining <= 0:
        return array
    pad_shape = list(array.shape)
    pad_shape[axis] = remaining
    if isinstance(array, jax.Array):
        fill = jnp.asarray(pad_value, dtype=array.dtype)
        padding = jnp.broadcast_to(fill, tuple(pad_shape))
        return jnp.concatenate([array, padding], axis=axis)
    fill = np.asarray(pad_value, dtype=array.dtype)
    padding = np.broadcast_to(fill, tuple(pad_shape))
    return np.concatenate([array, padding], axis=axis)


def _resolve_axes(spec, array):
    return spec(array) if callable(spec) else spec


def pad_features_by_schema(
    features: dict,
    axis_targets: dict[str, int],
    schema: dict[str, Any],
    pad_values: dict[str, Any] | None = None,
) -> dict:
    """Right-pad feature tensors strictly along named axes in `schema`."""
    pad_values = pad_values or {}
    result = {}
    for key, value in features.items():
        if value is None or key not in schema:
            result[key] = value
            continue
        axes = _resolve_axes(schema[key], value)
        fill = pad_values.get(key, 0)
        fill_val = fill(value) if callable(fill) else fill
        padded = value
        for axis_idx, axis_name in enumerate(axes):
            if axis_name is not None and axis_name in axis_targets:
                padded = pad_axis(
                    padded,
                    axis_targets[axis_name],
                    axis=axis_idx,
                    pad_value=fill_val,
                )
        result[key] = padded
    return result


def unpad_features_by_schema(
    features: dict,
    axis_limits: dict[str, int],
    schema: dict[str, Any],
) -> dict:
    """Slice feature tensors back to valid counts along named axes in `schema`."""
    result = {}
    for key, value in features.items():
        if value is None or key not in schema:
            result[key] = value
            continue
        axes = _resolve_axes(schema[key], value)
        slices = []
        needs_slice = False
        for axis_idx, axis_name in enumerate(axes):
            if axis_name is not None and axis_name in axis_limits:
                limit = int(axis_limits[axis_name])
                if limit < value.shape[axis_idx]:
                    needs_slice = True
                slices.append(slice(0, limit))
            else:
                slices.append(slice(None))
        result[key] = value[tuple(slices)] if needs_slice else value
    return result


def atom_array_to_atomX(atom_array, atom_to_token, num_atoms=14):
    residue_atom_count = atom_to_token.sum(axis=0)
    atomx_index = jnp.repeat(
        jnp.arange(num_atoms)[None], atom_to_token.shape[1], axis=0
    )
    atomx_mask = atomx_index < residue_atom_count[:, None]
    atomx_conindex = jnp.maximum(
        0,
        jnp.cumsum(atomx_mask.reshape(-1), axis=0).reshape(*atomx_mask.shape) - 1,
    )
    return atom_array[atomx_conindex], atomx_mask


def broadcast_array_to_atomX(residue_array, atom_to_token, num_atoms=14):
    residue_atom_count = atom_to_token.sum(axis=0)
    atomx_index = jnp.repeat(
        jnp.arange(num_atoms)[None], atom_to_token.shape[1], axis=0
    )
    atomx_mask = atomx_index < residue_atom_count[:, None]
    atomx = jnp.repeat(residue_array[:, None], num_atoms, axis=1)
    return atomx, atomx_mask


def atomX_to_atom_array(atomx, atom_to_token):
    consecutive_atoms = jnp.arange(atom_to_token.shape[0])
    consecutive_x, _ = atom_array_to_atomX(
        consecutive_atoms, atom_to_token, num_atoms=atomx.shape[1]
    )
    atom_residue_index = jnp.argmax(atom_to_token, axis=1)
    first_atom_in_residue = consecutive_x[atom_residue_index, 0]
    start = atom_residue_index * atomx.shape[1]
    flat_atomx = atomx.reshape(-1, *atomx.shape[2:])
    atomx_to_atom_array_index = jnp.clip(
        consecutive_atoms + start - first_atom_in_residue,
        0,
        flat_atomx.shape[0] - 1,
    )
    gathered = flat_atomx[atomx_to_atom_array_index]
    atom_valid = atom_to_token.sum(axis=1) > 0
    expand_shape = (atom_valid.shape[0],) + (1,) * (gathered.ndim - 1)
    return jnp.where(atom_valid.reshape(expand_shape), gathered, 0)


_LEGACY_BOLTZ_ATOM_SCHEMA = {
    "atom_pad_mask": (None, "atom"),
    "atom_resolved_mask": lambda arr: (
        (None, None, "atom") if arr.ndim == 3 else (None, "atom")
    ),
    "atom_to_token": (None, "atom", "token"),
    "token_to_rep_atom": (None, "token", "atom"),
    "token_to_center_atom": (None, "token", "atom"),
    "r_set_to_rep_atom": (None, "token", "atom"),
    "ref_pos": (None, "atom", None),
    "ref_space_uid": (None, "atom"),
    "ref_charge": (None, "atom"),
    "ref_element": (None, "atom", None),
    "ref_atom_name_chars": (None, "atom", None, None),
    "ref_chirality": (None, "atom"),
    "atom_backbone_feat": (None, "atom", None),
    "coords": lambda arr: (
        (None, None, "atom", None) if arr.ndim == 4 else (None, "atom", None)
    ),
    "bfactor": (None, "atom"),
    "plddt": (None, "atom"),
}


def pad_boltz_atom_features_for_compilation(features):
    """Deprecated: align Boltz atom features to a multiple of 32 via explicit schema."""
    warnings.warn(
        "pad_boltz_atom_features_for_compilation is deprecated; "
        "use JoltzInput.pad_to or JoltzInput.pad_to_nearest_multiple instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    num_atoms = int(features["atom_pad_mask"].shape[1])
    target_atoms = align_up(num_atoms, 32)
    return pad_features_by_schema(
        features,
        axis_targets={"atom": target_atoms},
        schema=_LEGACY_BOLTZ_ATOM_SCHEMA,
        pad_values={"ref_space_uid": -1, "atom_pad_mask": 0.0},
    )


def get_contact_atom(atom24, mol_type):
    protein = positions_to_ncacocb(atom24)[:, 4]
    smolecule = atom24[:, 0]
    dna = atom24[:, 11]
    rna = atom24[:, 12]
    protein = jnp.where((mol_type == 0)[:, None], protein, 0.0)
    dna = jnp.where((mol_type == 1)[:, None], dna, 0.0)
    rna = jnp.where((mol_type == 2)[:, None], rna, 0.0)
    smolecule = jnp.where((mol_type == 3)[:, None], smolecule, 0.0)
    return protein + dna + rna + smolecule


def backbone_to_template_geometry(
    coords: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute Boltz Gram-Schmidt backbone frames `(rot, ca)` from `(L, >=3, 3)` coords."""
    n, ca, c = coords[:, 0], coords[:, 1], coords[:, 2]
    v1 = c - ca
    v2 = n - ca
    e1 = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-10)
    u2 = v2 - e1 * (e1 * v2).sum(axis=-1, keepdims=True)
    e2 = u2 / (jnp.linalg.norm(u2, axis=-1, keepdims=True) + 1e-10)
    e3 = jnp.cross(e1, e2, axis=-1)
    rot = jnp.stack([e1, e2, e3], axis=-1)
    return rot, ca
