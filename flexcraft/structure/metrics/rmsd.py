"""RMSD-derived metrics."""

from dataclasses import dataclass, field
from typing import Sequence

import jax
import jax.numpy as jnp
from salad.modules.utils.geometry import apply_alignment, index_align, index_kabsch

from flexcraft.data.data import DesignData


def _rmsd_default_atoms() -> tuple[str, ...]:
    return ("CA",)


def _data_to_pos(item):
    pos, _ = _data_to_pos_and_mask(item)
    return pos


def _data_to_pos_and_mask(item):
    if hasattr(item, "to_data"):
        item = item.to_data()
        assert isinstance(item, DesignData)
    if isinstance(item, DesignData):
        pos = jnp.asarray(item["atom_positions"])
        item_mask = jnp.asarray(item["mask"], dtype=jnp.bool_) if "mask" in item else None
        return pos, item_mask
    return jnp.asarray(item), None


@dataclass(frozen=True)
class RMSD:
    """Root mean square deviation (RMSD) metric over selected backbone/CB atoms."""

    atoms: Sequence[str] = field(default_factory=_rmsd_default_atoms)

    def __post_init__(self):
        object.__setattr__(self, "atoms", tuple(self.atoms))

    def __call__(
        self,
        x,
        y,
        index=None,
        mask=None,
        eval_mask=None,
        weight=None,
    ) -> jax.Array:
        """Compute Kabsch-aligned RMSD between ``x`` and ``y``.

        Args:
            x: Predicted coordinates or ``DesignData`` / ``Result`` object.
            y: Reference coordinates or ``DesignData`` / ``Result`` object.
            index: Optional group partition aligned independently.
            mask: Optional residue mask used for Kabsch alignment (and evaluation if ``eval_mask`` is None).
            eval_mask: Optional residue mask used for RMSD evaluation only.
            weight: Optional per-residue or per-selected-atom alignment weights.

        Returns:
            Scalar RMSD over selected valid atoms.
        """
        order = ["N", "CA", "C", "O", "CB"]
        atom_index = jnp.array([order.index(a) for a in self.atoms], dtype=jnp.int32)
        x_pos, x_mask = _data_to_pos_and_mask(x)
        y_pos, y_mask = _data_to_pos_and_mask(y)

        num_residues = x_pos.shape[0]
        num_atoms = atom_index.shape[0]
        data_mask = jnp.ones((num_residues,), dtype=jnp.bool_)
        if x_mask is not None:
            data_mask = data_mask & x_mask
        if y_mask is not None:
            data_mask = data_mask & y_mask

        if mask is None:
            align_mask = data_mask
        else:
            align_mask = (jnp.asarray(mask) > 0) & data_mask

        if eval_mask is None:
            rms_mask = align_mask
        else:
            rms_mask = (jnp.asarray(eval_mask) > 0) & data_mask

        if index is None:
            group_index = jnp.zeros((num_residues,), dtype=jnp.int32)
        else:
            group_index = jnp.asarray(index, dtype=jnp.int32)

        if weight is not None:
            weight = jnp.asarray(weight, dtype=jnp.float32)
            if weight.ndim == 1 and weight.shape[0] == num_residues:
                weight = jnp.broadcast_to(weight[:, None], (num_residues, num_atoms)).reshape(-1)

        align_mask = jnp.broadcast_to(align_mask[:, None], (num_residues, num_atoms)).reshape(-1)
        rms_mask = jnp.broadcast_to(rms_mask[:, None], (num_residues, num_atoms)).reshape(-1)
        group_index = jnp.broadcast_to(group_index[:, None], (num_residues, num_atoms)).reshape(-1)
        x_sel = x_pos[:, atom_index].reshape(-1, 3)
        y_sel = y_pos[:, atom_index].reshape(-1, 3)

        params = index_kabsch(x_sel, y_sel, group_index, align_mask, weight=weight)
        x_aligned = apply_alignment(x_sel[:, None], params)[:, 0]
        sq_error = jnp.where(rms_mask, ((y_sel - x_aligned) ** 2).sum(axis=-1), 0.0)
        denom = jnp.maximum(1, rms_mask.astype(jnp.int32).sum())
        return jnp.sqrt(sq_error.sum() / denom)
