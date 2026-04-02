"""AbsciBind ipTM scoring metric from Origin-1.

Reference: Levine, S. et al., 2026. Origin-1: a generative AI platform for de novo
antibody design against novel epitopes. https://doi.org/10.64898/2026.01.14.699389
"""

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from flexcraft.structure.af._data import AFResult


def _d0(L) -> jnp.ndarray:
    """TM-score normalization factor (Origin-1, Suppl. §7.5).

    d_0(L) = 1.34 * (L - 15)^(1/3) - 1.8   for L >= 19
    d_0(L) = 1                                for L <  19
    """
    return jnp.where(L >= 19, 1.34 * (L - 15) ** (1 / 3) - 1.8, 1.0)


def _ptm_matrix(probabilities: jnp.ndarray, L: int) -> jnp.ndarray:
    """Compute the element-wise pTM score matrix.

    pTM_ij(L) = sum_{b=1}^{64} p_ij^b * 1 / (1 + (Delta_b / d_0(L))^2)
    where Delta_b = (b - 0.5) / 2

    Args:
        probabilities: PAE bin probabilities, shape (L_seq, L_seq, 64).
        L: Normalization length used to compute d_0.

    Returns:
        pTM matrix of shape (L_seq, L_seq).
    """
    num_bins = probabilities.shape[-1]
    b = jnp.arange(1, num_bins + 1)
    delta = (b - 0.5) / 2          # Delta_b, shape (64,); in Angstroms
    d = _d0(L)
    weights = 1 / (1 + (delta / d) ** 2)   # shape (64,)
    return (probabilities * weights).sum(axis=-1)


def _iptm_A_given_B(
    ptm: jnp.ndarray,
    A_mask: jnp.ndarray,
    B_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute ipTM_A(B; L).

    ipTM_A(B; L) = max_{i in A} [ 1/|B| * sum_{j in B} pTM_ij(L) ]

    Args:
        ptm: pTM matrix computed for the desired normalization length, shape (L, L).
        A_mask: Boolean mask selecting chain A residues, shape (L,).
        B_mask: Boolean mask selecting chain B residues, shape (L,).

    Returns:
        Scalar ipTM value.
    """
    B_count = jnp.maximum(1, B_mask.astype(jnp.int32).sum())
    # For each residue i, sum pTM_ij over all j in B → shape (L,)
    sum_over_B = (ptm * B_mask[None, :]).sum(axis=1)
    mean_over_B = sum_over_B / B_count
    # Take maximum over residues i in A; mask non-A residues to -inf
    return jnp.where(A_mask, mean_over_B, -jnp.inf).max()


@dataclass
class AbsciBindIPTM:
    """AbsciBind ipTM scoring metric (Origin-1, Suppl. §7.5).

    Computes a pairwise-chain ipTM score that is robust to multi-chain
    antibody/antigen configurations by working on merged super-chains.

    The final score is the mean of:
      1. Default ipTM = max[ ipTM_Ab(Ag; L_tot), ipTM_Ag(Ab; L_tot) ]
      2. Antibody-Aligned ipTM = ipTM_Ab(Ag; L_Ag)
         (uses L_Ag for d_0 to remove sensitivity to antibody chain length)

    Usage::

        scorer = AbsciBindIPTM()
        ab_mask = result.chain_index == 0   # antibody chain(s)
        is_target = result.chain_index == 1   # antigen chain(s)
        scores = scorer(result, ab_mask, is_target)
        print(scores["iptm"])

    Reference: Levine et al. 2026, https://doi.org/10.64898/2026.01.14.699389
    """

    def __call__(
        self,
        result: AFResult,
        is_target: np.ndarray,
    ) -> dict:
        """Compute the AbsciBind ipTM score.

        Args:
            result: AFResult object produced by an AlphaFold prediction.
            is_target: Boolean mask selecting antigen (target) residues, shape (L,).

        Returns:
            Dictionary with keys:
                ``iptm``            – final score (mean of default and ab_aligned).
                ``default_iptm``    – max[ipTM_Ab(Ag;L_tot), ipTM_Ag(Ab;L_tot)].
                ``ab_aligned_iptm`` – ipTM_Ab(Ag; L_Ag).
        """
        is_target = jnp.asarray(is_target, dtype=jnp.bool_)
        ab_mask = jnp.asarray(~is_target, dtype=jnp.bool_)

        # PAE bin probabilities: shape (L, L, 64)
        logits = result.result["predicted_aligned_error"]["logits"]
        probabilities = jax.nn.softmax(logits, axis=-1)

        L_ab = ab_mask.astype(jnp.int32).sum()
        L_ag = is_target.astype(jnp.int32).sum()
        L_tot = L_ab + L_ag

        # pTM matrices for both normalization lengths
        ptm_tot = _ptm_matrix(probabilities, L_tot)
        ptm_ag  = _ptm_matrix(probabilities, L_ag)

        # Default ipTM: max[ipTM_Ab(Ag;L_tot), ipTM_Ag(Ab;L_tot)]
        iptm_ab  = _iptm_A_given_B(ptm_tot, ab_mask, is_target)
        iptm_ag  = _iptm_A_given_B(ptm_tot, is_target, ab_mask)
        default_iptm = jnp.maximum(iptm_ab, iptm_ag)

        # Antibody-Aligned ipTM: ipTM_Ab(Ag; L_Ag)
        ab_aligned_iptm = _iptm_A_given_B(ptm_ag, ab_mask, is_target)

        # Final score: mean of both
        iptm = (default_iptm + ab_aligned_iptm) / 2

        return dict(
            iptm=iptm,
            default_iptm=default_iptm,
            ab_aligned_iptm=ab_aligned_iptm,
        )
