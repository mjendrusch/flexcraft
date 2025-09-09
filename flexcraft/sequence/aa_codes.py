"""This module provides amino acid codes for AF2 and ProteinMPNN."""

import numpy as np
import jax.numpy as jnp

AF2_CODE = "ARNDCQEGHILKMFPSTWYV"
PMPNN_CODE = "".join(sorted(AF2_CODE))

def translate(aatype, from_code, to_code):
    """Translate an integer amino acid type array from one amino acid code to another.
    
    Args:
        aatype: integer array of amino acid types encoded with `from_code`.
        from_code: amino acid code to translate from (sequence of single-letter code amino acids).
        to_code: amino acid code to translate to (sequence of single-letter code amino acids).

    Returns:
        Sequence encoded by `aatype`, re-encoded according to `to_code`.
    """
    code_mapping = np.array([to_code.index(c) for c in from_code] + [20], dtype=np.int32)
    return code_mapping[aatype]

def decode(aatype: jnp.ndarray, code: str) -> str:
    """Decode an integer amino acid type array into a single-letter code string.
    
    Args:
        aatype: integer array of amino acid types encoded with `code`.
        code: amino acid code to use for decoding.

    Returns:
        single-letter code sequence corresponding to `aatype`.
    """
    code = code + "X"
    return "".join([code[c] for c in aatype])
