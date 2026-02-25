"""This module provides amino acid codes for AF2 and ProteinMPNN."""

import jax
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

def translate_onehot(one_hot, from_code, to_code):
    code_mapping = np.array([to_code.index(c) for c in from_code] + [20], dtype=np.int32)
    one_hot_mapping = jax.nn.one_hot(code_mapping, num_classes=21) # (from, to)
    return jnp.einsum("...f,ft->...t", one_hot, one_hot_mapping)

def encode(sequence: str, code: str) -> jnp.ndarray:
    """Encode a single-letter code amino acid sequence as an integer amino acid type array.
    
    Args:
        sequence: single-letter code sequence string.
        code: amino acid code used for encoding.

    Returns:
        Integer array encoding the input sequence.
    """
    return jnp.array([code.index(c) for c in sequence], dtype=jnp.int32)

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
