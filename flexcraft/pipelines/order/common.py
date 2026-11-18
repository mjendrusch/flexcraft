"""Shared oligo padding and codon-optimization helpers for ordering pipelines."""


def _load_dnachisel():
    """Lazily import `dnachisel` with installation instructions."""
    try:
        import dnachisel as dc
    except ImportError as exc:
        raise ImportError(
            "dnachisel is required for codon optimization in `flexcraft.pipelines.order`. "
            "Install it with `uv sync --extra order`."
        ) from exc
    return dc


def _codon_optimize(
    aa_sequence: str,
    prefix: str = "",
    suffix: str = "",
    *,
    homopolymer_as_objective: bool = False,
    species: str = "e_coli",
) -> str:
    """Reverse-translate and codon-optimize an amino acid sequence with flanking adapters."""
    dc = _load_dnachisel()
    prefix = prefix.upper()
    suffix = suffix.upper()
    sequence = (prefix + dc.reverse_translate(aa_sequence) + suffix).upper()
    coding_span = (len(prefix), len(prefix) + 3 * len(aa_sequence))
    homopolymers = [
        dc.AvoidPattern(dc.HomopolymerPattern(base, 4))
        for base in ("A", "T", "C", "G")
    ]
    constraints = [
        dc.AvoidPattern("BsaI_site"),
        dc.EnforceGCContent(mini=0.3, maxi=0.7, window=50),
        dc.EnforceTranslation(translation=aa_sequence, location=coding_span),
        dc.EnforceSequence(sequence=prefix, location=(0, len(prefix))),
        dc.EnforceSequence(sequence=suffix, location=(coding_span[1], len(sequence))),
    ]
    objectives = [dc.CodonOptimize(species=species, location=coding_span)]
    if homopolymer_as_objective:
        objectives.extend(homopolymers)
    else:
        constraints[1:1] = homopolymers
    problem = dc.DnaOptimizationProblem(
        sequence=sequence,
        constraints=constraints,
        objectives=objectives,
    )
    problem.resolve_constraints()
    problem.optimize()
    return problem.sequence


def _gs_string(count: int) -> str:
    """Build a GGS-repeat linker string of length `count`."""
    return "".join("G" if i % 3 < 2 else "S" for i in range(count))


def _gs_pad(sequence: str, num_aa: int = 72) -> str:
    """Symmetrically pad `sequence` with GGS linkers up to `num_aa` residues."""
    length = len(sequence)
    if length < num_aa:
        num_remaining = num_aa - length
        pad_left = num_remaining // 2
        pad_right = num_remaining - pad_left
        sequence = _gs_string(pad_left) + sequence + _gs_string(pad_right)
    return sequence


codon_optimize = _codon_optimize
gs_string = _gs_string
gs_pad = _gs_pad
