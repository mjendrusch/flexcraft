
MODEL_OPTIONS = dict(
    salad_config="default_ve_scaled",
    salad_params="salad_params/default_ve_scaled-200k.jax",
    pmpnn_params="pmpnn_params/v_48_030.pkl",
    af2_params="./",
)
SAMPLER_OPTIONS = dict(
    scaffold_relax_cutoff=-1.0,
    num_designs=48,
    num_sequences=10,
    num_success=1,
    relax_cutoff=3.0,
    prev_threshold=0.9,
)
TASK_OPTIONS = dict(
    clash_lr=5e-3,
    compact_lr=1e-4,
    contact_lr=1e-2,
)