# flexcraft

`flexcraft` is a JAX library for combining generative protein design models, sequence design, and structure predictors into single-script, end-to-end protein design pipelines à la BindCraft.

Out of the box, a single `uv` environment co-installs:
- **Generative backbone models**: [`salad`](https://github.com/mjendrusch/salad) and [`rfd4fx`](https://github.com/mjendrusch/rfd4fx)
- **Sequence design**: `ProteinMPNN` and `SolubleMPNN` (in pure JAX/Haiku)
- **Structure predictors**: `AlphaFold2` (`flexcraft.structure.af`), `Joltz2` / `Boltz2` (`flexcraft.structure.boltz`), `Protenix` (`flexcraft.structure.protenix`), and `ESMFold2` (`flexcraft.structure.esmfold2`)
- **Optional physics-based refinement & scoring**: `PyRosetta` (`FastRelax` and `InterfaceAnalyzer`), loaded lazily on demand

## Getting started

### Installation (`uv`, Python 3.12)

Clone the repository and run `uv sync` from the `flexcraft` workspace root so `[tool.uv.*]` sources, metadata overrides, and the CPU-only PyTorch index (`torch.version.cuda is None`) apply automatically across Linux `x86_64`, NVIDIA Grace ARM (`linux/aarch64`), and macOS Apple Silicon (`darwin/arm64`):

```bash
git clone https://github.com/mjendrusch/flexcraft.git
cd flexcraft

# 1a. CPU default (macOS Apple Silicon, Linux x86_64, Linux Grace ARM)
uv sync

# 1b. NVIDIA GPU on Linux (x86_64 and Grace ARM; mutually exclusive extras)
uv sync --extra cuda12   # CUDA 12 PJRT plugin
# or:
uv sync --extra cuda13   # CUDA 13 PJRT plugin

# Optional: include dnachisel for oligo/gene ordering pipelines
uv sync --extra cuda12 --extra order

# 2. Download model parameters (salad, ProteinMPNN, SolubleMPNN, AF2, DAlphaBall)
bash download_params.sh
```

#### Optional out-of-tree `PyRosetta`

`PyRosetta` is imported lazily only when Rosetta relaxation or interface scoring is executed (commercial use requires a [PyRosetta license](https://els2.comotion.uw.edu/product/pyrosetta)). Because `PyRosetta` is installed from an out-of-tree wheelhouse rather than `uv.lock`, use `--inexact` when syncing and `--no-sync` with `uv run` so `uv` preserves `pyrosetta` in `.venv`:

```bash
uv pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release

# Preserve out-of-tree pyrosetta on future environment updates and command runs:
uv sync --inexact --extra cuda12
uv run --no-sync python -m flexcraft.pipelines.denovo --help
```

### Getting your 1st protein

Verify your setup by designing a 100-residue protein *de novo* (pass `--relax False` if `PyRosetta` is not installed):

```bash
uv run --no-sync python -m flexcraft.pipelines.denovo \
    --num_aa "100" \
    --num_designs 10 \
    --relax False \
    --out_path test_runs/denovo-100-1/
```

On multi-GPU machines, prefix commands with `CUDA_VISIBLE_DEVICES=0` to target a single GPU.

This creates `test_runs/denovo-100-1/` with the directory layout shared across `flexcraft` pipelines:
- `attempts/*.pdb`: `salad` backbones prior to sequence design
- `success/*.pdb`: AF2 predictions passing designability filters (`sc_lddt > 0.7`)
- `relaxed/*.pdb`: Rosetta-relaxed structures (when `--relax True`)
- `all.csv`: sequences and metrics for all evaluated `ProteinMPNN` samples
- `success.csv`: sequences and metrics for designs passing all filters

## Pipelines

`flexcraft` includes ready-to-run pipelines for common design tasks:
- **Motif scaffolding**:
  - [`flexcraft.pipelines.graft.simple`](docs/pipeline_motif.md): small motifs without symmetry or large inter-motif gaps
  - [`flexcraft.pipelines.graft.general`](docs/pipeline_motif.md): multi-segment, shape-conditioned, or symmetric motif scaffolding
- **Binder design**:
  - [`flexcraft.pipelines.bind`](docs/pipeline_binder.md): `salad` + `ProteinMPNN` + `BindCraftProperties` binder pipeline (plus gradient hallucination pipelines `flexcraft.pipelines.bind.mosaic`, `flexcraft.pipelines.bind.hal.af2`, and `flexcraft.pipelines.bind.hunter`)
- **Unconditional *de novo* design**:
  - [`flexcraft.pipelines.denovo`](docs/pipeline_denovo.md)
- **Oligo & gene ordering** (`--extra order`):
  - `flexcraft.pipelines.order.{select,scramble,codon_optimize}`

## Writing `flexcraft` pipelines

### Canonical `atom24` `DesignData` & All 4 Structure Predictors

All models exchange structures through `DesignData` (`flexcraft.data.DesignData`), standardized on `(L, 24, 3)` `atom24` coordinates and `(L, 24)` `atom_mask` (`to_protein()`, `to_salad()`, and `AFInput` slice `[:, :14]` automatically). Across all four predictors (`AF2`, `Joltz2`, `Protenix`, `ESMFold2`), `result.ptm` evaluates complex-wide masked pTM while `result.chain_ptm(chain_id)` evaluates per-chain/block pTM:

```python
from flexcraft.structure import af3_like
from flexcraft.utils import Keygen

key = Keygen(42)

# Run any backend ("af2_*", "joltz", "protenix_*", or "esmfold2_*") via a uniform spec API:
model = af3_like.AF3Like(model="joltz", cache="params/boltz/")
predict = model.predictor(num_recycle=4, num_samples=1)

spec = (
    af3_like.AF3LikeSpec()
    .add_protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK")
    .pad_to_nearest_multiple(64, num_residue_atoms=14)
)
prediction = predict(key(), spec)
result = prediction.result  # already unpadded on host by eager .predictor()
print(float(result.mean_plddt), float(result.ptm))
prediction.save_pdb("outputs/example.pdb")
```

Direct AlphaFold2 evaluation with initial guess and templates also works directly on `DesignData`:

```python
from flexcraft.structure.af import AFInput

af_input = AFInput.from_data(af_data).add_guess(af_data).pad_to(128)
result = af_predictor(params, key(), af_input)
# Inside @jax.jit, result and result.to_data() retain static padded shapes with masks:
jit_safe_data = result.to_data()
# On host, .unpad() strips padded tokens/atoms (called automatically by save_pdb / save_cif):
result.unpad().save_pdb("outputs/af2_pred.pdb")
```

### Multi-Axis Padding (`pad_to` / `pad_to_nearest_multiple`) & GPU VRAM Guidance

Varying sequence length, sidechain atom count, MSA depth, or template count triggers XLA recompilation. Call `.pad_to(...)` or `.pad_to_nearest_multiple(...)` on any `InputSpec` or `ModelInput` / `AFInput` to bucket shapes into a single compiled trace:

```python
spec = spec.pad_to(
    256,
    num_residue_atoms=24,  # use 14 (or 16) for protein-only campaigns to save VRAM
    num_msa=512,
    num_templates=4,
)
```

- `num_residue_atoms=24` (default) accommodates canonical amino acids, ligands, DNA, and RNA without recompiling when sequences mutate at fixed token length (`Joltz2` and `ESMFold2` automatically align total atoms to multiples of `W=32`, and `ESMFold2` also buckets `ESMC-6B` token lengths).
- For **protein-only** campaigns on `Joltz2`, `Protenix`, or `ESMFold2`, passing `num_residue_atoms=14` (max heavy atoms in `TRP`) reduces diffusion atom-window counts by ~1.7x–3x.

Approximate inference / hallucination ceilings by GPU VRAM:

| GPU VRAM | Mode | Recommended Ceiling `(target_tokens, num_residue_atoms, num_samples)` | Notes |
|---|---|---|---|
| **16 GB** | Inference (`predictor`) | `(<= 384, 14, 1)` or `(<= 256, 24, 1)` | Prefer `load_esmc=False` on `ESMFold2` (`ESMC-6B` requires ~13 GB bf16 / ~25 GB fp32) |
| **24 GB** | Inference / Gradient Hallucination | Inference: `(<= 640, 14, 1–2)` · Grad: `(<= 256, 14, 1–2)` | Use `num_residue_atoms=14` for protein-only binder hallucination |
| **40 GB** | Inference / Gradient Hallucination | Inference: `(<= 1024, 24, 4)` · Grad: `(<= 384, 24, 4)` | Supports `ESMFold2` (`load_esmc=True` on GPU/CPU offload) |
| **80 GB** | Multi-Sample / Large Complexes | Inference: `(<= 1536+, 24, 4–8)` · Grad: `(<= 640, 24, 4)` | Comfortable headroom for multi-chain complexes + templates + MSAs |

## Example scripts

See `scripts/` for minimal self-contained examples:
- `scripts/simple_salad.py`: generate structures with `salad`, design sequences with `ProteinMPNN`, predict with `AlphaFold2`, and optionally relax with `PyRosetta` (`--relax False` to skip Rosetta).
- `scripts/test_hal.py`: unconditional `AlphaFold2` gradient hallucination.
