#!/usr/bin/env bash
set -euo pipefail

echo "Warning: scripts/run_boltz.sh is deprecated. Use flexcraft.structure.boltz.Joltz2 or 'uv run --no-sync python -m flexcraft.pipelines.psp.boltz' instead." >&2

input_path="${1:?Usage: scripts/run_boltz.sh <input_path> <output_path> [cuda_device]}"
output_path="${2:?Usage: scripts/run_boltz.sh <input_path> <output_path> [cuda_device]}"
cuda_device="${3:-}"

if [[ -n "$cuda_device" ]]; then
    export CUDA_VISIBLE_DEVICES="$cuda_device"
fi

if command -v uv >/dev/null 2>&1; then
    uv run --no-sync boltz predict "$input_path" --out_dir "$output_path" --diffusion_samples 5 --output_format pdb --override
else
    boltz predict "$input_path" --out_dir "$output_path" --diffusion_samples 5 --output_format pdb --override
fi
