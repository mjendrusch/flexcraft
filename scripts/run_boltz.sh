input_path=$1
output_path=$2
cuda_device=$3

source ~/.bashrc
conda activate boltz1
CUDA_VISIBLE_DEVICES=$cuda_device boltz predict $input_path --out_dir $output_path --diffusion_samples 5 --output_format pdb --override
