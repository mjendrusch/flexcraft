mkdir params
cd params
wget https://zenodo.org/records/14711580/files/salad_params.tar.gz
tar -xzf salad_params.tar.gz
mv params salad
mkdir pmpnn
cd pmpnn
for noise in 05 10 20 30; do
    wget https://github.com/sokrypton/ColabDesign/raw/refs/heads/main/colabdesign/mpnn/weights/v_48_0${noise}.pkl
done
cd ..
mkdir solmpnn
cd solmpnn
for noise in 05 10 20 30; do
    wget https://github.com/sokrypton/ColabDesign/raw/refs/heads/main/colabdesign/mpnn/weights/v_48_0${noise}.pkl
done
cd ..
mkdir af
cd af
mkdir params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar | tar x -C params
cd ../..
mkdir deps
cd deps
wget https://github.com/martinpacesa/BindCraft/raw/refs/heads/main/functions/DAlphaBall.gcc
chmod +x DAlphaBall.gcc
cd ..