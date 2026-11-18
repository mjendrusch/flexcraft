#!/usr/bin/env bash
set -euo pipefail

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

expected_pmpnn_sha256() {
    case "$1" in
        02) echo "cf9c5980389312d26e8d25699e3d1372a9563b81770b4ca04c02027dcceb27a2" ;;
        10) echo "7020cb9bda0c5722c7ededcfd00f8546af1510c66a44fb7acaebcaa33c1479ea" ;;
        20) echo "993194d9cbe9334f6c2c81bddeb1404ea831711bca8e38b591b4116060484ac3" ;;
        30) echo "e6b7882f9ca25c24379c29aeb9b8e78e8bcffab5699a3521fd22c50713220697" ;;
        *) return 1 ;;
    esac
}

expected_solmpnn_sha256() {
    case "$1" in
        02) echo "71f50e774950063da724870cb9377020b54ed1630cb025a44dc3079fe1f5af82" ;;
        10) echo "b54d7ddbfac98eedd4ee37c93515c91691abb681b59fa2717c178d31cd7dd18a" ;;
        20) echo "f3f705752e625a173ac5ea4d2b2e46bd10ff13905a73bcc1ca2ecd3420b361e1" ;;
        30) echo "36ee859552bca57676f7e1b4f094c5dea4d37534e0e27351fa7582ac14d0ab78" ;;
        *) return 1 ;;
    esac
}

mkdir -p params
cd params

if [ ! -d salad ]; then
    curl -fLo salad_params.tar.gz https://zenodo.org/records/14711580/files/salad_params.tar.gz
    tar -xzf salad_params.tar.gz
    mv params salad
    rm -f salad_params.tar.gz
fi

mkdir -p pmpnn
rm -f pmpnn/v_48_005.pkl
for noise in 02 10 20 30; do
    weight_file="pmpnn/v_48_0${noise}.pkl"
    curl -fLo "${weight_file}" \
        "https://github.com/sokrypton/ColabDesign/raw/refs/heads/main/colabdesign/mpnn/weights/v_48_0${noise}.pkl"
    actual_sha="$(sha256_file "${weight_file}")"
    expected_sha="$(expected_pmpnn_sha256 "${noise}")"
    if [ "${actual_sha}" != "${expected_sha}" ]; then
        echo "ERROR: SHA256 mismatch for ${weight_file}: got ${actual_sha}, expected ${expected_sha}" >&2
        exit 1
    fi
done

mkdir -p solmpnn
rm -f solmpnn/v_48_005.pkl
for noise in 02 10 20 30; do
    pmpnn_file="pmpnn/v_48_0${noise}.pkl"
    solmpnn_file="solmpnn/v_48_0${noise}.pkl"
    curl -fLo "${solmpnn_file}" \
        "https://github.com/sokrypton/ColabDesign/raw/refs/heads/main/colabdesign/mpnn/weights_soluble/v_48_0${noise}.pkl"
    pmpnn_sha="$(sha256_file "${pmpnn_file}")"
    solmpnn_sha="$(sha256_file "${solmpnn_file}")"
    expected_sha="$(expected_solmpnn_sha256 "${noise}")"
    if [ "${solmpnn_sha}" = "${pmpnn_sha}" ]; then
        echo "ERROR: ${solmpnn_file} has identical SHA256 to vanilla ${pmpnn_file} (${solmpnn_sha})" >&2
        exit 1
    fi
    if [ "${solmpnn_sha}" != "${expected_sha}" ]; then
        echo "ERROR: SHA256 mismatch for ${solmpnn_file}: got ${solmpnn_sha}, expected ${expected_sha}" >&2
        exit 1
    fi
done

if [ ! -f af/params/params_model_1.npz ]; then
    mkdir -p af/params
    curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar | tar x -C af/params
fi

cd ..
mkdir -p dependencies
if [ ! -f dependencies/DAlphaBall.gcc ]; then
    curl -fLo dependencies/DAlphaBall.gcc \
        https://github.com/martinpacesa/BindCraft/raw/refs/heads/main/functions/DAlphaBall.gcc
    chmod +x dependencies/DAlphaBall.gcc
fi
