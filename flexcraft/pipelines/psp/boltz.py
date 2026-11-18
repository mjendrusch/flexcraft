import os

from flexcraft.files import FastaFile, ScoreCSV
from flexcraft.structure.boltz import Joltz2, JoltzResult, JoltzSpec
from flexcraft.utils import Keygen, parse_options


def main():
    opt = parse_options(
        "Protein structure prediction with Boltz2",
        in_path="sequences.fa",
        out_path="outputs/",
        boltz_path="./params/boltz/",
        num_samples=1,
        num_recycle=4,
        seed=42,
    )
    os.makedirs(f"{opt.out_path}/predictions/", exist_ok=True)

    key = Keygen(opt.seed)
    joltz = Joltz2(cache=opt.boltz_path).predictor(
        num_recycle=opt.num_recycle, num_samples=opt.num_samples
    )
    scores = ScoreCSV(
        path=f"{opt.out_path}/scores.csv",
        keys=["name", "sequence", "plddt", "ptm", "iptm", "ipsae"],
    )
    sequences = FastaFile(path=opt.in_path)

    for name, sequence in sequences.items():
        name = name.replace(":", "_binds_")
        chain_sequences = sequence.split(":")
        prediction = joltz(
            key(), JoltzSpec().add_protein(*chain_sequences, use_msa=True)
        )
        result: JoltzResult = prediction.result.unpad()
        plddt = float(result.mean_plddt.mean())
        ptm = float(result.ptm.max())
        iptm_samples = result.iptm
        iptm = float(iptm_samples.max())
        ipsae = float(result.ipsae().max())
        for index in range(opt.num_samples):
            sample_iptm = (
                iptm if result.is_single_sample else float(iptm_samples[index])
            )
            prediction.save_pdb(
                f"{opt.out_path}/predictions/{name}_model_{index}_{int(sample_iptm * 100)}.pdb",
                sample_index=index,
            )
        scores.write_line(
            dict(
                name=name,
                sequence=sequence,
                plddt=plddt,
                ptm=ptm,
                iptm=iptm,
                ipsae=ipsae,
            )
        )


if __name__ == "__main__":
    main()
