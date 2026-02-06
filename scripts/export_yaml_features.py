import argparse
import os

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, help="Path to a BoltzGen design YAML")
    parser.add_argument("--moldir", required=True, help="Path to mols directory or zip")
    parser.add_argument("--out", required=True, help="Output NPZ path")
    parser.add_argument(
        "--compute-affinity",
        action="store_true",
        help="Build affinity-specific features (affinity masks/MW) when present in the YAML.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    from boltzgen.task.predict.data_from_yaml import DataConfig, FromYamlDataModule
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.data.feature.featurizer import Featurizer

    cfg = DataConfig(
        moldir=args.moldir,
        multiplicity=1,
        yaml_path=args.yaml,
        tokenizer=Tokenizer(atomize_modified_residues=False),
        featurizer=Featurizer(),
        backbone_only=False,
        atom14=True,
        atom37=False,
        design=True,
        compute_affinity=args.compute_affinity,
        disulfide_prob=1.0,
        disulfide_on=True,
        skip_existing=False,
        skip_offset=0,
        diffusion_samples=1,
        output_dir=None,
    )

    module = FromYamlDataModule(cfg, batch_size=1, num_workers=0, pin_memory=False)
    loader = module.predict_dataloader()
    batch = next(iter(loader))

    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().numpy()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"Saved features to {args.out} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
