#!/usr/bin/env python3
"""Export boltzgen moldir RDKit pickle files to a strict JSONL format.

This is a one-time preprocessing step used to build a Julia-native JLD2 cache.
Runtime Julia pathways do not depend on Python.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from rdkit.Chem.rdchem import Mol


def _get_conformer(mol: Mol):
    for conf in mol.GetConformers():
        try:
            if conf.GetProp("name") == "Computed":
                return conf
        except KeyError:
            pass
    for conf in mol.GetConformers():
        try:
            if conf.GetProp("name") == "Ideal":
                return conf
        except KeyError:
            pass
    confs = list(mol.GetConformers())
    if len(confs) > 0:
        return confs[0]
    raise ValueError("Conformer does not exist.")


def _load_record(path: Path) -> dict:
    with path.open("rb") as handle:
        mol = pickle.load(handle)

    if not isinstance(mol, Mol):
        raise TypeError(f"{path} did not deserialize to an RDKit Mol")

    code = path.stem.upper()
    atom_names: list[str] = []
    elements: list[int] = []
    charges: list[int] = []
    coords: list[list[float]] = []
    heavy_idx_to_local: dict[int, int] = {}
    heavy_atoms = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        heavy_atoms.append(atom)
        if not atom.HasProp("name"):
            raise ValueError(f"{code}: heavy atom missing required 'name' property")
        atom_name = atom.GetProp("name")
        if atom_name == "":
            raise ValueError(f"{code}: heavy atom has empty name")
        atom_names.append(atom_name)
        elements.append(int(atom.GetAtomicNum()))
        charges.append(int(atom.GetFormalCharge()))

    if len(heavy_atoms) > 0:
        conformer = _get_conformer(mol)
        for i, atom in enumerate(heavy_atoms, start=1):
            pos = conformer.GetAtomPosition(atom.GetIdx())
            coords.append([float(pos.x), float(pos.y), float(pos.z)])
            heavy_idx_to_local[atom.GetIdx()] = i  # 1-based

    edge_to_type: dict[tuple[int, int], str] = {}
    for bond in mol.GetBonds():
        i0 = bond.GetBeginAtomIdx()
        j0 = bond.GetEndAtomIdx()
        if i0 not in heavy_idx_to_local or j0 not in heavy_idx_to_local:
            continue
        i = heavy_idx_to_local[i0]
        j = heavy_idx_to_local[j0]
        if i == j:
            raise ValueError(f"{code}: self-bond encountered at heavy atom index {i}")
        a = min(i, j)
        b = max(i, j)
        bt = str(bond.GetBondType().name)
        key = (a, b)
        if key in edge_to_type and edge_to_type[key] != bt:
            raise ValueError(
                f"{code}: inconsistent bond type for edge {key}: {edge_to_type[key]} vs {bt}",
            )
        edge_to_type[key] = bt

    bonds = [[a, b, edge_to_type[(a, b)]] for (a, b) in sorted(edge_to_type.keys())]
    return {
        "id": code,
        "atom_names": atom_names,
        "elements": elements,
        "charges": charges,
        "coords": coords,
        "bonds": bonds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moldir", type=Path, required=True, help="Path to moldir with *.pkl molecules")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for smoke tests (0 = all)")
    args = parser.parse_args()

    moldir = args.moldir.resolve()
    out = args.out.resolve()
    if not moldir.is_dir():
        raise ValueError(f"--moldir is not a directory: {moldir}")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    paths = sorted(moldir.glob("*.pkl"))
    if len(paths) == 0:
        raise ValueError(f"No *.pkl files found under moldir: {moldir}")
    if args.limit > 0:
        paths = paths[: args.limit]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for idx, path in enumerate(paths, start=1):
            record = _load_record(path)
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")
            if idx % 1000 == 0:
                print(f"processed {idx}/{len(paths)}")

    print(f"Wrote {len(paths)} records to {out}")


if __name__ == "__main__":
    main()
