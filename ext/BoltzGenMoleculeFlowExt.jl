module BoltzGenMoleculeFlowExt

using BoltzGen
using MoleculeFlow

# Source-mode scripts include `src/BoltzGen.jl` directly and resolve MoleculeFlow
# dynamically in `smiles.jl`. This extension exists to keep MoleculeFlow optional
# in package mode via weakdeps/extensions.

end
