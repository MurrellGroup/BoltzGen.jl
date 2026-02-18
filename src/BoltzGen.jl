module BoltzGen

using Onion
using ProtInterop
using ProtInterop: BondViolation  # re-exported by BoltzGen
using NNlib
using Statistics
using LinearAlgebra
using Random
using SafeTensors
using HuggingFaceApi
using YAML

include("const.jl")
include("utils.jl")
include("features.jl")
include("smiles.jl")
include("ccd_cache.jl")
include("yaml_parser.jl")
include("checkpoint.jl")
include("transformers.jl")
include("encoders.jl")
include("trunk.jl")
include("confidence.jl")
include("affinity.jl")
include("diffusion_conditioning.jl")
include("diffusion.jl")
include("masker.jl")
include("output.jl")
include("bond_lengths.jl")
include("boltz.jl")
include("api.jl")

export BoltzModel
export boltz_forward
export write_pdb
export write_pdb_atom37
export write_mmcif
export collect_atom37_entries
export geometry_stats_atom37
export assert_geometry_sane_atom37!
export postprocess_atom14
export boltz_masker
export build_denovo_atom14_features
export build_denovo_atom14_features_from_sequence
export build_design_features
export build_design_features_from_structure
export load_structure_tokens
export load_msa_sequences
export tokens_from_sequence
export parse_design_yaml
export infer_config
export load_params!
export load_model_from_state
export load_model_from_safetensors
export resolve_weights_path
export default_weights_filename

# REPL API exports
export BoltzGenHandle, load_boltzgen, load_boltz2
export design_from_sequence, design_from_yaml, denovo_sample
export fold_from_sequence, fold_from_sequences, fold_from_structure, target_conditioned_design
export output_to_pdb, output_to_pdb_atom37, output_to_mmcif
export write_outputs, confidence_metrics
export check_bond_lengths, print_bond_length_report, BondViolation

end
