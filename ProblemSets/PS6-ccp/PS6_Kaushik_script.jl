# PS6_Kaushik_script.jl
# Main script to run the Rust (1987) Bus Engine Replacement Model
# using Conditional Choice Probabilities (CCP) estimation

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

# Change to script directory (ensures relative paths work correctly)
cd(@__DIR__)

# Set random seed for reproducibility
Random.seed!(1234)

# Include source code
# This reads in all functions defined in PS6_Kaushik_Source.jl
include("PS6_Kaushik_Source.jl")

# Execute main wrapper function with timing
# This runs the complete CCP estimation:
# 1. Loads and reshapes bus data
# 2. Estimates flexible logit model
# 3. Constructs state space and transition matrices
# 4. Computes future values using CCPs
# 5. Estimates structural parameters
@time main()