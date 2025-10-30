# PS5_Kaushik_script.jl
# Script to run Problem Set 5 estimation

# Load all required packages
using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

# Set working directory to script location
cd(@__DIR__)

# Include the create_grids function first
include("create_grids.jl")

# Include source code (contains only function definitions)
include("PS5_Kaushik_source.jl")

# Run main estimation
result = main()

