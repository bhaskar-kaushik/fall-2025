# ============================================================================
# Problem Set 4 Script File
# ECON 6343: Econometrics III
# Student: Bhaskar Kaushik
# Date: Fall 2025 
# Professor: Tyler Ransom, University of Oklahoma
# AI note (required by syllabus): "Used Claude and ChatGPT to debug and refine code/tests for ECON 6343 Fall 2025 PS4."
# ============================================================================

# Load required packages
using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, 
      GLM, FreqTables, Distributions

# Change to script directory (ensures relative paths work correctly)
cd(@__DIR__)

# Set random seed for reproducibility
# This ensures quadrature and Monte Carlo results are replicable
Random.seed!(1234)

# Include source code
# This reads in all functions defined in PS4_Kaushik_Source.jl
include("PS4_Kaushik_Source.jl")

# Execute main wrapper function
# This runs all analyses: multinomial logit, quadrature practice, 
# Monte Carlo practice, and mixed logit setup
allwrap()