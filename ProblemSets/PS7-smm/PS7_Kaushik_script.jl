################################################################################
# Problem Set 7 - Script File
# ECON 6343: Econometrics III
################################################################################

using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM


# Change to current directory
cd(@__DIR__)

# Load source code - FIXED TO MATCH YOUR FILENAME
include("PS7_Kaushik_source.jl")

# Run main estimation routine
println("\nStarting estimation...")
@time results = main()

println("\n\nAll results returned successfully!")
println("Results object contains:")
println("  - β_hat_gmm: OLS estimates via GMM")
println("  - α_hat_mle: Multinomial logit via MLE")
println("  - α_hat_gmm_mle_start: Multinomial logit via GMM (MLE starting values)")
println("  - α_hat_gmm_random_start: Multinomial logit via GMM (random starting values)")
println("  - α_hat_sim: Estimates from simulated data")
println("  - α_hat_smm: Multinomial logit via SMM")