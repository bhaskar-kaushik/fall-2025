# PS6_Kaushik_tests.jl
# Unit tests for Rust (1987) Bus Engine Replacement Model - CCP Estimation

# Load required packages
using Test
using DataFrames
using GLM
using HTTP
using CSV
using Statistics
using LinearAlgebra
using DataFramesMeta

# Include the source code
include("PS6_Kaushik_source.jl")

println("="^60)
println("Running Unit Tests for PS6 - Rust Model CCP Estimation")
println("="^60)

# Test 1: Data Loading and Reshaping
@testset "Test 1: load_and_reshape_data()" begin
    println("\nTest 1: Testing load_and_reshape_data()...")
    
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    
    # Test that output types are correct
    @test isa(df_long, DataFrame)
    @test isa(Xstate, Matrix)
    @test isa(Zstate, Vector)
    @test isa(Branded, Vector)
    
    # Test dimensions
    @test nrow(df_long) == 20000  # 1000 buses × 20 time periods
    @test size(Xstate) == (1000, 20)  # N buses × T time periods
    @test length(Zstate) == 1000
    @test length(Branded) == 1000
    
    # Test that required columns exist
    @test "bus_id" in names(df_long)
    @test "time" in names(df_long)
    @test "Y" in names(df_long)
    @test "Odometer" in names(df_long)
    @test "RouteUsage" in names(df_long)
    @test "Branded" in names(df_long)
    @test "Xstate" in names(df_long)
    @test "Zst" in names(df_long)
    
    # Test that data is sorted correctly
    @test issorted(df_long, [:bus_id, :time])
    
    # Test value ranges
    @test all(df_long.Y .∈ Ref([0, 1]))  # Binary outcome
    @test all(df_long.time .>= 1) && all(df_long.time .<= 20)
    @test all(df_long.Branded .∈ Ref([0, 1]))
    
    # Test no missing values in key columns
    @test !any(ismissing.(df_long.Y))
    @test !any(ismissing.(df_long.time))
    
    println("✓ All data loading tests passed!")
end

# Test 2: Flexible Logit Estimation
@testset "Test 2: estimate_flexible_logit()" begin
    println("\nTest 2: Testing estimate_flexible_logit()...")
    
    # Load data for testing
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, _, _, _ = load_and_reshape_data(url)
    
    # Estimate flexible logit
    flex_model = estimate_flexible_logit(df_long)
    
    # Test that output is a GLM model
    @test isa(flex_model, StatsModels.TableRegressionModel)
    
    # Test that model has coefficients
    @test length(coef(flex_model)) > 0
    
    # Test that model has expected number of parameters (29 with all interactions)
    @test length(coef(flex_model)) == 29
    
    # Test that predictions work
    preds = predict(flex_model, df_long)
    @test length(preds) == nrow(df_long)
    @test all(0 .<= preds .<= 1)  # Probabilities should be between 0 and 1
    
    # Test that we can access model deviance (indicates successful fit)
    @test isfinite(deviance(flex_model))
    
    println("✓ All flexible logit tests passed!")
end

# Test 3: State Space Construction
@testset "Test 3: construct_state_space()" begin
    println("\nTest 3: Testing construct_state_space()...")
    
    # Create dummy grids
    xbin = 10
    zbin = 5
    xval = collect(1:xbin) ./ 10
    zval = collect(1:zbin) ./ 10
    xtran = zeros(xbin * zbin, xbin)  # Dummy transition matrix
    
    # Construct state space
    state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
    
    # Test output type
    @test isa(state_df, DataFrame)
    
    # Test dimensions
    @test nrow(state_df) == xbin * zbin
    
    # Test that required columns exist
    @test "Odometer" in names(state_df)
    @test "RouteUsage" in names(state_df)
    @test "Branded" in names(state_df)
    @test "time" in names(state_df)
    
    # Test initial values
    @test all(state_df.Branded .== 0)
    @test all(state_df.time .== 0)
    
    # Test that Odometer values match xval (repeated for each z)
    unique_odo = unique(state_df.Odometer)
    @test length(unique_odo) == xbin
    
    # Test that RouteUsage values match zval (repeated for each x)
    unique_route = unique(state_df.RouteUsage)
    @test length(unique_route) == zbin
    
    println("✓ All state space construction tests passed!")
end

# Test 4: Future Value Computation
@testset "Test 4: compute_future_values()" begin
    println("\nTest 4: Testing compute_future_values()...")
    
    # Load data and estimate flexible logit
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, _, _, _ = load_and_reshape_data(url)
    flex_model = estimate_flexible_logit(df_long)
    
    # Get state grids
    zval, zbin, xval, xbin, xtran = create_grids()
    state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
    
    # Parameters
    T = 20
    β = 0.9
    
    # Compute future values
    FV = compute_future_values(state_df, flex_model, xtran, xbin, zbin, T, β)
    
    # Test output type and dimensions
    @test isa(FV, Array{Float64, 3})
    @test size(FV) == (xbin * zbin, 2, T + 1)
    
    # Test that first time period is all zeros (no future at t=1)
    @test all(FV[:, :, 1] .== 0)
    
    # Test that future values have reasonable range
    # FV = -β * log(p0), which can be positive or negative depending on p0
    # When p0 is small (likely to replace), log(p0) is large negative, so FV is positive
    # When p0 is close to 1 (unlikely to replace), log(p0) is close to 0, so FV is close to 0
    @test all(FV .>= -10)  # Reasonable lower bound
    @test all(FV .<= 10)   # Reasonable upper bound
    
    # Test that not all values are zero
    @test !all(FV .== 0)
    
    # Test that values are finite
    @test all(isfinite.(FV))
    
    println("✓ All future value computation tests passed!")
end

# Test 5: Future Value Mapping to Data
@testset "Test 5: compute_fvt1()" begin
    println("\nTest 5: Testing compute_fvt1()...")
    
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    flex_model = estimate_flexible_logit(df_long)
    
    # Get state grids and compute FV
    zval, zbin, xval, xbin, xtran = create_grids()
    state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
    FV = compute_future_values(state_df, flex_model, xtran, xbin, zbin, 20, 0.9)
    
    # Compute fvt1
    fvt1 = compute_fvt1(FV, xtran, Xstate, Zstate, xbin, Branded)
    
    # Test output type and length
    @test isa(fvt1, Vector{Float64})
    @test length(fvt1) == nrow(df_long)
    
    # Test that not all values are zero
    @test !all(fvt1 .== 0)
    
    # Test that values are finite
    @test all(isfinite.(fvt1))
    
    # Test reasonable range (based on actual output: mean ≈ -0.56, min ≈ -2.2, max = 0)
    @test minimum(fvt1) >= -5.0  # Conservative lower bound
    @test maximum(fvt1) <= 0.1   # Should be close to 0 or slightly positive due to numerical precision
    @test mean(fvt1) < 0  # Mean should be negative
    
    println("✓ All future value mapping tests passed!")
end

# Test 6: Structural Parameter Estimation
@testset "Test 6: estimate_structural_params()" begin
    println("\nTest 6: Testing estimate_structural_params()...")
    
    # Load data and compute everything needed
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    flex_model = estimate_flexible_logit(df_long)
    
    zval, zbin, xval, xbin, xtran = create_grids()
    state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
    FV = compute_future_values(state_df, flex_model, xtran, xbin, zbin, 20, 0.9)
    fvt1 = compute_fvt1(FV, xtran, Xstate, Zstate, xbin, Branded)
    
    # Estimate structural parameters
    theta_hat = estimate_structural_params(df_long, fvt1)
    
    # Test output type
    @test isa(theta_hat, StatsModels.TableRegressionModel)
    
    # Test number of parameters (intercept + Odometer + Branded = 3)
    @test length(coef(theta_hat)) == 3
    
    # Test coefficient names
    coef_names = String.(coefnames(theta_hat))
    @test "(Intercept)" in coef_names
    @test "Odometer" in coef_names
    @test "Branded" in coef_names
    
    # Test sign expectations (based on economic theory)
    coefs = coef(theta_hat)
    odometer_idx = findfirst(x -> x == "Odometer", coef_names)
    @test coefs[odometer_idx] < 0  # Higher mileage → more likely to replace
    
    # Test that we can access model deviance (indicates successful fit)
    @test isfinite(deviance(theta_hat))
    
    # Test predictions work
    @test length(predict(theta_hat)) == nrow(df_long)
    
    # Test coefficient magnitudes are reasonable (based on your output)
    @test abs(coefs[1]) < 5.0  # Intercept should be reasonable
    @test abs(coefs[odometer_idx]) < 1.0  # Odometer coefficient
    
    println("✓ All structural parameter estimation tests passed!")
end

# Test 7: Integration Test - Full Pipeline
@testset "Test 7: Full Pipeline Integration" begin
    println("\nTest 7: Testing full pipeline integration...")
    
    # This tests that all functions work together correctly
    # by running through the entire estimation procedure
    
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    
    # Step 1: Load data
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    @test !isnothing(df_long)
    
    # Step 2: Estimate flexible logit
    flex_model = estimate_flexible_logit(df_long)
    @test !isnothing(flex_model)
    
    # Step 3: Create grids and state space
    zval, zbin, xval, xbin, xtran = create_grids()
    state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
    @test !isnothing(state_df)
    
    # Step 4: Compute future values
    FV = compute_future_values(state_df, flex_model, xtran, xbin, zbin, 20, 0.9)
    @test !isnothing(FV)
    
    # Step 5: Map to data
    fvt1 = compute_fvt1(FV, xtran, Xstate, Zstate, xbin, Branded)
    @test !isnothing(fvt1)
    
    # Step 6: Estimate structural parameters
    theta_hat = estimate_structural_params(df_long, fvt1)
    @test !isnothing(theta_hat)
    
    # Test that final estimates are reasonable
    coefs = coef(theta_hat)
    @test all(isfinite.(coefs))
    @test length(coefs) == 3
    
    println("✓ Full pipeline integration test passed!")
end

println("\n" * "="^60)
println("All Unit Tests Passed Successfully! ✓")
println("="^60)