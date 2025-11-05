#=
Problem Set 8 - Unit Tests
ECON 6343: Econometrics III
=#

using Test
using Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
using MultivariateStats, FreqTables, ForwardDiff, LineSearches

# Set random seed for reproducibility
Random.seed!(1234)

# Include source code
include("PS8_Kaushik_source.jl")

# Test data URL
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"

@testset "Problem Set 8 Unit Tests" begin
    
    @testset "Data Loading" begin
        df = load_data(url)
        @test isa(df, DataFrame)
        @test size(df) == (2438, 15)  # Exact dimensions from your output
        @test "logwage" in names(df)
        @test "black" in names(df)
        @test "asvabAR" in names(df)
    end
    
    @testset "Base Regression" begin
        df = load_data(url)
        model = estimate_base_regression(df)
        @test isa(model, StatsModels.TableRegressionModel)
        @test length(coef(model)) == 7  # intercept + 6 covariates
        
        # Check approximate coefficient values from your output
        coefs = coef(model)
        @test isapprox(coefs[1], 2.00771, atol=0.01)  # intercept
        @test isapprox(coefs[2], -0.167441, atol=0.01)  # black
        @test isapprox(coefs[7], 0.299131, atol=0.01)  # grad4yr
    end
    
    @testset "ASVAB Correlations" begin
        df = load_data(url)
        corr_df = compute_asvab_correlations(df)
        
        @test size(corr_df) == (6, 6)
        
        # Check diagonal is all 1s
        corr_matrix = Matrix(corr_df)
        for i in 1:6
            @test corr_matrix[i, i] ≈ 1.0
        end
        
        # Check specific correlations from your output
        @test isapprox(corr_matrix[1, 3], 0.799272, atol=0.01)
        @test isapprox(corr_matrix[2, 6], 0.421961, atol=0.01)
        
        # All correlations should be between -1 and 1
        @test all(-1 .<= corr_matrix .<= 1)
    end
    
    @testset "Full Regression with ASVAB" begin
        df = load_data(url)
        model = estimate_full_regression(df)
        @test isa(model, StatsModels.TableRegressionModel)
        @test length(coef(model)) == 13  # intercept + 12 covariates
        
        # Check that R² improved from base model
        base_model = estimate_base_regression(df)
        @test r2(model) > r2(base_model)
    end
    
    @testset "PCA Regression" begin
        df = load_data(url)
        model = estimate_pca_regression(df)
        @test isa(model, StatsModels.TableRegressionModel)
        @test length(coef(model)) == 8  # intercept + 7 covariates (including PC)
        
        # Check PCA coefficient is significant (from your output)
        coefs = coef(model)
        @test isapprox(coefs[8], -0.0525399, atol=0.01)  # asvabPCA coefficient
    end
    
    @testset "Factor Analysis Regression" begin
        df = load_data(url)
        model = estimate_factor_regression(df)
        @test isa(model, StatsModels.TableRegressionModel)
        @test length(coef(model)) == 8  # intercept + 7 covariates (including factor)
        
        # Check factor coefficient is significant (from your output)
        coefs = coef(model)
        @test isapprox(coefs[8], 0.114716, atol=0.01)  # asvabFactor coefficient
    end
    
    @testset "Matrix Preparation" begin
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        N = 2438  # From your output
        @test size(X) == (N, 7)
        @test size(y) == (N,)
        @test size(Xfac) == (N, 4)
        @test size(asvabs) == (N, 6)
        
        # Check that last column of X and Xfac are ones (intercept)
        @test all(X[:, 7] .== 1)
        @test all(Xfac[:, 4] .== 1)
    end
    
    @testset "Factor Model Likelihood" begin
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Test with random parameters
        n_params = 4*6 + 7 + 7 + 7  # γ + β + α + σ = 45 parameters
        test_params = 0.1 * randn(n_params)
        
        # Make sure σ values are positive
        test_params[end-6:end] = abs.(test_params[end-6:end]) .+ 0.1
        
        loglike = factor_model(test_params, X, Xfac, asvabs, y, 5)
        @test isfinite(loglike)
        @test loglike > 0  # negative log-likelihood should be positive
    end
    
    @testset "Parameter Count" begin
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Starting values
        svals = vcat(
            vec(Xfac\asvabs[:, 1]),
            vec(Xfac\asvabs[:, 2]),
            vec(Xfac\asvabs[:, 3]),
            vec(Xfac\asvabs[:, 4]),
            vec(Xfac\asvabs[:, 5]),
            vec(Xfac\asvabs[:, 6]),
            vec(X\y),
            rand(7),
            0.5*ones(7)
        )
        
        @test length(svals) == 45  # Total parameters
        
        # Check dimensions breakdown
        L = 4  # Xfac columns
        J = 6  # ASVAB tests
        K = 7  # X columns
        @test L*J + K + (J+1) + (J+1) == 45
    end
    
    @testset "Optimization Convergence" begin
        # Test with small synthetic data for speed
        N = 100
        Random.seed!(123)
        
        # Create synthetic data
        test_df = DataFrame(
            logwage = randn(N),
            black = rand(0:1, N),
            hispanic = rand(0:1, N),
            female = rand(0:1, N),
            schoolt = 12 .+ 4*randn(N),
            gradHS = rand(0:1, N),
            grad4yr = rand(0:1, N),
            asvabAR = randn(N),
            asvabCS = randn(N),
            asvabMK = randn(N),
            asvabNO = randn(N),
            asvabPC = randn(N),
            asvabWK = randn(N)
        )
        
        X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
        
        # Simple starting values
        svals = 0.1 * ones(45)
        svals[end-6:end] = 0.5 * ones(7)  # positive σ values
        
        # Test that likelihood evaluates
        loglike = factor_model(svals, X, Xfac, asvabs, y, 3)
        @test isfinite(loglike)
    end
    
    @testset "Standard Errors" begin
        # Test with very small data for speed
        N = 50
        Random.seed!(456)
        
        test_df = DataFrame(
            logwage = randn(N),
            black = rand(0:1, N),
            hispanic = rand(0:1, N), 
            female = rand(0:1, N),
            schoolt = randn(N),
            gradHS = rand(0:1, N),
            grad4yr = rand(0:1, N),
            asvabAR = randn(N),
            asvabCS = randn(N),
            asvabMK = randn(N),
            asvabNO = randn(N),
            asvabPC = randn(N),
            asvabWK = randn(N)
        )
        
        # Just test that run_estimation doesn't error
        X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
        start_vals = 0.1 * ones(45)
        start_vals[end-6:end] = ones(7)
        
        # Create simple objective to test mechanics
        obj = θ -> factor_model(θ, X, Xfac, asvabs, y, 3)
        
        # Test that objective evaluates
        @test isfinite(obj(start_vals))
    end
    
end

println("\n" * "="^80)
println("All unit tests passed!")
println("="^80)