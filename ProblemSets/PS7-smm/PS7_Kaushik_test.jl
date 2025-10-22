################################################################################
# Problem Set 7 - Unit Tests
# ECON 6343: Econometrics III
# GMM and SMM Estimation
# Student: Bhaskar Kaushik
################################################################################

# Load all required packages
using Test, Optim
using Random, LinearAlgebra, Statistics
using DataFrames, CSV, HTTP

include("PS7_Kaushik_source.jl")

println("="^80)
println("Running Unit Tests for Problem Set 7")
println("="^80)

@testset "Problem Set 7 Tests" begin
    
    #--------------------------------------------------------------------------
    # Test 1: OLS-GMM ≈ OLS
    #--------------------------------------------------------------------------
    @testset "Test 1: OLS-GMM equals OLS" begin
        println("\nTest 1: Checking OLS-GMM ≈ OLS")
        
        # Create simple test data
        N = 1000
        X_test = [ones(N) randn(N) randn(N)]
        β_true = [1.0, 2.0, -0.5]
        y_test = X_test * β_true + 0.1 * randn(N)
        
        # Estimate via GMM
        β_gmm = optimize(b -> ols_gmm(b, X_test, y_test), 
                         rand(3), 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-8, iterations=10_000))
        
        # Estimate via closed-form OLS
        β_ols = X_test \ y_test
        
        # Test that they are close
        @test isapprox(β_gmm.minimizer, β_ols, atol=1e-6)
        println("✓ GMM and OLS estimates match within tolerance")
        println("  GMM: ", β_gmm.minimizer)
        println("  OLS: ", β_ols)
        println("  Difference: ", norm(β_gmm.minimizer - β_ols))
    end
    
    #--------------------------------------------------------------------------
    # Test 2: Softmax rows sum to 1
    #--------------------------------------------------------------------------
    @testset "Test 2: Softmax probabilities sum to 1" begin
        println("\nTest 2: Checking softmax probabilities sum to 1")
        
        # Create test data
        N = 100
        J = 4
        K = 3
        X_test = randn(N, K)
        α_test = randn(K, J)
        
        Xβ = X_test * α_test
        P, _ = stable_softmax(Xβ)
        
        # Check that each row sums to 1
        row_sums = sum(P, dims=2)
        @test all(isapprox.(row_sums, 1.0, atol=1e-10))
        println("✓ All probability rows sum to 1")
        println("  Max deviation from 1: ", maximum(abs.(row_sums .- 1.0)))
    end
    
    #--------------------------------------------------------------------------
    # Test 3: MLE reduces negative log-likelihood
    #--------------------------------------------------------------------------
    @testset "Test 3: MLE improves objective from random start" begin
        println("\nTest 3: Checking MLE reduces negative log-likelihood")
        
        # Simulate small dataset
        Random.seed!(42)
        N = 500
        J = 3
        X_test = hcat(ones(N), randn(N), randn(N) .> 0.5)
        β_true = hcat([1.0, -0.5, 0.3], [0.5, 0.8, -0.2], zeros(3))
        
        # Gumbel(0,1) via inverse CDF
        ε = -log.(-log.(rand(N, J)))
        y_test = argmax.(eachrow(X_test * β_true .+ ε))
        
        # Random starting values
        α_start = randn(3 * (J-1))
        nll_start = mlogit_mle(α_start, X_test, y_test)
        
        # Optimize
        result = optimize(a -> mlogit_mle(a, X_test, y_test), 
                          α_start, 
                          LBFGS(), 
                          Optim.Options(g_tol=1e-5, iterations=10_000))
        
        nll_end = result.minimum
        
        @test nll_end < nll_start
        println("✓ MLE reduced negative log-likelihood")
        println("  Initial NLL: ", nll_start)
        println("  Final NLL: ", nll_end)
        println("  Improvement: ", nll_start - nll_end)
    end
    
    #--------------------------------------------------------------------------
    # Test 4: Stable softmax vs naive softmax
    #--------------------------------------------------------------------------
    @testset "Test 4: Stable softmax handles extreme values" begin
        println("\nTest 4: Checking numerical stability of softmax")
        
        # Create data with extreme values
        N = 10
        J = 3
        Xβ_extreme = [[-100.0, 0.0, 100.0] for i in 1:N]
        Xβ_extreme = reduce(vcat, transpose.(Xβ_extreme))
        
        P, _ = stable_softmax(Xβ_extreme)
        
        # Check no NaN or Inf
        @test !any(isnan.(P))
        @test !any(isinf.(P))
        
        # Check rows sum to 1
        row_sums = sum(P, dims=2)
        @test all(isapprox.(row_sums, 1.0, atol=1e-10))
        
        println("✓ Stable softmax handles extreme values without NaN/Inf")
        println("  Example probabilities: ", P[1,:])
    end
    
    #--------------------------------------------------------------------------
    # Test 5: Simulation produces valid choices
    #--------------------------------------------------------------------------
    @testset "Test 5: Simulation produces valid discrete choices" begin
        println("\nTest 5: Checking simulation produces valid choices")
        
        Random.seed!(123)
        y_sim, X_sim, β_sim = sim_logit_w_gumbel(1000, 4)
        
        # Check all choices are in valid range
        @test all(y_sim .>= 1)
        @test all(y_sim .<= 4)
        @test all(isinteger.(y_sim))
        
        # Check all choices are represented
        @test length(unique(y_sim)) == 4
        
        println("✓ Simulation produces valid discrete choices")
        println("  Choice frequencies: ", [mean(y_sim .== j) for j in 1:4])
    end
    
    #--------------------------------------------------------------------------
    # Test 6: SMM objective decreases during optimization
    #--------------------------------------------------------------------------
    @testset "Test 6: SMM objective improves" begin
        println("\nTest 6: Checking SMM objective decreases")
        
        # Use small simulated dataset
        Random.seed!(456)
        N = 300
        J = 3
        X_test = hcat(ones(N), randn(N))
        β_true = hcat([1.0, -0.5], [0.5, 0.8], zeros(2))
        
        # Gumbel(0,1) via inverse CDF
        ε = -log.(-log.(rand(N, J)))
        y_test = argmax.(eachrow(X_test * β_true .+ ε))
        
        # Random starting values
        α_start = randn(2 * (J-1))
        obj_start = mlogit_smm_overid(α_start, X_test, y_test, 50)
        
        # Optimize
        result = optimize(a -> mlogit_smm_overid(a, X_test, y_test, 50),
                          α_start,
                          LBFGS(), 
                          Optim.Options(g_tol=1e-5, iterations=500))
        
        obj_end = result.minimum
        
        @test obj_end < obj_start
        println("✓ SMM objective decreased")
        println("  Initial objective: ", obj_start)
        println("  Final objective: ", obj_end)
        println("  Improvement: ", obj_start - obj_end)
    end
    
    #--------------------------------------------------------------------------
    # Test 7: Parameter recovery in simulation
    #--------------------------------------------------------------------------
    @testset "Test 7: Parameter recovery from simulated data" begin
        println("\nTest 7: Checking parameter recovery")
        
        Random.seed!(789)
        N = 10_000
        J = 3
        X_test = hcat(ones(N), randn(N))
        β_true = hcat([0.5, -1.0], [1.0, 0.5], zeros(2))
        
        # Simulate data with Gumbel(0,1) via inverse CDF
        ε = -log.(-log.(rand(N, J)))
        y_test = argmax.(eachrow(X_test * β_true .+ ε))
        
        # Recover parameters
        α_start = randn(2 * (J-1))
        result = optimize(a -> mlogit_mle(a, X_test, y_test),
                          α_start,
                          LBFGS(), 
                          Optim.Options(g_tol=1e-5, iterations=10_000))
        
        β_recovered = [reshape(result.minimizer, 2, J-1) zeros(2)]
        
        # Check recovery error is small (should be within ~0.1 with large N)
        recovery_error = norm(β_true - β_recovered)
        @test recovery_error < 0.15
        
        println("✓ Parameters recovered with small error")
        println("  True β:")
        println(β_true)
        println("  Recovered β:")
        println(β_recovered)
        println("  Recovery error (Frobenius norm): ", recovery_error)
    end
    
    #--------------------------------------------------------------------------
    # Test 8: Data loading and preparation
    #--------------------------------------------------------------------------
    @testset "Test 8: Data loading functions work" begin
        println("\nTest 8: Checking data loading functions")
        
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        
        # Test load_data
        df, X, y = load_data(url)
        @test size(X, 2) == 4  # intercept, age, race, collgrad
        @test size(X, 1) == length(y)
        @test !any(isnan.(y))
        
        # Test prepare_occupation_data
        df2, X2, y2 = prepare_occupation_data(df)
        @test size(X2, 2) == 4  # intercept, age, white, collgrad
        @test all(y2 .>= 1)
        @test all(y2 .<= 7)
        
        println("✓ Data loading and preparation functions work correctly")
        println("  Wage data: N = $(size(X,1)), K = $(size(X,2))")
        println("  Occupation data: N = $(size(X2,1)), J = $(length(unique(y2)))")
    end
    
end

println("\n" * "="^80)
println("All Tests Complete!")
println("="^80)