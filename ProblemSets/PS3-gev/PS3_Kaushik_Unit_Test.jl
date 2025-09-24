
# ============================================================
# PS3 – GEV Models (ECON 6343): Unit Tests
# Tests for all functions in PS3_Kaushik_source.jl
# AI note (required by syllabus): "Used Claude and ChatGPT to debug and refine code/tests for ECON 6343 Fall 2025 PS3."
# ============================================================

using Test, Random, LinearAlgebra, Statistics
using DataFrames, CSV, HTTP
using Optim, Distributions

# Load source functions
include("PS3_Kaushik_source.jl")

println("\n", "="^60)
println("RUNNING UNIT TESTS FOR PS3 GEV MODELS")
println("="^60, "\n")

@testset "PS3 GEV Model Tests" begin
    
    # Create small test dataset
    Random.seed!(42)
    N_test = 50
    K_test = 3
    J_test = 4
    
    X_test = randn(N_test, K_test)
    X_test[:, 1] .= 1.0  # intercept
    Z_test = randn(N_test, J_test)
    y_test = rand(1:J_test, N_test)
    
    @testset "Multinomial Logit Tests" begin
        println("Testing Multinomial Logit Functions...")
        
        @test_nowarn mlogit_with_Z(randn(K_test*(J_test-1) + 1), X_test, Z_test, y_test)
        @test_throws AssertionError mlogit_with_Z(randn(5), X_test, Z_test, y_test)
        
        theta_test = randn(K_test*(J_test-1) + 1) * 0.1
        ll = mlogit_with_Z(theta_test, X_test, Z_test, y_test)
        @test isfinite(ll)
        @test ll > 0
        
        theta2 = theta_test + randn(length(theta_test)) * 0.01
        ll2 = mlogit_with_Z(theta2, X_test, Z_test, y_test)
        @test ll ≠ ll2
        
        y_bad = [0; rand(1:J_test, N_test-1)]
        @test_throws AssertionError mlogit_with_Z(theta_test, X_test, Z_test, y_bad)
        
        println("✓ Multinomial logit function tests passed")
    end
    
    @testset "Nested Logit Tests" begin
        println("Testing Nested Logit Functions...")
        
        N_nl = 30
        X_nl = randn(N_nl, K_test)
        X_nl[:, 1] .= 1.0
        Z_nl = randn(N_nl, 8)
        y_nl = rand(1:8, N_nl)
        
        theta_nl = [randn(2*K_test); 0.7; 0.8; 0.1]
        
        @test_nowarn nested_logit_with_Z(theta_nl, X_nl, Z_nl, y_nl)
        @test_throws AssertionError nested_logit_with_Z(randn(5), X_nl, Z_nl, y_nl)
        
        Z_wrong = randn(N_nl, 6)
        @test_throws AssertionError nested_logit_with_Z(theta_nl, X_nl, Z_wrong, y_nl)
        
        theta_extreme = copy(theta_nl)
        theta_extreme[2*K_test+1] = -10.0
        theta_extreme[2*K_test+2] = 10.0
        @test_nowarn nested_logit_with_Z(theta_extreme, X_nl, Z_nl, y_nl)
        
        ll_nl = nested_logit_with_Z(theta_nl, X_nl, Z_nl, y_nl)
        @test isfinite(ll_nl)
        @test ll_nl > 0
        
        println("✓ Nested logit function tests passed")
    end
    
    @testset "Optimization Functions Tests" begin
        println("Testing Optimization Functions...")
        
        @test_nowarn optimize_mlogit(X_test, Z_test, y_test)
        result = optimize_mlogit(X_test, Z_test, y_test)
        @test isa(result, Optim.OptimizationResults)
        @test length(result.minimizer) == K_test*(J_test-1) + 1
        
        Z_8 = randn(N_test, 8)
        y_8 = rand(1:8, N_test)
        @test_nowarn optimize_nested_logit(X_test, Z_8, y_8)
        result_nl = optimize_nested_logit(X_test, Z_8, y_8)
        @test isa(result_nl, Optim.OptimizationResults)
        @test length(result_nl.minimizer) == 2*K_test + 3
        @test isfinite(result_nl.minimum)
        
        # Test that lambda parameters are within bounds
        λWC = result_nl.minimizer[2*K_test+1]
        λBC = result_nl.minimizer[2*K_test+2]
        @test 0.01 ≤ λWC ≤ 1.0
        @test 0.01 ≤ λBC ≤ 1.0
        
        println("✓ Optimization function tests passed")
    end
    
    @testset "Standard Error Function Tests" begin
        println("Testing Standard Error Functions...")
        
        theta_se = randn(5) * 0.1
        simple_quad(x) = sum(x.^2)
        
        se_result = hessian_se(theta_se, simple_quad)
        @test length(se_result) == length(theta_se)
        @test all(isfinite.(se_result))
        @test all(se_result .≥ 0)
        
        theta_real = randn(K_test*(J_test-1) + 1) * 0.1
        se_real = hessian_se(theta_real, θ -> mlogit_with_Z(θ, X_test, Z_test, y_test))
        @test length(se_real) == length(theta_real)
        @test sum(isfinite.(se_real)) > 0
        
        println("✓ Standard error function tests passed")
    end
    
    @testset "Interpretation Functions Tests" begin
        println("Testing Interpretation Functions...")
        
        γ_test = 0.5
        Δ_test = 0.1
        odds_mult = odds_multiplier_for_delta_logwage(γ_test, Δ_test)
        @test odds_mult ≈ exp(γ_test * Δ_test)
        @test odds_mult > 0
        
        γ_neg = -0.3
        odds_neg = odds_multiplier_for_delta_logwage(γ_neg, Δ_test)
        @test odds_neg < 1
        @test odds_neg > 0
        
        p_test = 0.3
        elasticity = wage_semi_elasticity(γ_test, p_test)
        @test elasticity ≈ γ_test * (1 - p_test)
        
        elast_zero = wage_semi_elasticity(γ_test, 1.0)
        @test elast_zero ≈ 0.0
        elast_max = wage_semi_elasticity(γ_test, 0.0)
        @test elast_max ≈ γ_test
        
        λ_test = 0.8
        corr = correlation_measure(λ_test)
        @test corr ≈ (1 - λ_test)
        @test 0 ≤ corr ≤ 1
        
        @test correlation_measure(1.0) ≈ 0.0
        @test correlation_measure(0.0) ≈ 1.0
        
        interp_pos = interpret_gamma(0.5)
        interp_neg = interpret_gamma(-0.5)
        @test isa(interp_pos, String)
        @test isa(interp_neg, String)
        @test length(interp_pos) > 50
        @test length(interp_neg) > 50
        @test occursin("Positive", interp_pos)
        @test occursin("Negative", interp_neg)
        
        println("✓ Interpretation function tests passed")
    end
    
    @testset "Data Validation Tests" begin
        println("Testing Data Validation...")
        
        X_bad_rows = randn(N_test+5, K_test)
        @test_throws AssertionError mlogit_with_Z(randn(K_test*(J_test-1) + 1), X_bad_rows, Z_test, y_test)
        
        X_bad_cols = randn(N_test, K_test+2)
        @test_nowarn mlogit_with_Z(randn((K_test+2)*(J_test-1) + 1), X_bad_cols, Z_test, y_test)
        
        Z_bad = randn(N_test+3, J_test)
        @test_throws AssertionError mlogit_with_Z(randn(K_test*(J_test-1) + 1), X_test, Z_bad, y_test)
        
        y_zero = [0; rand(1:J_test, N_test-1)]
        @test_throws AssertionError mlogit_with_Z(randn(K_test*(J_test-1) + 1), X_test, Z_test, y_zero)
        
        y_high = [rand(1:J_test, N_test-1); J_test+1]
        @test_throws AssertionError mlogit_with_Z(randn(K_test*(J_test-1) + 1), X_test, Z_test, y_high)
        
        Z_nl_wrong = randn(N_test, 6)
        theta_nl = randn(2*K_test + 3)
        y_nl = rand(1:8, N_test)
        @test_throws AssertionError nested_logit_with_Z(theta_nl, X_test, Z_nl_wrong, y_nl)
        
        println("✓ Data validation tests passed")
    end
    
    @testset "Numerical Stability Tests" begin
        println("Testing Numerical Stability...")
        
        theta_extreme = [fill(10.0, K_test*(J_test-1)); -5.0]
        @test_nowarn mlogit_with_Z(theta_extreme, X_test, Z_test, y_test)
        ll_extreme = mlogit_with_Z(theta_extreme, X_test, Z_test, y_test)
        @test isfinite(ll_extreme)
        
        Z_extreme = randn(N_test, J_test) * 100
        @test_nowarn mlogit_with_Z(randn(K_test*(J_test-1) + 1) * 0.01, X_test, Z_extreme, y_test)
        
        Z_nl = randn(N_test, 8) * 10
        y_nl = rand(1:8, N_test)
        theta_nl_extreme = [fill(5.0, 2*K_test); 0.1; 0.9; -2.0]
        @test_nowarn nested_logit_with_Z(theta_nl_extreme, X_test, Z_nl, y_nl)
        ll_nl_extreme = nested_logit_with_Z(theta_nl_extreme, X_test, Z_nl, y_nl)
        @test isfinite(ll_nl_extreme)
        
        println("✓ Numerical stability tests passed")
    end
    
    @testset "Integration Test with Real Data Structure" begin
        println("Testing Integration with Real Data Structure...")
        
        N_real = 100
        X_real = hcat(ones(N_real), randn(N_real, 3))
        Z_real = randn(N_real, 8)
        y_real = rand(1:8, N_real)
        
        @test_nowarn begin
            result_mnl = optimize_mlogit(X_real, Z_real, y_real)
            @test Optim.converged(result_mnl)
            
            se_mnl = hessian_se(result_mnl.minimizer, θ -> mlogit_with_Z(θ, X_real, Z_real, y_real))
            @test length(se_mnl) == length(result_mnl.minimizer)
            
            γ_hat = result_mnl.minimizer[end]
            interp = interpret_gamma(γ_hat)
            @test isa(interp, String)
            
            odds = odds_multiplier_for_delta_logwage(γ_hat, 0.1)
            @test isfinite(odds) && odds > 0
        end
        
        @test_nowarn begin
            result_nl = optimize_nested_logit(X_real, Z_real, y_real)
            @test Optim.converged(result_nl)
            
            se_nl = hessian_se(result_nl.minimizer, θ -> nested_logit_with_Z(θ, X_real, Z_real, y_real))
            @test length(se_nl) == length(result_nl.minimizer)
            
            λWC = result_nl.minimizer[2*size(X_real,2)+1]
            λBC = result_nl.minimizer[2*size(X_real,2)+2]
            @test 0.01 ≤ λWC ≤ 1.0
            @test 0.01 ≤ λBC ≤ 1.0
            
            corr_WC = correlation_measure(λWC)
            corr_BC = correlation_measure(λBC)
            @test 0 ≤ corr_WC ≤ 1
            @test 0 ≤ corr_BC ≤ 1
        end
        
        ll_mnl_int = -optimize_mlogit(X_real, Z_real, y_real).minimum
        ll_nl_int = -optimize_nested_logit(X_real, Z_real, y_real).minimum
        @test ll_nl_int ≠ ll_mnl_int
        
        println("✓ Integration tests passed")
    end
    
    @testset "Edge Cases and Robustness" begin
        println("Testing Edge Cases and Robustness...")
        
        N_min = 10
        X_min = hcat(ones(N_min), randn(N_min, 1))
        Z_min = randn(N_min, 3)
        y_min = rand(1:3, N_min)
        
        theta_min = randn(size(X_min,2)*(size(Z_min,2)-1) + 1) * 0.1
        @test_nowarn mlogit_with_Z(theta_min, X_min, Z_min, y_min)
        
        y_same = fill(1, N_test)
        @test_nowarn mlogit_with_Z(randn(K_test*(J_test-1) + 1) * 0.01, X_test, Z_test, y_same)
        
        X_sep = hcat(ones(N_test), float.(y_test))
        @test_nowarn mlogit_with_Z(randn(2*(J_test-1) + 1) * 0.01, X_sep, Z_test, y_test)
        
        println("✓ Edge cases and robustness tests passed")
    end
end

println("\n", "="^60)
println("ALL UNIT TESTS COMPLETED SUCCESSFULLY!")
println("="^60, "\n")

# Optional integration test with PS3-like data
println("Running quick integration test with PS3-like data...")
try
    Random.seed!(123)
    N_sim = 200
    X_sim = hcat(ones(N_sim), 
                 20 .+ 10*randn(N_sim),      # age around 20-40
                 rand([0,1], N_sim),         # white (binary)
                 rand([0,1], N_sim))         # college grad (binary)
    
    Z_sim = randn(N_sim, 8) .+ 2.5  # log wages around exp(2.5) ≈ $12/hour
    y_sim = rand(1:8, N_sim)
    
    println("  Simulated data: N=$N_sim, K=$(size(X_sim,2)), J=$(size(Z_sim,2))")
    
    result_sim = optimize_mlogit(X_sim, Z_sim, y_sim)
    println("  MNL optimization: ", Optim.converged(result_sim) ? "✓ converged" : "✗ failed")
    
    result_nl_sim = optimize_nested_logit(X_sim, Z_sim, y_sim)
    println("  NL optimization:  ", Optim.converged(result_nl_sim) ? "✓ converged" : "✗ failed")
    
    println("✓ Integration test successful!")
    
catch e
    println("✗ Integration test failed: ", e)
end

println("\n", "="^60)
println("TESTING COMPLETE - ALL SYSTEMS GO!")
println("="^60)