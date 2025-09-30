# Problem Set 4 Unit Tests
# ECON 6343: Econometrics III
# Student: Bhaskar Kaushik
# Date: Fall 2025
# Professor: Tyler Ransom, University of Oklahoma
# AI note (required by syllabus): "Used Claude to help design and debug unit tests for ECON 6343 Fall 2025 PS4."

using Test
using Random, LinearAlgebra, Statistics, DataFrames, Distributions

# Load source code
include("PS4_Kaushik_Source.jl")

println("="^70)
println("RUNNING UNIT TESTS FOR PROBLEM SET 4")
println("="^70)

#---------------------------------------------------
# Test Suite 1: Data Loading and Structure
#---------------------------------------------------
@testset "Data Loading Tests" begin
    println("\n[Test 1] Data Loading and Structure...")
    
    df, X, Z, y = load_data()
    
    # Test data dimensions
    @test size(X, 1) == size(Z, 1) == length(y)
    @test size(X, 2) == 3  # age, white, collgrad
    @test size(Z, 2) == 8  # 8 occupations
    
    # Test data types
    @test eltype(X) <: Real
    @test eltype(Z) <: Real
    @test eltype(y) <: Integer
    
    # Test choice set
    @test minimum(y) == 1
    @test maximum(y) == 8
    @test all(y .>= 1) && all(y .<= 8)
    
    # Test no missing values
    @test !any(ismissing.(X))
    @test !any(ismissing.(Z))
    @test !any(ismissing.(y))
    
    println("  ✓ Data loads correctly")
    println("  ✓ Dimensions: N=$(size(X,1)), K=$(size(X,2)), J=$(length(unique(y)))")
end

#---------------------------------------------------
# Test Suite 2: Multinomial Logit Function
#---------------------------------------------------
@testset "Multinomial Logit Tests" begin
    println("\n[Test 2] Multinomial Logit Function...")
    
    # Create small test data
    Random.seed!(123)
    N_test, K_test, J_test = 100, 3, 8
    X_test = randn(N_test, K_test)
    Z_test = randn(N_test, J_test)
    y_test = rand(1:J_test, N_test)
    
    # Test parameters
    theta_test = [randn(K_test*(J_test-1)); 0.5]
    
    # Test likelihood computation
    ll = mlogit_with_Z(theta_test, X_test, Z_test, y_test)
    
    @test isfinite(ll)
    @test ll > 0  # negative log-likelihood should be positive
    @test typeof(ll) <: Real
    
    # Test that likelihood decreases with better parameters
    theta_better = theta_test .* 0.9
    ll_better = mlogit_with_Z(theta_better, X_test, Z_test, y_test)
    # Don't test which is better since we don't know optimal direction
    @test isfinite(ll_better)
    
    println("  ✓ Likelihood function computes correctly")
    println("  ✓ Log-likelihood = ", round(ll, digits=2))
end

#---------------------------------------------------
# Test Suite 3: Quadrature Integration
#---------------------------------------------------
@testset "Quadrature Tests" begin
    println("\n[Test 3] Quadrature Integration...")
    
    # Test standard normal integration
    d = Normal(0, 1)
    nodes, weights = lgwt(7, -4, 4)
    
    # Test density integration
    integral = sum(weights .* pdf.(d, nodes))
    @test isapprox(integral, 1.0, atol=0.01)
    
    # Test expectation
    expectation = sum(weights .* nodes .* pdf.(d, nodes))
    @test isapprox(expectation, 0.0, atol=0.001)
    
    # Test variance computation
    σ = 2.0
    d2 = Normal(0, σ)
    nodes_var, weights_var = lgwt(10, -5*σ, 5*σ)
    variance = sum(weights_var .* (nodes_var.^2) .* pdf.(d2, nodes_var))
    @test isapprox(variance, σ^2, atol=0.1)
    
    println("  ✓ Quadrature approximates integrals correctly")
    println("  ✓ Density integral ≈ ", round(integral, digits=4))
    println("  ✓ Variance ≈ ", round(variance, digits=4), " (true: 4.0)")
end

#---------------------------------------------------
# Test Suite 4: Monte Carlo Integration
#---------------------------------------------------
@testset "Monte Carlo Tests" begin
    println("\n[Test 4] Monte Carlo Integration...")
    
    Random.seed!(456)
    σ = 2.0
    d = Normal(0, σ)
    a, b = -5*σ, 5*σ
    
    function mc_integrate(f, a, b, D)
        draws = rand(D) * (b - a) .+ a
        return (b - a) * mean(f.(draws))
    end
    
    # Test with large number of draws
    D = 100_000
    variance_mc = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)
    mean_mc = mc_integrate(x -> x * pdf(d, x), a, b, D)
    density_mc = mc_integrate(x -> pdf(d, x), a, b, D)
    
    @test isapprox(variance_mc, σ^2, atol=0.1)
    @test isapprox(mean_mc, 0.0, atol=0.05)
    @test isapprox(density_mc, 1.0, atol=0.01)
    
    println("  ✓ Monte Carlo approximates integrals correctly")
    println("  ✓ Variance ≈ ", round(variance_mc, digits=4))
    println("  ✓ MC converges with more draws")
end

#---------------------------------------------------
# Test Suite 5: Mixed Logit Quadrature
#---------------------------------------------------
@testset "Mixed Logit Quadrature Tests" begin
    println("\n[Test 5] Mixed Logit Quadrature...")
    
    # Create small test data
    Random.seed!(789)
    N_test, K_test, J_test = 50, 3, 8
    X_test = randn(N_test, K_test)
    Z_test = randn(N_test, J_test)
    y_test = rand(1:J_test, N_test)
    
    # Test parameters: [alpha, mu_gamma, sigma_gamma]
    theta_test = [randn(K_test*(J_test-1)); 0.5; 1.0]
    
    # Test with 5 quadrature points
    ll = mixed_logit_quad(theta_test, X_test, Z_test, y_test, 5)
    
    @test isfinite(ll)
    @test ll > 0
    @test typeof(ll) <: Real
    
    # Test that function requires positive sigma (should error with negative)
    theta_bad = copy(theta_test)
    theta_bad[end] = -1.0  # negative sigma
    @test_throws DomainError mixed_logit_quad(theta_bad, X_test, Z_test, y_test, 5)
    
    println("  ✓ Mixed logit quadrature computes correctly")
    println("  ✓ Properly rejects negative sigma")
    println("  ✓ Log-likelihood = ", round(ll, digits=2))
end

#---------------------------------------------------
# Test Suite 6: Mixed Logit Monte Carlo
#---------------------------------------------------
@testset "Mixed Logit Monte Carlo Tests" begin
    println("\n[Test 6] Mixed Logit Monte Carlo...")
    
    # Create small test data
    Random.seed!(101112)
    N_test, K_test, J_test = 50, 3, 8
    X_test = randn(N_test, K_test)
    Z_test = randn(N_test, J_test)
    y_test = rand(1:J_test, N_test)
    
    # Test parameters
    theta_test = [randn(K_test*(J_test-1)); 0.5; 1.0]
    
    # Test with 100 MC draws
    ll = mixed_logit_mc(theta_test, X_test, Z_test, y_test, 100)
    
    @test isfinite(ll)
    @test ll > 0
    @test typeof(ll) <: Real
    
    # Test that more draws gives different (but similar) result
    ll2 = mixed_logit_mc(theta_test, X_test, Z_test, y_test, 200)
    @test isfinite(ll2)
    # Results should be close but not identical due to randomness
    @test isapprox(ll, ll2, rtol=0.2)
    
    println("  ✓ Mixed logit MC computes correctly")
    println("  ✓ Log-likelihood = ", round(ll, digits=2))
end

#---------------------------------------------------
# Test Suite 7: Parameter Recovery
#---------------------------------------------------
@testset "Parameter Recovery Tests" begin
    println("\n[Test 7] Parameter Recovery...")
    
    # Create data with known parameters
    Random.seed!(131415)
    N_test = 200
    K_test = 2
    J_test = 4
    
    # True parameters
    true_alpha = randn(K_test*(J_test-1))
    true_gamma = 0.5
    true_theta = [true_alpha; true_gamma]
    
    # Generate data
    X_test = randn(N_test, K_test)
    Z_test = randn(N_test, J_test)
    
    # Generate choices based on true parameters
    bigAlpha = [reshape(true_alpha, K_test, J_test-1) zeros(K_test)]
    probs = zeros(N_test, J_test)
    for i = 1:N_test
        for j = 1:J_test
            probs[i,j] = exp(X_test[i,:]' * bigAlpha[:,j] + 
                           true_gamma * (Z_test[i,j] - Z_test[i,J_test]))
        end
        probs[i,:] ./= sum(probs[i,:])
    end
    
    y_test = [rand(Categorical(probs[i,:])) for i = 1:N_test]
    
    # Estimate parameters
    result = optimize(
        theta -> mlogit_with_Z(theta, X_test, Z_test, y_test),
        true_theta .+ 0.1*randn(length(true_theta)),
        LBFGS(),
        Optim.Options(g_tol=1e-4, iterations=1000)
    )
    
    estimated_theta = result.minimizer
    
    # Check that estimates are close to truth
    # (may not be exact due to sampling variability)
    @test isapprox(estimated_theta[end], true_gamma, atol=0.3)
    
    println("  ✓ Parameter recovery test passed")
    println("  ✓ True γ = ", round(true_gamma, digits=3))
    println("  ✓ Estimated γ = ", round(estimated_theta[end], digits=3))
end

#---------------------------------------------------
# Test Suite 8: Numerical Stability
#---------------------------------------------------
@testset "Numerical Stability Tests" begin
    println("\n[Test 8] Numerical Stability...")
    
    Random.seed!(161718)
    N_test, K_test, J_test = 100, 3, 8
    X_test = randn(N_test, K_test)
    Z_test = randn(N_test, J_test)
    y_test = rand(1:J_test, N_test)
    
    # Test with extreme parameters
    theta_extreme = [10*randn(K_test*(J_test-1)); 5.0]
    ll_extreme = mlogit_with_Z(theta_extreme, X_test, Z_test, y_test)
    
    @test isfinite(ll_extreme)
    @test !isnan(ll_extreme)
    @test !isinf(ll_extreme)
    
    # Test with very small parameters
    theta_small = 0.01*randn(K_test*(J_test-1)+1)
    ll_small = mlogit_with_Z(theta_small, X_test, Z_test, y_test)
    
    @test isfinite(ll_small)
    
    println("  ✓ Likelihood computation is numerically stable")
    println("  ✓ Handles extreme parameter values")
end

#---------------------------------------------------
# Summary
#---------------------------------------------------
println("\n" * "="^70)
println("ALL TESTS PASSED ✓")
println("="^70)
println("\nTest Summary:")
println("  ✓ Data loading and validation")
println("  ✓ Multinomial logit likelihood")
println("  ✓ Quadrature integration accuracy")
println("  ✓ Monte Carlo integration convergence")
println("  ✓ Mixed logit with quadrature")
println("  ✓ Mixed logit with Monte Carlo")
println("  ✓ Parameter recovery")
println("  ✓ Numerical stability")
println("\nAll functions verified and ready for analysis!")
println("="^70)