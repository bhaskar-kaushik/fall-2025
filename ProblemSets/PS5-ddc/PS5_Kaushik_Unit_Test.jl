# PS5_Kaushik_tests.jl
# Unit tests for Problem Set 5 functions

# Load all required packages
using Test
using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

# Set working directory and include source
cd(@__DIR__)

# Include the create_grids function first
include("create_grids.jl")

# Include source code (contains only function definitions)
include("PS5_Kaushik_source.jl")

println("="^70)
println("RUNNING UNIT TESTS FOR PROBLEM SET 5")
println("="^70)

@testset "PS5 Bus Engine Replacement Tests" begin
    
    #---------------------------------------------------------------------------
    # Test 1: Data Loading Functions
    #---------------------------------------------------------------------------
    @testset "Data Loading" begin
        println("\nTest 1: Testing data loading functions...")
        
        # Test static data loading
        df_long = load_static_data()
        @test size(df_long, 1) > 0
        # CORRECT:
        @test "Y" in names(df_long)
        @test "Odometer" in names(df_long)
        @test "Branded" in names(df_long)
        @test "bus_id" in names(df_long)
        @test "time" in names(df_long)
        @test maximum(df_long.time) == 20
        println("✓ Static data loaded correctly with ", nrow(df_long), " observations")
        
        # Test dynamic data loading
        d = load_dynamic_data()
        @test d.N == 1000  # Number of buses
        @test d.T == 20    # Time periods
        @test size(d.Y) == (d.N, d.T)
        @test size(d.X) == (d.N, d.T)
        @test size(d.Xstate) == (d.N, d.T)
        @test length(d.Zstate) == d.N
        @test length(d.B) == d.N
        @test d.β == 0.9
        @test d.xbin > 0
        @test d.zbin > 0
        println("✓ Dynamic data loaded correctly: N=", d.N, ", T=", d.T)
    end
    
    #---------------------------------------------------------------------------
    # Test 2: State Grids
    #---------------------------------------------------------------------------
    @testset "State Grids" begin
        println("\nTest 2: Testing state grid construction...")
        
        d = load_dynamic_data()
        
        # Test grid dimensions
        @test length(d.xval) == d.xbin
        @test size(d.xtran) == (d.zbin * d.xbin, d.xbin)
        
        # Test that transition matrix rows sum to 1 (probability requirement)
        row_sums = sum(d.xtran, dims=2)
        @test all(abs.(row_sums .- 1.0) .< 1e-10)
        
        # Test that all transition probabilities are non-negative
        @test all(d.xtran .>= 0)
        
        println("✓ State grids constructed correctly")
        println("  - xbin: ", d.xbin, " mileage bins")
        println("  - zbin: ", d.zbin, " route usage bins")
        println("  - Total states: ", d.xbin * d.zbin)
        println("  - Transition matrix rows sum to 1.0: ", all(abs.(row_sums .- 1.0) .< 1e-10))
    end
    
    #---------------------------------------------------------------------------
    # Test 3: Static Estimation
    #---------------------------------------------------------------------------
    @testset "Static Estimation" begin
        println("\nTest 3: Testing static logit estimation...")
        
        df_long = load_static_data()
        theta_static = estimate_static_model(df_long)
        
        # Test that we got a GLM model object
        @test isa(theta_static, StatsModels.TableRegressionModel)
        
        # Test that we have 3 coefficients (intercept, Odometer, Branded)
        θ = coef(theta_static)
        @test length(θ) == 3
        
        # Test coefficient signs (should be reasonable)
        # Intercept should be positive (baseline utility of continuing)
        # Mileage coefficient should be negative (higher mileage = less utility)
        # Branded coefficient should be positive (better brands = more utility)
        @test θ[1] > 0  # Intercept
        @test θ[2] < 0  # Odometer coefficient should be negative
        @test θ[3] > 0  # Branded coefficient should be positive
        
        println("✓ Static model estimated successfully")
        println("  - θ₀ (constant): ", round(θ[1], digits=4))
        println("  - θ₁ (mileage):  ", round(θ[2], digits=4))
        println("  - θ₂ (brand):    ", round(θ[3], digits=4))
    end
    
    #---------------------------------------------------------------------------
    # Test 4: Future Value Computation
    #---------------------------------------------------------------------------
    @testset "Future Value Computation" begin
        println("\nTest 4: Testing future value computation...")
        
        d = load_dynamic_data()
        θ_test = [2.0, -0.15, 1.0]
        
        # Initialize FV array
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        
        # Compute future values
        compute_future_value!(FV, θ_test, d)
        
        # Test terminal condition
        @test all(FV[:, :, d.T+1] .== 0)
        
        # Test that FV values are finite and reasonable
        @test all(isfinite.(FV[:, :, 1:d.T]))
        @test all(FV[:, :, 1:d.T] .< 1000)  # Shouldn't be absurdly large
        
        # Test that FV decreases as we get closer to terminal period
        # (generally true for discounted models)
        avg_fv_early = mean(FV[:, :, 1])
        avg_fv_late = mean(FV[:, :, d.T])
        @test avg_fv_early > avg_fv_late
        
        println("✓ Future values computed successfully")
        println("  - FV dimensions: ", size(FV))
        println("  - Terminal condition satisfied: ", all(FV[:, :, d.T+1] .== 0))
        println("  - Average FV at t=1: ", round(avg_fv_early, digits=4))
        println("  - Average FV at t=T: ", round(avg_fv_late, digits=4))
    end
    
    #---------------------------------------------------------------------------
    # Test 5: Log Likelihood Function
    #---------------------------------------------------------------------------
    @testset "Log Likelihood" begin
        println("\nTest 5: Testing log likelihood function...")
        
        d = load_dynamic_data()
        θ_test = [2.0, -0.15, 1.0]
        
        # Compute log likelihood
        neg_ll = log_likelihood_dynamic(θ_test, d)
        
        # Test that likelihood is finite
        @test isfinite(neg_ll)
        
        # Test that negative log likelihood is positive
        # (since we return the negative)
        @test neg_ll > 0
        
        # Test that likelihood is reasonable in magnitude
        # For 1000 buses × 20 periods = 20,000 observations
        # Log likelihood should be in thousands
        @test neg_ll > 1000
        @test neg_ll < 100000
        
        println("✓ Log likelihood computed successfully")
        println("  - Negative log likelihood: ", round(neg_ll, digits=2))
        println("  - Log likelihood: ", round(-neg_ll, digits=2))
    end
    
    #---------------------------------------------------------------------------
    # Test 6: Optimization Setup
    #---------------------------------------------------------------------------
    @testset "Optimization Function" begin
        println("\nTest 6: Testing optimization wrapper...")
        
        d = load_dynamic_data()
        θ_start = [2.0, -0.15, 1.0]
        
        # Test that objective function works
        objective = θ -> log_likelihood_dynamic(θ, d)
        initial_obj = objective(θ_start)
        
        @test isfinite(initial_obj)
        @test initial_obj > 0
        
        println("✓ Optimization setup working")
        println("  - Initial objective value: ", round(initial_obj, digits=2))
    end
    
    #---------------------------------------------------------------------------
    # Test 7: Data Consistency Checks
    #---------------------------------------------------------------------------
    @testset "Data Consistency" begin
        println("\nTest 7: Testing data consistency...")
        
        d = load_dynamic_data()
        
        # Test that Y contains only 0s and 1s
        @test all(y -> y in [0, 1], d.Y)
        
        # Test that Xstate indices are valid
        @test all(1 .<= d.Xstate .<= d.xbin)
        
        # Test that Zstate indices are valid
        @test all(1 .<= d.Zstate .<= d.zbin)
        
        # Test that B contains only 0s and 1s
        @test all(b -> b in [0, 1], d.B)
        
        # Test that X (mileage) is non-negative
        @test all(d.X .>= 0)
        
        println("✓ Data passes all consistency checks")
    end
    
end

println("\n" * "="^70)
println("ALL TESTS COMPLETED")
println("="^70)