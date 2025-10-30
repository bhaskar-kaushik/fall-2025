# Debug and Test Script for Implicit State Prices Analysis
# Use this to troubleshoot issues with your specific data

using DataFrames, CSV, HTTP, GLM, Statistics, LinearAlgebra

include("create_grids.jl")
include("rust_estimation.jl")

"""
Test each component separately to isolate issues
"""
function debug_analysis()
    
    println("="^70)
    println("DEBUG MODE: Testing each component")
    println("="^70)
    
    β = 0.9
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    
    # Test 1: Data loading
    println("\n[TEST 1] Loading data...")
    try
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        println("  ✓ Data loaded successfully")
        println("    Observations: ", nrow(df_long))
        println("    Buses: ", size(Xstate, 1))
    catch e
        println("  ✗ Error loading data:")
        println("    ", e)
        return
    end
    
    # Test 2: State space
    println("\n[TEST 2] Creating state space...")
    try
        zval, zbin, xval, xbin, xtran = create_grids()
        println("  ✓ State space created")
        println("    xbin: ", xbin, ", zbin: ", zbin)
        println("    Transition matrix: ", size(xtran))
        println("    Row sums check: ", extrema(sum(xtran, dims=2)))
    catch e
        println("  ✗ Error creating state space:")
        println("    ", e)
        return
    end
    
    # Test 3: Flexible logit
    println("\n[TEST 3] Estimating flexible logit...")
    try
        df_long, _, _, _ = load_and_reshape_data(url)
        flexlogit = estimate_flexible_logit(df_long)
        println("  ✓ Flexible logit estimated")
        println("    Coefficients: ", length(coef(flexlogit)))
    catch e
        println("  ✗ Error in flexible logit:")
        println("    ", e)
        return
    end
    
    # Test 4: Future values
    println("\n[TEST 4] Computing future values...")
    try
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        flexlogit = estimate_flexible_logit(df_long)
        zval, zbin, xval, xbin, xtran = create_grids()
        statedf = construct_state_space(xbin, zbin, xval, zval, xtran)
        
        FV = compute_future_values(statedf, flexlogit, xtran, xbin, zbin, 20, β)
        println("  ✓ Future values computed")
        println("    FV dimensions: ", size(FV))
        println("    FV range: [", round(minimum(FV), digits=4), ", ", 
                round(maximum(FV), digits=4), "]")
    catch e
        println("  ✗ Error computing future values:")
        println("    ", e)
        showerror(stdout, e, catch_backtrace())
        return
    end
    
    # Test 5: Structural estimation
    println("\n[TEST 5] Structural parameter estimation...")
    try
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        flexlogit = estimate_flexible_logit(df_long)
        zval, zbin, xval, xbin, xtran = create_grids()
        statedf = construct_state_space(xbin, zbin, xval, zval, xtran)
        FV = compute_future_values(statedf, flexlogit, xtran, xbin, zbin, 20, β)
        efvt1 = compute_fvt1(FV, xtran, Xstate, Zstate, xbin, Branded)
        
        theta_hat = estimate_structural_params(df_long, efvt1)
        println("  ✓ Structural parameters estimated")
        println("    Coefficients: ", coef(theta_hat))
    catch e
        println("  ✗ Error in structural estimation:")
        println("    ", e)
        showerror(stdout, e, catch_backtrace())
        return
    end
    
    # Test 6: Implicit prices
    println("\n[TEST 6] Computing implicit state prices...")
    try
        zval, zbin, xval, xbin, xtran = create_grids()
        
        # Simple version for testing
        nstates = size(xtran, 1)
        Q0 = β * xtran
        Q1 = zeros(nstates, size(xtran, 2))
        
        for z in 1:zbin
            row_start = (z-1)*xbin + 1
            row_end = min(z*xbin, nstates)
            
            if row_start <= nstates
                Q1[row_start:row_end, row_start] .= β
            end
        end
        
        println("  ✓ Implicit prices computed")
        println("    Q0 dimensions: ", size(Q0))
        println("    Q1 dimensions: ", size(Q1))
        println("    Q0 row sum (sample): ", round(sum(Q0[1,:]), digits=4))
        println("    Q1 row sum (sample): ", round(sum(Q1[1,:]), digits=4))
        
        # Test indexing
        println("\n  Testing safe indexing...")
        test_states = [1, min(100, nstates), min(nstates, size(Q0, 1))]
        for s in test_states
            if s <= size(Q0, 1)
                println("    State $s: Q0 sum = ", round(sum(Q0[s,:]), digits=4))
            end
        end
        
    catch e
        println("  ✗ Error computing implicit prices:")
        println("    ", e)
        showerror(stdout, e, catch_backtrace())
        return
    end
    
    println("\n" * "="^70)
    println("ALL TESTS PASSED!")
    println("="^70)
    println("\nYour setup is working correctly. You can now run:")
    println("  include(\"implicit_prices_analysis.jl\")")
    
end

# Run debug
debug_analysis()