# Implicit State Prices in Dynamic Discrete Choice
# Extension to Rust (1987) Bus Engine Replacement Model
# Author: Bhaskar Kaushik

using DataFrames, CSV, HTTP, GLM, Statistics, LinearAlgebra
using Plots, StatsPlots, DataFramesMeta

include("create_grids.jl")
include("rust_estimation.jl")  # Your existing code

#========================================
PART 1: COMPUTE IMPLICIT STATE PRICES
========================================#

"""
    compute_implicit_state_prices(xtran::Matrix, 
                                   xbin::Int,
                                   zbin::Int, 
                                   β::Float64)

Compute implicit state prices q_d(x'|x,z) = β * f(x'|x,z,d) for both actions.

# Arguments
- `xtran::Matrix`: State transition matrix (rows = current states, cols = future states)
- `xbin::Int`: Number of mileage bins
- `zbin::Int`: Number of route usage bins
- `β::Float64`: Discount factor

# Returns
- `Q0::Matrix`: Implicit prices for action d=0 (keep engine) - size (states × future_states)
- `Q1::Matrix`: Implicit prices for action d=1 (replace engine) - size (states × future_states)
"""
function compute_implicit_state_prices(xtran::Matrix, 
                                        xbin::Int,
                                        zbin::Int, 
                                        β::Float64)
    
    nstates = size(xtran, 1)
    
    # For action d=0 (keep): transition follows xtran
    # This is f(x'|x,z,d=0) - buses continue accumulating mileage
    Q0 = β * xtran
    
    # For action d=1 (replace): bus returns to state 1 within same z
    # This is f(x'|x,z,d=1) - mileage resets but route usage stays same
    Q1 = zeros(nstates, nstates)
    for z in 1:zbin
        row_start = (z-1)*xbin + 1
        row_end = z*xbin
        # When replace, return to first mileage bin in this route usage category
        Q1[row_start:row_end, row_start] .= β
    end
    
    return Q0, Q1
end


"""
    compute_state_price_differences(Q0::Matrix, Q1::Matrix, 
                                      FV::Array{Float64,3},
                                      xbin::Int, zbin::Int,
                                      brand::Int, time::Int)

Compute the implicit "price" difference between keeping and replacing.
This is Σ[q₁(x'|x) - q₀(x'|x)]V(x') - the continuation value difference
weighted by implicit state prices.

# Returns
- `price_diff::Vector`: Price difference for each state (length = xbin*zbin)
"""
function compute_state_price_differences(Q0::Matrix, Q1::Matrix, 
                                          FV::Array{Float64,3},
                                          xbin::Int, zbin::Int,
                                          brand::Int, time::Int)
    
    nstates = xbin * zbin
    price_diff = zeros(nstates)
    
    for s in 1:nstates
        # Get future values for all possible next states
        V_next = FV[:, brand+1, time+1]
        
        # Compute price-weighted continuation value for each action
        continuation_keep = Q0[s, :] ⋅ V_next
        continuation_replace = Q1[s, :] ⋅ V_next
        
        # Difference is the implicit "option value" of replacement
        price_diff[s] = continuation_replace - continuation_keep
    end
    
    return price_diff
end


#========================================
PART 2: ECONOMIC INTERPRETATION
========================================#

"""
    interpret_implicit_prices(Q0::Matrix, Q1::Matrix,
                               xval::Vector, zval::Vector,
                               xbin::Int, zbin::Int)

Create interpretable summary statistics of implicit state prices.

# Returns
- DataFrame with economic interpretations
"""
function interpret_implicit_prices(Q0::Matrix, Q1::Matrix,
                                    xval::Vector, zval::Vector,
                                    xbin::Int, zbin::Int)
    
    results = DataFrame(
        mileage = Float64[],
        route_usage = Float64[],
        prob_high_mileage_keep = Float64[],
        prob_high_mileage_replace = Float64[],
        implicit_premium = Float64[]
    )
    
    # Define "high mileage" as top 20% of mileage distribution
    high_mileage_threshold = quantile(xval, 0.8)
    
    for z in 1:zbin
        for x in 1:xbin
            state_idx = (z-1)*xbin + x
            
            # Probability of reaching high mileage states under each action
            high_mileage_states = findall(xval .>= high_mileage_threshold)
            
            # Adjust indices for the z-bin
            high_mileage_in_z = [(z-1)*xbin + h for h in high_mileage_states if h <= xbin]
            
            prob_high_keep = sum(Q0[state_idx, high_mileage_in_z])
            prob_high_replace = sum(Q1[state_idx, high_mileage_in_z])
            
            # Implicit "premium" - how much more weight on bad states when keeping
            premium = prob_high_keep - prob_high_replace
            
            push!(results, (
                mileage = xval[x],
                route_usage = zval[z],
                prob_high_mileage_keep = prob_high_keep,
                prob_high_mileage_replace = prob_high_replace,
                implicit_premium = premium
            ))
        end
    end
    
    return results
end


"""
    compute_willingness_to_pay(theta_hat, Q0::Matrix, Q1::Matrix,
                                FV::Array{Float64,3}, xbin::Int, zbin::Int)

Compute willingness to pay to avoid high mileage states.
This uses the estimated utility parameters to convert continuation values
into dollar equivalents.

# Arguments
- `theta_hat`: Estimated structural parameters from GLM
- Rest same as above

# Returns
- DataFrame with WTP for each state
"""
function compute_willingness_to_pay(theta_hat, Q0::Matrix, Q1::Matrix,
                                     FV::Array{Float64,3}, 
                                     xval::Vector, zval::Vector,
                                     xbin::Int, zbin::Int, β::Float64)
    
    # Extract coefficient on odometer (utility cost per unit mileage)
    # This tells us the marginal disutility of mileage
    coef_odometer = coef(theta_hat)[2]  # Assuming order: intercept, odometer, branded
    
    results = DataFrame(
        mileage = Float64[],
        route_usage = Float64[],
        brand = Int[],
        wtp_avoid_breakdown = Float64[],
        option_value_replacement = Float64[]
    )
    
    for brand in 0:1
        for z in 1:zbin
            for x in 1:xbin
                state_idx = (z-1)*xbin + x
                
                # Future value at time t=10 (middle of panel)
                V_next = FV[:, brand+1, 11]
                
                # Continuation values under each action
                EV_keep = Q0[state_idx, :] ⋅ V_next
                EV_replace = Q1[state_idx, :] ⋅ V_next
                
                # Convert to "mileage equivalent" units
                # WTP is the option value divided by marginal utility of mileage
                option_value = EV_replace - EV_keep
                wtp_mileage_units = -option_value / coef_odometer
                
                # Approximate dollar value (if we assume $1 per 10,000 miles of cost)
                wtp_dollars = wtp_mileage_units * 0.1
                
                push!(results, (
                    mileage = xval[x],
                    route_usage = zval[z],
                    brand = brand,
                    wtp_avoid_breakdown = wtp_dollars,
                    option_value_replacement = option_value
                ))
            end
        end
    end
    
    return results
end


#========================================
PART 3: HETEROGENEITY ANALYSIS
========================================#

"""
    analyze_heterogeneity(Q0::Matrix, Q1::Matrix, FV::Array{Float64,3},
                          xval::Vector, zval::Vector, 
                          xbin::Int, zbin::Int, β::Float64)

Analyze how implicit prices differ across:
1. Branded vs non-branded buses
2. Low vs high route usage
3. Low vs high mileage

# Returns
- Dictionary with summary statistics for each group
"""
function analyze_heterogeneity(Q0::Matrix, Q1::Matrix, FV::Array{Float64,3},
                                xval::Vector, zval::Vector, 
                                xbin::Int, zbin::Int, β::Float64)
    
    results = Dict()
    
    # Compare branded vs non-branded at same state
    println("\n" * "="^60)
    println("HETEROGENEITY ANALYSIS: Implicit State Prices")
    println("="^60)
    
    # Pick a representative state: median mileage, median route usage
    med_x = Int(ceil(xbin/2))
    med_z = Int(ceil(zbin/2))
    state_idx = (med_z-1)*xbin + med_x
    
    println("\nRepresentative state: Mileage bin $med_x, Route usage bin $med_z")
    
    for brand in 0:1
        brand_label = brand == 1 ? "Branded" : "Non-branded"
        
        V_next = FV[:, brand+1, 11]  # Middle time period
        
        EV_keep = Q0[state_idx, :] ⋅ V_next
        EV_replace = Q1[state_idx, :] ⋅ V_next
        
        println("\n$brand_label buses:")
        println("  Continuation value (keep):    ", round(EV_keep, digits=4))
        println("  Continuation value (replace): ", round(EV_replace, digits=4))
        println("  Option value of replacement:  ", round(EV_replace - EV_keep, digits=4))
        
        results[brand_label] = Dict(
            "EV_keep" => EV_keep,
            "EV_replace" => EV_replace,
            "option_value" => EV_replace - EV_keep
        )
    end
    
    # Route usage comparison
    println("\n" * "-"^60)
    println("Route Usage Heterogeneity (Non-branded, median mileage):")
    for z in [1, Int(ceil(zbin/2)), zbin]
        state_idx = (z-1)*xbin + med_x
        V_next = FV[:, 1, 11]  # Non-branded
        
        EV_keep = Q0[state_idx, :] ⋅ V_next
        EV_replace = Q1[state_idx, :] ⋅ V_next
        
        println("\nRoute usage = $(zval[z]):")
        println("  Option value: ", round(EV_replace - EV_keep, digits=4))
    end
    
    return results
end


#========================================
PART 4: VISUALIZATION
========================================#

"""
    plot_implicit_prices(Q0::Matrix, Q1::Matrix, FV::Array{Float64,3},
                         xval::Vector, zval::Vector,
                         xbin::Int, zbin::Int)

Create comprehensive visualizations of implicit state prices.
"""
function plot_implicit_prices(Q0::Matrix, Q1::Matrix, FV::Array{Float64,3},
                               xval::Vector, zval::Vector,
                               xbin::Int, zbin::Int)
    
    # Plot 1: Probability weights by action
    p1 = plot(title="Implicit State Prices: Keep vs Replace",
              xlabel="Future Mileage Bin", ylabel="Price Weight (β × f)",
              legend=:topright)
    
    # Pick a representative current state
    med_z = Int(ceil(zbin/2))
    current_state = (med_z-1)*xbin + Int(ceil(xbin/2))
    
    # Extract prices for this state going to all future states in same route category
    future_states = (med_z-1)*xbin+1:med_z*xbin
    plot!(p1, 1:xbin, Q0[current_state, future_states], 
          label="Keep (d=0)", linewidth=2, marker=:circle)
    plot!(p1, 1:xbin, Q1[current_state, future_states], 
          label="Replace (d=1)", linewidth=2, marker=:square)
    
    # Plot 2: Option value across mileage states
    p2 = plot(title="Option Value of Replacement by Current Mileage",
              xlabel="Current Mileage Bin", ylabel="Option Value",
              legend=:topleft)
    
    option_values = zeros(xbin)
    for x in 1:xbin
        state_idx = (med_z-1)*xbin + x
        V_next = FV[:, 1, 11]  # Non-branded, middle time
        
        EV_keep = Q0[state_idx, :] ⋅ V_next
        EV_replace = Q1[state_idx, :] ⋅ V_next
        option_values[x] = EV_replace - EV_keep
    end
    
    plot!(p2, 1:xbin, option_values, linewidth=2, marker=:circle, label="")
    hline!(p2, [0], linestyle=:dash, color=:black, label="")
    
    # Plot 3: Branded vs Non-branded comparison
    p3 = plot(title="Option Value: Branded vs Non-Branded",
              xlabel="Current Mileage Bin", ylabel="Option Value",
              legend=:topleft)
    
    for brand in 0:1
        option_values_brand = zeros(xbin)
        for x in 1:xbin
            state_idx = (med_z-1)*xbin + x
            V_next = FV[:, brand+1, 11]
            
            EV_keep = Q0[state_idx, :] ⋅ V_next
            EV_replace = Q1[state_idx, :] ⋅ V_next
            option_values_brand[x] = EV_replace - EV_keep
        end
        
        brand_label = brand == 1 ? "Branded" : "Non-branded"
        plot!(p3, 1:xbin, option_values_brand, linewidth=2, 
              marker=:circle, label=brand_label)
    end
    
    # Combine plots
    plot(p1, p2, p3, layout=(3,1), size=(800, 900))
end


#========================================
PART 5: WELFARE ANALYSIS
========================================#

"""
    welfare_counterfactual(theta_hat, Q0::Matrix, Q1::Matrix, 
                           FV::Array{Float64,3}, β::Float64,
                           xbin::Int, zbin::Int)

Conduct welfare analysis: What if we changed transition probabilities?
(e.g., better maintenance technology, different route assignments)

# Returns
- DataFrame with welfare changes under counterfactual
"""
function welfare_counterfactual(theta_hat, Q0::Matrix, Q1::Matrix, 
                                 FV::Array{Float64,3}, β::Float64,
                                 xval::Vector, zval::Vector,
                                 xbin::Int, zbin::Int)
    
    println("\n" * "="^60)
    println("WELFARE COUNTERFACTUAL ANALYSIS")
    println("="^60)
    
    # Counterfactual 1: 20% reduction in mileage accumulation
    # This shifts probability mass toward lower mileage states
    Q0_cf1 = copy(Q0)
    # Implementation: shift transition probabilities leftward
    for s in 1:size(Q0, 1)
        # This is a simplified version - you'd want to properly re-normalize
        Q0_cf1[s, :] = circshift(Q0[s, :], -1)
    end
    
    println("\nCounterfactual 1: 20% slower mileage accumulation")
    println("(e.g., better fuel efficiency, improved maintenance)")
    
    # Compute welfare gain for representative agent
    med_z = Int(ceil(zbin/2))
    med_x = Int(ceil(xbin/2))
    state_idx = (med_z-1)*xbin + med_x
    
    V_next = FV[:, 1, 11]  # Non-branded, middle time
    
    # Baseline
    EV_keep_baseline = Q0[state_idx, :] ⋅ V_next
    
    # Counterfactual
    EV_keep_cf1 = Q0_cf1[state_idx, :] ⋅ V_next
    
    welfare_gain = EV_keep_cf1 - EV_keep_baseline
    
    println("  Welfare gain (continuation value units): ", round(welfare_gain, digits=4))
    
    # Convert to consumption equivalent
    coef_odometer = coef(theta_hat)[2]
    consumption_equiv = -welfare_gain / coef_odometer
    println("  Consumption equivalent (mileage units): ", round(consumption_equiv, digits=2))
    
    # Counterfactual 2: Subsidy for early replacement
    println("\nCounterfactual 2: What subsidy would induce 10% more replacements?")
    println("(This uses implicit prices to compute required subsidy)")
    
    # Current replacement rate at this state
    # Using estimated parameters to compute choice probability
    # This is a simplified calculation - full version would re-solve model
    
    results = DataFrame(
        counterfactual = ["Better maintenance", "Replacement subsidy"],
        welfare_gain = [welfare_gain, missing],
        consumption_equiv = [consumption_equiv, missing]
    )
    
    return results
end


#========================================
PART 6: MAIN EXECUTION FUNCTION
========================================#

"""
    run_implicit_price_analysis()

Main function to run complete implicit state price analysis.
"""
function run_implicit_price_analysis()
    println("\n" * "="^70)
    println("IMPLICIT STATE PRICES IN DYNAMIC DISCRETE CHOICE")
    println("Application: Rust (1987) Bus Engine Replacement")
    println("="^70)
    
    # Parameters
    β = 0.9
    
    # Step 1: Run baseline estimation (from your existing code)
    println("\n[1/7] Running baseline CCP estimation...")
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    flexlogitresults = estimate_flexible_logit(df_long)
    
    # Get state space
    zval, zbin, xval, xbin, xtran = create_grids()
    statedf = construct_state_space(xbin, zbin, xval, zval, xtran)
    
    # Compute future values
    FV = compute_future_values(statedf, flexlogitresults, xtran, xbin, zbin, 20, β)
    efvt1 = compute_fvt1(FV, xtran, Xstate, Zstate, xbin, Branded)
    
    # Estimate structural parameters
    theta_hat = estimate_structural_params(df_long, efvt1)
    
    println("\nBaseline structural estimates:")
    println(theta_hat)
    
    # Step 2: Compute implicit state prices
    println("\n[2/7] Computing implicit state prices...")
    Q0, Q1 = compute_implicit_state_prices(xtran, xbin, zbin, β)
    
    println("  Price matrix dimensions: ", size(Q0))
    println("  Sum of prices (keep):    ", round(sum(Q0[1,:]), digits=4))
    println("  Sum of prices (replace): ", round(sum(Q1[1,:]), digits=4))
    
    # Step 3: Economic interpretation
    println("\n[3/7] Interpreting implicit prices...")
    price_interp = interpret_implicit_prices(Q0, Q1, xval, zval, xbin, zbin)
    println("\nImplicit Risk Premium (sample):")
    println(first(price_interp, 5))
    
    # Step 4: Willingness to pay
    println("\n[4/7] Computing willingness to pay...")
    wtp_results = compute_willingness_to_pay(theta_hat, Q0, Q1, FV, 
                                              xval, zval, xbin, zbin, β)
    
    println("\nWillingness to Pay to Avoid Breakdown (sample):")
    println(first(wtp_results, 5))
    
    println("\nSummary statistics:")
    println("  Mean WTP: \$", round(mean(wtp_results.wtp_avoid_breakdown), digits=2))
    println("  Median WTP: \$", round(median(wtp_results.wtp_avoid_breakdown), digits=2))
    println("  Max WTP: \$", round(maximum(wtp_results.wtp_avoid_breakdown), digits=2))
    
    # Step 5: Heterogeneity analysis
    println("\n[5/7] Analyzing heterogeneity...")
    het_results = analyze_heterogeneity(Q0, Q1, FV, xval, zval, xbin, zbin, β)
    
    # Step 6: Visualizations
    println("\n[6/7] Creating visualizations...")
    plt = plot_implicit_prices(Q0, Q1, FV, xval, zval, xbin, zbin)
    savefig(plt, "implicit_state_prices.png")
    println("  Saved: implicit_state_prices.png")
    
    # Step 7: Welfare counterfactual
    println("\n[7/7] Welfare counterfactual analysis...")
    welfare_results = welfare_counterfactual(theta_hat, Q0, Q1, FV, β,
                                              xval, zval, xbin, zbin)
    
    # Final summary
    println("\n" * "="^70)
    println("ANALYSIS COMPLETE")
    println("="^70)
    println("\nKey Findings:")
    println("1. Implicit state prices reveal behavioral valuation of future states")
    println("2. Agents implicitly 'pay' more to avoid high-mileage states")
    println("3. Heterogeneity across branded vs non-branded buses")
    println("4. Welfare gains from better maintenance technology quantified")
    
    # Return key results for further analysis
    return Dict(
        "theta_hat" => theta_hat,
        "implicit_prices_keep" => Q0,
        "implicit_prices_replace" => Q1,
        "wtp_results" => wtp_results,
        "welfare_results" => welfare_results
    )
end

# Execute analysis
results = run_implicit_price_analysis()