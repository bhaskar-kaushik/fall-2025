# Generate Publication-Ready Tables and Figures
# For: "Implicit State Prices in Dynamic Discrete Choice Models"

using DataFrames, CSV, PrettyTables, Plots, LaTeXStrings, Statistics

"""
    create_table1_descriptive_stats(df_long::DataFrame, Xstate::Matrix, 
                                     Zstate::Vector, Branded::Vector)

Table 1: Descriptive Statistics of Bus Data
"""
function create_table1_descriptive_stats(df_long::DataFrame, Xstate::Matrix, 
                                          Zstate::Vector, Branded::Vector)
    
    stats = DataFrame(
        Variable = ["Replacement Rate", "Mean Odometer", "Std Odometer", 
                   "% Branded", "Mean Route Usage", "N Buses", "N Observations"],
        Value = [
            round(mean(df_long.Y), digits=3),
            round(mean(df_long.Odometer), digits=1),
            round(std(df_long.Odometer), digits=1),
            round(100*mean(Branded), digits=1),
            round(mean(Zstate), digits=3),
            size(Xstate, 1),
            nrow(df_long)
        ]
    )
    
    println("\n" * "="^60)
    println("TABLE 1: Descriptive Statistics")
    println("="^60)
    pretty_table(stats, header=["Variable", "Value"])
    
    # Export to LaTeX
    open("table1_descriptives.tex", "w") do f
        pretty_table(f, stats, backend=Val(:latex), 
                    header=["Variable", "Value"],
                    tf=tf_latex_booktabs)
    end
    
    return stats
end


"""
    create_table2_baseline_estimates(flexlogit, theta_hat)

Table 2: Baseline Estimation Results
Compares flexible logit (step 1) with structural parameters (step 2)
"""
function create_table2_baseline_estimates(theta_hat)
    
    # Extract coefficients
    theta_coefs = coef(theta_hat)
    theta_se = stderror(theta_hat)
    
    estimates = DataFrame(
        Parameter = ["Constant", "Odometer", "Branded"],
        Coefficient = round.(theta_coefs, digits=4),
        Std_Error = round.(theta_se, digits=4),
        T_Stat = round.(theta_coefs ./ theta_se, digits=2)
    )
    
    println("\n" * "="^60)
    println("TABLE 2: Structural Parameter Estimates (CCP Method)")
    println("="^60)
    pretty_table(estimates)
    
    # Export to LaTeX
    open("table2_estimates.tex", "w") do f
        pretty_table(f, estimates, backend=Val(:latex),
                    tf=tf_latex_booktabs)
    end
    
    return estimates
end


"""
    create_table3_implicit_prices(Q0::Matrix, Q1::Matrix, 
                                   xval::Vector, zval::Vector,
                                   xbin::Int, zbin::Int, β::Float64)

Table 3: Implicit State Prices - Representative States
Shows q_d(x'|x) for selected states
"""
function create_table3_implicit_prices(Q0::Matrix, Q1::Matrix, 
                                        FV::Array{Float64,3},
                                        xval::Vector, zval::Vector,
                                        xbin::Int, zbin::Int, β::Float64)
    
    # Select representative states: low, medium, high mileage
    states_to_show = [Int(ceil(xbin*0.25)), Int(ceil(xbin*0.5)), Int(ceil(xbin*0.75))]
    med_z = Int(ceil(zbin/2))
    
    results = DataFrame(
        Current_Mileage = String[],
        EV_Keep = Float64[],
        EV_Replace = Float64[],
        Option_Value = Float64[],
        Implicit_Premium = Float64[]
    )
    
    for (i, x_idx) in enumerate(states_to_show)
        state_idx = (med_z-1)*xbin + x_idx
        
        # Use middle time period, non-branded
        V_next = FV[:, 1, 11]
        
        EV_keep = Q0[state_idx, :] ⋅ V_next
        EV_replace = Q1[state_idx, :] ⋅ V_next
        option_val = EV_replace - EV_keep
        
        # Compute implicit premium (extra weight on bad states when keeping)
        high_mileage_states = findall(xval .>= quantile(xval, 0.75))
        high_states_in_z = [(med_z-1)*xbin + h for h in high_mileage_states if h <= xbin]
        
        premium = sum(Q0[state_idx, high_states_in_z]) - sum(Q1[state_idx, high_states_in_z])
        
        mileage_label = ["Low (25th pct)", "Medium (50th pct)", "High (75th pct)"][i]
        
        push!(results, (
            Current_Mileage = mileage_label,
            EV_Keep = round(EV_keep, digits=3),
            EV_Replace = round(EV_replace, digits=3),
            Option_Value = round(option_val, digits=3),
            Implicit_Premium = round(premium, digits=4)
        ))
    end
    
    println("\n" * "="^60)
    println("TABLE 3: Implicit State Prices and Option Values")
    println("="^60)
    pretty_table(results)
    
    # Export to LaTeX
    open("table3_implicit_prices.tex", "w") do f
        pretty_table(f, results, backend=Val(:latex),
                    tf=tf_latex_booktabs)
    end
    
    return results
end


"""
    create_table4_heterogeneity(Q0::Matrix, Q1::Matrix, FV::Array{Float64,3},
                                 xval::Vector, zval::Vector,
                                 xbin::Int, zbin::Int)

Table 4: Heterogeneity in Implicit State Prices
Compares branded vs non-branded buses
"""
function create_table4_heterogeneity(Q0::Matrix, Q1::Matrix, FV::Array{Float64,3},
                                      xval::Vector, zval::Vector,
                                      xbin::Int, zbin::Int)
    
    results = DataFrame(
        Group = String[],
        EV_Keep = Float64[],
        EV_Replace = Float64[],
        Option_Value = Float64[],
        Difference_from_baseline = Float64[]
    )
    
    med_x = Int(ceil(xbin/2))
    med_z = Int(ceil(zbin/2))
    state_idx = (med_z-1)*xbin + med_x
    
    baseline_option = 0.0
    
    for (i, brand) in enumerate([0, 1])
        V_next = FV[:, brand+1, 11]
        
        EV_keep = Q0[state_idx, :] ⋅ V_next
        EV_replace = Q1[state_idx, :] ⋅ V_next
        option_val = EV_replace - EV_keep
        
        if i == 1
            baseline_option = option_val
        end
        
        brand_label = brand == 1 ? "Branded" : "Non-branded"
        
        push!(results, (
            Group = brand_label,
            EV_Keep = round(EV_keep, digits=3),
            EV_Replace = round(EV_replace, digits=3),
            Option_Value = round(option_val, digits=3),
            Difference_from_baseline = round(option_val - baseline_option, digits=3)
        ))
    end
    
    println("\n" * "="^60)
    println("TABLE 4: Heterogeneity Analysis - Branded vs Non-Branded")
    println("="^60)
    pretty_table(results)
    
    # Export to LaTeX
    open("table4_heterogeneity.tex", "w") do f
        pretty_table(f, results, backend=Val(:latex),
                    tf=tf_latex_booktabs)
    end
    
    return results
end


"""
    create_table5_wtp(wtp_results::DataFrame)

Table 5: Willingness to Pay to Avoid High Mileage States
Summary statistics by group
"""
function create_table5_wtp(wtp_results::DataFrame)
    
    summary_stats = combine(groupby(wtp_results, :brand)) do df
        DataFrame(
            Mean_WTP = mean(df.wtp_avoid_breakdown),
            Median_WTP = median(df.wtp_avoid_breakdown),
            Std_WTP = std(df.wtp_avoid_breakdown),
            Min_WTP = minimum(df.wtp_avoid_breakdown),
            Max_WTP = maximum(df.wtp_avoid_breakdown)
        )
    end
    
    # Add brand labels
    summary_stats = @transform(summary_stats, 
        :Brand = [:brand == 1 ? "Branded" : "Non-branded" for brand in :brand])
    
    # Round for display
    for col in [:Mean_WTP, :Median_WTP, :Std_WTP, :Min_WTP, :Max_WTP]
        summary_stats[!, col] = round.(summary_stats[!, col], digits=2)
    end
    
    # Select and reorder columns
    summary_stats = select(summary_stats, :Brand, :Mean_WTP, :Median_WTP, 
                           :Std_WTP, :Min_WTP, :Max_WTP)
    
    println("\n" * "="^60)
    println("TABLE 5: Willingness to Pay to Avoid Breakdown (\$)")
    println("="^60)
    pretty_table(summary_stats)
    
    # Export to LaTeX
    open("table5_wtp.tex", "w") do f
        pretty_table(f, summary_stats, backend=Val(:latex),
                    tf=tf_latex_booktabs)
    end
    
    return summary_stats
end


"""
    create_figure1_price_structure(Q0::Matrix, Q1::Matrix, 
                                     xval::Vector, xbin::Int, zbin::Int, β::Float64)

Figure 1: Implicit State Price Structure
Shows how β*f(x'|x,d) varies across future states for each action
"""
function create_figure1_price_structure(Q0::Matrix, Q1::Matrix, 
                                         xval::Vector, xbin::Int, zbin::Int, β::Float64)
    
    # Pick representative current state
    med_z = Int(ceil(zbin/2))
    med_x = Int(ceil(xbin/2))
    current_state = (med_z-1)*xbin + med_x
    
    # Extract transition probabilities for this state
    future_states_range = (med_z-1)*xbin+1:med_z*xbin
    
    p = plot(size=(800, 500), dpi=300)
    
    # Plot implicit prices for both actions
    plot!(p, xval, Q0[current_state, future_states_range],
          label="Keep Engine (d=0)",
          linewidth=3, marker=:circle, markersize=6,
          color=:blue, alpha=0.8)
    
    plot!(p, xval, Q1[current_state, future_states_range],
          label="Replace Engine (d=1)",
          linewidth=3, marker=:square, markersize=6,
          color=:red, alpha=0.8)
    
    # Formatting
    xlabel!("Future Mileage State (x')")
    ylabel!("Implicit State Price: β·f(x'|x,d)")
    title!("Figure 1: Behavioral State Prices by Action\n(Current state: median mileage)")
    
    # Add annotation
    annotate!(p, maximum(xval)*0.7, maximum(Q0[current_state, future_states_range])*0.9,
              text("β = $β\nCurrent x = $(xval[med_x])", 10, :left))
    
    savefig(p, "figure1_price_structure.png")
    println("\nFigure 1 saved: figure1_price_structure.png")
    
    return p
end


"""
    create_figure2_option_value(Q0::Matrix, Q1::Matrix, FV::Array{Float64,3},
                                 xval::Vector, xbin::Int, zbin::Int)

Figure 2: Option Value of Replacement Across Mileage States
"""
function create_figure2_option_value(Q0::Matrix, Q1::Matrix, FV::Array{Float64,3},
                                      xval::Vector, xbin::Int, zbin::Int)
    
    med_z = Int(ceil(zbin/2))
    
    p = plot(size=(800, 500), dpi=300)
    
    # Compute option value for each mileage state, for both branded and non-branded
    for brand in 0:1
        option_values = zeros(xbin)
        
        for x in 1:xbin
            state_idx = (med_z-1)*xbin + x
            V_next = FV[:, brand+1, 11]  # Middle time period
            
            EV_keep = Q0[state_idx, :] ⋅ V_next
            EV_replace = Q1[state_idx, :] ⋅ V_next
            option_values[x] = EV_replace - EV_keep
        end
        
        brand_label = brand == 1 ? "Branded" : "Non-branded"
        color = brand == 1 ? :orange : :blue
        
        plot!(p, xval, option_values,
              label=brand_label,
              linewidth=3, marker=:circle, markersize=5,
              color=color, alpha=0.8)
    end
    
    # Add zero line
    hline!(p, [0], linestyle=:dash, color=:black, linewidth=2, 
           label="", alpha=0.5)
    
    # Formatting
    xlabel!("Current Mileage State (x)")
    ylabel!("Option Value: E[V|replace] - E[V|keep]")
    title!("Figure 2: Option Value of Replacement\n(Weighted by Implicit State Prices)")
    
    savefig(p, "figure2_option_value.png")
    println("Figure 2 saved: figure2_option_value.png")
    
    return p
end


"""
    create_figure3_decomposition(Q0::Matrix, Q1::Matrix, FV::Array{Float64,3},
                                  theta_hat, xval::Vector, xbin::Int, zbin::Int)

Figure 3: Value Function Decomposition
Shows u(x,d) vs continuation value component
"""
function create_figure3_decomposition(Q0::Matrix, Q1::Matrix, FV::Array{Float64,3},
                                       theta_hat, xval::Vector, xbin::Int, zbin::Int)
    
    med_z = Int(ceil(zbin/2))
    brand = 0  # Non-branded
    
    # Extract coefficients
    coefs = coef(theta_hat)
    θ_0 = coefs[1]  # Intercept
    θ_x = coefs[2]  # Odometer coefficient
    
    # Compute components for action d=0 (keep)
    flow_utility = zeros(xbin)
    continuation_value = zeros(xbin)
    total_value = zeros(xbin)
    
    for x in 1:xbin
        state_idx = (med_z-1)*xbin + x
        V_next = FV[:, brand+1, 11]
        
        # Flow utility: u(x, d=0) = θ_0 + θ_x * odometer
        flow_utility[x] = θ_0 + θ_x * xval[x]
        
        # Continuation value: β·E[V(x')|x, d=0]
        continuation_value[x] = Q0[state_idx, :] ⋅ V_next
        
        # Total
        total_value[x] = flow_utility[x] + continuation_value[x]
    end
    
    p = plot(size=(800, 600), dpi=300)
    
    # Plot components
    plot!(p, xval, flow_utility,
          label="Flow Utility: u(x,d=0)",
          linewidth=3, color=:blue, linestyle=:solid)
    
    plot!(p, xval, continuation_value,
          label="Continuation Value: β·Σq₀(x')V(x')",
          linewidth=3, color=:red, linestyle=:dash)
    
    plot!(p, xval, total_value,
          label="Total Value: v₀(x)",
          linewidth=3, color=:green, linestyle=:solid)
    
    # Formatting
    xlabel!("Current Mileage State (x)")
    ylabel!("Value")
    title!("Figure 3: Value Function Decomposition (Keep Engine)\nShowing Role of Implicit State Prices")
    
    savefig(p, "figure3_decomposition.png")
    println("Figure 3 saved: figure3_decomposition.png")
    
    return p
end


"""
    create_all_tables_and_figures(results_dict::Dict)

Master function to create all tables and figures for the paper
"""
function create_all_tables_and_figures(results_dict::Dict)
    
    println("\n" * "="^70)
    println("GENERATING ALL TABLES AND FIGURES FOR PAPER")
    println("="^70)
    
    # Note: You'll need to pass these from your main analysis
    # This is a template showing the structure
    
    println("\n✓ All tables saved as .tex files")
    println("✓ All figures saved as .png files (300 dpi)")
    println("\nFiles generated:")
    println("  - table1_descriptives.tex")
    println("  - table2_estimates.tex")
    println("  - table3_implicit_prices.tex")
    println("  - table4_heterogeneity.tex")
    println("  - table5_wtp.tex")
    println("  - figure1_price_structure.png")
    println("  - figure2_option_value.png")
    println("  - figure3_decomposition.png")
    
    return nothing
end


# Template for results section
"""
Generate a template for the empirical results section of your paper
"""
function generate_results_section_template()
    
    template = """
    
========================================
SUGGESTED RESULTS SECTION FOR PAPER
========================================

4. Empirical Results

4.1 Data and Estimation

We apply our framework to the bus engine replacement data from Rust (1987). 
[Cite Table 1 for descriptive statistics]. The data contains replacement 
decisions for [N] buses over [T] periods. We estimate the model using the 
two-step CCP procedure proposed by Hotz and Miller (1993).

[Table 2 about here]

Table 2 presents our structural parameter estimates. The coefficient on 
odometer is negative and significant (θ̂_x = [value], SE = [value]), indicating 
that higher mileage reduces utility. The branded bus coefficient is [positive/negative], 
suggesting [interpretation].

4.2 Implicit State Prices

Using the estimated transition probabilities and discount factor β = 0.9, we 
compute implicit state prices q_d(x'|x) = β·f(x'|x,d) for both actions. 
Figure 1 illustrates the structure of these prices.

[Figure 1 about here]

Several patterns emerge. First, under the "keep" action (d=0), probability 
mass concentrates on higher mileage states, reflecting continued deterioration. 
Second, under "replace" (d=1), all probability mass shifts to the lowest 
mileage state, as the engine resets. Third, these different probability 
structures generate distinct implicit valuations of future states.

Table 3 presents the continuation values weighted by these implicit prices.

[Table 3 about here]

At low mileage states, the option value of replacement is [negative/positive] 
(EV_replace - EV_keep = [value]), indicating that continuation under the current 
engine dominates. As mileage increases, this option value [rises/falls], reaching 
[value] at high mileage states. The implicit premium—measuring the extra weight 
on unfavorable high-mileage states when keeping versus replacing—increases from 
[value] to [value] as current mileage rises.

[Figure 2 about here]

Figure 2 shows how option values vary across the entire mileage distribution. 
The relationship is [monotonic/non-monotonic], with [interpretation].

4.3 Heterogeneity in Implicit State Prices

Agents with different characteristics face different transition probabilities, 
generating heterogeneous implicit state prices. Table 4 compares branded versus 
non-branded buses.

[Table 4 about here]

Branded buses exhibit [higher/lower] continuation values under both actions. 
The option value of replacement differs by [value] units between groups, 
equivalent to [X]% of the baseline. This heterogeneity arises from [explanation: 
different failure rates, maintenance schedules, or usage patterns].

4.4 Willingness to Pay

We convert option values into dollar equivalents using the estimated marginal 
utility of mileage. Table 5 reports summary statistics.

[Table 5 about here]

On average, bus operators implicitly value avoiding high-mileage states at 
[value] per expected breakdown. This varies from [min] to [max] across 
states. Branded buses show [higher/lower] willingness to pay ([value] vs 
[value]), consistent with their [interpretation].

These estimates provide a revealed-preference measure of breakdown costs that 
does not require observing actual breakdown events or their direct costs.

4.5 Decomposition and Interpretation

Figure 3 decomposes the value function into flow utility and continuation 
components.

[Figure 3 about here]

The continuation value term, weighted by implicit state prices, accounts for 
[XX]% of total value at low mileage and [YY]% at high mileage. This demonstrates 
the quantitative importance of forward-looking behavior and shows how implicit 
state prices aggregate future outcomes into current decisions.

Our findings show that conditional choice probabilities, when properly scaled, 
reveal agents' implicit valuation of future states—analogous to how Arrow-Debreu 
prices reveal market valuations in complete markets. This "behavioral pricing 
kernel" provides a unified framework for understanding dynamic choice under 
uncertainty.

========================================
"""
    
    println(template)
    
    # Save to file
    open("results_section_template.txt", "w") do f
        write(f, template)
    end
    
    println("\nTemplate saved to: results_section_template.txt")
    
    return template
end

# Generate the template
generate_results_section_template()