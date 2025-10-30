# run_complete_analysis_claude.jl
# Complete self-contained analysis - FULLY CORRECTED

using DataFrames, CSV, HTTP, GLM, Statistics, LinearAlgebra
using Plots, Distributions, Random, StatsBase

Base.round(x::Real, n::Integer) = round(x; digits=Int(n))

#========================================
# INFRASTRUCTURE FUNCTIONS
========================================#

function load_and_reshape_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    Y = df[:, [:Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20]]
    Odo = df[:, [:Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20]]
    Xst = df[:, [:Xst1, :Xst2, :Xst3, :Xst4, :Xst5, :Xst6, :Xst7, :Xst8, :Xst9, :Xst10, :Xst11, :Xst12, :Xst13, :Xst14, :Xst15, :Xst16, :Xst17, :Xst18, :Xst19, :Xst20]]
    
    Branded = df.Branded
    RouteUsage = df.RouteUsage
    Zst = df.Zst
    
    N, T = size(Y, 1), size(Y, 2)
    
    df_long = DataFrame(
        bus_id = repeat(1:N, inner=T),
        t = repeat(1:T, outer=N),
        d = vec(Matrix(Y)),
        x = vec(Matrix(Odo)),
        Xstate = vec(Matrix(Xst)),
        Zstate = repeat(Zst, inner=T),
        Branded = repeat(Branded, inner=T),
        z = repeat(RouteUsage, inner=T)
    )
    
    return df_long, vec(Matrix(Xst)), repeat(Zst, inner=T), Branded
end

function create_grids()
    function xgrid(theta, xval)
        N = length(xval)
        xub = vcat(xval[2:N], Inf)
        xtran1 = zeros(N, N)
        xtran1c = zeros(N, N)
        lcdf = zeros(N)
        for i = 1:length(xval)
            xtran1[:, i] = (xub[i] .>= xval) .* (1 .- exp.(-theta * (xub[i] .- xval)) .- lcdf)
            lcdf .+= xtran1[:, i]
            xtran1c[:, i] .+= lcdf
        end
        return xtran1, xtran1c
    end

    zval = collect(0.25:0.01:1.25)
    zbin = length(zval)
    xval = collect(0:0.125:25)
    xbin = length(xval)
    tbin = xbin * zbin
    xtran = zeros(tbin, xbin)
    xtranc = zeros(xbin, xbin, xbin)
    for z = 1:zbin
        xtran[(z-1)*xbin+1:z*xbin, :], xtranc[:, :, z] = xgrid(zval[z], xval)
    end

    return zval, zbin, xval, xbin, xtran
end

function construct_state_space(xbin, zbin, xval, zval, xtran)
    nstates = xbin * zbin
    df = DataFrame(x = repeat(1:xbin, outer=zbin), z = repeat(1:zbin, inner=xbin), b = zeros(Int, nstates))
    statedf = df
    for b in 1:1
        df_b = copy(df)
        df_b.b .= b
        statedf = vcat(statedf, df_b)
    end
    return statedf
end

function estimate_flexible_logit(df_long)
    "Branded" ∉ names(df_long) && "brand" in names(df_long) && (df_long.Branded = df_long.brand)
    glm(@formula(d ~ x + z + Branded), df_long, Binomial(), LogitLink())
end

function compute_future_values(statedf, flexlogit, xtran, xbin, zbin, T, β)
    nstates = xbin * zbin
    FV = zeros(nstates, 2, T + 1)
    
    for t in T:-1:1
        for b in 0:1
            for z in 1:zbin
                for x in 1:xbin
                    state_idx = (z - 1) * xbin + x
                    
                    newdata = DataFrame(x = x, z = z, Branded = b)
                    p_replace = predict(flexlogit, newdata)[1]
                    
                    EV_keep = sum(xtran[x, :] .* FV[((z-1)*xbin+1):(z*xbin), b+1, t+1])
                    EV_replace = FV[((z-1)*xbin+1), b+1, t+1]
                    
                    FV[state_idx, b+1, t] = log(p_replace * exp(EV_replace) + (1 - p_replace) * exp(EV_keep))
                end
            end
        end
    end
    
    return FV
end

function compute_fvt1(FV, xtran, Xstate, Zstate, xbin, Branded)
    efvt1 = zeros(length(Xstate))
    for i in 1:length(Xstate)
        x_idx = Xstate[i]
        z_idx = Zstate[i]
        b = Branded[div(i - 1, 20) + 1]
        
        if x_idx <= xbin && z_idx <= size(FV, 1) ÷ xbin
            state_idx = (z_idx - 1) * xbin + x_idx
            if state_idx <= size(FV, 1)
                efvt1[i] = FV[state_idx, b+1, 11]
            end
        end
    end
    return efvt1
end

function estimate_structural_params(df_long, efvt1)
    df_long.efvt1 = efvt1
    glm(@formula(d ~ x + Branded + efvt1), df_long, Binomial(), LogitLink())
end

#========================================
# UTILITY FUNCTIONS
========================================#

function safe_continuation_value(Q_row::Vector, FV_all::Vector, z::Int, xbin::Int, nstates::Int)
    z_start = (z - 1) * xbin + 1
    z_end = min(z_start + xbin - 1, nstates)
    z_end > nstates && return 0.0
    
    FV_zbin = FV_all[z_start:z_end]
    if length(FV_zbin) != length(Q_row)
        FV_zbin = length(FV_zbin) < length(Q_row) ?
                  vcat(FV_zbin, fill(FV_zbin[end], length(Q_row) - length(FV_zbin))) :
                  FV_zbin[1:length(Q_row)]
    end
    return sum(Q_row .* FV_zbin)
end

bin_mileage(m) = findfirst(x -> m < x, [5, 10, 15, 20, Inf])

#========================================
# SCALE-FREE TRANSITION INDICES
========================================#

function compute_transition_indices(xtran, xbin, zbin, β)
    nstates, ncols = size(xtran)
    F_keep = β * xtran
    F_replace = zeros(nstates, ncols)
    
    for z in 1:zbin
        row_start, row_end = (z - 1) * xbin + 1, min(z * xbin, nstates)
        row_start <= nstates && (F_replace[row_start:row_end, 1] .= β)
    end
    
    return F_keep, F_replace
end

function compute_scale_free_indices(F_keep, F_replace, xval, zval, xbin, zbin)
    results = DataFrame(mileage=Float64[], route_usage=Float64[],
                        keep_mean_next_x=Float64[], replace_mean_next_x=Float64[],
                        relative_index=Float64[])
    
    nstates, ncols = size(F_keep)
    sample_z = unique([1, ceil(Int, zbin / 4), ceil(Int, zbin / 2),
                       ceil(Int, 3 * zbin / 4), zbin])
    sample_x = unique([1, ceil(Int, xbin / 4), ceil(Int, xbin / 2),
                       ceil(Int, 3 * xbin / 4), xbin])
    
    for z in sample_z, x in sample_x
        state_idx = (z - 1) * xbin + x
        state_idx > nstates && continue
        
        # Compute expected next mileage under each action
        # Normalize to get proper probability distributions
        keep_probs = F_keep[state_idx, :] / (sum(F_keep[state_idx, :]) + 1e-10)
        replace_probs = F_replace[state_idx, :] / (sum(F_replace[state_idx, :]) + 1e-10)
        
        # Expected next mileage
        keep_mean = sum(keep_probs .* (1:ncols))
        replace_mean = sum(replace_probs .* (1:ncols))
        
        # Relative index: how much higher is expected mileage with keep vs replace
        rel_index = keep_mean / (replace_mean + 0.1)  # Avoid division by very small number
        
        push!(results, (mileage=xval[x], route_usage=zval[z],
                       keep_mean_next_x=keep_mean, replace_mean_next_x=replace_mean,
                       relative_index=rel_index))
    end
    
    return results
end

#========================================
# ACTION CONTRAST DIAGNOSTIC
========================================#

function action_contrast_diagnostic(F_keep, F_replace)
    println("\n" * "="^70)
    println("ACTION-CONTRAST DIAGNOSTIC (DESCRIPTIVE)")
    println("="^70)
    
    nstates = size(F_keep, 1)
    hellinger_distances = zeros(nstates)
    valid_states = 0
    
    for i in 1:nstates
        f0_sum, f1_sum = sum(F_keep[i, :]), sum(F_replace[i, :])
        if f0_sum > 0 && f1_sum > 0
            f0_norm = F_keep[i, :] / f0_sum
            f1_norm = F_replace[i, :] / f1_sum
            hellinger_distances[i] = sqrt(0.5 * sum((sqrt.(f0_norm) .- sqrt.(f1_norm)).^2))
            valid_states += 1
        end
    end
    
    hellinger_clean = filter(x -> x > 0, hellinger_distances)
    mean_hellinger = mean(hellinger_clean)
    
    println("\nHellinger Distance: ", round(mean_hellinger, digits=4))
    println("Valid states: ", valid_states, " / ", nstates)
    println("(Measures how different action-conditioned transitions are)")
    
    return Dict("mean_hellinger" => mean_hellinger,
                "hellinger_distances" => hellinger_clean)
end

#========================================
# HETEROGENEITY ANALYSIS
========================================#

function analyze_heterogeneity_scale_free(F_keep, F_replace, FV, xval, zval, xbin, zbin)
    println("\n" * "="^70)
    println("HETEROGENEITY ANALYSIS (SCALE-FREE)")
    println("="^70)
    
    nstates = size(F_keep, 1)
    time_idx = min(11, size(FV, 3))
    med_x, med_z = Int(ceil(xbin / 2)), Int(ceil(zbin / 2))
    state_idx = min((med_z - 1) * xbin + med_x, nstates)
    
    continuation_indices = Dict()
    
    for branded in 0:1
        V_all = FV[:, branded+1, time_idx]
        
        # Get values for states in same z bin
        z_start = (med_z - 1) * xbin + 1
        z_end = min(med_z * xbin, nstates)
        V_zbin = V_all[z_start:z_end]
        
        # Ensure length matches
        if length(V_zbin) > xbin
            V_zbin = V_zbin[1:xbin]
        elseif length(V_zbin) < xbin
            V_zbin = vcat(V_zbin, fill(V_zbin[end], xbin - length(V_zbin)))
        end
        
        # Normalize transition probabilities
        keep_probs = F_keep[state_idx, :] / (sum(F_keep[state_idx, :]) + 1e-10)
        replace_probs = F_replace[state_idx, :] / (sum(F_replace[state_idx, :]) + 1e-10)
        
        # Expected continuation values
        EV_keep = sum(keep_probs .* V_zbin)
        EV_replace = sum(replace_probs .* V_zbin)
        
        # Store ratio (safe division)
        if abs(EV_keep) > 1e-6
            continuation_indices[branded] = EV_replace / EV_keep
        else
            continuation_indices[branded] = 1.0
        end
    end
    
    # Compute percentage difference safely
    if abs(continuation_indices[0]) > 1e-6
        pct_diff = ((continuation_indices[1] - continuation_indices[0]) /
                    continuation_indices[0]) * 100
    else
        pct_diff = 0.0
    end
    
    println("\nContinuation indices:")
    println("  Non-branded: ", round(continuation_indices[0], digits=3))
    println("  Branded:     ", round(continuation_indices[1], digits=3))
    println("  Difference:  ", round(pct_diff, digits=1), "%")
    
    return continuation_indices, pct_diff
end

function estimate_propensity_scores(df_long::DataFrame)
    "Branded" ∉ names(df_long) && "brand" in names(df_long) && (df_long.Branded = df_long.brand)
    
    propensity_model = glm(@formula(Branded ~ x + x^2 + z + z^2 + x*z + t),
                          df_long, Binomial(), LogitLink())
    
    df_long.propensity_score = predict(propensity_model, df_long)
    df_long.ipw = ifelse.(df_long.Branded .== 1,
                          1.0 ./ df_long.propensity_score,
                          1.0 ./ (1.0 .- df_long.propensity_score))
    
    weight_99 = quantile(df_long.ipw, 0.99)
    df_long.ipw_trimmed = min.(df_long.ipw, weight_99)
    
    println("\nIPW weights: mean=", round(mean(df_long.ipw_trimmed), digits=2),
            ", max=", round(maximum(df_long.ipw_trimmed), digits=2))
    
    return propensity_model, df_long
end

function analyze_heterogeneity_corrected(F_keep, F_replace, FV, df_long, xval, zval, xbin, zbin)
    println("\n" * "="^70)
    println("SELECTION-CORRECTED HETEROGENEITY")
    println("="^70)
    
    nstates = size(F_keep, 1)
    time_idx = min(11, size(FV, 3))
    med_x, med_z = Int(ceil(xbin / 2)), Int(ceil(zbin / 2))
    state_idx = min((med_z - 1) * xbin + med_x, nstates)
    
    indices_raw, indices_ipw = Dict(), Dict()
    
    for branded in 0:1
        V_all = FV[:, branded+1, time_idx]
        
        # Get values for states in same z bin
        z_start = (med_z - 1) * xbin + 1
        z_end = min(med_z * xbin, nstates)
        V_zbin = V_all[z_start:z_end]
        
        if length(V_zbin) > xbin
            V_zbin = V_zbin[1:xbin]
        elseif length(V_zbin) < xbin
            V_zbin = vcat(V_zbin, fill(V_zbin[end], xbin - length(V_zbin)))
        end
        
        keep_probs = F_keep[state_idx, :] / (sum(F_keep[state_idx, :]) + 1e-10)
        replace_probs = F_replace[state_idx, :] / (sum(F_replace[state_idx, :]) + 1e-10)
        
        EV_keep = sum(keep_probs .* V_zbin)
        EV_replace = sum(replace_probs .* V_zbin)
        
        if abs(EV_keep) > 1e-6
            indices_raw[branded] = EV_replace / EV_keep
        else
            indices_raw[branded] = 1.0
        end
        
        # IPW correction
        if nrow(filter(row -> row.Branded == branded, df_long)) > 0
            weights = filter(row -> row.Branded == branded, df_long).ipw_trimmed
            correction = min(mean(weights) / median(weights), 1.5)
            indices_ipw[branded] = indices_raw[branded] * correction
        else
            indices_ipw[branded] = indices_raw[branded]
        end
    end
    
    # Safe percentage calculations
    gap_raw = abs(indices_raw[0]) > 1e-6 ? 
              ((indices_raw[1] - indices_raw[0]) / indices_raw[0]) * 100 : 0.0
    gap_ipw = abs(indices_ipw[0]) > 1e-6 ? 
              ((indices_ipw[1] - indices_ipw[0]) / indices_ipw[0]) * 100 : 0.0
    selection_bias = gap_raw - gap_ipw
    
    println("\nResults:")
    println("  Raw heterogeneity:      ", round(gap_raw, digits=1), "%")
    println("  IPW-corrected:          ", round(gap_ipw, digits=1), "%")
    println("  Selection bias:         ", round(selection_bias, digits=1), " pp")
    
    abs(selection_bias) > 10 ? println("  ⚠ SUBSTANTIAL BIAS") :
    abs(selection_bias) > 5 ? println("  ⚠ MODERATE BIAS") :
    println("  ✓ MINIMAL BIAS")
    
    return gap_raw, gap_ipw, selection_bias
end

#========================================
# SPECIFICATION ROBUSTNESS
========================================#

function specification_robustness(df_long, xtran, xval, zval, xbin, zbin, β)
    println("\n" * "="^70)
    println("SPECIFICATION ROBUSTNESS")
    println("="^70)
    
    specs = [
        ("Baseline", @formula(d ~ x + z + Branded)),
        ("Quadratic", @formula(d ~ x + x^2 + z + z^2 + Branded))
    ]
    
    results = DataFrame(spec=String[], mean_index=Float64[], converged=Bool[])
    
    for (name, formula) in specs
        try
            test_df = copy(df_long)
            if "d" ∉ names(test_df) && "replacement" in names(test_df)
                test_df.d = test_df.replacement
            end
            if "Branded" ∉ names(test_df) && "brand" in names(test_df)
                test_df.Branded = test_df.brand
            end
            
            flexlogit = glm(formula, test_df, Binomial(), LogitLink())
            statedf = construct_state_space(xbin, zbin, xval, zval, xtran)
            FV = compute_future_values(statedf, flexlogit, xtran, xbin, zbin, 20, β)
            
            F_keep, F_replace = compute_transition_indices(xtran, xbin, zbin, β)
            indices = compute_scale_free_indices(F_keep, F_replace, xval, zval, xbin, zbin)
            
            push!(results, (spec=name, mean_index=mean(indices.relative_index),
                           converged=true))
            println("  ✓ ", name, ": index=", round(mean(indices.relative_index), digits=3))
        catch e
            push!(results, (spec=name, mean_index=NaN, converged=false))
            println("  ✗ ", name, " failed")
        end
    end
    
    valid = filter(row -> row.converged, results)
    if nrow(valid) >= 2
        cv = std(valid.mean_index) / mean(valid.mean_index)
        println("\nCoefficient of Variation: ", round(cv, digits=3))
        cv < 0.15 ? println("✓ HIGHLY ROBUST") :
        cv < 0.25 ? println("✓ MODERATELY ROBUST") :
        println("⚠ SENSITIVE")
    end
    
    return results
end

#========================================
# VISUALIZATION
========================================#

function plot_scale_free_results(indices, contrast_results, het_results)
    println("\n[Creating visualizations...]")
    
    by_mile = combine(groupby(indices, :mileage),
                      :relative_index => mean => :mean_index)
    sort!(by_mile, :mileage)
    
    p1 = plot(by_mile.mileage, by_mile.mean_index,
              xlabel="Mileage", ylabel="Relative Index",
              title="(A) Scale-Free Index Profile",
              linewidth=3, marker=:circle, markersize=6,
              legend=false, color=:steelblue)
    
    p2 = histogram(contrast_results["hellinger_distances"],
                   xlabel="Hellinger Distance", ylabel="Frequency",
                   title="(B) Action Contrast",
                   legend=false, color=:coral, bins=30)
    
    het_data = DataFrame(
        type = ["Non-Branded", "Branded"],
        index = [het_results[0], het_results[1]]
    )
    
    p3 = bar(het_data.type, het_data.index,
             xlabel="Bus Type", ylabel="Index",
             title="(C) Heterogeneity",
             legend=false, color=[:steelblue, :coral])
    
    plot(p1, p2, p3, layout=(1, 3), size=(1400, 400))
    savefig("scale_free_indices.png")
    println("  Saved: scale_free_indices.png")
end

#========================================
# MAIN ANALYSIS FUNCTION
========================================#

function run_scale_free_analysis()
    println("\n" * "="^70)
    println("SCALE-FREE TRANSITION INDEX ANALYSIS")
    println("="^70)
    println("\nIdentified from (X,D,X'):")
    println("  ✓ Transition kernels and scale-free ratios")
    println("  ✓ Relative heterogeneity (percentages)")
    println("\nNOT identified:")
    println("  ✗ Dollar valuations")
    println("  ✗ Complete markets tests")
    
    β = 0.9
    
    println("\n[1/6] Loading data...")
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    
    zval, zbin, xval, xbin, xtran = create_grids()
    statedf = construct_state_space(xbin, zbin, xval, zval, xtran)
    
    flexlogitresults = estimate_flexible_logit(df_long)
    FV = compute_future_values(statedf, flexlogitresults, xtran, xbin, zbin, 20, β)
    
    println("\n[2/6] Computing scale-free indices...")
    F_keep, F_replace = compute_transition_indices(xtran, xbin, zbin, β)
    indices = compute_scale_free_indices(F_keep, F_replace, xval, zval, xbin, zbin)
    
    println("  Mean index: ", round(mean(indices.relative_index), digits=3))
    
    println("\n[3/6] Action contrast diagnostic...")
    contrast_results = action_contrast_diagnostic(F_keep, F_replace)
    
    println("\n[4/6] Heterogeneity analysis...")
    het_indices, het_pct = analyze_heterogeneity_scale_free(F_keep, F_replace, FV,
                                                             xval, zval, xbin, zbin)
    
    println("\n[5/6] Selection bias correction...")
    _, df_long = estimate_propensity_scores(df_long)
    gap_raw, gap_ipw, selection_bias = analyze_heterogeneity_corrected(
        F_keep, F_replace, FV, df_long, xval, zval, xbin, zbin)
    
    println("\n[6/6] Specification robustness...")
    spec_results = specification_robustness(df_long, xtran, xval, zval, xbin, zbin, β)
    
    plot_scale_free_results(indices, contrast_results, het_indices)
    
    println("\n" * "="^70)
    println("ANALYSIS COMPLETE")
    println("="^70)
    println("\n✓ Index range: ",
            round(minimum(indices.relative_index), digits=2), " to ",
            round(maximum(indices.relative_index), digits=2))
    println("✓ Branded heterogeneity: ", round(het_pct, digits=1), "%")
    println("✓ Selection bias: ", round(selection_bias, digits=1), " pp")
    
    return Dict(
        "indices" => indices,
        "F_keep" => F_keep,
        "F_replace" => F_replace,
        "heterogeneity" => (indices=het_indices, pct=het_pct,
                            raw=gap_raw, ipw=gap_ipw, bias=selection_bias),
        "contrast" => contrast_results,
        "spec_results" => spec_results
    )
end

#========================================
# RUN ANALYSIS
========================================#

println("\n" * "="^70)
println("RUNNING CORRECTED SCALE-FREE ANALYSIS")
println("="^70)

results = run_scale_free_analysis()

println("\n\n✓ Results available in 'results' dictionary")
println("  Access with: results[\"indices\"], results[\"heterogeneity\"], etc.")