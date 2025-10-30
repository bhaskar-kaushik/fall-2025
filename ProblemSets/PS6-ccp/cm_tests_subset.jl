# ================================================
# cm_tests_subset.jl
# Minimal add-on for CM action-invariance + Euler tests
# Works with your existing outputs: Q0, Q1, xtran, xval, zval, xbin, zbin, β
# ================================================

using Statistics, LinearAlgebra, DataFrames, StatsBase

# --- utilities

"""
    _row_normalize!(A)

Row-normalize a matrix in place so each row sums to 1 (if the row sum > 0).
Returns A.
"""
function _row_normalize!(A::AbstractMatrix{<:Real})
    rs = sum(A, dims=2)
    for i in 1:size(A,1)
        s = rs[i]
        if s > 0
            @inbounds A[i, :] ./= s
        end
    end
    return A
end

"""
    _hellinger_row_distance(p, q)

Hellinger distance between two probability vectors (assumes nonnegative, sums to 1).
H(p,q) = (1/√2) * ||√p - √q||_2
"""
function _hellinger_row_distance(p::AbstractVector, q::AbstractVector)
    @assert length(p) == length(q)
    s = 0.0
    @inbounds for j in eachindex(p)
        s += (sqrt(max(p[j],0.0)) - sqrt(max(q[j],0.0)))^2
    end
    return sqrt(s) / sqrt(2)
end

"""
    _kl_row_divergence(p, q; ϵ=1e-12)

KL(p||q) with small ϵ for numerical stability. p, q must be nonnegative and sum ~ 1.
"""
function _kl_row_divergence(p::AbstractVector, q::AbstractVector; ϵ=1e-12)
    @assert length(p) == length(q)
    d = 0.0
    @inbounds for j in eachindex(p)
        pj = max(p[j], 0.0) + ϵ
        qj = max(q[j], 0.0) + ϵ
        d += pj * log(pj / qj)
    end
    return d
end

# ------------------------------------------------
# 1) CM ACTION-INVARIANCE TEST
# ------------------------------------------------

"""
    cm_action_invariance_test(Q0, Q1; metric=:hellinger, summary=true)

Choice-Market (CM) test: compare the **normalized** continuation kernels
for d=keep vs d=replace. Under action-invariance (complete-markets analogue),
the *pricing kernel shape* (over next states) should not depend on the action.

We implement it as a distributional distance between row-normalized Q0 and Q1.

Arguments:
- Q0, Q1: implicit price matrices β f(⋅|X, d) (you already compute these)
- metric: :hellinger (default) or :kl (KL divergence)
- summary: if true, prints a compact summary table

Returns: DataFrame with per-row distances and a short summary Dict.
"""
function cm_action_invariance_test(Q0::AbstractMatrix, Q1::AbstractMatrix;
                                   metric::Symbol=:hellinger, summary::Bool=true)
    @assert size(Q0) == size(Q1) "Q0 and Q1 must have the same shape"
    n, k = size(Q0)

    # Normalize rows to compare shapes (remove β and level effects)
    P0 = Array{Float64}(undef, n, k); P0 .= Q0
    P1 = Array{Float64}(undef, n, k); P1 .= Q1
    _row_normalize!(P0)
    _row_normalize!(P1)

    dist = Vector{Float64}(undef, n)
    if metric == :hellinger
        @inbounds for i in 1:n
            dist[i] = _hellinger_row_distance(view(P0,i,:), view(P1,i,:))
        end
    elseif metric == :kl
        @inbounds for i in 1:n
            # symmetrized KL for stability: 0.5*(KL(p||q)+KL(q||p))
            dist[i] = 0.5 * (_kl_row_divergence(view(P0,i,:), view(P1,i,:)) +
                             _kl_row_divergence(view(P1,i,:), view(P0,i,:)))
        end
    else
        error("Unknown metric. Use :hellinger or :kl")
    end

    stats = Dict(
        "metric" => String(metric),
        "mean_distance" => mean(dist),
        "median_distance" => median(dist),
        "p90_distance" => quantile(dist, 0.90),
        "p95_distance" => quantile(dist, 0.95),
        "max_distance" => maximum(dist)
    )

    if summary
        println("\n==============================")
        println("CM ACTION-INVARIANCE TEST")
        println("==============================")
        println("Metric: ", stats["metric"])
        println("Mean distance:   ", round(stats["mean_distance"], digits=4))
        println("Median:          ", round(stats["median_distance"], digits=4))
        println("90th pct:        ", round(stats["p90_distance"], digits=4))
        println("95th pct:        ", round(stats["p95_distance"], digits=4))
        println("Max:             ", round(stats["max_distance"], digits=4))
        println("Interpretation:")
        if stats["mean_distance"] < 0.05
            println("  ✓ Near action-invariance (complete-market-like).")
        elseif stats["mean_distance"] < 0.15
            println("  ⚠ Mild violations—some action dependence (moderate wedges).")
        else
            println("  ✗ Strong violations—kernels differ by action (incomplete markets).")
        end
    end

    return DataFrame(row = 1:n, distance = dist), stats
end

# ------------------------------------------------
# 2) EULER / PRICING TEST WITH SYNTHETIC PAYOFFS
# ------------------------------------------------

"""
    build_synthetic_payoffs(xval; high_q=0.8, medium_q=0.5)

Construct simple state-measurable payoff vectors R(x') for testing:
- R_high = 1{x' in top (1 - high_q) mileage}, else 0
- R_med  = 1{x' above median mileage}, else 0
- R_lin  = scaled linear payoff in mileage ∈ [0,1]

Returns Dict of payoff vectors of length K (number of next-state mileage bins).
"""
function build_synthetic_payoffs(xval::AbstractVector; high_q=0.8, medium_q=0.5)
    k = length(xval)
    thresh_high = quantile(xval, high_q)
    thresh_med  = quantile(xval, medium_q)

    R_high = [x >= thresh_high ? 1.0 : 0.0 for x in xval]
    R_med  = [x >= thresh_med  ? 1.0 : 0.0 for x in xval]

    # linear payoff scaled to [0,1]
    xmin, xmax = minimum(xval), maximum(xval)
    span = xmax > xmin ? (xmax - xmin) : 1.0
    R_lin  = [(x - xmin)/span for x in xval]

    return Dict("R_high" => R_high, "R_med" => R_med, "R_lin" => R_lin)
end

"""
    euler_pricing_test(Qd, R; normalize=true)

Generic pricing test: checks if E[q_d · R] is "action-invariant" up to a scale.
- Qd: n×k pricing weights (β f_d)
- R:  length k payoff over next-state mileage bins

We compute row-wise prices π_i = sum_j Qd[i,j] * R[j], then summarize across i.

If `normalize=true`, we report π normalized by β·E[R] under the *row's* implied prob.
(That removes level effects and focuses on relative pricing tightness.)

Returns: DataFrame (row-wise prices) and a Dict summary (mean, CV, quantiles).
"""
function euler_pricing_test(Qd::AbstractMatrix, R::AbstractVector; normalize::Bool=true)
    n, k = size(Qd)
    @assert length(R) == k "Payoff length must match number of next-state bins"

    # Row-normalized implied probabilities
    P = Array{Float64}(undef, n, k); P .= Qd
    _row_normalize!(P)

    prices = Vector{Float64}(undef, n)
    for i in 1:n
        π = 0.0
        @inbounds for j in 1:k
            π += Qd[i,j] * R[j]
        end
        if normalize
            μR = 0.0
            @inbounds for j in 1:k
                μR += P[i,j] * R[j]
            end
            prices[i] = μR > 0 ? π / μR : 0.0   # equals β if Qd = β f_d
        else
            prices[i] = π
        end
    end

    df = DataFrame(row = 1:n, price = prices)
    s = Dict(
        "mean" => mean(prices),
        "median" => median(prices),
        "cv" => (std(prices) / max(mean(prices), 1e-12)),
        "p10" => quantile(prices, 0.10),
        "p90" => quantile(prices, 0.90)
    )

    println("\n==============================")
    println("EULER / PRICING TEST (synthetic R)")
    println("==============================")
    println("Normalization: ", normalize ? "row-wise E[R]" : "none")
    println("Mean(price):   ", round(s["mean"]; digits=4),
            "   CV: ", round(s["cv"]; digits=3),
            "   [p10,p90]=[", round(s["p10"]; digits=4), ", ", round(s["p90"]; digits=4), "]")
    println("Interpretation:")
    if normalize
        # Under tight pricing, normalized prices should be nearly constant across rows.
        # Their level equals β; dispersion (CV) is the key diagnostic.
        if s["cv"] < 0.10
            println("  ✓ Near-constant normalized prices (tight pricing relations).")
        elseif s["cv"] < 0.25
            println("  ⚠ Moderate dispersion—evidence of wedges or heterogeneity.")
        else
            println("  ✗ Large dispersion—pricing varies strongly across states.")
        end
    else
        println("  (Raw levels shown; compare d=0 vs d=1 or report relative indices.)")
    end

    return df, s
end


# ------------------------------------------------
# 3) ONE-STOP WRAPPERS THAT USE YOUR OBJECTS
# ------------------------------------------------

"""
    run_cm_and_euler_tests(Q0, Q1, xval; high_q=0.8)

Runs:
  (i) CM action-invariance (Hellinger)
  (ii) Euler/pricing tests on three synthetic payoffs (R_high, R_med, R_lin)
Returns a Dict with all outputs.
"""
function run_cm_and_euler_tests(Q0::AbstractMatrix, Q1::AbstractMatrix, xval::AbstractVector; high_q=0.8)
    # CM test
    cm_rows, cm_stats = cm_action_invariance_test(Q0, Q1; metric=:hellinger, summary=true)

    # Synthetic payoffs
    R = build_synthetic_payoffs(xval; high_q=high_q)

    # Euler tests for each action and payoff
    euler = Dict{String,Any}()
    for (name, rval) in R
        df0, s0 = euler_pricing_test(Q0, rval; normalize=true)
        df1, s1 = euler_pricing_test(Q1, rval; normalize=true)
        euler[name] = Dict("keep_df"=>df0, "keep_stats"=>s0, "replace_df"=>df1, "replace_stats"=>s1)
        println("\n--- Payoff: $name ---")
        println(" keep:    mean=", round(s0["mean"]; digits=4), "  CV=", round(s0["cv"]; digits=3))
        println(" replace: mean=", round(s1["mean"]; digits=4), "  CV=", round(s1["cv"]; digits=3))
    end

    return Dict("cm_rows"=>cm_rows, "cm_summary"=>cm_stats, "euler"=>euler)
end


# ================================================
# HOW TO USE (in your current script, after you have Q0,Q1,xtran,xval,...):
#
# tests = run_cm_and_euler_tests(Q0, Q1, xval; high_q=0.8)
# tests["cm_summary"]        # => CM test summary
# tests["euler"]["R_high"]   # => Dict with keep/replace df & stats
#
# You can also switch the CM metric:
# cm_action_invariance_test(Q0, Q1; metric=:kl)
# ================================================
