# AI note (required by syllabus): "Used Claude and ChatGPT to debug and refine code/tests for ECON 6343 Fall 2025 PS3."
# Need to import Normal distribution for p-value calculation
using Distributions

# ============================================================
# PS3 – GEV Models (ECON 6343): Source Functions
# Clean: only functions; no package imports; no top-level execution
# ============================================================

# ---------- Core likelihoods ----------

"""
    mlogit_with_Z(theta, X, Z, y) -> Float64

Negative log-likelihood for multinomial logit with alt-specific wage Z.
Normalization: β_8 = 0. X includes an intercept (ASC) as its first column.
theta length = K*(J-1) + 1 (γ).
"""
function mlogit_with_Z(theta, X, Z, y)
    N, K = size(X)
    J = size(Z, 2)
    @assert length(theta) == K*(J-1) + 1 "theta length must be K*(J-1)+1"
    @assert size(Z,1) == N "Z must have same rows as X"
    @assert all(1 ≤ yy ≤ J for yy in y) "y must be in 1..J"

    α = theta[1:end-1]          # stacked K*(J-1)
    γ = theta[end]
    bigAlpha = [reshape(α, K, J-1) zeros(K)]  # K×J, last col = 0

    # utilities (N×J)
    V = similar(Z, N, J)
    for j in 1:J
        if j < J
            @inbounds V[:, j] = X * bigAlpha[:, j] .+ γ .* (Z[:, j] .- Z[:, J])
        else
            @inbounds V[:, j] = zero.(Z[:, j])          # γ*(Z_J - Z_J) = 0
        end
    end

    # softmax row-wise
    Vmax = maximum(V, dims=2)
    expV = exp.(V .- Vmax)
    P = expV ./ sum(expV, dims=2)

    # nll
    nll = 0.0
    @inbounds for i in 1:N
        nll -= log(max(P[i, y[i]], 1e-15))
    end
    return nll
end

"""
    nested_logit_with_Z(theta, X, Z, y) -> Float64

Negative log-likelihood for PS3 nested logit (WC={1,2,3}, BC={4,5,6,7}, Other=8).
Spec: V_ij = X_i'β_k + γ(Z_ij - Z_i8) if j in nest k; V_i8 = 0.
theta = [β_WC(K); β_BC(K); λ_WC; λ_BC; γ].
"""
function nested_logit_with_Z(theta, X, Z, y)
    N, K = size(X)
    J = size(Z, 2)
    @assert J == 8 "PS3 uses 8 alternatives"
    @assert length(theta) == 2*K + 3 "theta must be 2*K + 3"
    @assert size(Z,1) == N
    @assert all(1 ≤ yy ≤ J for yy in y)

    βWC = theta[1:K]
    βBC = theta[K+1:2*K]
    λWC = clamp(theta[2*K+1], 0.01, 1.0)
    λBC = clamp(theta[2*K+2], 0.01, 1.0)
    γ   = theta[end]

    WC = (1,2,3)
    BC = (4,5,6,7)

    nll = 0.0
    @inbounds for i in 1:N
        Zi8 = Z[i, 8]

        # inclusive values (scaled by λ)
        SWC = sum(exp((dot(X[i, :], βWC) + γ*(Z[i, j] - Zi8))/λWC) for j in WC)
        SBC = sum(exp((dot(X[i, :], βBC) + γ*(Z[i, j] - Zi8))/λBC) for j in BC)

        D = 1.0 + SWC^λWC + SBC^λBC

        yi = y[i]
        if yi in WC
            Vyi = dot(X[i,:], βWC) + γ*(Z[i, yi] - Zi8)
            P = exp(Vyi/λWC) * SWC^(λWC - 1.0) / D
        elseif yi in BC
            Vyi = dot(X[i,:], βBC) + γ*(Z[i, yi] - Zi8)
            P = exp(Vyi/λBC) * SBC^(λBC - 1.0) / D
        else
            P = 1.0 / D
        end
        nll -= log(max(P, 1e-15))
    end
    return nll
end

# ---------- Optimizers ----------

function optimize_mlogit(X, Z, y; rng_seed=123, g_tol=1e-6, iterations=100_000)
    N, K = size(X); J = size(Z,2)
    counts = [mean(y .== j) for j in 1:J]
    base = max(counts[J], 1e-6)
    start = Float64[]
    for j in 1:J-1
        push!(start, log(max(counts[j],1e-6) / base))
        for _ in 2:K
            push!(start, 0.01)
        end
    end
    push!(start, 0.01) # γ

    return Optim.optimize(θ -> mlogit_with_Z(θ, X, Z, y),
                          start, Optim.LBFGS(),
                          Optim.Options(g_tol=g_tol, iterations=iterations, show_trace=false))
end

function optimize_nested_logit(X, Z, y; rng_seed=123, g_tol=1e-6, iterations=100_000)
    N, K = size(X)
    start = vcat(fill(0.01, 2*K), [0.7, 0.7, 0.01])
    lower = fill(-Inf, length(start)); upper = fill( Inf, length(start))
    lower[2*K+1:2*K+2] .= 0.01;  upper[2*K+1:2*K+2] .= 1.0
    return Optim.optimize(θ -> nested_logit_with_Z(θ, X, Z, y),
                          lower, upper, start,
                          Optim.Fminbox(Optim.LBFGS()),
                          Optim.Options(g_tol=g_tol, iterations=iterations, show_trace=false))
end

# ---------- Standard Errors ----------

function hessian_se(theta, f; h=1e-5)
    n = length(theta)
    H = zeros(n,n)
    for i in 1:n, j in 1:n
        θpp = copy(theta); θpp[i]+=h; θpp[j]+=h
        θpm = copy(theta); θpm[i]+=h; θpm[j]-=h
        θmp = copy(theta); θmp[i]-=h; θmp[j]+=h
        θmm = copy(theta); θmm[i]-=h; θmm[j]-=h
        H[i,j] = (f(θpp) - f(θpm) - f(θmp) + f(θmm)) / (4h^2)
    end
    try
        V = inv(H)
        se = sqrt.(max.(diag(V), 0))
        return se
    catch
        return fill(NaN, n)
    end
end

# ---------- Helper Functions ----------

odds_multiplier_for_delta_logwage(γ, Δ) = exp(γ*Δ)
wage_semi_elasticity(γ, p) = γ * (1 - p)
correlation_measure(λ) = 1 - λ

function interpret_gamma(gamma_hat)
    odds10 = exp(0.1*gamma_hat)
    s = "γ measures the change in latent utility V_ij from a one-unit increase in the relative expected log wage (Z_ij - Z_i8).\n"*
        "A 10% wage increase changes the odds by exp(0.1·γ) = $(round(odds10, digits=3)).\n"*
        (gamma_hat > 0 ?
            "Positive γ is consistent with higher wages raising the likelihood of choosing occupation j." :
            "Negative γ can arise from compensating differentials, measurement error in expected wages, or omitted amenities.")
    return s
end

# ---------- Main Analysis Function ----------

function allwrap(X, Z, y; df=nothing)
    N, K = size(X); J = size(Z,2)
    println("\n", "="^70)
    println("OCCUPATIONAL CHOICE AND WAGES: GEV MODEL ANALYSIS")
    println("ECON 6343: Econometrics III")
    println("Dataset: NLSW88 (PS3)")
    println("="^70, "\n")

    println("Data summary:")
    counts = [sum(y .== j) for j in 1:J]
    println("  N=$(N), K=$(K) (incl. ASC), J=$(J)")
    println("  Choice counts: ", counts)

    # ----- Q1: MNL -----
    println("\n", "="^70)
    println("QUESTION 1: MULTINOMIAL LOGIT WITH ALT-SPECIFIC WAGES")
    println("="^70)
    print("Estimating MNL... "); flush(stdout)
    res_mnl = optimize_mlogit(X, Z, y)
    println(Optim.converged(res_mnl) ? "done (converged)" : "done (NOT converged)")
    println("Computing SEs (MNL)... ",); flush(stdout)
    θm = res_mnl.minimizer
    sem = hessian_se(θm, θ -> mlogit_with_Z(θ, X, Z, y))
    println("done.\n")

    ll_mnl = -Optim.minimum(res_mnl)
    γm = θm[end]; γm_se = sem[end]; γm_t = isfinite(γm_se) && γm_se>0 ? γm/γm_se : NaN
    println("MNL: Log-likelihood = ", round(ll_mnl, digits=3))
    println("γ (log wage) = ", round(γm, digits=4), "  SE=", round(γm_se, digits=4), "  t=", round(γm_t, digits=2))
    println("Odds multiplier for 10% wage ↑: ", round(odds_multiplier_for_delta_logwage(γm, 0.1), digits=3))
    
    println("\nBasic Interpretation:")
    println(interpret_gamma(γm))

    # ----- Q2: Interpretation of γ -----
    println("\n", "="^70)
    println("QUESTION 2: INTERPRETATION OF γ COEFFICIENT")
    println("="^70)
    println("Economic Meaning of γ = ", round(γm, digits=4), ":")
    println("• γ measures the marginal utility of relative expected log wage (Z_ij - Z_i8)")
    println("• One unit increase in relative log wage changes log-odds by γ")
    println("• For a 10% wage increase: odds multiplier = ", round(odds_multiplier_for_delta_logwage(γm, 0.1), digits=3))
    
    if γm > 0
        println("• Positive γ: Higher relative wages increase occupation choice probability")
        println("• Economic interpretation: Workers prefer higher-paying occupations")
    else
        println("• Negative γ: Higher relative wages decrease occupation choice probability")
        println("• Possible explanations:")
        println("  - Compensating differentials (high wage = undesirable job characteristics)")
        println("  - Measurement error in expected wages")
        println("  - Omitted variables (job amenities, working conditions)")
        println("  - Selection effects or unobserved heterogeneity")
    end
    
    println("\nSemi-elasticity Analysis:")
    println("• At mean choice probability p=", round(1/J, digits=3), ":")
    mean_elasticity = wage_semi_elasticity(γm, 1/J)
    println("  Semi-elasticity = ", round(mean_elasticity, digits=4))
    println("  A 1% wage increase changes choice probability by ≈", round(mean_elasticity*0.01*100, digits=3), " percentage points")
    
    if isfinite(γm_se) && γm_se > 0
        println("\nStatistical Significance:")
        println("• t-statistic = ", round(γm_t, digits=2))
        p_val_approx = 2 * (1 - cdf(Normal(), abs(γm_t)))
        println("• Approximate p-value ≈ ", round(p_val_approx, digits=4))
        if abs(γm_t) > 1.96
            println("• Result: Statistically significant at 5% level")
        else
            println("• Result: Not statistically significant at 5% level")
        end
    end

    # ----- Q3: NL -----
    println("\n", "="^70)
    println("QUESTION 3: NESTED LOGIT (WC={1,2,3}, BC={4,5,6,7}, Other={8})")
    println("="^70)
    print("Estimating unrestricted NL... "); flush(stdout)
    res_nl = optimize_nested_logit(X, Z, y)
    println(Optim.converged(res_nl) ? "done (converged)" : "done (NOT converged)")
    θn = res_nl.minimizer
    println("Computing SEs (NL)... "); flush(stdout)
    sen = hessian_se(θn, θ -> nested_logit_with_Z(θ, X, Z, y))
    println("done.\n")

    ll_nl = -Optim.minimum(res_nl)
    λWC = θn[2*K+1]; λBC = θn[2*K+2]; γn = θn[end]
    se_λWC = sen[2*K+1]; se_λBC = sen[2*K+2]; se_γn = sen[end]

    println("NL: Log-likelihood = ", round(ll_nl, digits=4))
    println("λ_WC=", round(λWC, digits=4), "  (1-λ_WC=", round(correlation_measure(λWC), digits=4), ")  SE=", round(se_λWC, digits=4))
    println("λ_BC=", round(λBC, digits=4), "  (1-λ_BC=", round(correlation_measure(λBC), digits=4), ")  SE=", round(se_λBC, digits=4))
    println("γ (log wage) = ", round(γn, digits=4), "   SE=", round(se_γn, digits=4))

    # LR test of IIA
    print("\nEstimating restricted NL (λ_WC=λ_BC=1)... "); flush(stdout)
    
    function restricted_objective(theta_reduced)
    theta_full = zeros(2*K + 3)
    theta_full[1:2*K] = theta_reduced[1:2*K]
    theta_full[2*K+1] = 1.0
    theta_full[2*K+2] = 1.0
        theta_full[end] = theta_reduced[end]
        return nested_logit_with_Z(theta_full, X, Z, y)
    end

    start_reduced = vcat(θn[1:2*K], θn[end])
    res_restr = Optim.optimize(restricted_objective, start_reduced,
                              Optim.LBFGS(),
                              Optim.Options(g_tol=1e-6, iterations=50_000, show_trace=false))

    println(Optim.converged(res_restr) ? "done (converged)" : "done (NOT converged)")
    ll_restr = -Optim.minimum(res_restr)

    LR = 2*(ll_nl - ll_restr)
    println("LR test (H₀: λ_WC=λ_BC=1):  LR = ", round(LR, digits=4), "   df=2   (χ²₀.₉₅ ≈ 5.991) → ",
            (LR > 5.991 ? "Reject IIA (nesting matters)" : "Fail to reject IIA (MNL sufficient)"))

    println("\nInterpretation:")
    println("  H₀: λ_WC = λ_BC = 1 (no within-nest correlation, IIA holds)")
    println("  H₁: At least one λ ≠ 1 (within-nest correlation exists)")
    if LR > 5.991
        println("  → Evidence of within-nest correlation; nested structure is important")
    else
        println("  → No strong evidence against IIA; simple MNL may be adequate")
    end

    # Prediction accuracy
    function argmax_col(v)
        jbest, vbest = 1, v[1]
        for j in 2:length(v)
            if v[j] > vbest; jbest, vbest = j, v[j]; end
        end
        return jbest
    end
    
    # MNL accuracy
    begin
        α = θm[1:end-1]; γ = θm[end]
        bigA = [reshape(α, K, J-1) zeros(K)]
        V = similar(Z, N, J)
        for j in 1:J
            if j < J
                V[:, j] = X*bigA[:, j] .+ γ .* (Z[:, j] .- Z[:, J])
            else
                V[:, j] = zero.(Z[:, j])
            end
        end
        Vmax = maximum(V, dims=2); P = exp.(V .- Vmax); P ./= sum(P, dims=2)
        pred = [argmax_col(@view P[i, :]) for i in 1:N]
        acc = sum(pred .== y) / N
        println("\nPrediction Accuracy:")
        println("  MNL: ", round(100*acc, digits=1), "%")
    end
    
    # NL accuracy
    begin
        WC = (1,2,3); BC = (4,5,6,7)
        βWC = θn[1:K]; βBC = θn[K+1:2K]; λWC = θn[2K+1]; λBC = θn[2K+2]; γ = θn[end]
        P = zeros(N, J)
        for i in 1:N
            Zi8 = Z[i,8]
            SWC = sum(exp((dot(X[i,:],βWC)+γ*(Z[i,j]-Zi8))/λWC) for j in WC)
            SBC = sum(exp((dot(X[i,:],βBC)+γ*(Z[i,j]-Zi8))/λBC) for j in BC)
            D = 1 + SWC^λWC + SBC^λBC
            for j in WC
                Vij = dot(X[i,:],βWC)+γ*(Z[i,j]-Zi8)
                P[i,j] = exp(Vij/λWC) * SWC^(λWC-1) / D
            end
            for j in BC
                Vij = dot(X[i,:],βBC)+γ*(Z[i,j]-Zi8)
                P[i,j] = exp(Vij/λBC) * SBC^(λBC-1) / D
            end
            P[i,8] = 1/D
        end
        pred = [argmax_col(@view P[i, :]) for i in 1:N]
        acc = sum(pred .== y) / N
        println("  NL : ", round(100*acc, digits=1), "%")
    end

    println("\n", "="^70)
    println("ANALYSIS COMPLETE")
    println("="^70)
    return nothing
end