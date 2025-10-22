################################################################################
# Problem Set 7 - Complete Solution (Functions + Main)
# ECON 6343: Econometrics III
# GMM and SMM Estimation
# Student: Bhaskar Kaushik
################################################################################


################################################################################
# Core Functions
################################################################################

"""
    load_data(url)

Load wage data and create design matrix for OLS regression.
Returns: DataFrame, X matrix, log wage vector
"""
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1)) df.age df.race .== 1 df.collgrad .== 1]
    y = log.(df.wage)
    return df, X, y
end

"""
    prepare_occupation_data(df)

Prepare occupation data for multinomial logit.
Collapse occupation categories and create covariates.
"""
function prepare_occupation_data(df)
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    df.white = df.race .== 1
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation
    return df, X, y
end

"""
    ols_gmm(β, X, y)

GMM objective function for OLS regression (identity weighting).
J(β) = (y - Xβ)'(y - Xβ)
"""
function ols_gmm(β, X, y)
    ŷ = X * β
    g = y .- ŷ
    return dot(g, g)
end

"""
    stable_softmax(Xβ)

Compute softmax probabilities with numerical stability using log-sum-exp.
Returns: (P, logden), where P are probabilities and logden is the log denom.
"""
function stable_softmax(Xβ)
    m = maximum(Xβ; dims=2)
    logden = m .+ log.(sum(exp.(Xβ .- m); dims=2))
    P = exp.(Xβ .- logden)
    return P, logden
end

"""
    mlogit_mle(α, X, y)

Negative log-likelihood for multinomial logit with stability.
α stacks the first (J-1) columns of the K×J coefficient matrix (last column 0).
"""
function mlogit_mle(α, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)

    # indicator matrix for observed choices
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = (y .== j)
    end

    bigα = [reshape(α, K, J-1) zeros(K)]
    Xβ = X * bigα
    # log-sum-exp
    m = maximum(Xβ; dims=2)
    logden = m .+ log.(sum(exp.(Xβ .- m); dims=2))
    logP = Xβ .- logden
    # negative log-likelihood
    return -sum(bigY .* logP)
end

"""
    mlogit_gmm(α, X, y)

Just-identified GMM for multinomial logit:
E[X_i (d_ij - P_ij)] = 0 for j=1..J-1 (K×(J-1) moments)
"""
function mlogit_gmm(α, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)

    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = (y .== j)
    end

    bigα = [reshape(α, K, J-1) zeros(K)]
    P, _ = stable_softmax(X * bigα)

    g = zeros((J-1) * K)
    for j = 1:(J-1)
        for k = 1:K
            g[(j-1)*K + k] = mean((bigY[:, j] .- P[:, j]) .* X[:, k])
        end
    end
    return N * dot(g, g)
end

"""
    mlogit_gmm_overid(α, X, y)

Over-identified GMM for multinomial logit with moment vector vec(d - P).
"""
function mlogit_gmm_overid(α, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)

    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = (y .== j)
    end

    bigα = [reshape(α, K, J-1) zeros(K)]
    P, _ = stable_softmax(X * bigα)

    g = bigY[:] .- P[:]  # N*J moments
    return dot(g, g)
end

"""
    sim_logit(N, J)

Simulate multinomial logit data using inverse CDF on probabilities.
Returns (Y::Vector{Int}, X, β) with last column of β normalized to 0.
"""
function sim_logit(N=100_000, J=4)
    X = hcat(ones(N), randn(N), randn(N) .> 0.5, 10 .* rand(N))

    if J == 4
        β = hcat([1, -1, 0.5, 0.25],
                 [0, 0.5, 0.3, -0.4],
                 [0, -0.5, 2, 1],
                 zeros(4))
    else
        β = -2 .+ 4 .* rand(size(X,2), J)
        β = β .- β[:, end]  # normalize last column to zero
    end

    P, _ = stable_softmax(X * β)
    draw = rand(N)
    Y = zeros(N)
    for j = 1:J
        Y .+= (vec(sum(P[:, j:J]; dims=2)) .>= draw)
    end
    return Int.(Y), X, β
end

"""
    sim_logit_w_gumbel(N, J)

Simulate multinomial logit via Gumbel i.i.d. shocks: argmax_j (Xβ_j + ε_ij).
"""
function sim_logit_w_gumbel(N=100_000, J=4)
    X = hcat(ones(N), randn(N), randn(N) .> 0.5, 10 .* rand(N))

    if J == 4
        β = hcat([1, -1, 0.5, 0.25],
                 [0, 0.5, 0.3, -0.4],
                 [0, -0.5, 2, 1],
                 zeros(4))
    else
        β = -2 .+ 4 .* rand(size(X,2), J)
        β = β .- β[:, end]
    end

    # Gumbel(0,1) via inverse CDF: ε = -log(-log(U)), U ~ U(0,1)
    ε = -log.(-log.(rand(N, J)))
    Y = Int.(argmax.(eachrow(X * β .+ ε)))
    return Y, X, β
end

"""
    mlogit_smm_overid(α, X, y, D)

SMM objective for multinomial logit: match average simulated choice indicators.
"""
function mlogit_smm_overid(α, X, y, D)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)

    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = (y .== j)
    end

    bigỸ = zeros(N, J)
    bigα = [reshape(α, K, J-1) zeros(K)]

    Random.seed!(1234)  # CRN
    for _ = 1:D
        ε = -log.(-log.(rand(N, J)))
        ỹ = Int.(argmax.(eachrow(X * bigα .+ ε)))
        for j = 1:J
            bigỸ[:, j] .+= (ỹ .== j) .* (1/D)
        end
    end

    g = bigY[:] .- bigỸ[:]
    return dot(g, g)
end

"""
    ovr_ols_starts(X, y, J)

Simple, dependency-free starting values for multinomial logit:
One-vs-rest OLS of d_ij on X for each class j, then normalize on class J.
Returns α vectorized for the first (J-1) classes.
"""
function ovr_ols_starts(X::AbstractMatrix, y::AbstractVector{<:Integer}, J::Int)
    K = size(X, 2)
    B = zeros(K, J)
    for j = 1:J
        d = (y .== j) .* 1.0
        B[:, j] = X \ d
    end
    for j = 1:J-1
        B[:, j] .-= B[:, J]   # normalize
    end
    return vec(B[:, 1:J-1])
end

################################################################################
# Main Function (script-style)
################################################################################

"""
    main()

Runs all estimation procedures and prints summaries.
"""
function main()
    println("="^80)
    println("Problem Set 7: GMM and SMM Estimation")
    println("="^80)

    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df, X_wage, y_wage = load_data(url)
    df, X, y = prepare_occupation_data(df)

    #--------------------------------------------------------------------------
    # Q1: OLS via GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 1: OLS Estimation via GMM")
    println("="^80)

    β_hat_gmm = optimize(b -> ols_gmm(b, X_wage, y_wage),
                         rand(size(X_wage, 2)),
                         LBFGS(),
                         Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=false))

    β_ols = X_wage \ y_wage

    println("\nGMM estimates: ", β_hat_gmm.minimizer)
    println("OLS estimates: ", β_ols)
    println("Difference (norm): ", norm(β_hat_gmm.minimizer - β_ols))

    #--------------------------------------------------------------------------
    # Q2: Multinomial Logit via MLE and GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 2: Multinomial Logit via MLE and GMM")
    println("="^80)

    # Dependency-free starts (OVR OLS)
    println("\nComputing starting values (OVR-OLS)...")
    J = length(unique(y))
    svals = ovr_ols_starts(X, y, J)

    # (a) MLE
    println("\nPart (a): MLE (this may take a minute...)")
    α_hat_mle = optimize(a -> mlogit_mle(a, X, y),
                         svals,
                         LBFGS(),
                         Optim.Options(g_tol=1e-4, iterations=1000, show_trace=false))
    println("MLE estimates: ", α_hat_mle.minimizer)
    println("MLE objective: ", α_hat_mle.minimum)

    # (b) Just-ID GMM (MLE start)
    println("\nPart (b): Just-identified GMM with MLE starting values")
    α_hat_gmm_just = optimize(a -> mlogit_gmm(a, X, y),
                              α_hat_mle.minimizer,
                              LBFGS(),
                              Optim.Options(g_tol=1e-4, iterations=1000, show_trace=false))
    println("Just-ID GMM estimates: ", α_hat_gmm_just.minimizer)
    println("Just-ID GMM objective: ", α_hat_gmm_just.minimum)

    # (c) Over-ID GMM (MLE start)
    println("\nPart (c): Over-identified GMM with MLE starting values")
    α_hat_gmm_mle_start = optimize(a -> mlogit_gmm_overid(a, X, y),
                                   α_hat_mle.minimizer,
                                   LBFGS(),
                                   Optim.Options(g_tol=1e-4, iterations=1000, show_trace=false))
    println("Over-ID GMM (MLE start) estimates: ", α_hat_gmm_mle_start.minimizer)
    println("Over-ID GMM objective: ", α_hat_gmm_mle_start.minimum)

    # (d) Over-ID GMM (random start)
    println("\nPart (d): Over-identified GMM with random starting values")
    α_hat_gmm_random_start = optimize(a -> mlogit_gmm_overid(a, X, y),
                                      rand(length(svals)),
                                      LBFGS(),
                                      Optim.Options(g_tol=1e-4, iterations=1000, show_trace=false))
    println("Over-ID GMM (random start) estimates: ", α_hat_gmm_random_start.minimizer)
    println("Over-ID GMM objective: ", α_hat_gmm_random_start.minimum)

    # Compare
    println("\nComparison:")
    println("Diff (MLE vs Just-ID GMM): ", norm(α_hat_mle.minimizer - α_hat_gmm_just.minimizer))
    println("Diff (MLE vs Over-ID GMM-MLE): ", norm(α_hat_mle.minimizer - α_hat_gmm_mle_start.minimizer))
    println("Diff (MLE vs Over-ID GMM-random): ", norm(α_hat_mle.minimizer - α_hat_gmm_random_start.minimizer))
    println("\nIs objective globally concave? Compare objectives from different starting values.")
    println("If estimates are very different, objective is likely NOT globally concave.")

    #--------------------------------------------------------------------------
    # Q3: Simulate and Recover Parameters
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 3: Simulate Data and Recover Parameters")
    println("="^80)

    println("\nSimulating data with Gumbel method...")
    Random.seed!(1234)
    ySim, XSim, β_true = sim_logit_w_gumbel(100_000, 4)
    println("Choice frequencies: ", [mean(ySim .== j) for j = 1:4])

    println("\nTrue parameters (β):")
    println(β_true)

    println("\nRecovering parameters via MLE...")
    K_sim, J_sim = size(XSim, 2), 4
    svals_sim = randn(K_sim * (J_sim - 1))
    α_hat_recovered = optimize(a -> mlogit_mle(a, XSim, ySim),
                               svals_sim,
                               LBFGS(),
                               Optim.Options(g_tol=1e-5, iterations=1000, show_trace=false))
    β_recovered = [reshape(α_hat_recovered.minimizer, K_sim, J_sim-1) zeros(K_sim)]

    println("\nRecovered parameters (β̂):")
    println(β_recovered)
    println("\nParameter recovery error (Frobenius norm):")
    println(norm(β_true - β_recovered))

    #--------------------------------------------------------------------------
    # Q5: Multinomial Logit via SMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 5: Multinomial Logit via SMM")
    println("="^80)

    println("\nEstimating via SMM with 100 simulation draws (this will take a while)...")
    α_hat_smm = optimize(th -> mlogit_smm_overid(th, X, y, 100),
                         α_hat_mle.minimizer,
                         LBFGS(),
                         Optim.Options(g_tol=1e-4, iterations=500, show_trace=false))

    println("\nComparison of estimates:")
    println("MLE: ",            α_hat_mle.minimizer)
    println("Just-ID GMM: ",     α_hat_gmm_just.minimizer)
    println("Over-ID GMM: ",     α_hat_gmm_mle_start.minimizer)
    println("SMM: ",             α_hat_smm.minimizer)

    println("\nDifferences from MLE:")
    println("Just-ID GMM vs MLE: ", norm(α_hat_gmm_just.minimizer - α_hat_mle.minimizer))
    println("Over-ID GMM vs MLE:  ", norm(α_hat_gmm_mle_start.minimizer - α_hat_mle.minimizer))
    println("SMM vs MLE:          ", norm(α_hat_smm.minimizer - α_hat_mle.minimizer))

    println("\n" * "="^80)
    println("Estimation Complete!")
    println("="^80)

    return nothing
end
