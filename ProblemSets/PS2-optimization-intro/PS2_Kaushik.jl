#!/usr/bin/env julia
# ================================================================
# PS2_Kaushik.jl
# ECON 6343 – PS2 solution (Fall 2025)
# AI note (required by syllabus): "Used ChatGPT and Grok to debug and refine code/tests for ECON 6343 Fall 2025 PS2."
#
# How to run:
#   julia --project=. ProblemSets/PS2-optimization-intro/PS2_Kaushik.jl
#   julia --project=. ProblemSets/PS2-optimization-intro/PS2_Kaushik.jl --profile
# ================================================================

# ---------- Activate the repo env BEFORE any imports ----------
const DIR_ROOT = normpath(joinpath(@__DIR__, "..", ".."))  # -> fall-2025
if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    Pkg.activate(DIR_ROOT)
    Pkg.instantiate()
    @info "Active project" Base.active_project()
    @assert isfile(joinpath(DIR_ROOT, "Project.toml")) "Project.toml not found at DIR_ROOT = $DIR_ROOT"
end
@info "Running file" @__FILE__

# ---------- Imports (only what Prof listed) ----------
using Random, LinearAlgebra, Statistics, Printf
using DataFrames, CSV, HTTP
using Optim, GLM
using ForwardDiff
using Distributions
using FreqTables

# ---------- Portable paths ----------
const DIR_DATA = joinpath(DIR_ROOT, "ProblemSets", "PS2-optimization-intro", "data")
const DIR_OUT  = joinpath(DIR_ROOT, "ProblemSets", "PS2-optimization-intro", "out")
for d in (DIR_DATA, DIR_OUT); isdir(d) || mkpath(d); end

# ================================================================
# Utilities
# ================================================================
log1pexp(x::Real) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
σ(x) = 1 / (1 + exp(-x))                       # logistic
clamp01(x) = x < 0 ? 0.0 : (x > 1 ? 1.0 : x)   # safety for predicted probs

# Case/spacing/punct insensitive column finder
normalize_key(s) = replace(lowercase(strip(String(s))), r"[^a-z0-9]+" => "")
function find_col(df::AbstractDataFrame, name::AbstractString)
    m = Dict{String,Symbol}()
    for n in names(df)
        k = normalize_key(n)
        haskey(m,k) || (m[k] = Symbol(n))
    end
    get(m, normalize_key(name), nothing)
end

# Load NLSW88 from the course repo (used in Q2–Q5)
function load_nlsw88(; cachefile=joinpath(DIR_DATA, "nlsw88.csv"))
    if isfile(cachefile)
        return CSV.read(cachefile, DataFrame)
    end
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    resp = HTTP.get(url)
    @assert resp.status == 200 "Failed to download nlsw88.csv"
    df = CSV.read(resp.body, DataFrame)
    CSV.write(cachefile, df)
    return df
end

# ================================================================
# Q1. One-dimensional optimization (maximize quartic)
# ================================================================
# Expose f exactly as tests expect (both scalar and vector signatures)
f(x::Real) = -(x^4) - 10x^3 - 2x^2 - 3x - 2
f(z::AbstractVector) = f(z[1])
const f_scalar = f
minusf(z) = -f(z)

function q1(; rng=Random.default_rng())
    x0  = rand(rng) .* 2 .- 1             # random start in [-1,1]
    res = optimize(minusf, [x0], BFGS(), Optim.Options(; iterations=10_000, g_tol=1e-10); autodiff=:forward)
    xstar = Optim.minimizer(res)[1]
    fstar = f(xstar)
    println("\n[Q1] argmax x* = ", @sprintf("%.6f", xstar), " ; f(x*) = ", @sprintf("%.6f", fstar))
    return (; xstar, fstar, res)
end

# ================================================================
# Q2. OLS via Optim vs closed form vs GLM.lm
# ================================================================
# Design: y = married==1 ; regress on 1, age, white, collgrad
function build_design(df::DataFrame)
    age   = find_col(df, "age")
    race  = find_col(df, "race")
    coll  = find_col(df, "collgrad")
    marr  = find_col(df, "married")
    @assert !any(isnothing, (age,race,coll,marr)) "One of age/race/collgrad/married not found."

    keep = .!ismissing.(df[!,age]) .& .!ismissing.(df[!,race]) .& .!ismissing.(df[!,coll]) .& .!ismissing.(df[!,marr])
    df2  = df[keep, :]

    X = [ones(nrow(df2)) df2[!,age] (df2[!,race].==1) (df2[!,coll].==1)]
    y = (df2[!,marr].==1) .* 1.0
    return X, y, df2
end

ols_ssr(β, X, y) = begin
    e = y .- X*β
    return dot(e,e)
end

function q2(df::DataFrame)
    X, y, df2 = build_design(df)

    # Optim OLS
    β0  = zeros(size(X,2))
    res = optimize(b -> ols_ssr(b, X, y), β0, LBFGS(), Optim.Options(; iterations=100_000, g_tol=1e-8); autodiff=:forward)
    β_optim = Optim.minimizer(res)

    # Closed form
    β_closed = (X' * X) \ (X' * y)

    # GLM (sanity)
    df2.white = df2.race .== 1
    fit_lm = lm(@formula(married ~ age + white + collgrad), df2)

    println("\n[Q2] OLS β (Optim)  = ", round.(β_optim; digits=6))
    println("[Q2] OLS β (closed) = ", round.(β_closed; digits=6))
    println("[Q2] OLS β (GLM.lm) = ", round.(coef(fit_lm); digits=6))
    return (; β_optim, β_closed, fit_lm)
end

# ================================================================
# Q3. Binary Logit MLE (married on age, white, collgrad)
# ================================================================
function logit_negll(β, X, y)
    η = X*β
    ll = -sum( y .* log1pexp.(-η) .+ (1 .- y) .* log1pexp.(η) )
    return -ll
end

function q3(df::DataFrame)
    X, y, df2 = build_design(df)
    β_start = (X' * X) \ (X' * y)
    res = optimize(b -> logit_negll(b, X, y), β_start, LBFGS(), Optim.Options(; iterations=200_000, g_tol=1e-8); autodiff=:forward)
    β_logit = Optim.minimizer(res)

    df2.white = df2.race .== 1
    fit_glm = glm(@formula(married ~ age + white + collgrad), df2, Binomial(), LogitLink())
    println("\n[Q3] Logit β (Optim) = ", round.(β_logit; digits=6))
    println("[Q3] Logit β (GLM)   = ", round.(coef(fit_glm); digits=6))
    return (; β_logit, res, fit_glm)
end

# Predicted probabilities and quick calibration summary (used in Q4)
function q4_predicted_probs(df::DataFrame; β=nothing)
    X, y, _ = build_design(df)
    if β === nothing
        β = (X' * X) \ (X' * y)
    end
    p = σ.(X*β)
    p = clamp01.(p)
    dfp = DataFrame(p=p, y=y)
    return p, dfp
end

# ================================================================
# Q4. “Understanding the model” – quick checks
# ================================================================
function q4(df::DataFrame, β_logit)
    p, dfp = q4_predicted_probs(df; β=β_logit)
    @printf "\n[Q4] mean(y)=%.3f  mean(p̂)=%.3f  corr(y,p̂)=%.3f\n" mean(dfp.y) mean(dfp.p) cor(dfp.y, dfp.p)
    return (; mean_y=mean(dfp.y), mean_p=mean(dfp.p), corr_yp=cor(dfp.y, dfp.p))
end

# ================================================================
# Q5. Multinomial Logit (occupation collapsed to 7 groups)
# ================================================================
function prep_mnl(df::DataFrame)
    occ = find_col(df, "occupation")
    @assert !isnothing(occ) "occupation column not found"
    df2 = deepcopy(df)
    df2 = dropmissing(df2, occ)
    for k in 8:13
        df2[df2[!,occ].==k, occ] .= 7
    end

    age   = find_col(df2, "age")
    race  = find_col(df2, "race")
    coll  = find_col(df2, "collgrad")
    keep = .!ismissing.(df2[!,age]) .& .!ismissing.(df2[!,race]) .& .!ismissing.(df2[!,coll])
    df2 = df2[keep, :]

    X = [ones(nrow(df2)) df2[!,age] (df2[!,race].==1) (df2[!,coll].==1)]
    y = convert(Vector{Int}, df2[!,occ])   # 1..7
    @assert all(1 .<= y .<= 7)
    return X, y, df2
end

# Row-wise softmax with base alternative (utility of alt 1 = 0)
function row_softmax_with_base(U::AbstractMatrix)
    N, Jm1 = size(U)
    T = eltype(U)
    M = fill(zero(T), N, Jm1 + 1)
    @views M[:, 2:end] .= U
    m = maximum(M, dims = 2)
    M .-= m
    E = exp.(M)
    S = sum(E, dims = 2)
    return E ./ S
end

# Negative log-likelihood for MNL with base alt normalized to 0
function mlogit_negll(θ::AbstractVector, X::AbstractMatrix, y::Vector{Int}; J::Int)
    N, K = size(X)
    @assert length(θ) == K*(J-1)
    Θ = reshape(θ, K, J-1)
    U = X * Θ
    P = row_softmax_with_base(U)           # N×J
    T = eltype(P)
    ll = zero(T)
    @inbounds for i in 1:N
        ll += log(P[i, y[i]])
    end
    return -ll
end

function q5(df::DataFrame; nstarts::Int=5, rng=Random.default_rng())
    X, y, _ = prep_mnl(df)
    N, K = size(X)
    J = 7
    L = K*(J-1)

    best = nothing
    rows = Vector{NamedTuple}()
    for s in 1:nstarts
        θ0 = 0.1 .* randn(rng, L)
        res = optimize(θ -> mlogit_negll(θ, X, y; J=J), θ0, LBFGS(), Optim.Options(; iterations=200_000, g_tol=1e-7); autodiff=:forward)
        θ̂ = Optim.minimizer(res)
        nll = Optim.minimum(res)
        push!(rows, (; start=s, NLL=nll))
        if best === nothing || nll < best.NLL
            best = (; start=s, θ=θ̂, NLL=nll, K=K, J=J)
        end
    end
    table = DataFrame(rows)
    println("\n[Q5] multistart NLL summary:")
    show(table; allcols=true); println()
    println("[Q5] best start = ", best.start, " ; best NLL = ", @sprintf("%.6f", best.NLL))

    Θ̂ = reshape(best.θ, K, J-1)
    U  = X*Θ̂
    P  = row_softmax_with_base(U)
    ŷ = map(i -> argmax(view(P, i, :)), 1:size(P,1))

    # Use FreqTables to stay within allowed packages
    actual = freqtable(y)
    pred   = freqtable(ŷ)

    println("[Q5] actual occ dist: ", actual)
    println("[Q5] predicted top-1 occ dist: ", pred)

    return (; best, table, actual, pred)
end

# ================================================================
# Entrypoint
# ================================================================
function main(; profile=false, seed=1234)
    Random.seed!(seed)
    t0 = time()

    out1 = q1(); t1 = time()
    df = load_nlsw88()
    out2 = q2(df); t2 = time()
    out3 = q3(df); t3 = time()
    out4 = q4(df, out3.β_logit); t4 = time()
    out5 = q5(df; nstarts=5); t5 = time()

    println("\nDone. Artifacts in: ", DIR_OUT)
    if profile
        @printf "Timing (s): Q1=%.3f  Q2=%.3f  Q3=%.3f  Q4=%.3f  Q5=%.3f  total=%.3f\n" (t1-t0) (t2-t1) (t3-t2) (t4-t3) (t5-t4) (t5-t0)
    end
    return (; out1, out2, out3, out4, out5)
end

# Thin wrapper some graders like:
allwrap() = main()

if abspath(PROGRAM_FILE) == @__FILE__
    main(profile="--profile" in ARGS)
end

allwrap(; kwargs...) = main(; kwargs...)
