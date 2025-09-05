# ECON 6343 – PS1 solution (Fall 2025)
# AI note (required by syllabus): "Used ChatGPT and Grok to debug and refine code/tests for ECON 6343 Fall 2025 PS1."
#
# How to run:
#   julia --project=. PS1_Kaushik.jl
#   julia --project=. PS1_Kaushik.jl --profile   # prints simple timing

using Random, LinearAlgebra, Statistics
using CSV, DataFrames
using Distributions
using JLD
using Test

# -------------------------- Reproducible env --------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# -------------------------- Portable paths --------------------------
const DIR_ROOT  = @__DIR__
const DIR_DATA  = joinpath(DIR_ROOT, "data")
const DIR_RAW   = joinpath(DIR_DATA, "raw")        # put nlsw88.csv here
const DIR_PROC  = joinpath(DIR_DATA, "processed")  # generated processed data
const DIR_OUT   = joinpath(DIR_ROOT, "out")        # generated matrices & summaries
for d in (DIR_DATA, DIR_RAW, DIR_PROC, DIR_OUT)
    isdir(d) || mkpath(d)
end

# -------------------------- Constants --------------------------
const SEED      = 1234
const N         = 15169
const K         = 6
const T         = 5
const SIGMA_Y   = 0.36
const BIN_N1_P  = (20, 0.6)   # for X[:,5,:]
const BIN_N2_P  = (20, 0.5)   # for X[:,6,:]

# -------------------------- Helpers --------------------------

"""
    normalize_key(s)

Lowercase and replace non-alphanumerics with `_`. Used for case/spacing-robust
column name resolution.
"""
normalize_key(s::AbstractString) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

"""
    find_col(df, target) -> Symbol | nothing

Return the `Symbol` for column `target` in `df` using case/spacing/punctuation
insensitive matching; return `nothing` if not found.
"""
function find_col(df::AbstractDataFrame, target::AbstractString)
    key = normalize_key(target)
    table = Dict(normalize_key(n) => Symbol(n) for n in names(df))
    return get(table, key, nothing)
end

"""
    replace_nothing_with_missing!(df) -> df

Mutate `df` by converting any `nothing` cells to `missing`
(helps when writing CSVs).
"""
function replace_nothing_with_missing!(df::DataFrame)
    for name in names(df)
        col = df[!, name]
        if any(x -> x === nothing, col)
            allowmissing!(df, name)
            replace!(df[!, name], nothing => missing)
        end
    end
    return df
end

# -------------------------- Q1 --------------------------

"""
    q1() -> (A,B,C,D)

Builds matrices per PS1 Q1, prints requested counts, and writes artifacts to `out/`:
- `matrixpractice.jld`, `firstmatrix.jld`, `Cmatrix.csv`, `Dmatrix.dat`
"""
function q1()
    Random.seed!(SEED)

    A = -5 .+ 15 .* rand(10,7)              # U[-5,10]
    B = -2 .+ 15 .* randn(10,7)             # N(-2,15)
    C = hcat(A[1:5,1:5], B[1:5,6:7])        # 5×7
    D = map(x -> x <= 0 ? x : 0.0, A)

    println("nelem(A) = ", length(A))
    println("nunique(D) = ", length(unique(vec(D))))

    # Q1(d): flatten B
    E  = reshape(B, :)
    E2 = vec(B)

    # Q1(e,f): 3-D array and permute
    F = Array{Float64,3}(undef,10,7,2); F[:,:,1]=A; F[:,:,2]=B
    F = permutedims(F, (3,1,2))             # 2 × 10 × 7

    # Q1(g): Kronecker
    G = kron(B, C)
    try
        kron(C, F)                          # not defined for 3-D
    catch err
        @info "C ⊗ F errors because F is 3D" error=err
    end

    JLD.save(joinpath(DIR_OUT,"matrixpractice.jld"),
             "A",A,"B",B,"C",C,"D",D,"E",E,"E2",E2,"F",F,"G",G)
    JLD.save(joinpath(DIR_OUT,"firstmatrix.jld"), "A",A,"B",B,"C",C,"D",D)
    CSV.write(joinpath(DIR_OUT,"Cmatrix.csv"), DataFrame(C, :auto))
    CSV.write(joinpath(DIR_OUT,"Dmatrix.dat"), DataFrame(D, :auto); delim='\t')

    return A,B,C,D
end

# -------------------------- Q2 --------------------------

"""
    _q2_compute(A,B,C) -> NamedTuple

Internal worker for Q2 that returns all derived objects used by tests.
"""
function _q2_compute(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    if size(A)!=(10,7) || size(B)!=(10,7)
        throw(DimensionMismatch("A and B must be 10×7; got $(size(A)), $(size(B))"))
    end
    if size(C)!=(5,7)
        throw(DimensionMismatch("C must be 5×7; got $(size(C))"))
    end

    # (a) elementwise product (loop & vectorized)
    AB = similar(A)
    for i in axes(A,1), j in axes(A,2)
        AB[i,j] = A[i,j]*B[i,j]
    end
    AB2 = A .* B

    # (b) values of C in [-5,5]
    Cprime = Float64[]
    for x in C
        if -5 <= x <= 5
            push!(Cprime, x)
        end
    end
    Cprime2 = vec(C[(C .>= -5) .& (C .<= 5)])

    # (c) build X (N×K×T)
    X = Array{Float64,3}(undef, N, K, T)
    X[:,1,:] .= 1.0
    X[:,5,:] .= rand(Binomial(BIN_N1_P[1], BIN_N1_P[2]), N, 1)
    X[:,6,:] .= rand(Binomial(BIN_N2_P[1], BIN_N2_P[2]), N, 1)
    for t in 1:T
        p = 0.75 * (6 - t) / 5
        X[:,2,t] = rand(N) .< p
        μ3, σ3 = 15 + (t-1), 5*(t-1)
        X[:,3,t] = μ3 .+ (σ3 == 0 ? zeros(N) : σ3 .* randn(N))
        μ4, σ4 = pi*(6 - t)/3, 1/exp(1)
        X[:,4,t] = μ4 .+ σ4 .* randn(N)
    end

    # (d) β (K×T)
    β = [ k==1 ? 1 + 0.25*(t-1) :
          k==2 ? log(t) :
          k==3 ? -sqrt(t) :
          k==4 ? exp(t)-exp(t+1) :
          k==5 ? t : t/3
          for k in 1:K, t in 1:T ]

    # (e) Y (N×T)
    Y = [ sum(X[n, :, t] .* β[:, t]) + 0.36*randn()
          for n in 1:N, t in 1:T ]

    return (; AB, AB2, Cprime, Cprime2, X, β, Y)
end

"""
    q2(A,B,C) -> nothing

Wrapper that validates inputs and computes Q2 quantities.
"""
function q2(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    _ = _q2_compute(A,B,C)
    return nothing
end

# -------------------------- Q3 --------------------------

"""
    q3() -> nothing

Reads `data/raw/nlsw88.csv`, computes required summaries, and writes:
- `out/nlsw88_summarystats.csv`
- `out/mean_wage_by_ind_occ.csv`
- `data/processed/nlsw88_processed.csv`
"""
function q3()
    infile = joinpath(DIR_RAW, "nlsw88.csv")
    if !isfile(infile)
        throw(ArgumentError("Place nlsw88.csv at $(infile)."))
    end
    nlsw = CSV.read(infile, DataFrame; missingstring=[".", "NA", "", "nan", "NaN"])

    nm  = find_col(nlsw, "never_married")
    mar = find_col(nlsw, "married")
    col = find_col(nlsw, "collgrad")
    grd = find_col(nlsw, "grade")
    rac = find_col(nlsw, "race")
    ind = find_col(nlsw, "industry")
    occ = find_col(nlsw, "occupation")
    wag = find_col(nlsw, "wage")

    # (b) percents
    pct_never = if !isnothing(nm)
        mean(skipmissing(nlsw[!, nm]) .== 1)
    elseif !isnothing(mar)
        mean(skipmissing(nlsw[!, mar]) .== 0)
    else
        @warn "Could not find never_married or married in: $(names(nlsw))"
        missing
    end

    pct_coll = if !isnothing(col)
        mean(skipmissing(nlsw[!, col]) .== 1)
    elseif !isnothing(grd)
        mean(skipmissing(nlsw[!, grd]) .>= 16)
    else
        @warn "Could not find collgrad or grade in: $(names(nlsw))"
        missing
    end

    !ismissing(pct_never) && println("% never married = ", round(pct_never*100; digits=2))
    !ismissing(pct_coll)  && println("% college grads = ", round(pct_coll*100; digits=2))

    # (c) race proportions (DataFrames version)
    isnothing(rac) && throw(ArgumentError("race column not found"))
    race_counts = combine(groupby(nlsw, rac), nrow => :count)
    race_counts[!, :prop] = race_counts.count ./ sum(race_counts.count)
    println("race distribution (prop):\n", race_counts)

    # (d) summary stats + missing grade
    summarystats = describe(nlsw, :mean, :median, :std, :min, :max, :nunique)
    isnothing(grd) && throw(ArgumentError("grade column not found"))
    nmiss_grade = sum(ismissing, nlsw[!, grd])
    println("missing grade obs = ", nmiss_grade)
    replace_nothing_with_missing!(summarystats)
    CSV.write(joinpath(DIR_OUT, "nlsw88_summarystats.csv"), summarystats)

    # (e) industry × occupation counts (long form)
    isnothing(ind) && throw(ArgumentError("industry column not found"))
    isnothing(occ) && throw(ArgumentError("occupation column not found"))
    jt = combine(groupby(nlsw, [ind, occ]), nrow => :count)
    println("industry x occupation (long-form counts):\n", jt)

    # (f) mean wage by (industry, occupation)
    isnothing(wag) && throw(ArgumentError("wage column not found"))
    df_w  = select(nlsw, ind, occ, wag)
    df_grp = combine(groupby(df_w, [ind, occ]), wag => (x -> mean(skipmissing(x))) => :wage_mean)
    CSV.write(joinpath(DIR_OUT, "mean_wage_by_ind_occ.csv"), df_grp)

    # processed copy
    nlsw_processed = deepcopy(nlsw)
    replace_nothing_with_missing!(nlsw_processed)
    CSV.write(joinpath(DIR_PROC, "nlsw88_processed.csv"), nlsw_processed)
    return nothing
end

# -------------------------- Q4 --------------------------

"""
    matrixops(A,B) -> (A∘B, A'B, sum(A+B))

Hadamard product, crossproduct, and scalar sum. Errors if sizes differ.
"""
function matrixops(A::AbstractArray, B::AbstractArray)
    if size(A) != size(B)
        throw(ArgumentError("inputs must have the same size."))
    end
    hadamard  = A .* B
    crossprod = transpose(A) * B
    total_sum = sum(A .+ B)
    return hadamard, crossprod, total_sum
end

"""
    q4() -> nothing

Loads matrices from `out/firstmatrix.jld`, runs `matrixops` on (A,B) and on
(ttl_exp, wage) vectors from `data/processed/nlsw88_processed.csv`.
"""
function q4()
    jldfile = joinpath(DIR_OUT, "firstmatrix.jld")
    if !isfile(jldfile)
        throw(ArgumentError("Run q1() first to create $(jldfile)"))
    end
    d = JLD.load(jldfile)
    A,B,C,D = d["A"], d["B"], d["C"], d["D"]

    hAB, cpAB, sAB = matrixops(A,B)
    println("matrixops(A,B): sum = ", sAB, " ; hadamard size = ", size(hAB))
    try
        matrixops(C,D)
    catch err
        println("matrixops(C,D) correctly errored: ", err)
    end

    procfile = joinpath(DIR_PROC, "nlsw88_processed.csv")
    if !isfile(procfile)
        throw(ArgumentError("Run q3() first to create $(procfile)"))
    end
    nlsw = CSV.read(procfile, DataFrame)
    texp = find_col(nlsw, "ttl_exp")
    wage = find_col(nlsw, "wage")
    isnothing(texp) && throw(ArgumentError("ttl_exp not found"))
    isnothing(wage) && throw(ArgumentError("wage not found"))

    mask = .!ismissing.(nlsw[!, texp]) .& .!ismissing.(nlsw[!, wage])
    v1 = nlsw[!, texp][mask]
    v2 = nlsw[!, wage][mask]
    n  = length(v1)
    V1, V2 = reshape(v1, n, 1), reshape(v2, n, 1)
    hVV, cpVV, sVV = matrixops(V1,V2)
    println("matrixops(ttl_exp,wage): sum = ", sVV, " ; hadamard size = ", size(hVV))
    return nothing
end

# -------------------------- Entrypoint --------------------------

"""
    run_pipeline(; profile=false) -> nothing

Runs Q1→Q4 end-to-end. If `profile=true`, prints simple step timing.
"""
function run_pipeline(; profile::Bool=false)
    println("\n--- Running PS1 pipeline ---")
    t0 = time()
    A,B,C,D = q1()
    t1 = time()
    q2(A,B,C)
    t2 = time()
    if isfile(joinpath(DIR_RAW, "nlsw88.csv"))
        q3()
    else
        println("nlsw88.csv not found in $(DIR_RAW); skipping q3()")
    end
    t3 = time()
    if isfile(joinpath(DIR_OUT, "firstmatrix.jld")) &&
       isfile(joinpath(DIR_PROC, "nlsw88_processed.csv"))
        q4()
    end
    t4 = time()
    println("Done. Artifacts in: ", DIR_OUT, " and ", DIR_PROC)
    if profile
        println("Timing (s): q1=$(round(t1-t0,digits=3))  q2=$(round(t2-t1,digits=3))  q3=$(round(t3-t2,digits=3))  q4=$(round(t4-t3,digits=3))  total=$(round(t4-t0,digits=3))")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline(profile="--profile" in ARGS)
end