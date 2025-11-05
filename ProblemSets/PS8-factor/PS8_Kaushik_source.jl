#=
Problem Set 8 - Source Code
ECON 6343: Econometrics III
Factor Models and Dimension Reduction
=#


# Include quadrature file
include("lgwt.jl")

#==================================================================================
# Question 1: Load data and estimate base regression
==================================================================================#

function load_data(url::String)
    return CSV.read(HTTP.get(url).body, DataFrame)
end

function estimate_base_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
end

#==================================================================================
# Question 2: Compute correlations among ASVAB scores
==================================================================================#

function compute_asvab_correlations(df::DataFrame)
    asvabs = Matrix(df[:, end-5:end])
    correlation = cor(asvabs)
    cordf = DataFrame( 
        cor1 = correlation[:,1], 
        cor2 = correlation[:,2], 
        cor3 = correlation[:,3], 
        cor4 = correlation[:,4], 
        cor5 = correlation[:,5], 
        cor6 = correlation[:,6]
    )
    return cordf
end

#==================================================================================
# Question 3: Regression with all ASVAB scores
==================================================================================#

function estimate_full_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
                       asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
end

#==================================================================================
# Question 4: PCA regression
==================================================================================#

function estimate_pca_regression(df::DataFrame)
    asvabs = Matrix(df[:, end-5:end])' # transpose
    M = fit(PCA, asvabs; maxoutdim=1)
    asvabPCA = MultivariateStats.transform(M, asvabs)
    df_pca = copy(df)
    df_pca = @transform(df_pca, :asvabPCA = asvabPCA[:])
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df_pca)
end

#==================================================================================
# Question 5: Factor Analysis regression
==================================================================================#

function estimate_factor_regression(df::DataFrame)
    asvabs = Matrix(df[:, end-5:end])' # transpose
    M = fit(FactorAnalysis, asvabs; maxoutdim=1)
    asvabFactor = MultivariateStats.transform(M, asvabs)
    df_factor = copy(df)
    df_factor = @transform(df_factor, :asvabFactor = asvabFactor[:])
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), df_factor)
end

#==================================================================================
# Question 6: Full factor model with MLE
==================================================================================#

function prepare_factor_matrices(df::DataFrame)
    X = [df.black df.hispanic df.female df.schoolt df.gradHS df.grad4yr ones(size(df, 1))]    
    y = df.logwage
    Xfac = [df.black df.hispanic df.female ones(size(df, 1))]
    asvabs = Matrix(df[:, end-5:end])
    return X, y, Xfac, asvabs
end

function factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix, 
                     y::Vector, R::Integer) where T<:Real
    
    # Get dimensions
    K = size(X, 2)
    L = size(Xfac, 2)
    J = size(Meas, 2)
    N = length(y)
    
    # Unpack parameters
    γ = reshape(θ[1:J*L], L, J)
    β = θ[J*L+1:J*L+K]
    α = θ[J*L+K+1:J*L+K+J+1]
    σ = θ[end-J:end]

    # Get quadrature nodes and weights
    ξ, ω = lgwt(R, -5, 5)

    # Initialize likelihood
    like = zeros(T, N)
    
    # Loop over quadrature points
    for r in 1:R
        # ASVAB test contribution
        Mlike = zeros(T, N, J)
        for j in 1:J
            Mres = Meas[:, j] .- Xfac * γ[:, j] .- α[j] * ξ[r]
            sdj = sqrt(σ[j]^2)
            Mlike[:, j] = (1 ./ sdj) .* pdf.(Normal(0, 1), Mres ./ sdj)
        end

        # Wage contribution
        Yres = y .- X * β .- α[end] * ξ[r]
        sdy = sqrt(σ[end]^2)
        Ylike = (1 ./ sdy) .* pdf.(Normal(0, 1), Yres ./ sdy)

        # Construct overall likelihood
        like += ω[r] .* prod(Mlike; dims = 2)[:] .* Ylike .* pdf.(Normal(0,1), ξ[r])
    end
    
    return -sum(log.(like))
end

function run_estimation(df::DataFrame, start_vals::Vector)
    # Prepare data matrices
    X, y, Xfac, asvabs = prepare_factor_matrices(df)

    # Optimize
    td = TwiceDifferentiable(th -> factor_model(th, X, Xfac, asvabs, y, 9), 
                            start_vals, autodiff = :forward)

    result = optimize(td, start_vals, Newton(linesearch = BackTracking()), 
                     Optim.Options(g_tol = 1e-5, iterations = 100_000, 
                     show_trace = true, show_every = 1))

    # Compute standard errors
    H = Optim.hessian!(td, result.minimizer)
    se = sqrt.(diag(inv(H)))

    return result.minimizer, se, -result.minimum
end

function main()
    # Data URL
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS8-factor/nlsy.csv"
    
    println("="^80)
    println("Problem Set 8: Factor Models and Dimension Reduction")
    println("="^80)
    
    # Load data
    println("\nLoading data...")
    df = load_data(url)
    println("Data loaded successfully. Dimensions: ", size(df))
    
    # Question 1
    println("\n" * "="^80)
    println("Question 1: Base Regression (without ASVAB)")
    println("="^80)
    OLSnoASVAB = estimate_base_regression(df)
    println(OLSnoASVAB)

    # Question 2
    println("\n" * "="^80)
    println("Question 2: ASVAB Correlations")
    println("="^80)
    cordf = compute_asvab_correlations(df)
    println(cordf)
    
    # Question 3
    println("\n" * "="^80)
    println("Question 3: Full Regression (with all ASVAB)")
    println("="^80)
    OLSwASVAB = estimate_full_regression(df)
    println(OLSwASVAB)
    
    # Question 4
    println("\n" * "="^80)
    println("Question 4: PCA Regression")
    println("="^80)
    OLSPCA = estimate_pca_regression(df)
    println(OLSPCA)
    
    # Question 5
    println("\n" * "="^80)
    println("Question 5: Factor Analysis Regression")
    println("="^80)
    OLSFA = estimate_factor_regression(df)
    println(OLSFA)
    
    # Question 6
    println("\n" * "="^80)
    println("Question 6: Full Factor Model (MLE)")
    println("="^80)
    
    # Prepare starting values
    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    svals = vcat(
        vec(Xfac\asvabs[:, 1]),
        vec(Xfac\asvabs[:, 2]),
        vec(Xfac\asvabs[:, 3]),
        vec(Xfac\asvabs[:, 4]),
        vec(Xfac\asvabs[:, 5]),
        vec(Xfac\asvabs[:, 6]),
        vec(X\y),
        rand(7),
        0.5*ones(7)
    )
    
    println("\nEstimating full factor model...")
    θ̂, se, loglike = run_estimation(df, svals)
    println("\nEstimation Results:")
    println("Estimate    Std.Error    Z-statistic")
    println(hcat(θ̂, se, θ̂ ./ se))
    println("Log-Likelihood: ", loglike)
    
    println("\n" * "="^80)
    println("Analysis complete!")
    println("="^80)
end