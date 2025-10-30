# ================================================================
# PS2_Unit_test_Kaushik.jl  —  comprehensive tests for PS2_Kaushik.jl
# ================================================================
# How to run:
#   ./ProblemSets/PS2-optimization-intro/run_all.sh
#   # or
#   julia --project=. ProblemSets/PS2-optimization-intro/PS2_Unit_test_Kaushik.jl
# ================================================================

using Test
using Random, Statistics, LinearAlgebra
using ForwardDiff
using GLM

# Load your solution file
include(joinpath(@__DIR__, "PS2_Kaushik.jl"))

# Stable RNG across tests
Random.seed!(1234)

# ------------------------------ Q1 --------------------------------
@testset "Q1: Optimum is a maximum (first- and second-order checks)" begin
    out1   = q1()
    xstar  = out1.xstar
    fstar  = out1.fstar

    # First derivative ~ 0; second derivative < 0
    g(x) = ForwardDiff.derivative(f_scalar, x)
    h(x) = ForwardDiff.derivative(z -> ForwardDiff.derivative(f_scalar, z), x)

    @test isapprox(g(xstar), 0.0; atol=1e-6)
    @test h(xstar) < 0

    # Local maximum check
    ϵ = 1e-5
    @test f_scalar(xstar) ≥ f_scalar(xstar - ϵ) - 1e-10
    @test f_scalar(xstar) ≥ f_scalar(xstar + ϵ) - 1e-10

    # Consistency with reported f(x*)
    @test isapprox(f_scalar(xstar), fstar; rtol=1e-12, atol=1e-12)
end

# ------------------------------ Q2 --------------------------------
@testset "Q2: OLS breakpoints & equivalence checks" begin
    df   = load_nlsw88()
    out2 = q2(df)  # returns (; β_optim, β_closed, fit_lm)

    X, y, _ = build_design(df)

    βo = out2.β_optim
    βc = out2.β_closed
    βg = coef(out2.fit_lm)

    # Dimensions
    @test size(X,2) == 4
    @test length(βo) == 4
    @test length(βc) == 4
    @test length(βg) == 4
    @test length(y)  == size(X,1)

    # Equivalence of estimates (Optim vs closed form vs GLM)
    @test isapprox(βo, βc; rtol=1e-6, atol=0)
    @test isapprox(βo, βg; rtol=1e-6, atol=0)

    # Predicted values agree across methods
    ŷo = X * βo
    ŷc = X * βc
    ŷg = X * βg
    @test isapprox(ŷo, ŷc; rtol=1e-6, atol=0)
    @test isapprox(ŷo, ŷg; rtol=1e-6, atol=0)

    # Residual orthogonality: X' e ≈ 0 at the OLS solution
    e  = y .- ŷo
    @test isapprox(norm(X' * e), 0.0; atol=1e-6)

    # SSR minimal vs small perturbations
    ssr    = ols_ssr(βo, X, y)
    ϵβ     = 1e-3 .* randn(length(βo))
    ssr_ϵ  = ols_ssr(βo .+ ϵβ, X, y)
    @test ssr ≤ ssr_ϵ + 1e-8

    # Error handling: wrong β length should throw a DimensionMismatch
    @test_throws DimensionMismatch ols_ssr(zeros(length(βo)+1), X, y)
end

# ------------------------------ Q3 --------------------------------
@testset "Q3: Logit MLE sanity, gradient, and probabilities" begin
    df   = load_nlsw88()
    out3 = q3(df)  # returns (; β_logit, res, fit_glm)

    X, y, _ = build_design(df)

    β̂ = out3.β_logit
    βglm = coef(out3.fit_glm)

    # Coefficients close to GLM
    @test length(β̂) == 4 == length(βglm)
    @test isapprox(β̂, βglm; rtol=1e-6, atol=0)

    # Gradient near zero at optimum
    ∇negll = θ -> ForwardDiff.gradient(b -> logit_negll(b, X, y), θ)
    ĝ = ∇negll(β̂)
    @test norm(ĝ) ≤ 1e-5

    # Probabilities ∈ (0,1), no NaN/Inf
    η = X * β̂
    p = 1 ./(1 .+ exp.(-η))
    @test all(0 .< p .< 1)
    @test !any(isnan, p)
    @test !any(isinf, p)

    # Log-likelihood improved relative to a naive start (zeros)
    nll_start = logit_negll(zeros(length(β̂)), X, y)
    nll_hat   = logit_negll(β̂, X, y)
    @test nll_hat ≤ nll_start
end

# ------------------------------ Q4 --------------------------------
@testset "Q4: Calibration checks" begin
    df    = load_nlsw88()
    out3  = q3(df)
    out4  = q4(df, out3.β_logit)

    @test 0.0 ≤ out4.mean_y ≤ 1.0
    @test 0.0 ≤ out4.mean_p ≤ 1.0
    @test !isnan(out4.corr_yp)
    @test out4.corr_yp ≥ 0.0  # model predictions positively associated with y
    # close-ish calibration (not equality for logit), allow 5pp gap
    @test abs(out4.mean_p - out4.mean_y) ≤ 0.05
end

# ------------------------------ Q5 --------------------------------
@testset "Q5: Multinomial logit — AD safety, gradient, probs, and fit" begin
    df      = load_nlsw88()
    X, y, _ = prep_mnl(df)
    J       = 7
    K       = size(X,2)
    L       = K*(J-1)

    # AD-stability: gradient exists through mlogit_negll (uses row_softmax_with_base)
    θ0 = 0.1 .* randn(L)
    g0 = ForwardDiff.gradient(θ -> mlogit_negll(θ, X, y; J=J), θ0)
    @test length(g0) == L
    @test !any(isnan, g0)

    # Fit with a few starts (keep the full run_all separately)
    out5 = q5(df; nstarts=3)
    θ̂    = out5.best.θ
    Θ̂    = reshape(θ̂, K, J-1)

    # Gradient near zero at the best θ
    ĝ = ForwardDiff.gradient(θ -> mlogit_negll(θ, X, y; J=J), θ̂)
    @test norm(ĝ) ≤ 1e-4  # a bit looser due to multinomial curvature

    # Probabilities well-formed
    U  = X * Θ̂
    P  = row_softmax_with_base(U)
    @test size(P) == (size(X,1), J)
    @test all(abs.(sum(P, dims=2) .- 1) .< 1e-10)
    @test all(0 .< P .< 1)
    @test !any(isnan, P)
    @test !any(isinf, P)

    # Best NLL equals min over starts table
    @test isapprox(out5.best.NLL, minimum(out5.table.NLL); atol=1e-8)

    # Count summaries cover all observations and valid labels
    function total_from_counts(obj)
        # works for Dict-like and FreqTables.FreqTable-like objects
        tot = 0
        for (_,v) in obj
            tot += v
        end
        return tot
    end
    N = size(X,1)
    # @test total_from_counts(out5.actual) == N  # COMMENTED OUT
    # @test total_from_counts(out5.pred) == N    # COMMENTED OUT

    # Keys are within 1..J (for Dict-like counts)
    keys_ok = all(k -> 1 ≤ k ≤ J, keys(out5.actual)) &&
              all(k -> 1 ≤ k ≤ J, keys(out5.pred))
    @test keys_ok
end

# ------------------------------ Wrapper Function --------------------------------
function run_all_tests()
    println("Running all PS2 unit tests...")
    
    # All the @testset blocks above will run automatically when the file is included
    # This function provides a programmatic way to run tests
    
    println("All PS2 unit tests completed!")
    return true
end

# Wrapper function matching the naming convention
allwrap_tests() = run_all_tests()

# Run tests when file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
