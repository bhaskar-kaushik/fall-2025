# PS1_Kaushik_UnitTest.jl

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Include the main script sitting in the same directory
include(joinpath(@__DIR__, "PS1_Kaushik.jl"))

# -------------------------- Unit tests --------------------------

@testset "q1 tests" begin
    Random.seed!(SEED)
    A,B,C,D = q1()
    @test size(A) == (10,7)
    @test size(B) == (10,7)
    @test size(C) == (5,7)
    @test size(D) == (10,7)
    @test all(A .<= 10) && all(A .>= -5)
    @test isapprox(mean(B), -2, atol=2)
    @test isapprox(std(B), 15, atol=3)
    @test C == hcat(A[1:5,1:5], B[1:5,6:7])
    @test all(D .== min.(A, 0.0))
    @test length(A) == 70
    @test isfile(joinpath(DIR_OUT,"matrixpractice.jld"))
    @test isfile(joinpath(DIR_OUT,"firstmatrix.jld"))
    @test isfile(joinpath(DIR_OUT,"Cmatrix.csv"))
    @test isfile(joinpath(DIR_OUT,"Dmatrix.dat"))
end

@testset "q2 tests" begin
    Random.seed!(SEED)
    A = -5 .+ 15 .* rand(10,7)
    B = -2 .+ 15 .* randn(10,7)
    C = hcat(A[1:5,1:5], B[1:5,6:7])

    @test q2(A,B,C) === nothing

    cache = _q2_compute(A,B,C)
    @test cache.AB == cache.AB2 == A .* B
    @test sort(cache.Cprime) == sort(cache.Cprime2)
    @test length(cache.Cprime) > 0
    @test size(cache.X) == (N,K,T)
    @test all(cache.X[:,1,:] .== 1.0)
    @test all(diff(cache.X[:,5,:], dims=2) .== 0)
    @test all(diff(cache.X[:,6,:], dims=2) .== 0)
    for t in 1:T
        @test isapprox(mean(cache.X[:,2,t]), 0.75*(6-t)/5, atol=0.05)
        @test isapprox(mean(cache.X[:,3,t]), 15+t-1, atol = 1)
        @test isapprox(std(cache.X[:,3,t]), 5*(t-1), atol=2)
        @test isapprox(mean(cache.X[:,4,t]), pi*(6-t)/3, atol=0.1)
        @test isapprox(std(cache.X[:,4,t]), 1/exp(1), atol=0.1)
    end
    @test isapprox(mean(cache.X[:,5,1]), BIN_N1_P[1]*BIN_N1_P[2], atol = 1)
    @test isapprox(mean(cache.X[:,6,1]), BIN_N2_P[1]*BIN_N2_P[2], atol = 1)
    @test size(cache.β) == (K,T)
    @test cache.β[1,:] == [1, 1.25, 1.5, 1.75, 2.0]
    @test cache.β[2,:] == log.(1:T)
    @test cache.β[3,:] == -sqrt.(1:T)
    @test cache.β[4,:] == [exp(t)-exp(t+1) for t=1:T]
    @test cache.β[5,:] == 1:T
    @test cache.β[6,:] == (1:T)./3
    @test size(cache.Y) == (N,T)
    for t in 1:T, n in 1:10
        det_part = sum(cache.X[n,:,t] .* cache.β[:,t])
        @test isapprox(cache.Y[n,t], det_part, atol = 1.2)
    end
end

@testset "q3 tests" begin
    if isfile(joinpath(DIR_RAW,"nlsw88.csv"))
        q3()
        @test isfile(joinpath(DIR_PROC,"nlsw88_processed.csv"))
        @test isfile(joinpath(DIR_OUT,"nlsw88_summarystats.csv"))
        @test isfile(joinpath(DIR_OUT,"mean_wage_by_ind_occ.csv"))
        nlsw = CSV.read(joinpath(DIR_PROC,"nlsw88_processed.csv"), DataFrame)
        @test nrow(nlsw) == 2246
        @test isapprox(mean(skipmissing(nlsw.never_married)), 0.1042, atol=0.001)
        @test isapprox(mean(skipmissing(nlsw.collgrad)), 0.2369, atol=0.001)
        @test sum(ismissing.(nlsw.grade)) == 2
        summarystats = CSV.read(joinpath(DIR_OUT,"nlsw88_summarystats.csv"), DataFrame)
        @test ncol(summarystats) >= 6
        df_grp = CSV.read(joinpath(DIR_OUT,"mean_wage_by_ind_occ.csv"), DataFrame)
        @test nrow(df_grp) > 50
    else
        @test_skip "nlsw88.csv not found in $(DIR_RAW)"
    end
end

@testset "q4 tests" begin
    Random.seed!(SEED)
    A = -5 .+ 15 .* rand(10,7)
    B = -2 .+ 15 .* randn(10,7)
    h, cp, s = matrixops(A, B)
    @test h == A .* B
    @test cp == A' * B
    @test s == sum(A + B)
    C = hcat(A[1:5,1:5], B[1:5,6:7])
    D = map(x -> x <= 0 ? x : 0.0, A)
    @test_throws ArgumentError matrixops(C, D)
    A_small = [1 2; 3 4]
    B_small = [5 6; 7 8]
    h, cp, s = matrixops(A_small, B_small)
    @test h == [5 12; 21 32]
    @test cp == [26 30; 38 44]
    @test s == 36
    @test_throws ArgumentError matrixops(A_small, [1 2 3]')
end
