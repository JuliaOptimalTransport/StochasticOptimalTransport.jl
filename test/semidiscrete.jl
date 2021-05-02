@testset "semidiscrete.jl" begin
    @testset "equal source and target" begin
        c(x, y) = abs(x - y)

        xs = randn(3)
        ps = rand(3)
        ps ./= sum(ps)
        μ = DiscreteNonParametric(xs, ps)
        ν = SOT.DiscreteMeasure(xs, ps)

        @test SOT.wasserstein(c, μ, ν) ≈ 0 atol = 2e-2
        @test SOT.wasserstein(c, μ, ν, 1e-6) ≈ 0 atol = 2e-2
        @test SOT.wasserstein(c, μ, ν, 1e-3) ≈ 0 atol = 2e-2

        @test SOT.wasserstein(c, ν, μ) ≈ 0 atol = 2e-2
        @test SOT.wasserstein(c, ν, μ, 1e-6) ≈ 0 atol = 2e-2
        @test SOT.wasserstein(c, ν, μ, 1e-3) ≈ 0 atol = 2e-2
    end

    @testset "uniform weights" begin
        c(x, y) = abs(x - y)

        n = 5
        xs = randn(n)
        ys = randn(n)
        ps = fill(1 / n, n)
        μ = DiscreteNonParametric(xs, ps)
        ν = SOT.DiscreteMeasure(ys, ps)

        # analytic Wasserstein distance
        d = sum(abs, sort(xs) .- sort(ys)) / n

        @test SOT.wasserstein(c, μ, ν) ≈ d atol = 2e-2
        @test SOT.wasserstein(c, μ, ν, 1e-6) ≈ d atol = 2e-2
        @test SOT.wasserstein(c, μ, ν, 1e-3) ≈ d atol = 2e-2

        @test SOT.wasserstein(c, ν, μ) ≈ d atol = 2e-2
        @test SOT.wasserstein(c, ν, μ, 1e-6) ≈ d atol = 2e-2
        @test SOT.wasserstein(c, ν, μ, 1e-3) ≈ d atol = 2e-2
    end
end
