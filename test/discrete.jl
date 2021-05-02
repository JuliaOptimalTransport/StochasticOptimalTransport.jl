@testset "discrete.jl" begin
    @testset "equal source and target" begin
        c(x, y) = abs(x - y)

        xs = randn(3)
        ps = rand(3)
        ps ./= sum(ps)
        μ = SOT.DiscreteMeasure(xs, ps)
        ν = SOT.DiscreteMeasure(xs, ps)

        @test SOT.wasserstein(c, μ, ν; stepsize=0.05) ≈ 0 atol = 1e-4
        @test SOT.wasserstein(c, μ, ν, 1e-6; stepsize=0.05) ≈ 0 atol = 1e-4
        @test SOT.wasserstein(c, μ, ν, 1e-3; stepsize=0.05) ≈ 0 atol = 1e-4
    end

    @testset "uniform weights" begin
        c(x, y) = abs(x - y)

        n = 5
        xs = randn(n)
        ys = randn(n)
        ps = fill(1 / n, n)
        μ = SOT.DiscreteMeasure(xs, ps)
        ν = SOT.DiscreteMeasure(ys, ps)

        # analytic Wasserstein distance
        d = sum(abs, sort(xs) .- sort(ys)) / n

        @test SOT.wasserstein(c, μ, ν; stepsize=0.05) ≈ d atol = 5e-2
        @test SOT.wasserstein(c, μ, ν, 1e-6; stepsize=0.05) ≈ d atol = 5e-2
        @test SOT.wasserstein(c, μ, ν, 1e-3; stepsize=0.05) ≈ d atol = 5e-2
    end
end
