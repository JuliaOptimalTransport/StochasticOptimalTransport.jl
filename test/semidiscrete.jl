@testset "semidiscrete.jl" begin
    @testset "gradient step" begin
        c(x, y) = abs(x - y)
        τ = rand()
        x = randn()
        xs = randn(100)
        ps = rand(100)
        ps ./= sum(ps)
        ν = SOT.DiscreteMeasure(xs, ps)
        v = zeros(100)

        # unregularized and regularized approach
        for ε in (nothing, abs(randn()))
            # out-of-place method
            z0 = SOT.gradient_step(c, τ, ν, x, ε)

            # in-place method with `v = 0`
            z = zero(z0)
            tmp = similar(z)
            SOT.gradient_step!(c, z, τ, v, ν, x, tmp, ε)
            @test z0 ≈ z0
        end
    end

    @testset "example" begin
        c(x, y) = abs(x - y)

        # equal source and target distribution
        xs = randn(3)
        ps = rand(3)
        ps ./= sum(ps)
        μ = DiscreteNonParametric(xs, ps)
        ν = SOT.DiscreteMeasure(xs, ps)
        @test SOT.wasserstein(c, μ, ν) ≈ 0 atol=2e-2
        @test SOT.wasserstein(c, μ, ν, 1e-6) ≈ 0 atol=2e-2
        @test SOT.wasserstein(c, μ, ν, 1e-3) ≈ 0 atol=2e-2
    end
end
