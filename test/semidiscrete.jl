@testset "semidiscrete.jl" begin
    @testset "gradient step" begin
        c(x, y) = abs(x - y)
        τ = rand()
        x = randn()
        ys = randn(100)
        v = zeros(100)
        ν = rand(100)
        ν ./= sum(ν)

        # unregularized and regularized approach
        for ε in (nothing, abs(randn()))
            # out-of-place method
            z0 = SOT.gradient_step(c, τ, ν, x, ys, ε)

            # in-place method with `v = 0`
            z = zero(z0)
            tmp = similar(z)
            SOT.gradient_step!(c, z, τ, v, ν, x, ys, tmp, ε)
            @test z0 ≈ z0
        end
    end

    @testset "example" begin
        c(x, y) = abs(x - y)

        # equal source and target distribution
        ys = randn(3)
        ν = rand(3)
        ν ./= sum(ν)
        μ = DiscreteNonParametric(ys, ν)
        @test SOT.wasserstein_SGA(c, μ, ν, ys) ≈ 0 atol=2e-2
        @test SOT.wasserstein_SGA(c, μ, ν, ys, 1e-6) ≈ 0 atol=2e-2
        @test SOT.wasserstein_SGA(c, μ, ν, ys, 1e-3) ≈ 0 atol=2e-2
    end
end
