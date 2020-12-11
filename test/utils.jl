@testset "utils.jl" begin
    @testset "DiscreteMeasure" begin
        @test_throws ErrorException SOT.DiscreteMeasure(randn(3), rand(2))
    end

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
end
