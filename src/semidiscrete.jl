function dual_cost(rng::Random.AbstractRNG, c, v, μ::DiscreteMeasure, ν, ε; kwargs...)
    return dual_cost(rng, c, v, ν, μ, ε; kwargs...)
end
function dual_cost(
    rng::Random.AbstractRNG,
    c,
    v,
    μ,
    ν::DiscreteMeasure,
    ε;
    montecarlo_samples=10_000,
    kwargs...,
)
    # compute MC estimate of the expected c-transform with respect to `μ`
    mean_ctransform = Statistics.mean(
        ctransform(c, v, rand(rng, μ), ν, ε) for _ in 1:montecarlo_samples
    )

    return LinearAlgebra.dot(v, ν.ps) + mean_ctransform
end

function dual_v(rng::Random.AbstractRNG, c, μ::DiscreteMeasure, ν, ε; kwargs...)
    return dual_v(rng, c, ν, μ, ε; kwargs...)
end
function dual_v(
    rng::Random.AbstractRNG,
    c,
    μ,
    ν::DiscreteMeasure,
    ε;
    maxiters::Int=10_000,
    stepsize=1,
    warmup_phase=1,
    atol=0,
    rtol=iszero(atol) ? typeof(float(atol))(1//10_000) : 0,
)
    # initial iterates
    k = 1
    x = rand(rng, μ)
    τ = stepsize / (1 + √(0 / warmup_phase))
    ṽ = gradient_step(c, τ, ν, x, ε)

    # initial dual solution
    v = copy(ṽ)

    # error estimates
    Δv = similar(ṽ)
    errors = @. Δv / (atol + rtol * abs(v))

    converged = false
    while !converged && k < maxiters
        # update iterates
        k += 1
        x = rand(rng, μ)
        τ = stepsize / (1 + √((k - 1) / warmup_phase))
        gradient_step!(c, ṽ, τ, ṽ, ν, x, Δv, ε)

        # estimate error
        @. Δv = (ṽ - v) / k
        @. errors = Δv / (atol + rtol * abs(v))
        converged = maximum(abs, errors) < 1

        # update dual solution
        v .+= Δv
    end

    if k == maxiters && !converged
        @warn "Optimization algorithm did not converge. Try to adjust the parameters."
    end

    return v
end
