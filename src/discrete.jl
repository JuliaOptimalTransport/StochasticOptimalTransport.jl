function dual_cost(
    rng::Random.AbstractRNG,
    c,
    v,
    μ::DiscreteMeasure,
    ν::DiscreteMeasure,
    ε;
    kwargs...,
)
    # compute mean c-transform
    mean_ctransform = sum(p * ctransform(c, v, x, ν, ε) for (x, p) in zip(μ.xs, μ.ps))

    return LinearAlgebra.dot(v, ν.ps) + mean_ctransform
end

function dual_v(
    rng::Random.AbstractRNG,
    c,
    μ::DiscreteMeasure,
    ν::DiscreteMeasure,
    ε;
    stepsize = 1,
    maxiters::Int = 10_000,
    atol = 0,
    rtol = iszero(atol) ? typeof(float(atol))(1 // 10_000) : 0,
)
    # initial iterates
    k = 1
    sampler = Random.Sampler(rng, 1:length(μ))
    i = rand(rng, sampler)
    d = gradient_step(c, μ.ps[i], ν, μ.xs[i], ε)
    G = zeros(eltype(d), length(d), length(μ))
    copyto!(view(G, :, i), d)

    # initial dual solution
    v = stepsize * d

    # error estimates
    Δv = similar(v)
    errors = @. Δv / (atol + rtol * abs(v))

    converged = false
    while !converged && k < maxiters
        # update iterates
        k += 1
        i = rand(rng, sampler)
        g = view(G, :, i)
        d .-= g
        gradient_step!(c, g, μ.ps[i], v, ν, μ.xs[i], Δv, ε; reset=true)
        d .+= g

        # estimate error
        LinearAlgebra.mul!(Δv, stepsize, d)
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
