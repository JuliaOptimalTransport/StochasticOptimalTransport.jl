wasserstein_SGA(args...; kwargs...) = wasserstein_SGA(Random.GLOBAL_RNG, args...; kwargs...)
function wasserstein_SGA(
    rng::Random.AbstractRNG,
    c,
    μ,
    ν::AbstractVector,
    ys::AbstractVector,
    ε::Union{Real,Nothing} = nothing;
    montecarlo_samples = 10_000,
    kwargs...,
)
    # approximate solution `v` of the dual problem
    v = dual_v_SGA(rng, c, μ, ν, ys, ε; kwargs...)

    # compute MC estimate of the expected c-transform with respect to `μ`
    mean_ctransform = Statistics.mean(
        ctransform(c, v, rand(rng, μ), ys, ν, ε) for _ in 1:montecarlo_samples
    )

    return LinearAlgebra.dot(v, ν) + mean_ctransform
end

dual_v_SGA(args...; kwargs...) = dual_v_SGA(Random.GLOBAL_RNG, args...; kwargs...)
function dual_v_SGA(
    rng::Random.AbstractRNG,
    c,
    μ,
    ν::AbstractVector,
    ys::AbstractVector,
    ε::Union{Real,Nothing} = nothing;
    maxiters::Int = 10_000,
    initial_stepsize = 1,
    warmup_phase = 10,
    atol = 0,
    rtol = iszero(atol) ? typeof(float(atol))(1 // 10_000) : 0,
)
    # initial iterates
    k = 1
    x = rand(rng, μ)
    τ = initial_stepsize / (1 + √(1 / warmup_phase))
    ṽ = gradient_step(c, τ, ν, x, ys, ε)

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
        τ = initial_stepsize / (1 + √(k / warmup_phase))
        gradient_step!(c, ṽ, τ, ṽ, ν, x, ys, Δv, ε)

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

# initial gradient step (unregularized subgradient)
function gradient_step(c, τ, ν::AbstractVector, x, ys::AbstractVector, ::Nothing)
    tmp = @. c((x,), ys)
    i = argmin(tmp)
    z = τ .* ν
    z[i] -= τ
    return z
end

# initial gradient step (regularized gradient)
function gradient_step(c, τ, ν::AbstractVector, x, ys::AbstractVector, ε::Real)
    tmp = @. - c((x,), ys) / ε
    StatsFuns.softmax!(tmp)
    z = @. τ * (ν - tmp)
    return z
end

# gradient step (unregularized subgradient)
function gradient_step!(
    c,
    z::AbstractVector,
    τ,
    v::AbstractVector,
    ν::AbstractVector,
    x,
    ys::AbstractVector,
    tmp::AbstractVector,
    ::Nothing,
)
    @. tmp = c((x,), ys) - v
    @. z += τ * ν
    z[argmin(tmp)] -= τ
    return z
end

# gradient step (regularized gradient)
function gradient_step!(
    c,
    z::AbstractVector,
    τ,
    v::AbstractVector,
    ν::AbstractVector,
    x,
    ys::AbstractVector,
    tmp::AbstractVector,
    ε::Real,
)
    @. tmp = (v - c((x,), ys)) / ε
    StatsFuns.softmax!(tmp)
    @. z += τ * (ν - tmp)
    return z
end