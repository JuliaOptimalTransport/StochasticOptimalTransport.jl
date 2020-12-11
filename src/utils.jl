struct DiscreteMeasure{X<:AbstractVector,P<:AbstractVector}
    xs::X
    ps::P

    function DiscreteMeasure{X,P}(xs::X, ps::P) where {X,P}
        length(xs) == length(ps) ||
            error("length of support `xs` and probabilities `ps` must be equal")
        new{X,P}(xs, ps)
    end
end

"""
    DiscreteMeasure(xs::AbstractVector, ps::AbstractVector)

Construct a discrete measure with support `xs` and corresponding weights `ps`.
"""
function DiscreteMeasure(xs::AbstractVector, ps::AbstractVector)
    return DiscreteMeasure{typeof(xs),typeof(ps)}(xs, ps)
end

Base.length(μ::DiscreteMeasure) = length(μ.xs)

# initial gradient step (unregularized subgradient)
function gradient_step(c, τ, ν::DiscreteMeasure, x, ::Nothing)
    tmp = @. c((x,), ν.xs)
    i = argmin(tmp)
    z = τ .* ν.ps
    z[i] -= τ
    return z
end

# initial gradient step (regularized gradient)
function gradient_step(c, τ, ν::DiscreteMeasure, x, ε::Real)
    tmp = @. - c((x,), ν.xs) / ε
    StatsFuns.softmax!(tmp)
    z = @. τ * (ν.ps - tmp)
    return z
end

# gradient step (unregularized subgradient)
function gradient_step!(
    c,
    z::AbstractVector,
    τ,
    v::AbstractVector,
    ν::DiscreteMeasure,
    x,
    tmp::AbstractVector,
    ::Nothing;
    reset::Bool = false,
)
    @. tmp = c((x,), ν.xs) - v
    if reset
        @. z = τ * ν.ps
    else
        @. z += τ * ν.ps
    end
    z[argmin(tmp)] -= τ
    return z
end

# gradient step (regularized gradient)
function gradient_step!(
    c,
    z::AbstractVector,
    τ,
    v::AbstractVector,
    ν::DiscreteMeasure,
    x,
    tmp::AbstractVector,
    ε::Real;
    reset::Bool = false,
)
    @. tmp = (v - c((x,), ν.xs)) / ε
    StatsFuns.softmax!(tmp)
    if reset
        @. z = τ * (ν.ps - tmp)
    else
        @. z += τ * (ν.ps - tmp)
    end
    return z
end

@doc raw"""
    ctransform(c, v, x, ν, ε)

Compute the c-transform
```math
v^{c,ε}(x) = \begin{cases}
- ε \log\bigg(\int \exp{\Big(\frac{v_y - c(x, y)}{ε}\Big)} \, ν(\mathrm{d}y)\bigg) & \text{if } ε > 0,\\
\min_{y} c(x, y) - v_y & \text{otherwise}.
\end{cases}
```
"""
function ctransform(
    c,
    v::AbstractVector,
    x,
    ν::DiscreteMeasure,
    ::Nothing,
)
    return minimum(c(x, yᵢ) - vᵢ for (vᵢ, yᵢ) in zip(v, ν.xs))
end
function ctransform(
    c,
    v::AbstractVector,
    x,
    ν::DiscreteMeasure,
    ε::Real,
)
    t = StatsFuns.logsumexp(
        (vᵢ - c(x, yᵢ)) / ε + log(νᵢ) for (vᵢ, yᵢ, νᵢ) in zip(v, ν.xs, ν.ps)
    )
    return - ε * (t + 1)
end

