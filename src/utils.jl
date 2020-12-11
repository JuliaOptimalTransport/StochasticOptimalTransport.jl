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

