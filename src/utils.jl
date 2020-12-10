"""
    ctransform(c, v, x, ys, ν, ε)

Compute the c-transform
```math
v^{c,ε}(x) = \\begin{cases}
\\- ε \\log\\bigg(\\sum_{i=1}^n \\exp{\\Big(\\frac{v[i] - c(x, ys[i])}{ε}\\Big)} ν[i]\\bigg) & \\text{if } ε > 0,\\\\
\\min_{i} c(x, y[i]) - v[i] & \\text{otherwise}.
\\end{cases}
```
"""
function ctransform(
    c,
    v::AbstractVector,
    x,
    ys::AbstractVector,
    ν::AbstractVector,
    ::Nothing,
)
    return minimum(c(x, yᵢ) - vᵢ for (vᵢ, yᵢ) in zip(v, ys))
end
function ctransform(
    c,
    v::AbstractVector,
    x,
    ys::AbstractVector,
    ν::AbstractVector,
    ε::Real,
)
    t = StatsFuns.logsumexp(
        (vᵢ - c(x, yᵢ)) / ε + log(νᵢ) for (vᵢ, yᵢ, νᵢ) in zip(v, ys, ν)
    )
    return - ε * (t + 1)
end

