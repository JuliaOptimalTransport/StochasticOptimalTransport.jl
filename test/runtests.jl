using StochasticOptimalTransport

using Distributions

using Random
using Test

Random.seed!(1234)

const SOT = StochasticOptimalTransport

@testset "StochasticOptimalTransport.jl" begin
    @testset "Utilities" begin include("utils.jl") end
    @testset "Discrete OT" begin include("discrete.jl") end
    @testset "Semi-discrete OT" begin include("semidiscrete.jl") end
end
