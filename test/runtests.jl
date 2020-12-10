using StochasticOptimalTransport

using Distributions
using Test

const SOT = StochasticOptimalTransport

@testset "StochasticOptimalTransport.jl" begin
    @testset "Semi-discrete OT" begin include("semidiscrete.jl") end
end
