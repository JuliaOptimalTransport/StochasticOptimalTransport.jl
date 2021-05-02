using StochasticOptimalTransport
using Distributions
using Documenter
using Random
using Test

Random.seed!(1234)

const SOT = StochasticOptimalTransport

@testset "StochasticOptimalTransport.jl" begin
    @testset "Utilities" begin
        include("utils.jl")
    end
    @testset "Discrete OT" begin
        include("discrete.jl")
    end
    @testset "Semi-discrete OT" begin
        include("semidiscrete.jl")
    end
    @testset "doctests" begin
        DocMeta.setdocmeta!(
            StochasticOptimalTransport,
            :DocTestSetup,
            :(using StochasticOptimalTransport);
            recursive=true,
        )
        doctest(StochasticOptimalTransport)
    end
end
