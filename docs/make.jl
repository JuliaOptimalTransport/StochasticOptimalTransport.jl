using Documenter

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

using StochasticOptimalTransport

DocMeta.setdocmeta!(
    StochasticOptimalTransport,
    :DocTestSetup,
    :(using StochasticOptimalTransport);
    recursive=true,
)

makedocs(;
    modules=[StochasticOptimalTransport],
    authors="David Widmann <david.widmann@it.uu.se>",
    repo="https://github.com/JuliaOptimalTransport/StochasticOptimalTransport.jl/blob/{commit}{path}#L{line}",
    sitename="StochasticOptimalTransport.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliaoptimaltransport.github.io/StochasticOptimalTransport.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/JuliaOptimalTransport/StochasticOptimalTransport.jl",
    push_preview=true,
    devbranch="main",
)
