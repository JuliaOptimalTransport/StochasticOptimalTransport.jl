using StochasticOptimalTransport
using Documenter

makedocs(;
    modules=[StochasticOptimalTransport],
    authors="David Widmann <david.widmann@it.uu.se>",
    repo="https://github.com/devmotion/StochasticOptimalTransport.jl/blob/{commit}{path}#L{line}",
    sitename="StochasticOptimalTransport.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://devmotion.github.io/StochasticOptimalTransport.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/devmotion/StochasticOptimalTransport.jl",
)
