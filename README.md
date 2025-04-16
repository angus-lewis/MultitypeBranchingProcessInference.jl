# MultitypeBranchingProcessInference

[![Build Status](https://github.com/angus-lewis/MultitypeBranchingProcessInference.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/angus-lewis/MultitypeBranchingProcessInference.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Installation

The easiest way to install is using Pkg.develop

julia> import Pkg

julia> Pkg.develop([
    (; url="https://github.com/angus-lewis/MultitypeBranchingProcessInference.jl"),
])

This will probably put the repo in ~/.julia/dev/. See Pkg.develop documentation for more.

You can then use using/import to use the package, or work on the package itself in ~/.julia/dev/.

Alternatively, git clone the repo then navigate to the repo so that you are in the MultitypeBranchingProcessInference.jl directory.

Then run 

    % julia --project=. "import Pkg; Pkg.build()"

To run the examples you may need to also cd into examples/ and run 

    %julia --project=. deps/build.jl