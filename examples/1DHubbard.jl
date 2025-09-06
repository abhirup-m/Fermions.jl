#### Iterative diagonalisation solution of the 1D Hubbard model ####
using Plots, Measures, LinearAlgebra
include("../src/base.jl")
include("../src/correlations.jl")
include("../src/constants.jl")
include("../src/iterDiag.jl")


function Hubbard1D(
        U::Float64,
        sites::Int64,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    for i in 1:(sites-1)
        push!(hamiltonian, ("nn", [i, i+1], U))
        push!(hamiltonian, ("+-", [i, i+1], 1.0))
        push!(hamiltonian, ("+-", [i+1, i], 1.0))
    end
    return hamiltonian
end

sites = 10
maxSize = 300
hubbardHamiltonian = Hubbard1D(2.0, sites)
hamiltonianFamily = MinceHamiltonian(hubbardHamiltonian, collect(2:sites))
results = IterDiag(hamiltonianFamily, maxSize; symmetries=['N'], correlationDefDict=Dict("nn-$(i)-$(i+1)"=> [("nn", [i, i+1], 1.0)] for i in 1:3:(sites-1)), specFuncDefDict=Dict("any" => [("+", [1], 1.0)]))
display(results)

sites = 5
maxSize = 300
hubbardHamiltonian = Hubbard1D(2.0, sites)
hamiltonianFamily = MinceHamiltonian(hubbardHamiltonian, [10])
println(length(hamiltonianFamily))
results = IterDiag(hamiltonianFamily, maxSize; symmetries=['N'], correlationDefDict=Dict("nn-$(i)-$(i+1)"=> [("nn", [i, i+1], 1.0)] for i in 1:3:(sites-1)), specFuncDefDict=Dict("any" => [("+", [1], 1.0)]))
display(results)
