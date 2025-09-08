#### Iterative diagonalisation solution of the 1D Hubbard model ####
using Plots, Measures, Fermions


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

sites = 30
maxSize = 3000
hubbardHamiltonian = Hubbard1D(2.0, sites)
hamiltonianFamily = MinceHamiltonian(hubbardHamiltonian, collect(2:sites))
savePaths, resultsDict = IterDiag(hamiltonianFamily, maxSize; correlationDefDict=Dict("nn-$(i)-$(i+1)"=> [("nn", [i, i+1], 1.0)] for i in 1:(sites-1)))
display(resultsDict)
