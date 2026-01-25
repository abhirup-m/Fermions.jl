######## TEST RUNS FOR BENCHMARKING ########
# Calculates time and memory allocation for
# some convoluted artificial Hamiltonians
####################################################

using BenchmarkTools, ProgressMeter, LinearAlgebra
include("../src/constants.jl")
include("../src/base.jl")
include("../src/eigen.jl")
include("../src/correlations.jl")
include("../src/eigenstateRG.jl")
include("../src/iterDiag.jl")

function Hamiltonian(num_sites, bandwidth)
    Ek_values = range(-bandwidth, stop=bandwidth, length=num_sites)
    Ek_values_spindegeneracy = repeat(Ek_values, inner=2)
    kinetic_energy = [("n", [i], Ek_values_spindegeneracy[i])  for i in 1:2*num_sites]
    hubbard_term = [("+-+-", [2 * i - 1, 2 * j, 2 * k - 1, 2 * l], rand()) for (i, j, k, l) in Iterators.product(1:num_sites, 1:num_sites, 1:num_sites, 1:num_sites)]
    return vcat(kinetic_energy, vec(hubbard_term), Dagger(vec(hubbard_term)))
end

#=@btime begin=#
    ########
    # 4.149 s (2438760 allocations: 1.22 GiB)
    # lowered to:  46.347 ms (1267488 allocations: 98.89 MiB)
    ########
#==#
#=    num_sites = 4=#
#=    bandwidth = 1=#
#=    hamiltonian = Hamiltonian(num_sites, bandwidth)=#
#==#
#=    basis = BasisStates(2 * num_sites)=#
#=    E, V = Spectrum(hamiltonian, basis)=#
#==#
#=    corrOperator = [("+-n", [i, j, k], 1.) for i in 1:2*num_sites for j in 1:2*num_sites for k in 1:2*num_sites]=#
#=    _ = GenCorrelation(V[1], corrOperator)=#
#==#
#=    mutInfoReducingIndices = (collect(1:4), collect(5:8))=#
#=    _ = MutInfo(V[1], mutInfoReducingIndices)=#
#=end=#

begin
    ########
    # 17.696 s (140438681 allocations: 7.30 GiB)
    # lowered to: 410.687 ms (3549953 allocations: 382.19 MiB)
    ########

    num_sites = 3
    bandwidth = 1
    hamiltonian = Hamiltonian(num_sites, bandwidth)
    basis = BasisStates(2 * num_sites)
    E, V = Spectrum(hamiltonian, basis)
    initState = V[1]
    numSteps = 10
    alphaValues = rand(numSteps)

    function unitaryOperatorFunction(alpha, num_entangled, sectors)
        IOMposition = 2 * num_entangled + 1
        unitaryTerms = Tuple{String,Vector{Int64},Float64}[]
        for i in 1:2:(2*num_entangled-1)
            push!(unitaryTerms, ("+-", [i, IOMposition], 1.))
            push!(unitaryTerms, ("+-", [IOMposition, i], 1.))
            push!(unitaryTerms, ("+-", [i+1, IOMposition+1], 1.))
            push!(unitaryTerms, ("+-", [IOMposition+1, i+1], 1.))
        end
        return unitaryTerms
    end
    function stateExpansionFunction(state, sectors)
        newstate = typeof(state)()
        for (k, v) in state
            newstate[vcat(k, [0, 0])] = v
            newstate[vcat(k, [1, 1])] = v
        end
        return newstate
    end
    _ = getWavefunctionRG(initState, alphaValues, numSteps, unitaryOperatorFunction, stateExpansionFunction, "ph")
end


#=@btime begin=#
    ########
    # 13.220 s (2328913 allocations: 28.15 GiB)
    # lowered to: 8.853 s (2335834 allocations: 20.93 GiB)
    ########
#=    num_sites = 6=#
#=    bandwidth = 1=#
#=    hamiltonian = Hamiltonian(num_sites, bandwidth)=#
#=    family = MinceHamiltonian(hamiltonian, collect(2:2:2*num_sites))=#
#=    corrOperator = Dict("$i-$j-$k" => [("+-n", [i, j, k], 1.)] for i in 1:2*num_sites for j in 1:2:2*num_sites for k in 1:2:2*num_sites)=#
#=    mutInfo = Dict("$i-$j" => ([i, i+1], [j, j+1]) for i in 1:2:2*num_sites for j in 1:2:2*num_sites)=#
#=    _ = IterDiag(family, 300; correlationDefDict=corrOperator, mutInfoDefDict=mutInfo)=#
#=end=#
