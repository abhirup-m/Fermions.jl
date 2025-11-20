function HubbardModel(
        sites::Int64,
        hubbardU::Float64,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    for i in 1:(sites-1)
        push!(hamiltonian, ("+-", [2*i-1, 2*i+1], -1.))
        push!(hamiltonian, ("+-", [2*i+1, 2*i-1], -1.))
        push!(hamiltonian, ("+-", [2*i, 2*i+2], -1.))
        push!(hamiltonian, ("+-", [2*i+2, 2*i], -1.))
        push!(hamiltonian, ("nn", [2*i-1, 2*i], hubbardU))
    end
    push!(hamiltonian, ("nn", [2*sites-1, 2*sites], hubbardU))
    return hamiltonian
end

sites = 5
hubbardU = 0.5
hamiltonian = HubbardModel(sites, hubbardU)
hamiltonianFamily = MinceHamiltonian(hamiltonian, 4:2:2*sites)
probes = Dict("create" => [("+", [2 * sites - 1], 1.)], "destroy" => [("-", [2 * sites - 1], 1.)])
results = IterDiag(hamiltonianFamily, 
                   100; 
                   correlationDefDict=Dict("T12" => [("+-", [1, 3], 1.)]),
                   vneDefDict=Dict("SEE1" => [1, 2]),
                   mutInfoDefDict=Dict("I2_12" => ([1, 2], [3, 4])),
                   specFuncDefDict=Dict("A12" => probes)
                  )
println(results["A12"])
