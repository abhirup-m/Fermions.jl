using Distributed

include("../src/base.jl")
include("../src/constants.jl")
include("../src/modelHamiltonians.jl")
include("../src/iterDiag.jl")

numBathSites = 8
kondoJ = [Dict{NTuple{2, Int64}, Float64}((i,j)=>rand() for i in 1:numBathSites for j in 1:numBathSites)]

hamiltonian = KondoModel(numBathSites, 1., kondoJ)
hamiltonianFamily = MinceHamiltonian(hamiltonian, 4:2:(2 + 2*numBathSites) |> collect)
@time output = IterDiag(hamiltonianFamily, 2500, silent=false);
