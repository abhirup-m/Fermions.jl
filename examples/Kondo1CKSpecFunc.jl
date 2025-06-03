@everywhere using Fermions, ProgressMeter, Plots
@everywhere include("../src/modelHamiltonians.jl")
@everywhere include("../src/correlations.jl")

numSites = 30
dispersion1CK = [ 0 .* collect(range(-1, 1, length=numSites))]
dispersion2CK = [0 .* collect(range(-1, 1, length=div(numSites, 2))) for _ in 1:2]
J = 4.
kondoJ1CK = [Dict((i, i) => J for i in 1:numSites)]
kondoJ2CK = [Dict((i, i) => 0.5 for i in 1:div(numSites, 2)), Dict((i, i) => 0.5 for i in 1:div(numSites, 2))]

freqValues = collect(-2:0.01:2)
deltaOmega = (maximum(freqValues) - minimum(freqValues)) / (length(freqValues) - 1)

hamiltonian1CK = KondoModel(dispersion1CK, kondoJ1CK)
hamFamily1CK = MinceHamiltonian(hamiltonian1CK, collect(4:2:2*(numSites+1)))
savePaths, results, specFuncOperators = IterDiag(hamFamily1CK, 1000; maxMaxSize=2000, specFuncDefDict=Dict("imp" => [("+-+", [1,2,2 + 2*i], 1.0) for i in 1:numSites]))
specFunc = IterSpecFunc(savePaths, specFuncOperators["imp"], freqValues, 0.01)
specFunc ./= sum(specFunc) * deltaOmega
p = plot(freqValues, specFunc)
savefig(p, "1CK.pdf")

hamiltonian2CK = KondoModel(dispersion2CK, kondoJ2CK)
hamFamily2CK = MinceHamiltonian(hamiltonian2CK, collect(6:4:2*(numSites+1)))
savePaths, results, specFuncOperators = IterDiag(hamFamily2CK, 1000; maxMaxSize=2000, specFuncDefDict=Dict("imp" => [("+-+", [1,2,2+2*i], 1.0) for i in 1:div(numSites,2)]))
specFunc = IterSpecFunc(savePaths, specFuncOperators["imp"], freqValues, 0.01)
specFunc ./= sum(specFunc) * deltaOmega
p = plot(freqValues, specFunc)
savefig(p, "2CK.pdf")
