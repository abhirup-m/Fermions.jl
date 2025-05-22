using ProgressMeter, LinearAlgebra, Plots

include("../src/base.jl")
include("../src/correlations.jl")

U = 1.0
numSites = 10
basis = BasisStates(numSites)
probes = Dict("create" => OperatorMatrix(basis, [("+", [1], 1.0)]), "destroy" => OperatorMatrix(basis, [("-", [1], 1.0)]))
H = [("n", [i], cos(2π*i/numSites)) for i in 1:numSites]
#=H0 = [("+-", [1, 3], 1.0), ("+-", [3, 1], 1.0), ("+-", [2, 4], 1.0), ("+-", [4, 2], 1.0)] #, ("nn", [1, 2], U), ("nn", [3, 4], U)]=#
E, X = eigen(Hermitian(OperatorMatrix(basis, H)))
sc0 = SpectralCoefficients([collect(Xi) for Xi in eachcol(X)], E, probes);

for i in 1:numSites
    append!(H, [("+-", [i, 1], U)])
end
E, X = eigen(Hermitian(OperatorMatrix(basis, H)))
sc = SpectralCoefficients([collect(Xi) for Xi in eachcol(X)], E, probes);

freqValues = collect(-2:0.01:2)
SE = SelfEnergy(sc, sc0, freqValues)
p1 = plot(freqValues, real(SE))
p2 = scatter(freqValues, imag(SE))
display(plot(p1, p2))

