using ProgressMeter, LinearAlgebra, Plots

include("../src/base.jl")
include("../src/correlations.jl")

V = 0.1
Ed = -0.0
numSites = 12
basis = BasisStates(numSites)
probes = Dict("create" => OperatorMatrix(basis, [("+", [1], 1.0)]), "destroy" => OperatorMatrix(basis, [("-", [1], 1.0)]))
Ek = 10.0 .^ range(-3, stop=0, length=div(numSites-1, 2))
H = [("n", [2 * i], Ek[i]) for i in 1:div(numSites-1, 2)]
append!(H, [("n", [2 * i + 1], -Ek[i]) for i in 1:div(numSites-1, 2)])
append!(H, [("n", [1], Ed)])
E, X = eigen(Hermitian(OperatorMatrix(basis, H)))
println(E[1:10])
sc0 = SpectralCoefficients([collect(Xi) for Xi in eachcol(X)], E, probes);

for i in 1:div(numSites-1, 2)
    append!(H, [("+-", [1, 2 * i], Ek[i]^2), ("+-", [2 * i, 1], Ek[i]^2)])
    append!(H, [("+-", [1, 2 * i + 1], Ek[i]^2), ("+-", [2 * i + 1, 1], Ek[i]^2)])
end
E, X = eigen(Hermitian(OperatorMatrix(basis, H)))
println(E[1:10])
sc = SpectralCoefficients([collect(Xi) for Xi in eachcol(X)], E, probes);

freqValues = collect(-5:1e-3:5)
G, G0, SE = SelfEnergy(sc, sc0, freqValues; standDev=1e-1)
p1 = plot(freqValues, real(SE))
p2 = plot(freqValues, imag(SE))
p3 = plot(freqValues, -imag(G0))
p4 = plot(freqValues, -imag(G))
display(plot(p1, p2, p3, p4, size=(1200, 1000)))

