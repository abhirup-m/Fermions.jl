using Fermions, Plots
include("../src/eigen.jl")

t1 = 1.
t2 = 2.
halfSizeX = 16
halfSizeY = 5
basis = BasisStates1p((2 * halfSizeX) * (2 * halfSizeY))
hamiltonian = SSHModel((t1, t1), (t1, t2), (halfSizeX, halfSizeY); joinEnds=(true, false))
translationOperatorX = Tuple{String, Vector{Int64}, Float64}[]
for i in 1:2*halfSizeY # y
    xIndices = (i - 1) * 2 * halfSizeX .+ (1:2*halfSizeX)
    xIndicesTranslated = circshift(xIndices, 1)
    for (j, jprime) in zip(xIndices, xIndicesTranslated)
        push!(translationOperatorX, ("+-", [jprime, j], 1.))
    end
end
bandStructure = BandStructure(hamiltonian, basis, translationOperatorX; dimensionSymmetry=2 * halfSizeX)
p = plot(legend=false)
for (kx, E) in bandStructure
   scatter!(p, repeat([kx], length(E)), E)
end
display(p)
