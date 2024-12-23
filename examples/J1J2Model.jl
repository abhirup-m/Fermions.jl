using Fermions, Serialization, Plots
include("../src/modelHamiltonians.jl")

global maxSize = 2000

function GetGap(
        J1J2Ratio::Float64,
        totalSites::Int64,
    )
    partitions = collect(2:2:totalSites)
    hamiltonian = J1J2Model(J1J2Ratio, totalSites)
    hamiltonianFlow = MinceHamiltonian(hamiltonian, partitions)
    savePaths, results = IterDiag(hamiltonianFlow, maxSize; symmetries=['N'], silent=true)
    energies = deserialize(savePaths[end-1])["eigVals"]
    quantumNos = first.(deserialize(savePaths[end-1])["quantumNos"]) .- totalSites/2
    display(quantumNos[1:5])
    return energies[findfirst(==(0.), quantumNos)] - energies[findfirst(==(1.), quantumNos)]
end

totalSitesList = [80] # round.(Int, 10 .^ (1:0.3:2))
for (i, totalSites) in enumerate(totalSitesList)
    if totalSites % 2 ≠ 0
        totalSitesList[i] += 1
    end
end
display(totalSitesList)
J1J2RatioCrit = zeros(length(totalSitesList))
for (i, totalSites) in enumerate(totalSitesList)
    J1J2RatioBounds = [0., 4.]
    tolerance = 1e-2
    gapBounds = [GetGap(J1J2RatioBounds[1], totalSites), GetGap(J1J2RatioBounds[2], totalSites)]
    println(gapBounds)
    @assert gapBounds[1] * gapBounds[2] < 0
    while abs(J1J2RatioBounds[2] - J1J2RatioBounds[1]) > tolerance
        midPoint = 0.5 * sum(J1J2RatioBounds)
        gap = GetGap(midPoint, totalSites)
        if gap * gapBounds[1] > 0
            J1J2RatioBounds[1] = midPoint
            gapBounds[1] = gap
        else
            J1J2RatioBounds[2] = midPoint
            gapBounds[2] = gap
        end
    end
    J1J2RatioCrit[i] = sum(J1J2RatioBounds)/2
    println(J1J2RatioCrit[i])
end
plot(1 ./ totalSitesList, J1J2RatioCrit; xscale=:log10)
