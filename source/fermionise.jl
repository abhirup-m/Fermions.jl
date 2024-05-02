using Combinatorics
using ProgressMeter
using LinearAlgebra
using Distributed

"""
Returns a set of basis states in the form
[[0,0,...,0],
 [0,0,...,1],
       .
       .
       .
 [1,1,...,1]
],
where each inner vector represents a configuration of the electronic fock states.
Optionally accepts a parameter totOccupancy that restricts the basis states to only
those with the specified total occupancy, and a parameter totSpin that does the same
but for a specified total magnetisation.
"""
function BasisStates(numLevels::Int64; totOccupancy::Union{Vector{Int64},Nothing}=nothing)

    # if totOccupancy is not set, all occupancies from 0 to N are allowed,
    # otherwise use only the provided totOccupancy
    if isnothing(totOccupancy)
        allowedOccupancies = 0:numLevels
    else
        @assert all(0 .<= totOccupancy .<= numLevels)
        allowedOccupancies = totOccupancy
    end

    # create dictionary to store configurations classified by their total occupancies.
    # For eg, basisStates = {0: [[0, 0]], 1: [[1, 0], [0, 1]], 2: [[1, 1]]}}
    basisStates = Dict{Tuple{Int64,Int64},Vector{BitArray}}()

    # generate all possible configs, by running integers from 0 to 2^N-1, and converting
    # them to binary forms
    allConfigs = digits.(0:2^numLevels-1, base=2, pad=numLevels) |> reverse

    # loop over allowed occupancies, and store the configs in the dict
    # classified by their total occupancies. We will also store the subdimensions
    # while we are creating the dictionary.

    for totOcc in allowedOccupancies
        totoccMatchingConfigs = allConfigs[sum.(allConfigs).==totOcc]
        for config in totoccMatchingConfigs
            sigmaz = sum(config[1:2:end]) - sum(config[2:2:end])
            if !((totOcc, sigmaz) in keys(basisStates))
                basisStates[(totOcc, sigmaz)] = BitArray[]
            end
            push!(basisStates[(totOcc, sigmaz)], config)
        end
    end

    return basisStates
end

@everywhere function TransformBit(qubit::Bool, operator::Char)
    if operator == 'n'
        return qubit, qubit
    elseif operator == 'h'
        return qubit, 1 - qubit
    elseif (operator == '+' && qubit == 0) || (operator == '-' && qubit == 1)
        return 1 - qubit, 1
    else
        return qubit, 0
    end
end


@everywhere function applyOperatorOnState(stateDict::Dict{BitVector,Float64}, operatorList::Dict{Tuple{String,Vector{Int64}},Float64})
    # define a dictionary for the final state obtained after applying 
    # the operators on all basis states
    completeOutputState = Dict{BitVector,Float64}()

    # loop over all operator tuples within operatorList
    for ((opType, opMembers), opStrength) in pairs(operatorList)

        holeProjectionSites = [m for (i, m) in enumerate(reverse(opMembers)) if reverse(opType)[i] in ['+', 'h'] && (i == 1 || m ∉ reverse(opMembers)[1:i-1])]
        particleProjectionSites = [m for (i, m) in enumerate(reverse(opMembers)) if reverse(opType)[i] in ['-', 'n'] && (i == 1 || m ∉ reverse(opMembers)[1:i-1])]

        for (state, coefficient) in stateDict

            # from the outset, ignore states that are creating at occupied bits, destroying 
            # at empty bits, applying n at empty bits or checking 1-n at occupied bits.
            if 1 in state[holeProjectionSites] || 0 in state[particleProjectionSites]
                continue
            end

            newCoefficient = coefficient
            newState = copy(state)

            # for each basis state, obtain a modified state after applying the operator tuple
            for (siteIndex, operator) in zip(reverse(opMembers), reverse(opType))
                newQubit, factor = TransformBit(newState[siteIndex], operator)
                if factor == 0
                    newCoefficient = 0
                    break
                end
                # calculate the fermionic exchange sign by counting the number of
                # occupied states the operator has to "hop" over
                exchangeSign = operator in ['+', '-'] ? (-1)^sum(newState[1:siteIndex-1]) : 1
                @inbounds newState[siteIndex] = newQubit
                newCoefficient *= exchangeSign * factor
            end

            # if the coefficient of the transformation is non-zero, add this to the output state
            # dictionary for the operator tuple we are currently looping over
            if newCoefficient != 0
                if newState in keys(completeOutputState)
                    completeOutputState[newState] += opStrength * newCoefficient
                else
                    completeOutputState[newState] = opStrength * newCoefficient
                end
            end
        end
    end
    return completeOutputState
end


@everywhere function generalOperatorMatrix(basisStates::Dict{Tuple{Int64,Int64},Vector{BitArray}}, operatorList::Dict{Tuple{String,Vector{Int64}},Float64}; tolerance::Float64=0.0)
    operatorFullMatrix = Dict(key => zeros(length(value), length(value)) for (key, value) in basisStates)
    Threads.@threads for (key, bstates) in collect(basisStates)
        for (index, state) in collect(enumerate(bstates))
            for (nState, coeff) in applyOperatorOnState(Dict(state => 1.0), operatorList)
                operatorFullMatrix[key][index, bstates.==[nState]] .= tolerance == 0 ? coeff : round(coeff, digits=trunc(Int, -log10(tolerance)))
            end
        end
    end
    return operatorFullMatrix
end


function generalOperatorMatrix(basisStates::Dict{Tuple{Int64,Int64},Vector{BitArray}}, operatorList::Dict{Tuple{String,Vector{Int64}},Vector{Float64}}; tolerance::Float64=0.0)
    operatorFullMatrices = [Dict{Tuple{Int64,Int64},Matrix{Float64}}() for _ in length(collect(values(operatorList))[1])]
    for (key, couplingSet) in operatorList
        subMatrix = generalOperatorMatrix(basisStates, Dict(key => 1.0); tolerance=tolerance)
        operatorFullMatrices = [merge(+, dict, Dict(k => v .* coupling for (k,v) in subMatrix)) for (dict, coupling) in zip(operatorFullMatrices, couplingSet)]
    end
    return operatorFullMatrices
end


function getSpectrum(hamiltonian::Dict{Tuple{Int64,Int64},Matrix{Float64}})
    eigvals = Dict{Tuple{Int64,Int64},Vector{Float64}}()
    eigvecs = Dict{Tuple{Int64,Int64},Vector{Vector{Float64}}}()
    for (index, matrix) in collect(hamiltonian)
        F = eigen(Hermitian(matrix))
        eigvals[index] = sort(F.values)
        eigvecs[index] = [F.vectors[:, i] for i in sortperm(F.values)]
    end
    return eigvals, eigvecs
end
