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


function roundTo(val, tolerance)
    return round(val, digits=trunc(Int, -log10(tolerance)))
end


function multiplyDict(dicts, multipliers)
    @assert length(dicts) == length(multipliers)
    return mergewith(+, fetch.([Threads.@spawn ((dict, m) -> Dict(k => v * m for (k, v) in dict))(dict, m) for (dict, m) in zip(dicts, multipliers) if m != 0])...)
end


function BasisStates(numLevels::Int64; totOccCriteria::Function=(x, N) -> true, magzCriteria::Function=x -> true, localCriteria::Function=x -> true)
    @assert numLevels > 0 && numLevels % 2 == 0
    return [Dict(BitVector(config) => 1.0) for config in digits.(0:2^numLevels-1, base=2, pad=numLevels) |> reverse
            if totOccCriteria(config, numLevels) && magzCriteria(config) && localCriteria(config)]
end


function TransformBit(qubit::Bool, operator::Char)
    @assert operator in ('n', 'h', '+', '-')
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


function ApplyOperator(operator::Vector{Tuple{String,Vector{Int64},Float64}}, incomingState::Dict{BitVector,Float64}; tolerance=1e-10)
    @assert maximum([maximum(positions) for (_, positions, _) in operator]) ≤ length.(keys(incomingState))[1]

    outgoingState = typeof(incomingState)()

    # loop over all operator tuples within operatorList
    for (opType, opMembers, opStrength) in operator
        for (incomingBasisState, coefficient) in incomingState
            newCoefficient = coefficient
            outgoingBasisState = copy(incomingBasisState)

            # for each basis state, obtain a modified state after applying the operator tuple
            for (siteIndex, operator) in zip(reverse(opMembers), reverse(opType))
                newQubit, factor = TransformBit(outgoingBasisState[siteIndex], operator)
                if factor == 0
                    newCoefficient = 0
                    break
                end
                # calculate the fermionic exchange sign by counting the number of
                # occupied states the operator has to "hop" over
                exchangeSign = ifelse(operator in ["+", "-"], (-1)^sum(outgoingBasisState[1:siteIndex]), 1)

                outgoingBasisState[siteIndex] = newQubit
                newCoefficient *= exchangeSign * factor
            end

            if abs(newCoefficient) > tolerance
                if haskey(outgoingState, outgoingBasisState)
                    outgoingState[outgoingBasisState] += opStrength * newCoefficient
                else
                    outgoingState[outgoingBasisState] = opStrength * newCoefficient
                end
            end
        end
    end

    return outgoingState
end


function OperatorMatrix(basisStates::Vector{Dict{BitVector,Float64}}, operator::Vector{Tuple{String,Vector{Int64},Float64}})
    operatorMatrix = zeros(length(basisStates), length(basisStates))

    for (incomingIndex, incomingState) in enumerate(basisStates)
        newState = ApplyOperator(operator, incomingState)

        for (outgoingIndex, outgoingState) in enumerate(basisStates)
            overlap = 0
            for k in keys(newState)
                if haskey(outgoingState, k)
                    overlap += newState[k] * outgoingState[k]
                end
            end
            operatorMatrix[outgoingIndex, incomingIndex] = overlap
        end
    end
    return operatorMatrix
end


function prettyPrint(state::BitVector)
    output = ""
    @assert length(state) % 2 == 0
    for i in 1:2:length(state)
        if state[[i, i + 1]] == [0, 0]
            output = output * "0"
        elseif state[[i, i + 1]] == [1, 0]
            output = output * "↑"
        elseif state[[i, i + 1]] == [0, 1]
            output = output * "↓"
        else
            output = output * "2"
        end
    end
    return output
end


function StateOverlap(state1::Dict{BitVector,Float64}, state2::Dict{BitVector,Float64})
    commonKeys = intersect(keys(state1), keys(state2))
    if isempty(commonKeys)
        return 0
    end
    overlap = sum(fetch.([Threads.@spawn (k -> state1[k] * state2[k])(key) for key in commonKeys]))
    return overlap
end
