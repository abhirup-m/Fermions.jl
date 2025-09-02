"""
    BasisStates(numLevels, totOccReq, magzReq, localCriteria)

Creates a set of basis states for the `numLevels` Fock states, having the
required total occupancy and magnetization, and subject to any criteria
imposed locally.

# Examples
```jldoctest
julia> BasisStates(2, [1], [0], x->true)
Dict{BitVector, Float64}[]
julia> BasisStates(3, [1], [-1], x->true)
1-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 1, 0] => 1.0)
julia> BasisStates(4, [2], [0], x->true)
4-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 0, 1, 1] => 1.0)
 Dict([0, 1, 1, 0] => 1.0)
 Dict([1, 0, 0, 1] => 1.0)
 Dict([1, 1, 0, 0] => 1.0)
julia> BasisStates(4, [2], [0], x->sum(x[1:2])==1)
2-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 1, 1, 0] => 1.0)
 Dict([1, 0, 0, 1] => 1.0)
```
"""
function BasisStates(
        numLevels,
        totOccReq,
        magzReq,
        localCriteria
    )
    @assert !isempty(totOccReq) && !isempty(magzReq)
    basis = Dict{BitVector, ComplexF64}[]
    for decimalNum in 0:2^numLevels-1
        config = digits(decimalNum, base=2, pad=numLevels) |> reverse
        if !isempty(totOccReq)
            totOcc = sum(config)
            if totOcc ∉ totOccReq
                continue
            end
        end
        if !isempty(magzReq)
            magz = sum(config[1:2:end]) - sum(config[2:2:end])
            if magz ∉ magzReq
                continue
            end
        end
        if localCriteria(config)
            push!(basis, Dict(BitVector(config) => 1.0))
        end
    end
    return basis
end
export BasisStates


"""
    BasisStates(numLevels; totOccReq, magzReq, localCriteria)

Extends BasisStates() by making the last three arguments optional.
When skipped, these arguments are filled with all possible values
and then passed to BasisStates().

# Examples
```jldoctest
julia> BasisStates(2)
4-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 0] => 1.0)
 Dict([0, 1] => 1.0)
 Dict([1, 0] => 1.0)
 Dict([1, 1] => 1.0)

julia> BasisStates(2; totOccReq=1)
2-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 1] => 1.0)
 Dict([1, 0] => 1.0)

julia> BasisStates(2; magzReq=0)
2-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 0] => 1.0)
 Dict([1, 1] => 1.0)
```
"""
function BasisStates(
        numLevels;
        totOccReq=nothing,
        magzReq=nothing,
        localCriteria=x -> true
    )
    if isnothing(totOccReq)
        totOccReq = collect(0:numLevels)
    elseif typeof(totOccReq) == Int64
        totOccReq = [totOccReq]
    end
    if isnothing(magzReq)
        magzReq = collect(-div(numLevels, 2):numLevels - div(numLevels, 2))
    elseif typeof(magzReq) == Int64
        magzReq = [magzReq]
    end
    return BasisStates(numLevels, totOccReq, magzReq, localCriteria)
end
export BasisStates


"""
    BasisStates(numLevels)

Specialises BasisStates() by restricting to the 1-particle case;
returns all 1-particle basis states for the given number of levels.
Equivalent to BasisStates(numLevels; totOccReq=1), but much much faster.

# Examples
```jldoctest
julia> BasisStates1p(10)
10-element Vector{Dict{BitVector, Float64}}:
 Dict([1, 0, 0, 0, 0, 0, 0, 0, 0, 0] => 1.0)
 Dict([0, 1, 0, 0, 0, 0, 0, 0, 0, 0] => 1.0)
 Dict([0, 0, 1, 0, 0, 0, 0, 0, 0, 0] => 1.0)
 Dict([0, 0, 0, 1, 0, 0, 0, 0, 0, 0] => 1.0)
 Dict([0, 0, 0, 0, 1, 0, 0, 0, 0, 0] => 1.0)
 Dict([0, 0, 0, 0, 0, 1, 0, 0, 0, 0] => 1.0)
 Dict([0, 0, 0, 0, 0, 0, 1, 0, 0, 0] => 1.0)
 Dict([0, 0, 0, 0, 0, 0, 0, 1, 0, 0] => 1.0)
 Dict([0, 0, 0, 0, 0, 0, 0, 0, 1, 0] => 1.0)
 Dict([0, 0, 0, 0, 0, 0, 0, 0, 0, 1] => 1.0)
```
"""
function BasisStates1p(
        numLevels,
    )
    config = BitVector(fill(0, numLevels))
    config[1] = 1
    basis = Dict{BitVector, ComplexF64}[]
    for shift in 1:numLevels
        push!(basis, Dict(copy(config) => 1.0))
        circshift!(config, 1)
    end
    return basis
end
export BasisStates1p


"""
    TransformBit(qubit, operator)

Apply the single qubit operator ('n', 'h', '+' or '-') on a single fock state.

# Examples
```jldoctest
julia> TransformBit(Bool(0), '+')
(1, 1)

julia> TransformBit(Bool(1), 'h')
(1, 0)

julia> TransformBit(Bool(1), '-')
(0, 1)
```
"""
function TransformBit(
        qubit,
        operator
    )
    @assert operator in ('n', 'h', '+', '-')
    if operator == 'n'
        return 0 + qubit, 0 + qubit
    elseif operator == 'h'
        return 0 + qubit, 1 - qubit
    elseif (operator == '+' && qubit == 0) || (operator == '-' && qubit == 1)
        return 1 - qubit, 1
    else
        return 0 + qubit, 0
    end
end
export TransformBit


"""
    ApplyOperatorChunk(opType, opMembers, opStrength, incomingState; tolerance)

Apply a single tensor product operator chunk (for eg., c^†_1 c_2 or n_1 c^†_3 c_4) 
on a general state and return the new state.

# Examples
```jldoctest
julia> state = Dict(Bool.([1, 0]) => 1.0, Bool.([0, 1]) => -0.5)
Dict{BitVector, Float64} with 2 entries:
  [1, 0] => 1.0
  [0, 1] => -0.5

julia> opType, opMembers, opStrength = ("+-", [1, 2], 0.1)
("+-", [1, 2], 0.1)

julia> ApplyOperatorChunk(opType, opMembers, opStrength, state)
Dict{BitVector, Float64} with 1 entry:
  [1, 0] => -0.05
```
"""
function ApplyOperatorChunk(
        opType,
        opMembers,
        opStrength,
        incomingState;
        tolerance=1e-16
    )
    outgoingState = Dict{BitVector, ComplexF64}()
    for i in eachindex(opMembers)[end:-1:2]
        if opMembers[i] ∈ opMembers[i+1:end]
            continue
        end
        if opType[i] == '+' || opType[i] == 'h'
            filter!(p -> p[1][opMembers[i]] == 0, incomingState)
        else
            filter!(p -> p[1][opMembers[i]] == 1, incomingState)
        end
    end
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
            exchangeSign = ifelse(operator in ['+', '-'], (-1)^sum(outgoingBasisState[1:siteIndex-1]), 1)

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
    return outgoingState
end
export ApplyOperatorChunk


"""
    ApplyOperator(operator, incomingState; tolerance)

Extends ApplyOperatorChunk() by applying a more general operator (consisting
of multiple operator chunks) on a general state.

# Examples
```jldoctest
julia> state = Dict(Bool.([1, 0]) => 1.0, Bool.([0, 1]) => -0.5)
Dict{BitVector, Float64} with 2 entries:
  [1, 0] => 1.0
  [0, 1] => -0.5

julia> operator = [("+-", [1, 2], 0.1), ("nh", [2, 1], 1.0)]
2-element Vector{Tuple{String, Vector{Int64}, Float64}}:
 ("+-", [1, 2], 0.1)
 ("nh", [2, 1], 1.0)

julia> ApplyOperator(operator, state)
Dict{BitVector, Float64} with 2 entries:
  [1, 0] => -0.05
  [0, 1] => -0.5
```
"""
function ApplyOperator(
        operator,
        incomingState;
        tolerance=1e-16
    )
    @assert !isempty(operator)
    @assert maximum([maximum(positions) for (_, positions, _) in operator]) ≤ length.(keys(incomingState))[1]

    return mergewith(+, fetch.([Threads.@spawn ApplyOperatorChunk(opType, opMembers, opStrength, copy(incomingState); tolerance=tolerance) 
                                for (opType, opMembers, opStrength) in operator])...)

    return outgoingState
end
export ApplyOperator


"""
    OperatorMatrix(basisStates, operator)

Return the matrix representation of the operator in the given basis.

# Examples
```jldoctest
julia> basis = BasisStates(2)
4-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 0] => 1.0)
 Dict([0, 1] => 1.0)
 Dict([1, 0] => 1.0)
 Dict([1, 1] => 1.0)

julia> operator = [("+-", [1, 2], 0.5), ("n", [2], -1.0)]
2-element Vector{Tuple{String, Vector{Int64}, Float64}}:
 ("+-", [1, 2], 0.5)
 ("n", [2], -1.0)

julia> OperatorMatrix(basis, operator)
4×4 Matrix{Float64}:
 0.0   0.0  0.0   0.0
 0.0  -1.0  0.0   0.0
 0.0   0.5  0.0   0.0
 0.0   0.0  0.0  -1.0
```
"""
function OperatorMatrix(
        basisStates,
        operator;
        tolerance=1e-16,
    )
    operatorMatrix = Matrix{ComplexF64}(zeros(length(basisStates), length(basisStates)))
    newStates = fetch.([Threads.@spawn ApplyOperator(operator, incomingState; tolerance=tolerance)
                        for incomingState in basisStates])
    Threads.@threads for incomingIndex in findall(!isempty, newStates)
        Threads.@threads for outgoingIndex in eachindex(basisStates)
            operatorMatrix[outgoingIndex, incomingIndex] = StateOverlap(basisStates[outgoingIndex], newStates[incomingIndex])
        end
    end
    return operatorMatrix
end
export OperatorMatrix


"""
    StateOverlap(state1, state2)

Compute the inner product ⟨state1|state2⟩.

# Examples
```jldoctest
julia> state1 = Dict(Bool.([1, 0]) => 1.0, Bool.([0, 1]) => -0.5)
Dict{BitVector, Float64} with 2 entries:
  [1, 0] => 1.0
  [0, 1] => -0.5

julia> state2 = Dict(Bool.([1, 1]) => 0.5, Bool.([0, 1]) => 0.5)
Dict{BitVector, Float64} with 2 entries:
  [1, 1] => 0.5
  [0, 1] => 0.5

julia> StateOverlap(state1, state2)
-0.25
```
"""
function StateOverlap(
        state1,
        state2
    )
    overlap = 0.
    keys2 = keys(state2)
    for (key, val) in state1
        if key ∈ keys2
            overlap += val * state2[key]
        end
    end
    return overlap
end
export StateOverlap


"""
    ExpandIntoBasis(state, basisStates)

Returns the basis decomposition of the provided state. That is, 
if the basis is {b_i} and the state is {b_1=>c_1, b_2=>c_2, ...},
the function returns [c_1, c_2, ...].
"""
function ExpandIntoBasis(
        state,
        basisStates,
    )
    coefficients = zeros(length(basisStates))
    for (index, bstate) in enumerate(basisStates)
        coefficients[index] = StateOverlap(bstate, state)
    end
    return coefficients
end
export ExpandIntoBasis


"""
    GetSector(state, symmetries)

Returns the symmetry sector of the provided state. If symmetries only
has 'N'('Z'), returns the total occupancy (total magnetization).
"""
function GetSector(
        state,
        symmetries,
    )
    bstate = sort(collect(keys(state)), by=k->abs(state[k]))[end]
    totOcc = sum(bstate)
    magz = sum(bstate[1:2:end]) - sum(bstate[2:2:end])
    if symmetries == ['N']
        return (totOcc,)
    elseif symmetries == ['Z']
        return (magz,)
    elseif symmetries == ['N', 'Z']
        return (totOcc, magz)
    else
        @assert false "Incorrect format of `symmetries`."
    end
end
export GetSector


"""
    RoundTo(val, tolerance)

Round the provided float to the specified tolerance.

# Examples
```jldoctest
julia> fermions.roundTo(1.122323, 1e-3)
1.122
```
"""
function RoundTo(
        val,
        tolerance
    )
    return round(val, digits=trunc(Int, -log10(tolerance)))
end
export RoundTo


"""
    PermuteSites(state, permutation)

Given a basis state |n_{ν1} n_{ν2} n_{ν3} … ⟩ corresponding to
a particular ordering choice [ν_1, ν_2, …], this function permutes
the basis into P(B) = [P(ν1), P(ν2),P(ν3) … ], and returns the 
basis state written in this new basis, accounting for any fermions 
signs arising from index permutations.

# Examples
```jldoctest
julia> state = Bool.([1, 1])
2-element BitVector:
 1
 1

julia> PermuteSites(state, [2, 1])
(Bool[1, 1], -1)
```
"""
function PermuteSites(
        state::BitVector,
        permutation::Vector{Int64},
    )
    @assert length(state) == length(permutation)
    sign = 1
    currentPerm = collect(1:length(state))
    while currentPerm ≠ permutation
        for index in eachindex(currentPerm[1:end-1])
            if findfirst(==(currentPerm[index]), permutation) > findfirst(==(currentPerm[index + 1]), permutation)
                currentPerm[index], currentPerm[index+1] = currentPerm[index+1], currentPerm[index]
                sign *= ifelse(state[index] == 1 && state[index+1] == 1, -1, 1)
                state[index], state[index+1] = state[index+1], state[index]
            end
        end
    end
    return state, sign
end
export PermuteSites


"""
    PermuteSites(state, permutation)

Given a state that is written in a particular basis
B = {|n_{ν1} n_{ν2} n_{ν3} … ⟩}, this function permutes
the basis into P(B) = {|n_{P(ν1)} n_{P(ν2)} n_{P(ν3)} … ⟩}
and returns the state written in this new basis, accounting
for any fermions signs arising from index permutations.

# Examples
```jldoctest
julia> state
Dict{BitVector, Float64} with 2 entries:
  [1, 1] => 0.5
  [0, 1] => 0.3

julia> PermuteSites(state, [2, 1])
Dict{BitVector, Float64} with 2 entries:
  [1, 1] => -0.5
  [1, 0] => 0.3
```
"""
function PermuteSites(
        state::Dict{BitVector, ComplexF64},
        permutation::Vector{Int64},
    )
    newState = Dict{BitVector, ComplexF64}()
    for (k,v) in state
        newBasisState, sign = PermuteSites(k, permutation)
        newState[newBasisState] = sign * v
    end
    return newState
end
export PermuteSites


"""
    Dagger(operator, members)

Native method to calculate hermitian conjugate of operator.

# Examples
```jldoctest
julia> Dagger("+--+", [1,4,3,2])
("-++-", [2, 3, 4, 1])
 ```
"""
function Dagger(
        operator,
        members,
    )
    newOpType = reverse(operator)
    newOpType = replace(newOpType, '+' => '-', '-' => '+')
    newMembers = reverse(members)
    return newOpType, newMembers
end
export Dagger


"""
    Dagger(operator, members)

Native method to calculate hermitian conjugate of operator.

# Examples
```jldoctest
julia> operator = [("++", [1, 2], 1.), ("+n", [3, 4], 2.)]
2-element Vector{Tuple{String, Vector{Int64}, Float64}}:
 ("++", [1, 2], 1.0)
 ("+n", [3, 4], 2.0)

julia> Dagger(operator)
2-element Vector{Tuple{String, Vector{Int64}, Float64}}:
 ("--", [2, 1], 1.0)
 ("n-", [4, 3], 2.0)
 ```
"""
function Dagger(
        operator,
    )
    for i in eachindex(operator)
        newOpType, newMembers = Dagger(operator[i][1:2]...)
        insert!(operator, i+1, (newOpType, newMembers, operator[i][3]'))
        deleteat!(operator, i)
    end
    return operator
end
export Dagger


"""
    VacuumState(basisStates)

Returns the vacuum (completely unoccupied) state for the 

# Examples
```jldoctest
julia> operator = [("++", [1, 2], 1.), ("+n", [3, 4], 2.)]
2-element Vector{Tuple{String, Vector{Int64}, Float64}}:
 ("++", [1, 2], 1.0)
 ("+n", [3, 4], 2.0)

julia> Dagger(operator)
2-element Vector{Tuple{String, Vector{Int64}, Float64}}:
 ("--", [2, 1], 1.0)
 ("n-", [4, 3], 2.0)
 ```
"""
function VacuumState(
        basisStates;
        tolerance=1e-14,
    )
    numSites = basisStates |> first |> keys |> first |> length
    numberOperators = [("n", [i], 1.) for i in 1:numSites]
    for state in basisStates
        if GenCorrelation(state, numberOperators) < tolerance
            return state
        end
    end
    @assert false "Vacuum state not found!"
end
export VacuumState


function DoesCommute(
        operatorLeft,
        operatorRight,
        basisStates;
        tolerance=1e-15,
    )
    matrixLeft = OperatorMatrix(basisStates, operatorLeft)
    matrixRight = OperatorMatrix(basisStates, operatorRight)
    commutator = matrixLeft * matrixRight - matrixRight * matrixLeft
    if commutator .|> abs |> maximum < tolerance
        return true
    else
        return false
    end
end
export DoesCommute
