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
        numLevels::Int64, 
        totOccReq::Vector{Int64},
        magzReq::Vector{Int64},
        localCriteria::Function
    )
    basis = Dict{BitVector,Float64}[]
    config = falses(numLevels)
    for decimalNum in 0:2^numLevels-1
        for bit in 1:numLevels
            config[bit] = (decimalNum >> (numLevels - bit)) & 1 == 1
        end
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
        numLevels::Int64;
        totOccReq::Union{Vector{Int64}, Int64}=Int64[],
        magzReq::Union{Vector{Int64}, Int64}=Int64[],
        localCriteria::Function=x -> true
    )
    if typeof(totOccReq) == Int64
        totOccReq = [totOccReq]
    end
    if typeof(magzReq) == Int64
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
        numLevels::Int64, 
    )
    config = BitVector(fill(0, numLevels))
    config[1] = 1
    basis = Dict{BitVector,Float64}[]
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
@inline function TransformBit(qubit::Bool, operator::Char)
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
@inline function ApplyOperatorChunk(
        opType::String,
        opMembers::Vector{Int64},
        opStrength::Float64,
        incomingState::Dict{BitVector,Float64};
        tolerance::Float64=0.,
    )
    outgoingState = Dict{BitVector,Float64}()
    outgoingBasisState = similar(first(keys(incomingState)))
    for (incomingBasisState, coefficient) in incomingState

        if allunique(opMembers)
            skip = false
            for (t, m) in zip(opType, opMembers)
                if t in ('+', 'h') && incomingBasisState[m] == 1
                    skip = true
                    break
                elseif (t in ('-', 'n')) && incomingBasisState[m] == 0
                    skip = true
                    break
                end
            end
            if skip
                continue
            end
        end
        newCoefficient = coefficient
        copyto!(outgoingBasisState, incomingBasisState)

        # for each basis state, obtain a modified state after applying the operator tuple
        for (siteIndex, operator) in zip(reverse(opMembers), reverse(opType))
            newQubit, factor = TransformBit(outgoingBasisState[siteIndex], operator)
            if factor == 0
                newCoefficient = 0
                break
            end
            # calculate the fermionic exchange sign by counting the number of
            # occupied states the operator has to "hop" over
            @inbounds @views exchangeSign = ifelse(operator in ['+', '-'], (-1)^sum(outgoingBasisState[1:siteIndex-1]), 1)

            @inbounds outgoingBasisState[siteIndex] = newQubit
            newCoefficient *= exchangeSign * factor
        end
        if abs(newCoefficient) > tolerance
            if haskey(outgoingState, outgoingBasisState)
                @inbounds outgoingState[copy(outgoingBasisState)] += opStrength * newCoefficient
            else
                @inbounds outgoingState[copy(outgoingBasisState)] = opStrength * newCoefficient
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
        operator::Vector{Tuple{String,Vector{Int64},Float64}},
        incomingState::Dict{BitVector,Float64};
        tolerance::Float64=0.,
    )
    @assert !isempty(operator)
    @assert maximum([maximum(positions) for (_, positions, _) in operator]) ≤ length.(keys(incomingState))[1]

    outgoingState = empty(incomingState)
    for (opType, opMembers, opStrength) in operator
        if opStrength ≠ 0
            mergewith!(+, outgoingState, ApplyOperatorChunk(opType, opMembers, opStrength, copy(incomingState); tolerance=tolerance))
        end
    end
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
        basisStates::Vector{Dict{BitVector,Float64}},
        operator::Vector{Tuple{String,Vector{Int64},Float64}};
        tolerance::Float64=0.,
    )
    operatorMatrix = zeros(length(basisStates), length(basisStates))
    newStates = [ApplyOperator(operator, incomingState; tolerance=tolerance)
                        for incomingState in basisStates]
    for incomingIndex in findall(!isempty, newStates)
        for outgoingIndex in eachindex(basisStates)
            @inbounds operatorMatrix[outgoingIndex, incomingIndex] = StateOverlap(basisStates[outgoingIndex], newStates[incomingIndex])
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
@inline function StateOverlap(
        state1::Dict{BitVector,Float64}, 
        state2::Dict{BitVector,Float64}
    )
    overlap = 0.
    #=keys2 = keys(state2)=#
    for (key, val) in state1
        if haskey(state2, key)
            @inbounds overlap += val * state2[key]
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
        state::Dict{BitVector,Float64}, 
        basisStates::Vector{Dict{BitVector,Float64}},
    )
    coefficients = zeros(length(basisStates))
    for (index, bstate) in enumerate(basisStates)
        @inbounds coefficients[index] = StateOverlap(bstate, state)
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
        state::Dict{BitVector, Float64}, 
        symmetries::Vector{Char},
    )
    pivotKey = nothing
    pivotVal = -Inf
    for (k,v) in state
        if abs(v) > pivotVal
            pivotVal = abs(v)
            pivotKey = k
        end
    end
    bstate = pivotKey
    #=bstate = sort(collect(keys(state)), by=k->abs(state[k]))[end]=#
    totOcc = sum(bstate)
    @views magz = sum(bstate[1:2:end]) - sum(bstate[2:2:end])
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
@inline function RoundTo(
        val::Union{Int64, Float64}, 
        tolerance::Float64
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
        state::Dict{BitVector, Float64},
        permutation::Vector{Int64},
    )
    newState = Dict{BitVector, Float64}()
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
        operator::String,
        members::Vector{Int64};
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
        operator::Vector{Tuple{String,Vector{Int64},Float64}};
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

Returns the vacuum (completely unoccupied) state for the given basis

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
        basisStates::Vector{Dict{BitVector,Float64}};
        tolerance::Float64=0.,
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
        operatorLeft::Vector{Tuple{String,Vector{Int64},Float64}},
        operatorRight::Vector{Tuple{String,Vector{Int64},Float64}},
        basisStates::Vector{Dict{BitVector,Float64}};
        tolerance=0.,
    )
    matrixLeft = OperatorMatrix(basisStates, operatorLeft)
    matrixRight = OperatorMatrix(basisStates, operatorRight)
    commutator = matrixLeft * matrixRight - matrixRight * matrixLeft
    if commutator .|> abs |> maximum ≤ tolerance
        return true
    else
        return false
    end
end
export DoesCommute


"""
    Product(operatorLeft, operatorRight)

Multiplies two operators, in the abstract basis-indepdendent form.

# Examples
```jldoctest
julia> Product([("+-",[1,2],1.0)], [("-+", [1,2], 1.0)])
1-element Vector{Tuple{String, Vector{Int64}, Float64}}:
 ("nh", [1, 2], -1.0)

julia> Product([("+-",[1,2],1.0)], [("n+", [1,2], 1.0)])
Tuple{String, Vector{Int64}, Float64}[]
 ```
"""
function Product(
        operatorLeft::Vector{Tuple{String,Vector{Int64},Float64}},
        operatorRight::Vector{Tuple{String,Vector{Int64},Float64}},
    )
    prodOperator = Tuple{String,Vector{Int64},Float64}[]
    for (type1, inds1, strength1) in operatorLeft
        for (type2, inds2, strength2) in operatorRight
            prodTerm = [collect(type1*type2), vcat(inds1, inds2), strength1 * strength2]
            if !isdisjoint(inds1, inds2)
                prodTable = Dict(
                                 'n' => Dict('n' => 'n', 'h' => nothing, '+' => '+', '-' => nothing),
                                 'h' => Dict('h' => 'h', 'n' => nothing, '-' => '-', '+' => nothing),
                                 '+' => Dict('-' => 'n', '+' => nothing, 'h' => '+', 'n' => nothing),
                                 '-' => Dict('+' => 'h', '-' => nothing, 'n' => '-', 'h' => nothing),
                                )
                #=println(prodTerm)=#
                for (pos1, i) in filter(p -> p[2] ∈ inds2, collect(enumerate(inds1)))
                    frontOffSet = length(inds1) + length(inds2) - length(prodTerm[1])
                    pos2 = findfirst(==(i), inds2)
                    if type1[pos1] ∈ "nh"
                        accumulatedSign = 1
                    else
                        accumulatedSign = (-1)^count(∈("+-"), prodTerm[1][(pos1 - frontOffSet + 1):(end-(length(inds2) - pos2)-1)])
                    end
                    #=println(accumulatedSign, prodTerm[1][(pos1 - frontOffSet + 1):(end-(length(inds2) - pos2)-1)])=#
                    newType = prodTable[type1[pos1]][type2[pos2]]
                    if isnothing(newType)
                        prodTerm[3] = 0.
                        break
                    end
                    prodTerm[1][end-(length(inds2) - pos2)] = newType
                    deleteat!(prodTerm[1], pos1 - frontOffSet)
                    deleteat!(prodTerm[2], pos1 - frontOffSet)
                    prodTerm[3] *= accumulatedSign
                    #=println(prodTerm)=#
                end
            end
            push!(prodOperator, (join(prodTerm[1]), prodTerm[2], prodTerm[3]))
        end
    end
    return prodOperator
end
export Product
