using LinearAlgebra

"""
    ClassifyBasis(basisStates)

Classify the set of basis states according to the total
occupancy and total magnetisation.

# Examples
```jldoctest
julia> basis = BasisStates(2)
4-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 0] => 1.0)
 Dict([0, 1] => 1.0)
 Dict([1, 0] => 1.0)
 Dict([1, 1] => 1.0)

julia> ClassifyBasis(basis, ['N', 'Z'])
Dict{Tuple{Int64, Int64}, Vector{Dict{BitVector, Float64}}} with 4 entries:
  (0, 0)  => [Dict([0, 0]=>1.0)]
  (1, -1) => [Dict([0, 1]=>1.0)]
  (1, 1)  => [Dict([1, 0]=>1.0)]
  (2, 0)  => [Dict([1, 1]=>1.0)]

```
"""
function ClassifyBasis(
        basisStates::Vector{Dict{BitVector,Float64}},
        symmetries::Vector{Char};
        energies::Vector{Float64}=Float64[],
    )
    @assert symmetries ∈ (['N', 'Z'], ['N'], ['Z'], ['Z', 'N'])
    classifiedBasis = Dict{NTuple{length(symmetries),Int64},typeof(basisStates)}()
    if !isempty(energies)
        classifiedEnergies = Dict{NTuple{length(symmetries),Int64},typeof(energies)}()
    end

    # Multhreading doesn't help here, already tried!
    for (i, stateDict) in enumerate(basisStates)
        key = GetSector(stateDict, symmetries)
        if key ∉ keys(classifiedBasis)
            classifiedBasis[key] = []
            if !isempty(energies)
                classifiedEnergies[key] = []
            end
        end
        push!(classifiedBasis[key], stateDict)
        if !isempty(energies)
            push!(classifiedEnergies[key], energies[i])
        end
    end
    if isempty(energies)
        return classifiedBasis
    else
        return classifiedBasis, classifiedEnergies
    end
end
export ClassifyBasis


"""
    TransformState(vector, basisStates)

Given a basis {B_i} and a vector {c_i}, return the basis representation
Σ_i c_i B_i of the vector

# Examples
```jldoctest
julia> basis = BasisStates(2)
4-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 0] => 1.0)
 Dict([0, 1] => 1.0)
 Dict([1, 0] => 1.0)
 Dict([1, 1] => 1.0)

julia> vector = [0, 0.5, -0.5, 0];

julia> TransformState(vector, basis)
Dict{BitVector, Float64} with 2 entries:
  [1, 0] => -0.5
  [0, 1] => 0.5
```
"""
function TransformState(
        vector::Union{Vector{ComplexF64}, Vector{Float64}},
        basisStates::Vector{Dict{BitVector,Float64}};
        tolerance::Float64=1e-16,
    )
    transformedState = Dict{BitVector, vector |> eltype}()
    keysArr = keys.(basisStates)
    valuesArr = values.(basisStates)
    for (i, c_i) in enumerate(vector)
        if abs(c_i) < tolerance
            continue
        end
        mergewith!(+, transformedState, Dict(keysArr[i] .=> c_i .* valuesArr[i]))
    end
    return transformedState
end
export TransformState


"""
    Spectrum(operator, basisStates)

Return eigenvalues and eigenvectors of the hermitian operator
in the subspace defined by the basis.

# Examples
```jldoctest
julia> basis = BasisStates(2)
4-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 0] => 1.0)
 Dict([0, 1] => 1.0)
 Dict([1, 0] => 1.0)
 Dict([1, 1] => 1.0)

julia> operator = [("+-", [1, 2], 1.0), ("+-", [2, 1], 1.0)]
2-element Vector{Tuple{String, Vector{Int64}, Float64}}:
 ("+-", [1, 2], 1.0)
 ("+-", [2, 1], 1.0)

julia> E, X = Spectrum(operator, basis);

julia> display(E)
4-element Vector{Float64}:
 -0.9999999999999989
  0.0
  0.0
  1.0

julia> display(X)
4-element Vector{Dict{BitVector, Float64}}:
 Dict([1, 0] => -0.7071067811865475, [0, 1] => 0.7071067811865477)
 Dict([0, 0] => 1.0)
 Dict([1, 1] => 1.0)
 Dict([1, 0] => 0.7071067811865477, [0, 1] => 0.7071067811865475)
```
"""
function Spectrum(
    operator::Vector{Tuple{String,Vector{Int64},Float64}},
    basisStates::Vector{Dict{BitVector,Float64}};
    diagElements::Vector{Float64}=Float64[],
    tolerance::Float64=1e-14,
    assumeHerm::Bool=true,
    )
    matrix = OperatorMatrix(basisStates, operator; tolerance=tolerance)
    if assumeHerm
        @assert maximum(abs.(matrix .- matrix')) < tolerance
    end
    if !isempty(diagElements)
        matrix += diagm(diagElements)
    end

    eigenVals, eigenVecs = assumeHerm ? eigen(Hermitian(matrix)) : eigen(matrix)
    eigenStates = [TransformState(collect(vector), basisStates; tolerance=tolerance)
                   for vector in eachcol(eigenVecs)]
    return eigenVals, eigenStates
end
export Spectrum


"""
    Spectrum(operator, basisStates)

Extends Spectrum() by making use of symmetries.

# Examples
```jldoctest
julia> basis = BasisStates(2)
4-element Vector{Dict{BitVector, Float64}}:
 Dict([0, 0] => 1.0)
 Dict([0, 1] => 1.0)
 Dict([1, 0] => 1.0)
 Dict([1, 1] => 1.0)

julia> operator = [("+-", [1, 2], 1.0), ("+-", [2, 1], 1.0)]
2-element Vector{Tuple{String, Vector{Int64}, Float64}}:
 ("+-", [1, 2], 1.0)
 ("+-", [2, 1], 1.0)

julia> E, X = Spectrum(operator, basis, ['N']; classify=true);

julia> display(X)
Dict{Tuple{Int64}, Vector{Dict{BitVector, Float64}}} with 3 entries:
  (0,) => [Dict([0, 0]=>1.0)]
  (2,) => [Dict([1, 1]=>1.0)]
  (1,) => [Dict([1, 0]=>0.707107, [0, 1]=>-0.707107), Dict([1, 0]=>0.707107, [0, 1]=>0.707107)]
```
"""
function Spectrum(
    operator::Vector{Tuple{String,Vector{Int64},Float64}},
    basisStates::Vector{Dict{BitVector,Float64}},
    symmetries::Vector{Char};
    diagElements::Vector{Float64}=Float64[],
    tolerance::Float64=1e-14,
    classify::Bool=false,
)
    if !isempty(diagElements)
        @assert length(diagElements) == size(basisStates)[1]
    end

    classifiedBasis = ClassifyBasis(basisStates, symmetries)
    classifiedEigvals = Dict(k => Float64[] for k in keys(classifiedBasis))
    classifiedEigvecs = Dict(k => Dict{BitVector,Float64}[] for k in keys(classifiedBasis))
    Threads.@threads for (k, basis) in collect(classifiedBasis)
        classifiedEigvals[k], classifiedEigvecs[k] = Spectrum(operator, basis; diagElements=diagElements, tolerance=tolerance)
    end
    if classify
        return classifiedEigvals, classifiedEigvecs
    else
        allEigvals = Float64[]
        allEigvecs = Dict{BitVector,Float64}[]
        for k in keys(classifiedBasis)
            append!(allEigvecs, classifiedEigvecs[k])
            append!(allEigvals, classifiedEigvals[k])
        end
        return sort(allEigvals), allEigvecs[sortperm(allEigvals)]
    end
end
export Spectrum


"""
    Spectrum(operator, basisStates)

Extends Spectrum() by accepting an already-classified basis.

# Examples
```jldoctest
julia> classifiedBasis = Dict((0,) => [Dict(Bool.([0, 0])=>1.0)], (1,) => [Dict(Bool.([0, 1]) => 1.0), Dict(Bool.([1, 0])=>1.0)], (2,)  => [Dict(Bool.([1, 1])=>1.0)])
Dict{Tuple{Int64}, Vector{Dict{BitVector, Float64}}} with 3 entries:
  (0,) => [Dict([0, 0]=>1.0)]
  (2,) => [Dict([1, 1]=>1.0)]
  (1,) => [Dict([0, 1]=>1.0), Dict([1, 0]=>1.0)]

julia> operator = [("+-", [1, 2], 1.0), ("+-", [2, 1], 1.0)]
2-element Vector{Tuple{String, Vector{Int64}, Float64}}:
 ("+-", [1, 2], 1.0)
 ("+-", [2, 1], 1.0)

julia> E, X = Spectrum(operator, classifiedBasis);

julia> display(X)
Dict{Tuple{Int64}, Vector{Dict{BitVector, Float64}}} with 3 entries:
  (0,) => [Dict([0, 0]=>1.0)]
  (2,) => [Dict([1, 1]=>1.0)]
  (1,) => [Dict([1, 0]=>0.707107, [0, 1]=>-0.707107), Dict([1, 0]=>0.707107, [0, 1]=>0.707107)]

```
"""
function Spectrum(
    operator::Vector{Tuple{String,Vector{Int64},Float64}},
    classifiedBasis::Union{Dict{Tuple{Int64}, Vector{Dict{BitVector,Float64}}}, Dict{Tuple{Int64, Int64}, Vector{Dict{BitVector,Float64}}}};
    diagElements::Dict{}=Dict(),
    tolerance::Float64=1e-14,
    )
    if !isempty(diagElements)
        @assert all(x -> true, [length(E) == size(classifiedBasis[k])[1] for (k,E) in diagElements])
    end

    classifiedEigvals = Dict(k => Float64[] for k in keys(classifiedBasis))
    classifiedEigvecs = Dict(k => Dict{BitVector,Float64}[] for k in keys(classifiedBasis))
    Threads.@threads for (k, basis) in collect(classifiedBasis)
        if k in keys(diagElements)
            classifiedEigvals[k], classifiedEigvecs[k] = Spectrum(operator, basis; diagElements=diagElements[k], tolerance=tolerance)
        else
            classifiedEigvals[k], classifiedEigvecs[k] = Spectrum(operator, basis; tolerance=tolerance)
        end
    end
    return classifiedEigvals, classifiedEigvecs
end
export Spectrum


function BandStructure(
        hamiltonian::Vector{Tuple{String,Vector{Int64},Float64}},
        basisStates::Vector{Dict{BitVector,Float64}},
        eigenOperator::Vector{Tuple{String,Vector{Int64},Float64}};
        dimensionSymmetry::Int64=-1,
        tolerance::Float64=1e-10
    )
    @assert DoesCommute(hamiltonian, eigenOperator, basisStates)
    eigenVals, eigenStates = Spectrum(hamiltonian, basisStates)
    if dimensionSymmetry < 0
        dimensionSymmetry = length(eigenVals)
    end
    classifiedEnergies = Dict{Int64, Vector{Float64}}()
    count = 1
    while count ≤ length(eigenStates)
        degenerateSubspace = findall(Ei -> abs(Ei - eigenVals[count]) < tolerance, eigenVals)
        degenerateBasis = [eigenStates[i] for i in degenerateSubspace]
        if length(degenerateBasis) == 1
            flag, symmetryValue = IsEigenState(eigenStates[degenerateSubspace[1]], eigenOperator; tolerance=tolerance)
            @assert flag eigenStates[degenerateSubspace[1]]
            quantumNums = [trunc(Int, dimensionSymmetry * atan(imag(symmetryValue), real(symmetryValue)) / π)]
        else
            symmetryValues, _ = Spectrum(eigenOperator, degenerateBasis; assumeHerm=false)
            quantumNums = trunc.(Int, dimensionSymmetry .* atan.(imag.(symmetryValues), real.(symmetryValues)) ./ π)
        end
        for quantumNum in quantumNums
            if quantumNum in keys(classifiedEnergies)
                push!(classifiedEnergies[quantumNum], eigenVals[count])
            else
                classifiedEnergies[quantumNum] = [eigenVals[count]]
            end
        end
        count += length(degenerateSubspace)
    end
    return classifiedEnergies
end
export BandStructure


function IsEigenState(
        state::Dict{BitVector,Float64},
        operator::Vector{Tuple{String,Vector{Int64},Float64}};
        tolerance::Float64=1e-15,
    )
    @assert !isempty(state) 
    pivotVal, pivotKey = findmax(state)
    if pivotVal == 0
        pivotVal, pivotKey = findmin(state)
        @assert pivotVal ≠ 0
    end
    newState = ApplyOperator(operator, state)
    if keys(newState) ≠ keys(state)
        return false, 0.
    end
    if newState[pivotKey] == 0
        if all(c -> abs(c) < tolerance, values(newState))
            println("New state is all zero, hard to tell if eigenstate or simply the vacuum.")
            return true, 0.
        else
            return false, 0.
        end
    else
        eigenValue = newState[pivotKey] / state[pivotKey]
        if all(k -> abs(eigenValue - newState[k] / state[k]) < tolerance, keys(state))
            return true, eigenValue
        else
            return false, 0.
        end
    end
end
export IsEigenState
