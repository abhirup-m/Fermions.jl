function OperatorTimeEvol(
        operator::Vector{Tuple{String,Vector{Int64},Float64}},
        hamiltonian::Vector{Tuple{String,Vector{Int64},Float64}},
        initState::Dict{BitVector,Float64}, 
        basisStates::Vector{Dict{BitVector,Float64}},
        deltaTime::Float64,
        numSteps::Int64;
    )
    initStateVector = ExpandIntoBasis(initState, basisStates)
    initStateVector ./= norm(initStateVector)
    operatorMatrix = convert(Matrix{ComplexF32}, OperatorMatrix(basisStates, operator))
    hamiltonianMatrix = OperatorMatrix(basisStates, hamiltonian)
    deltaUnitary = convert(Matrix{ComplexF32}, (I(size(hamiltonianMatrix)[1]) .- 1im .* hamiltonianMatrix .* deltaTime ./ 2) / (I(size(hamiltonianMatrix)[1]) .+ 1im .* hamiltonianMatrix .* deltaTime ./ 2))
    expecValueTimeEvol = zeros(numSteps)
    operatorMatrixTimeEvol = Matrix{ComplexF64}[]
    @showprogress for step in 1:numSteps
        push!(operatorMatrixTimeEvol, operatorMatrix)
        expecValueTimeEvol[step] = real(initStateVector' * operatorMatrix * initStateVector)
        operatorMatrix = deltaUnitary' * operatorMatrix * deltaUnitary
    end
    return expecValueTimeEvol, range(0, step=deltaTime, length=numSteps), operatorMatrixTimeEvol
end
export OperatorTimeEvol


function OTOC(
        operatorMatrixTimeEvol::Vector{Matrix{ComplexF64}},
        staticMatrix::Union{Matrix{ComplexF64}, Matrix{Float64}},
        hamiltonianMatrix::Matrix{Float64};
    )
    otoc = zeros(length(operatorMatrixTimeEvol))
    groundState = eigen(hamiltonianMatrix).vectors[:, 1]
    @showprogress for step in eachindex(operatorMatrixTimeEvol)
        commutator = operatorMatrixTimeEvol[step] * staticMatrix - staticMatrix * operatorMatrixTimeEvol[step]
        otoc[step] = real(groundState' 
                          * commutator' * commutator
                          * groundState
                         )
    end
    return otoc
end
export OTOC


function StateEvolution(
        initState::Vector{ComplexF64}, 
        hamiltonian::Matrix{Float64},
        timeSteps::Vector{Float64},
    )
    stateEvolution = Vector{Complex}[initState]
    for step in eachindex(timeSteps)[2:end]
        deltaTime = timeSteps[step] - timeSteps[step-1]
        newState = inv(I + 0.5im * deltaTime * hamiltonian) * (I - 0.5im * deltaTime * hamiltonian) * stateEvolution[end]
        push!(stateEvolution, newState)
    end
    for (i, v) in enumerate(stateEvolution)
        stateEvolution[i] = v/norm(v)
    end
    return stateEvolution
end
export StateEvolution


function TimeEvolve(
        initState::Dict{BitVector,Float64},
        operator::Vector{Tuple{String,Vector{Int64},Float64}},
        hamltFlow::Vector{Vector{Tuple{String,Vector{Int64},Float64}}},
        maxSize::Int64,
        timeDef::Dict{String, Number};
        symmetries::Vector{Char}=Char[],
    )
    @assert haskey(timeDef, "start") && haskey(timeDef, "stop") && haskey(timeDef, "numSteps")

    results = IterDiag(
                       hamltFlow,
                       maxSize;
                       symmetries=symmetries,
                       specFuncDefDict=Dict("operator" => Dict("create" => operator, "destroy" => Dagger(operator))),
                       stateDict = Dict("initState" => initState),
                       save=true,
                      )
    operatorFirst = results["specFuncOperators"]["operator"]["create"][1]
    operatorMatrix = results["specFuncOperators"]["operator"]["create"][end]
    eigVals = deserialize(results["savePaths"][end-1])["eigVals"]
    rotation = deserialize(results["savePaths"][end-1])["basis"]
    operatorLastStep = rotation' * operatorMatrix * rotation
    operatorLastStep *= √(sum(operatorFirst^2) / sum(operatorLastStep^2))
    initState = deserialize(results["savePaths"][end-1])["stateVectors"]["initState"]
    referenceState = rotation' * initState

    timeSteps = range(timeDef["start"], stop=timeDef["stop"], length=timeDef["numSteps"])
    timeEvol = 0 .* collect(timeSteps)
    deltaUnitary = exp.(-1im * eigVals * (timeSteps[2] - timeSteps[1]))
    for (i, time) in enumerate(timeSteps)
        referenceState /= norm(referenceState)
        evolvedVal = referenceState' * operatorLastStep * referenceState
        timeEvol[i] = 0.5 * (evolvedVal + evolvedVal')
        referenceState = deltaUnitary .* referenceState
    end
    return timeEvol
end
