function getWavefunctionRG(initState::Dict{BitVector,Float64}, alphaValues::Vector{Float64}, numSteps::Integer, unitaryOperatorFunction::Function, stateExpansionFunction::Function, sectors::String; maxSize::Int64=0, RGtolerance::Float64=1e-16)

    @assert numSteps ≤ length(alphaValues)
    numEntangled = div(length(collect(keys(initState))[1]), 2)
    stateFlowArray = Dict{BitVector,Float64}[]
    push!(stateFlowArray, initState)

    for alpha in alphaValues[1:numSteps]
        newState = stateExpansionFunction(stateFlowArray[end], sectors)
        unitaryOperatorList = unitaryOperatorFunction(alpha, numEntangled, sectors)
        numEntangled = div(length(collect(keys(newState))[1]), 2)

        # apply unitary evolution operator to generate new terms
        newTerms = fetch.([Threads.@spawn ApplyOperator([operator], newState) for operator in unitaryOperatorList])
        mergewith!(+, newState, newTerms...)

        # remove wavefunction coefficients below a threshold
        maxv = maximum(abs.(values(newState)))
        filter!(x -> abs(x[2])/maxv > RGtolerance, newState)

        # retained only maxSize number of coefficients in wavefunction
        if maxSize > 0 && maxSize < length(newState)
            println("Drop ratio ~ ", sum(sort(abs.(values(newState)), rev=true)[maxSize:end] .^ 2) / sum(values(newState) .^ 2))
            newState = Dict(sort(collect(newState), by=x->x|>last|>abs, rev=true)[1:maxSize])
        else
            println("Drop ratio ~ ", 0)
        end
        total_norm = sum(values(newState) .^ 2)^0.5
        map!(x->x/total_norm, values(newState))
        push!(stateFlowArray, newState)
    end

    return stateFlowArray
end
export getWavefunctionRG
