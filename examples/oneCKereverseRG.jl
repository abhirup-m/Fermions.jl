using Fermions
function stateExpansion(state::Dict{BitVector,Float64}, sectors::String)
    #=@assert sectors in ["p", "h", "ph"] "Provided IOM sectors not among p, h or ph/hp."=#
    if sectors == "p"
        state = Dict{keytype(state),valtype(state)}([key; [1, 1]] => val for (key, val) in state)
    elseif sectors == "h"
        state = Dict{keytype(state),valtype(state)}([key; [0, 0]] => val for (key, val) in state)
    elseif sectors == "ph"
        state = Dict{keytype(state),valtype(state)}([key; [1, 1, 0, 0]] => val for (key, val) in state)
        newState = Dict{BitVector, Float64}()
        for (key, val) in state
            newState[[key; [1, 0, 0, 1]]] = val
            newState[[key; [0, 1, 1, 0]]] = val
        end
        return newState
    else
        newState = Dict{BitVector, Float64}()
        for (key, val) in state
            newState[[key; [1, 0, 0, 1]]] = val
            newState[[key; [0, 1, 1, 0]]] = val
        end
    end
    return state
end
export stateExpansion1CK

operator = [("+-+-", [1, 2, 4, 3], 1.0)]
num_sites = 3
bandwidth = 1
hamiltonian = KondoModel(1, 0., 1.)
basis = BasisStates(2 * num_sites)
E, V = Spectrum(hamiltonian, basis)
initState = V[1]
numSteps = 6
alphaValues = ones(numSteps)
@time stateFlow = getWavefunctionRG(initState, alphaValues, numSteps, unitaries1CK, stateExpansion, "ph")
corr1 = [GenCorrelation(state, operator) for state in stateFlow]

#=@time stateFlow = getWavefunctionRG(initState, alphaValues, numSteps, unitaries1CK, stateExpansion, "hp")=#
#=corr2 = [GenCorrelation(state, operator) for state in stateFlow]=#
#==#
display(corr1)
#=display(corr2)=#
