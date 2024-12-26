using LinearAlgebra, Fermions, Plots

L = 40
timeMax = 20
deltaTime = 1e-2
basis = BasisStates1p(L)
basisMb = BasisStates(L)
numberOperators = [OperatorMatrix(basis, [("n", [i], 1.)]) for i in 1:L]
collatzHamiltonian = CollatzModel(L)
collatzHamiltonianMatrix = OperatorMatrix(basis, collatzHamiltonian)
emptyState = VacuumState(basisMb)
initState = ExpandIntoBasis(ApplyOperator([("+", [L], 1.)], emptyState), basis) .+ 0im
stateTimeEvol = StateEvolution(initState, collatzHamiltonianMatrix, range(0, stop=timeMax, step=deltaTime) |> collect)
maxDetectPos = [argmax([real(state' * operator * state) for operator in numberOperators]) for state in stateTimeEvol]
plot(maxDetectPos)
