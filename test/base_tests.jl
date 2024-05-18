function HubbardDimerMatrix(eps, U, hop_t)
    hubbardDimerMatrix = Dict()
    hubbardDimerMatrix[(0, 0)] = [0;;]
    hubbardDimerMatrix[(1, 1)] = [eps[3] hop_t[1]; hop_t[1] eps[1]]
    hubbardDimerMatrix[(1, -1)] = [eps[4] hop_t[2]; hop_t[2] eps[2]]
    hubbardDimerMatrix[(2, 2)] = [eps[1] + eps[3];;]
    hubbardDimerMatrix[(2, 0)] = (
                                        [U[2] + sum(eps[3:4]) hop_t[1] -hop_t[2] 0;
                                         hop_t[1] sum(eps[[1,4]]) 0 hop_t[2];
                                         -hop_t[2] 0 sum(eps[[2,3]]) -hop_t[1];
                                         0 hop_t[2] -hop_t[1] sum(eps[[1,2]]) + U[1]]
                                       )
    hubbardDimerMatrix[(2, -2)] = [eps[2] + eps[4];;]
    hubbardDimerMatrix[(3, 1)] = [U[2] + sum(eps[[1,3,4]]) -hop_t[2]; -hop_t[2] U[1] + sum(eps[1:3])]
    hubbardDimerMatrix[(3, -1)] = [U[2] + sum(eps[2:4]) -hop_t[1]; -hop_t[1] U[1] + sum(eps[[1,2,4]])]
    hubbardDimerMatrix[(4, 0)] = [sum(eps) + sum(U);;]
    return hubbardDimerMatrix
end

@testset "BasisStates" begin

    @test BasisStates(1) == Dict((1,1) => [[1]], 
                                 (0,0) => [[0]]
                                )
    @test BasisStates(1; totOccupancy=1) == Dict((1,1) => [[1]]) 
    @test BasisStates(1; totOccupancy=0) == Dict((0,0) => [[0]]) 

    @test BasisStates(2) == Dict(
                                (0,0) => [[0, 0]],
                                (1,1) => [[1, 0]],
                                (1,-1) => [[0, 1]],
                                (2,0) => [[1, 1]],
                                )
    @test BasisStates(2; totOccupancy=0) == Dict((0,0) => [[0, 0]]) 
    @test BasisStates(2; totOccupancy=1) == Dict(
                                                (1,1) => [[1, 0]],
                                                (1,-1) => [[0, 1]],
                                               )
    @test BasisStates(2; totOccupancy=2) == Dict((2,0) => [[1, 1]]) 

    @test BasisStates(3) == Dict((0,0) => [[0, 0, 0]],
                                (1,1) => [[0, 0, 1], [1, 0, 0]],
                                (1,-1) => [[0, 1, 0]],
                                (2,2) => [[1, 0, 1]],
                                (2,0) => [[0, 1, 1], [1, 1, 0]],
                                (3,1) => [[1, 1, 1]],
                                )
    @test BasisStates(3; totOccupancy=0) == Dict((0,0) => [[0, 0, 0]]) 
    @test BasisStates(3; totOccupancy=1) == Dict(
                                                (1,1) => [[0, 0, 1], [1, 0, 0]],
                                                (1,-1) => [[0, 1, 0]],
                                               )
    @test BasisStates(3; totOccupancy=2) == Dict(
                                                (2,2) => [[1, 0, 1]],
                                                (2,0) => [[0, 1, 1], [1, 1, 0]],
                                                )
    @test BasisStates(3; totOccupancy=3) == Dict((3,1) => [[1, 1, 1]]) 
end

@testset "TransformBit" begin
    @test TransformBit(false, 'n') == (0, 0)
    @test TransformBit(false, 'h') == (0, 1)
    @test TransformBit(false, '+') == (1, 1)
    @test TransformBit(false, '-') == (0, 0)
    @test TransformBit(true, 'n') == (1, 1)
    @test TransformBit(true, 'h') == (1, 0)
    @test TransformBit(true, '+') == (1, 0)
    @test TransformBit(true, '-') == (0, 1)
end

@testset "applyOperatorOnState" begin
    # checking linearity
    b4 = BasisStates(4)
    for key in [(0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0)]
        stateDict = Dict(k => v for (k, v) in zip(b4[key], rand(length(b4[key]))))
        partialStates = [Dict(k => v) for (k, v) in stateDict]
        ops = ["n", "h", "+", "-"]
        combinedOpList = Dict{Tuple{String,Vector{Int64}},Float64}()
        totalApplication = []
        for op in Iterators.product(ops, ops, ops, ops)
            oplist = Dict((string(op...), [1,2,3,4]) => 1.0)
            mergewith!(+, combinedOpList, oplist)
            partialApplications = [applyOperatorOnState(state, oplist) for state in partialStates]
            completeApplication = applyOperatorOnState(stateDict, oplist)
            @test completeApplication == merge(+, partialApplications...)
            totalApplication = [totalApplication; [completeApplication]]
        end
        @test all([merge(+, totalApplication...)[k] ≈ applyOperatorOnState(stateDict, combinedOpList)[k] for k in keys(merge(+, totalApplication...))])
    end

    # quantitative checking through basis states
    state = Dict(BitVector([0, 0]) => 0.1)
    @testset "state = [0, 0], operator = $op" for op in ["+", "-", "n", "h"]
        oplist = Dict((op, [1]) => 0.5)
        if op == "+"
            @test applyOperatorOnState(state, oplist) == Dict(BitVector([1, 0]) => 0.05)
        elseif op == "h"
            @test applyOperatorOnState(state, oplist) == Dict(BitVector([0, 0]) => 0.05)
        else
            @test isempty(applyOperatorOnState(state, oplist))
        end
        oplist = Dict((op, [2]) => 0.5)
        if op == "+"
            @test applyOperatorOnState(state, oplist) == Dict(BitVector([0, 1]) => 0.05)
        elseif op == "h"
            @test applyOperatorOnState(state, oplist) == Dict(BitVector([0, 0]) => 0.05)
        else
            @test isempty(applyOperatorOnState(state, oplist))
        end
        @testset for op2 in ["+", "-", "n", "h"]
            oplist = Dict((op * op2, [1, 2]) => 0.5)
            if occursin("-", op * op2) || occursin("n", op * op2)
                @test isempty(applyOperatorOnState(state, oplist))
            elseif op * op2 == "++"
                @test applyOperatorOnState(state, oplist) == Dict(BitVector([1, 1]) => 0.05)
            elseif op * op2 == "+h"
                @test applyOperatorOnState(state, oplist) == Dict(BitVector([1, 0]) => 0.05)
            elseif op * op2 == "h+"
                @test applyOperatorOnState(state, oplist) == Dict(BitVector([0, 1]) => 0.05)
            elseif op * op2 == "hh"
                @test applyOperatorOnState(state, oplist) == Dict(BitVector([0, 0]) => 0.05)
            end
        end
    end

    state = Dict(BitVector([1, 1]) => 0.1)
    @testset "state = [1, 1], operator = $op" for op in ["+", "-", "n", "h"]
        oplist = Dict((op, [1]) => 0.5 )
        if op == "-"
            @test applyOperatorOnState(state, oplist) == Dict(BitVector([0, 1]) => 0.05)
        elseif op == "n"
            @test applyOperatorOnState(state, oplist) == Dict(BitVector([1, 1]) => 0.05)
        else
            @test isempty(applyOperatorOnState(state, oplist))
        end
        oplist = Dict((op, [2]) => 0.5 )
        if op == "-"
            @test applyOperatorOnState(state, oplist) == Dict(BitVector([1, 0]) => -0.05)
        elseif op == "n"
            @test applyOperatorOnState(state, oplist) == Dict(BitVector([1, 1]) => 0.05)
        else
            @test isempty(applyOperatorOnState(state, oplist))
        end
        @testset for op2 in ["+", "-", "n", "h"]
        oplist = Dict((op * op2, [1, 2]) => 0.5 )
            if occursin("+", op * op2) || occursin("h", op * op2)
                @test isempty(applyOperatorOnState(state, oplist))
            elseif op * op2 == "--"
                @test applyOperatorOnState(state, oplist) == Dict(BitVector([0, 0]) => -0.05)
            elseif op * op2 == "-n"
                @test applyOperatorOnState(state, oplist) == Dict(BitVector([0, 1]) => 0.05)
            elseif op * op2 == "n-"
                @test applyOperatorOnState(state, oplist) == Dict(BitVector([1, 0]) => -0.05)
            elseif op * op2 == "nn"
                @test applyOperatorOnState(state, oplist) == Dict(BitVector([1, 1]) => 0.05)
            end
        end
    end
end


@testset "GeneralOperatorMatrix" begin
    basisStates = BasisStates(4)
    eps = rand(4)
    hop_t = rand(2)
    U = rand(2)
    operatorList = Dict{Tuple{String, Vector{Int64}}, Float64}()
    operatorList[("n", [1])] = eps[1]
    operatorList[("n", [2])] = eps[2]
    operatorList[("n", [3])] = eps[3]
    operatorList[("n", [4])] = eps[4]
    operatorList[("nn", [1, 2])] = U[1]
    operatorList[("nn", [3, 4])] = U[2]
    operatorList[("+-", [1, 3])] = hop_t[1]
    operatorList[("+-", [3, 1])] = hop_t[1]
    operatorList[("+-", [2, 4])] = hop_t[2]
    operatorList[("+-", [4, 2])] = hop_t[2]
    computedMatrix = generalOperatorMatrix(basisStates, operatorList)
    comparisonMatrix = HubbardDimerMatrix(eps, U, hop_t)
    @test keys(comparisonMatrix) == keys(computedMatrix)
    for key in keys(comparisonMatrix)
        @test computedMatrix[key] ≈ comparisonMatrix[key]
    end
end


@testset "GeneralOperatorMatrixSet" begin
    basisStates = BasisStates(4)
    eps_arr = [rand(4) for _ in 1:3]
    hop_t_arr = [rand(2) for _ in 1:3]
    U_arr = [rand(2) for _ in 1:3]
    operatorList = [("n", [1]), ("n", [2]), ("n", [3]), ("n", [4]), ("nn", [1, 2]), ("nn", [3, 4]), ("+-", [1, 3]), ("+-", [3, 1]), ("+-", [2, 4]), ("+-", [4, 2])]
    couplingMatrix = [[eps_arr[i][1], eps_arr[i][2], eps_arr[i][3], eps_arr[i][4], U_arr[i][1], U_arr[i][2], hop_t_arr[i][1], hop_t_arr[i][1], hop_t_arr[i][2], hop_t_arr[i][2]] for i in 1:3]
    computedMatrixSet = generalOperatorMatrix(basisStates, operatorList, couplingMatrix)
    comparisonMatrixSet = [HubbardDimerMatrix(eps_arr[i], U_arr[i], hop_t_arr[i]) for i in 1:3]
    @test length(computedMatrixSet) == length(comparisonMatrixSet)
    for (comparisonMatrix, computedMatrix) in zip(comparisonMatrixSet, computedMatrixSet)
        @test keys(comparisonMatrix) == keys(computedMatrix)
        for key in keys(comparisonMatrix)
            @test computedMatrix[key] ≈ comparisonMatrix[key]
        end
    end
end

@testset "Spectrum" begin
    basisStates = BasisStates(4)
    eps = rand(4)
    hop_t = rand(2)
    U = rand(2)
    operatorList = Dict{Tuple{String, Vector{Int64}}, Float64}()
    operatorList[("n", [1])] = eps[1]
    operatorList[("n", [2])] = eps[2]
    operatorList[("n", [3])] = eps[3]
    operatorList[("n", [4])] = eps[4]
    operatorList[("nn", [1, 2])] = U[1]
    operatorList[("nn", [3, 4])] = U[2]
    operatorList[("+-", [1, 3])] = hop_t[1]
    operatorList[("+-", [3, 1])] = hop_t[1]
    operatorList[("+-", [2, 4])] = hop_t[2]
    operatorList[("+-", [4, 2])] = hop_t[2]
    computedMatrix = generalOperatorMatrix(basisStates, operatorList)
    eigvals, eigvecs = getSpectrum(computedMatrix)
    comparisonMatrix = HubbardDimerMatrix(eps, U, hop_t)
    @test keys(comparisonMatrix) == keys(computedMatrix)
    for key in keys(comparisonMatrix)
        eigvalTest, eigvecTest = eigen(comparisonMatrix[key])
        @test eigvals[key] ≈ eigvalTest
        @test eigvecs[key] ≈ [eigvecTest[:, i] for i in eachindex(eachrow(eigvecTest))]
    end
end
