using Random

@testset "BasisStates" begin

    @test issetequal(BasisStates(2),
        [
            Dict([0, 0] => 1.0),
            Dict([1, 0] => 1.0),
            Dict([0, 1] => 1.0),
            Dict([1, 1] => 1.0),
        ]
    )
    @test issetequal(BasisStates(3),
        [
            Dict([0, 0, 0] => 1.0),
            Dict([1, 0, 0] => 1.0),
            Dict([0, 1, 0] => 1.0),
            Dict([1, 1, 0] => 1.0),
            Dict([0, 0, 1] => 1.0),
            Dict([1, 0, 1] => 1.0),
            Dict([0, 1, 1] => 1.0),
            Dict([1, 1, 1] => 1.0),
        ]
    )
end


@testset "RoundTo" begin
    @test RoundTo(1, 1e-10) == 1
    @test RoundTo(1 + 1e-11, 1e-10) == 1
    @test RoundTo(1e-11, 1e-10) == 0
    @test RoundTo(1e-10, 1e-10) == 1e-10
    @test RoundTo(-1e-10, 1e-10) == -1e-10
    @test RoundTo(0, 1e-10) == 0
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

@testset "ApplyOperator" begin
    # checking linearity
    basis = BasisStates(4)
    coeffs1 = rand(4^4)
    coeffs2 = rand(length(basis))
    allOperators = [[(o1 * o2 * o3 * o4, [1, 2], coeffs1[i])]
                    for (i, (o1, o2,)) in enumerate(Iterators.product(repeat([["+", "-", "n", "h"]], 2)...))]
    totalOperator = vcat(allOperators...)
    allStates = [Dict(k => coeffs2[i] * v for (k, v) in dict) for (i, dict) in enumerate(basis)]
    totalState = mergewith(+, allStates...)
    totalOutgoingState = ApplyOperator(totalOperator, totalState)
    pieceWiseAddedState = mergewith(+, [ApplyOperator(operator, state) for operator in allOperators for state in allStates]...)

    @test keys(totalOutgoingState) == keys(pieceWiseAddedState)
    for key in keys(totalOutgoingState)
        @test totalOutgoingState[key] ≈ pieceWiseAddedState[key]
    end

    # quantitative checking through basis states
    state = Dict(BitVector([0, 0]) => 0.1)
    @testset "state = [0, 0], operator = $op" for op in ["+", "-", "n", "h"]
        oplist = [(op, [1], 0.5)]
        if op == "+"
            @test ApplyOperator(oplist, state) == Dict(BitVector([1, 0]) => 0.05)
        elseif op == "h"
            @test ApplyOperator(oplist, state) == Dict(BitVector([0, 0]) => 0.05)
        else
            @test isempty(ApplyOperator(oplist, state))
        end
        oplist = [(op, [2], 0.5)]
        if op == "+"
            @test ApplyOperator(oplist, state) == Dict(BitVector([0, 1]) => 0.05)
        elseif op == "h"
            @test ApplyOperator(oplist, state) == Dict(BitVector([0, 0]) => 0.05)
        else
            @test isempty(ApplyOperator(oplist, state))
        end
        @testset for op2 in ["+", "-", "n", "h"]
            oplist = [(op * op2, [1, 2], 0.5)]
            if occursin("-", oplist[1][1]) || occursin("n", oplist[1][1])
                @test isempty(ApplyOperator(oplist, state))
            elseif oplist[1][1] == "++"
                @test ApplyOperator(oplist, state) == Dict(BitVector([1, 1]) => 0.05)
            elseif oplist[1][1] == "+h"
                @test ApplyOperator(oplist, state) == Dict(BitVector([1, 0]) => 0.05)
            elseif oplist[1][1] == "h+"
                @test ApplyOperator(oplist, state) == Dict(BitVector([0, 1]) => 0.05)
            elseif oplist[1][1] == "hh"
                @test ApplyOperator(oplist, state) == Dict(BitVector([0, 0]) => 0.05)
            end
        end
    end

    state = Dict(BitVector([1, 1]) => 0.1)
    @testset "state = [1, 1], operator = $op" for op in ["+", "-", "n", "h"]
        oplist = [(op, [1], 0.5)]
        if op == "-"
            @test ApplyOperator(oplist, state) == Dict(BitVector([0, 1]) => 0.05)
        elseif op == "n"
            @test ApplyOperator(oplist, state) == Dict(BitVector([1, 1]) => 0.05)
        else
            @test isempty(ApplyOperator(oplist, state))
        end
        oplist = [(op, [2], 0.5)]
        if op == "-"
            @test ApplyOperator(oplist, state) == Dict(BitVector([1, 0]) => -0.05)
        elseif op == "n"
            @test ApplyOperator(oplist, state) == Dict(BitVector([1, 1]) => 0.05)
        else
            @test isempty(ApplyOperator(oplist, state))
        end
        @testset "state = [1, 1], operator2 = $op2" for op2 in ["+", "-", "n", "h"]
            oplist = [(op * op2, [1, 2], 0.5)]
            if occursin("+", op * op2) || occursin("h", op * op2)
                @test isempty(ApplyOperator(oplist, state))
            elseif op * op2 == "--"
                @test ApplyOperator(oplist, state) == Dict(BitVector([0, 0]) => -0.05)
            elseif op * op2 == "-n"
                @test ApplyOperator(oplist, state) == Dict(BitVector([0, 1]) => 0.05)
            elseif op * op2 == "n-"
                @test ApplyOperator(oplist, state) == Dict(BitVector([1, 0]) => -0.05)
            elseif op * op2 == "nn"
                @test ApplyOperator(oplist, state) == Dict(BitVector([1, 1]) => 0.05)
            end
        end
    end
end

@testset "OperatorMatrix" begin
    eps = 1.281
    hop_t = 0.284
    U = 3.132
    operatorList = HubbardDimerOplist(eps, U, hop_t)
    @testset "sector=$((n, m))" for (n, m) in [(0, 0), (1, 1), (1, -1), (2, 2), (2, 0), (2, -2), (3, 1), (3, -1), (4, 0)]
        basisStates = BasisStates(4; totOccReq=n, magzReq=m)
        if (n,m) == (2,0)
            @test Set([collect(keys(b))[1] for b in basisStates]) == Set([[0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0]])
            basisStates = [Dict(BitVector([1, 0, 0, 1]) => 1.0), Dict(BitVector([0, 1, 1, 0]) => 1.0), Dict(BitVector([1, 1, 0, 0]) => 1.0), Dict(BitVector([0, 0, 1, 1]) => 1.0)]
        end
        computedMatrix = OperatorMatrix(basisStates, operatorList)
        comparisonMatrix = HubbardDimerMatrix(eps, U, hop_t)[(n, m)]
        @test comparisonMatrix ≈ computedMatrix
    end
end


@testset "StateOverlap" begin
    basis = BasisStates(4)
    for (b1, b2) in Iterators.product(basis, basis)
        @test StateOverlap(b1, b2) == ifelse(b1 == b2, 1, 0)
    end
    coeffs1 = rand(length(basis))
    totalState1 = mergewith(+, [Dict(k => coeffs1[i] * v for (k, v) in dict) for (i, dict) in enumerate(basis)]...)
    coeffs2 = rand(length(basis))
    totalState2 = mergewith(+, [Dict(k => coeffs2[i] * v for (k, v) in dict) for (i, dict) in enumerate(basis)]...)
    @test StateOverlap(totalState1, totalState2) ≈ sum(coeffs1 .* coeffs2)
end


@testset "Dagger" begin
    qubitOperators = ['+', '-', 'n', 'h', '+', '-', 'n', 'h', '+', '-', 'n', 'h']
    numSites = 8
    basis = BasisStates(numSites)
    @testset for numLegs in 1:numSites
        operator = [(join(shuffle(qubitOperators)[1:numLegs]), collect(shuffle(1:numSites)[1:numLegs]), rand()) for _ in 1:10]
        hermConj = Dagger(deepcopy(operator))
        errorMatrix = OperatorMatrix(basis, hermConj) - OperatorMatrix(basis, operator)'
        @test errorMatrix .|> abs |> maximum == 0
    end
end


# ---------------------------------------------------------------
# BasisStates
# ---------------------------------------------------------------
@testset "BasisStates" begin
    # no restrictions → all 2^N states
    bs = BasisStates(3)
    @test length(bs) == 8

    # total occupancy filtering
    bs1 = BasisStates(3; totOccReq=1)
    @test length(bs1) == 3
    for st in bs1
        @test sum(first(keys(st))) == 1
    end

    # magnetization filtering
    bs2 = BasisStates(4; magzReq=0)
    for st in bs2
        bv = first(keys(st))
        @test sum(bv[1:2:end]) - sum(bv[2:2:end]) == 0
    end

    # local criteria
    bs3 = BasisStates(4; totOccReq=2, localCriteria = x->x[1]==1)
    for st in bs3
        bv = first(keys(st))
        @test sum(bv)==2 && bv[1]==1
    end

    # explicit-form constructor (Vector inputs)
    bs4 = BasisStates(3, [1], [1, -1], x->true)
    @test all(sum(first(keys(s))) == 1 for s in bs4)
end



# ---------------------------------------------------------------
# BasisStates1p
# ---------------------------------------------------------------
@testset "BasisStates1p" begin
    bs = BasisStates1p(5)
    @test length(bs) == 5
    for (i,st) in enumerate(bs)
        @test first(keys(st)) == BitVector([j==i for j=1:5])
    end
end



# ---------------------------------------------------------------
# TransformBit
# ---------------------------------------------------------------
@testset "TransformBit" begin
    @test TransformBit(false, '+') == (true, 1)
    @test TransformBit(true, '-')  == (false, 1)

    # number operator
    @test TransformBit(true, 'n')  == (true, 1)
    @test TransformBit(false, 'n') == (false, 0)

    # hole operator
    @test TransformBit(true, 'h')  == (true, 0)
    @test TransformBit(false, 'h') == (false, 1)

    # invalid op
    @test_throws AssertionError TransformBit(true, 'x')
end



# ---------------------------------------------------------------
# ApplyOperatorChunk
# ---------------------------------------------------------------
@testset "ApplyOperatorChunk" begin
    st = Dict(BitVector([1,0])=>1.0, BitVector([0,1])=>-0.5)

    # simple hopping c1† c2
    out = ApplyOperatorChunk("+-", [1,2], 1.0, st)
    @test haskey(out, BitVector([1,0]))
    @test out[BitVector([1,0])] ≈ (-0.5)

    # operator that kills state
    out2 = ApplyOperatorChunk("--", [1,2], 1.0, st)
    @test isempty(out2)

    # fermionic sign check: hopping over an occupied site
    st2 = Dict(BitVector([1,1,0])=>1.0)
    out3 = ApplyOperatorChunk("++", [3,1], 1.0, st2)
    # site1 occupied → c1 kills → should be empty
    @test isempty(out3)
end



# ---------------------------------------------------------------
# ApplyOperator
# ---------------------------------------------------------------
@testset "ApplyOperator" begin
    st = Dict(BitVector([1,0])=>1.0, BitVector([0,1])=>-0.5)
    op = [("+-", [1,2], 0.1), ("nh", [2,1], 1.0)]

    out = ApplyOperator(op, st)
    @test length(out) == 2
    @test out[BitVector([1,0])] ≈ -0.05
    @test out[BitVector([0,1])] ≈ -0.5
end



# ---------------------------------------------------------------
# OperatorMatrix
# ---------------------------------------------------------------
@testset "OperatorMatrix" begin
    basis = BasisStates(2)
    op = [("+-", [1,2], 0.5), ("n", [2], -1.0)]

    mat = OperatorMatrix(basis, op)
    @test size(mat) == (4,4)

    # test a couple of values
    @test mat[2,2] ≈ -1.0
    @test mat[3,2] ≈ 0.5
end



# ---------------------------------------------------------------
# StateOverlap
# ---------------------------------------------------------------
@testset "StateOverlap" begin
    s1 = Dict(BitVector([1,0])=>1.0, BitVector([0,1])=>-0.5)
    s2 = Dict(BitVector([1,1])=>0.5, BitVector([0,1])=>0.5)
    @test StateOverlap(s1,s2) ≈ -0.25
end



# ---------------------------------------------------------------
# ExpandIntoBasis
# ---------------------------------------------------------------
@testset "ExpandIntoBasis" begin
    basis = BasisStates(2)
    st = Dict(BitVector([1,0])=>2.0)
    coeffs = ExpandIntoBasis(st, basis)
    @test coeffs[3] == 2.0
    @test all(coeffs[[1,2,4]] .== 0)
end



# ---------------------------------------------------------------
# GetSector
# ---------------------------------------------------------------
@testset "GetSector" begin
    st = Dict(BitVector([1,0,1,0])=>1.0)
    @test GetSector(st, ['N']) == (2,)
    @test GetSector(st, ['Z']) == ( (1+1)-(0+0), )    # =2
    @test GetSector(st, ['N','Z']) == (2,2)
end



# ---------------------------------------------------------------
# RoundTo
# ---------------------------------------------------------------
@testset "RoundTo" begin
    @test RoundTo(1.122323, 1e-3) == 1.122
    @test RoundTo(1.122323, 1e-2) == 1.12
end



# ---------------------------------------------------------------
# PermuteSites (BitVector)
# ---------------------------------------------------------------
@testset "PermuteSites BitVector" begin
    bv = BitVector([1,1])
    new, s = PermuteSites(copy(bv), [2,1])
    @test new == BitVector([1,1])
    @test s == -1    # two fermions swapped → minus sign

    bv2 = BitVector([1,0,1])
    new2, s2 = PermuteSites(copy(bv2), [3,2,1])
    # manually check: two swaps, but middle site empty → only 1 fermion swap
    @test s2 == -1
end



# ---------------------------------------------------------------
# PermuteSites (Dict state)
# ---------------------------------------------------------------
@testset "PermuteSites Dict" begin
    st = Dict(BitVector([1,1])=>0.5, BitVector([0,1])=>0.3)
    new = PermuteSites(st, [2,1])

    @test new[BitVector([1,1])] == -0.5
    @test new[BitVector([1,0])] == 0.3
end



# ---------------------------------------------------------------
# Dagger
# ---------------------------------------------------------------
@testset "Dagger" begin
    opT, mem = Dagger("+--+", [1,4,3,2])
    @test opT == "-++-"
    @test mem == [2,3,4,1]

    op = [("++", [1,2], 1.), ("+n", [3,4], 2.)]
    out = Dagger(op)
    @test out[1][1] == "--"
    @test out[2][1] == "n-"
end



# ---------------------------------------------------------------
# VacuumState
# ---------------------------------------------------------------
@testset "VacuumState" begin
    b = BasisStates(3)
    vac = VacuumState(b)
    @test first(keys(vac)) == BitVector([0,0,0])
end



# ---------------------------------------------------------------
# DoesCommute
# ---------------------------------------------------------------
@testset "DoesCommute" begin
    basis = BasisStates(2)

    # n1  and n2 commute
    op1 = [("n", [1], 1.0)]
    op2 = [("n", [2], 1.0)]
    @test DoesCommute(op1, op2, basis)

    # creation at different sites anticommute → commutator ≠ 0
    ca = [("+", [1], 1.0)]
    cb = [("+", [2], 1.0)]
    @test !DoesCommute(ca, cb, basis)
end
