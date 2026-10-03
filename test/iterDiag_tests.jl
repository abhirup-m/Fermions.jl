using Random
using LinearAlgebra
using Random

# set deterministic seed for reproducible property tests
Random.seed!(12345)

# ---- Helpers ----
# quiet convenience for checking approximate equality with tolerance
isapprox_rel(a, b; atol=1e-8, rtol=1e-8) = isapprox(a, b; atol=atol, rtol=rtol)

# For temporary filesystem use
mktempdir_local(fn) = (dir = mktempdir(); try fn(dir) finally rm(dir; recursive=true) end)

# ---- Basic sanity tests for BasisStates family ----
@testset "BasisStates and BasisStates1p" begin
    # full basis 2 levels
    b2 = BasisStates(2)
    @test length(b2) == 4
    # totOccReq
    b2_one = BasisStates(2; totOccReq=1)
    @test length(b2_one) == 2
    # explicit totOcc/magz/local criterion variant
    b4 = BasisStates(4, [2], [0], x->true)
    @test length(b4) == 4
    # localCriteria restricting to first two sites sum==1
    b4_restricted = BasisStates(4, [2], [0], x->sum(x[1:2])==1)
    @test length(b4_restricted) == 2
    # BasisStates1p
    b1p10 = BasisStates1p(10)
    @test length(b1p10) == 10
    # each element is a Dict with exactly one key whose sum==1
    for s in b1p10
        @test length(keys(s)) == 1
        config = first(keys(s))
        @test sum(config) == 1
    end
end

# ---- OperatorMatrix / ApplyOperator / StateOverlap consistency tests ----
# We rely on OperatorMatrix, ApplyOperator, StateOverlap to be available in scope.
@testset "OperatorMatrix and GenCorrelation consistency on small examples" begin
    # 2-site example from docstring: operator = [("+-",[1,2],0.5), ("n",[2],-1.0)]
    basis = BasisStates(2)
    op = [("+-", [1,2], 0.5), ("n", [2], -1.0)]
    mat = OperatorMatrix(basis, op)
    @test size(mat) == (4, 4)
    # check diagonal entries for n(2) - they should be -1 when second bit==1
    # basis ordering is as in BasisStates doc (binary increasing)
    # We compute occupations from keys to be robust to any ordering assumptions
    diag_expected = zeros(4)
    for (i, ks) in enumerate(basis)
        cfg = first(keys(ks))
        diag_expected[i] = -1.0 * Float64(cfg[2])  # n at site2 contribution (coupling -1.0)
    end
    @test isapprox.(diag(mat), diag_expected) |> all

    # Build a normalized random state vector and compare GenCorrelation(dict, op) vs vector-matrix version
    for N in (2, 3, 4)
        bs = BasisStates(N)
        dim = length(bs)
        # random real vector
        v = randn(dim)
        v ./= norm(v)
        # transform vector -> dict form (use TransformState)
        state_dict = TransformState(v, bs)
        # operator matrix: identity for trivial test
        # create a random hermitian matrix and convert to OperatorMatrix by direct construction:
        # We'll use OperatorMatrix for builtin operators, so we instead construct matrix directly
        # using OperatorMatrix on a "constructed" operator via diagonal elements (use diagElements)
        # For generality, compute operator directly with OperatorMatrix: pick number operator on site1
        operator_def = [("n",[1],1.0)]
        M = OperatorMatrix(bs, operator_def)
        # GenCorrelation can accept vector+matrix or dict+operator-def
        # dict-version should call ApplyOperator internally; compare numeric
        val_dict = GenCorrelation(state_dict, operator_def)
        val_vec = GenCorrelation(v, M)
        @test isapprox_rel(val_dict, val_vec; atol=1e-8)
    end
end

# ---- CheckTrivial, OrganiseOperator tests ----
@testset "CheckTrivial and OrganiseOperator" begin
    # trivial case example: operator "++--" with members [1,1,2,3] -> trivial because first site repeated
    @test CheckTrivial(['+', '+', '-', '-'], [1,1,2,3]) == true
    # non-trivial example: distinct sites
    @test CheckTrivial(['+', '-'], [1,2]) == false

    # OrganiseOperator: when members are sorted -> identity sign 1
    op, mem, s = OrganiseOperator("+-", [1,2])
    @test op == "+-"
    @test mem == [1,2]
    @test s == 1

    # We test using a typical input per code: operator string of length same as members, characters like '+' or '-'
    op3, mem3, s3 = OrganiseOperator("+-+-", [1,3,2,4])
    @test issorted(mem3)
    @test abs(s3) == 1  # sign must be ±1

    # Sanity: reorganising an already sorted operator should not change sign
    op_sorted, mem_sorted, sign_sorted = OrganiseOperator("+-+-", [1,2,3,4])
    @test mem_sorted == [1,2,3,4]
    @test sign_sorted == 1
end

# ---- CreateDNH tests ----
@testset "CreateDNH properties" begin
    # For single-site basis, the "+ operator" matrix should exist via OperatorMatrix
    b1 = BasisStates(1)
    cdag = OperatorMatrix(b1, [("+", [1], 1.0)])
    ops = Dict{Tuple{String, Vector{Int64}}, Matrix{Float64}}()
    ops[("+",[1])] = cdag
    CreateDNH(ops, 1)
    @test haskey(ops, ("-", [1]))
    @test haskey(ops, ("n", [1]))
    @test haskey(ops, ("h", [1]))
    nmat = ops[("n",[1])]
    hmat = ops[("h",[1])]
    # For fermions, expectation: n + h = Identity (since h = c c† = 1 - n). So n + h == I
    Id = Matrix{Float64}(I, 2, 2)
    @test isapprox_rel(nmat + hmat, Id; atol=1e-10)
end

# ---- CombineRequirements and QuantumNosForBasis tests ----
@testset "CombineRequirements & QuantumNosForBasis" begin
    # simple occReq: occupancy equal to 1
    occReq = (o, N) -> o == 1
    comb = CombineRequirements(occReq, nothing)
    # produce BasisStates for N=3 and check
    basis3 = BasisStates(3)
    qnos = QuantumNosForBasis(collect(1:3), ['N'], basis3)
    @test length(qnos) == length(basis3)
    # The combined function should return a boolean for a tuple q
    @test isa(comb, Function)
    # Two-requirement combination: occ and magz
    magzReq = (m, N) -> m == 0
    comb2 = CombineRequirements(occReq, magzReq)
    @test isa(comb2, Function)
end

# ---- UpdateRequirements / MinceHamiltonian tests ----
@testset "UpdateRequirements / MinceHamiltonian logic" begin
    # simple ham flow for 4-site nearest neighbour hopping
    ham = [("+-",[1,2], -1.0), ("+-",[2,3], -1.0), ("+-",[3,4], -1.0)]
    # Partition into two blocks [1..2] and [3..4]
    hamFlow = MinceHamiltonian(ham, [2,4])
    @test length(hamFlow) == 2
    # first block should contain operator with max index ≤ 2
    for term in hamFlow[1]
        @test maximum(term[2]) ≤ 2
    end
    # UpdateRequirements should produce create/basket/newSitesFlow shapes consistent with hamFlow
    create, basket, newSitesFlow = UpdateRequirements(hamFlow)
    @test length(create) == length(hamFlow)
    @test length(basket) == length(hamFlow)
    @test length(newSitesFlow) == length(hamFlow)
end

# ---- Spectrum / Diagonalise / TruncateSpectrum / UpdateOldOperators tests ----
@testset "Spectrum & Diagonalisation consistency checks" begin
    # simple 2-site hopping Hamiltonian as docstring example
    basis2 = BasisStates(2)
    ham_op = [("+-", [1,2], -1.0), ("+-", [2,1], -1.0)]
    E, X = Spectrum(ham_op, basis2)
    # Compare with direct matrix diagonalization using OperatorMatrix
    Hmat = OperatorMatrix(basis2, ham_op)
    F = eigen(Hermitian(Hmat))
    # sort eigenvalues for comparsion
    @test length(E) == length(F.values)
    @test isapprox_rel(sort(E), sort(F.values); atol=1e-10)

    # Test Diagonalise with provided quantumNos: create simple block-diagonal ham
    Hblock = diagm(0 => [0.0, 1.0, 2.0, 3.0])
    # artificially set quantum numbers (2 blocks): first two states quantumNo=(0,), last two (1,)
    qnos = [(0,), (0,), (1,), (1,)]
    eigVals, eigVecs, qout = Diagonalise(Hblock, qnos)
    # eigenvalues should be sorted ascending
    @test issorted(eigVals)
    # quantum numbers must be permuted to align with sorted eigenvalues
    @test length(qout) == length(qnos)

    # TruncateSpectrum tests: create a rotation (identity) and degenerate eigenvalues
    rot = Matrix{Float64}(I, 6, 6)
    eigs = [0.0, 0.1, 0.1, 0.2, 0.2, 0.3]
    # no corrQuantumNoReq: request maxSize=3; with degTol should include equal energies at cutoff
    rot2, eigs2, q2 = TruncateSpectrum(nothing, rot, eigs, 3, 1e-10, [1,2], nothing, 10)
    @test length(eigs2) ≥ 3
    @test maximum(eigs2) ≤ eigs[3] + 1e-10
end

# ---- Randomised property tests: GenCorrelation equivalence & OperatorMatrix linearity ----
@testset "Randomized property-based tests" begin
    for N in (2, 3, 4, 5)
        bs = BasisStates(N)
        dim = length(bs)
        # create a few random operator-defs made from single-site number operators and random couplings
        # we will compare GenCorrelation(dict, operatorVector) with vector-matrix calculation
        for trial in 1:20
            v = randn(dim)
            v ./= norm(v)
            dict_state = TransformState(v, bs)
            # create a random Hermitian operator matrix by combining some basic operator-def terms
            operator_terms = Vector{Tuple{String, Vector{Int64}, Float64}}()
            # randomly select up to 3 single-site number operators and random couplings
            for site in rand(1:N, rand(1:3))
                push!(operator_terms, ("n", [site], randn()))
            end
            M = OperatorMatrix(bs, operator_terms)
            val1 = GenCorrelation(dict_state, operator_terms)
            val2 = GenCorrelation(v, M)
            @test isapprox_rel(val1, val2; atol=1e-10)
        end
    end
end

#=
Tests for the `transform` keyword of `IterDiag`.

Semantics being tested (from iterDiag.jl): after step `s` is diagonalised
(and possibly truncated), the Hamiltonian is rewritten in the eigenbasis and
enlarged as kron(diagm(eigVals), identityEnv). `transform` is applied to that
matrix *before* the terms of step `s+1` are added. It is therefore called
exactly length(hamltFlow) - 1 times, never on the first step's bare matrix
and never after the final step.

Run with `julia --project test/test_iterdiag_transform.jl`, or `include` it
from test/runtests.jl.
=#

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Open spinless chain with a generic (non-repeating) on-site potential so
# that spectra are free of accidental degeneracies.
function ChainTerms(L::Int64; t::Float64=1.0)
    terms = Tuple{String, Vector{Int64}, Float64}[]
    for i in 1:L-1
        push!(terms, ("+-", [i, i+1], -t))
        push!(terms, ("+-", [i+1, i], -t))
    end
    for i in 1:L
        push!(terms, ("n", [i], 0.3 * cos(1.7 * i)))
    end
    return terms
end

# Flow that starts with sites {1,2} and adds one site per step: L-1 steps.
ChainFlow(L::Int64) = MinceHamiltonian(ChainTerms(L), 2:L)

# IterDiag mutates both hamltFlow and correlationDefDict in place, so every
# call gets fresh copies.
function RunIterDiag(flow, maxSize; corrDefs=nothing, kwargs...)
    corr = isnothing(corrDefs) ?
        Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}}() :
        deepcopy(corrDefs)
    return IterDiag(deepcopy(flow), maxSize; correlationDefDict=corr, silent=true, kwargs...)
end

# Exact diagonalisation of a list of terms on sites 1..n.
function ExactEigen(terms, n::Int64)
    basis = BasisStates(n)
    return eigen(Hermitian(OperatorMatrix(basis, terms))), basis
end

function ExactExpectation(terms, operator, n::Int64)
    F, basis = ExactEigen(terms, n)
    ψ = F.vectors[:, 1]
    return ψ' * OperatorMatrix(basis, operator) * ψ
end

const CORR_DEFS = Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}}(
    "n1"    => [("n", [1], 1.0)],
    "hop12" => [("+-", [1, 2], 1.0), ("+-", [2, 1], 1.0)],
    "nLast" => [("n", [6], 1.0)],
)

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


L = 6
flow = ChainFlow(L)
nSteps = length(flow)
fullSize = 2^L        # large enough that nothing is ever truncated

@testset "default (nothing) matches identity transform" begin
    rDefault = RunIterDiag(flow, fullSize; corrDefs=CORR_DEFS)
    rNothing = RunIterDiag(flow, fullSize; corrDefs=CORR_DEFS, transform=nothing)
    rIdentity = RunIterDiag(flow, fullSize; corrDefs=CORR_DEFS, transform=H -> H)
    for key in ["energyPerSite"; collect(keys(CORR_DEFS))]
        @test rDefault[key] ≈ rNothing[key] atol=1e-12
        @test rDefault[key] ≈ rIdentity[key] atol=1e-12
    end
    # sanity check of the harness: untruncated run is exact
    F, _ = ExactEigen(ChainTerms(L), L)
    @test rIdentity["energyPerSite"] ≈ F.values[1] / L atol=1e-10
end

@testset "call count and input received by transform" begin
    seen = Matrix{Float64}[]
    recorder = H -> (push!(seen, copy(H)); H)
    RunIterDiag(flow, fullSize; transform=recorder)

    # called once between every pair of consecutive steps
    @test length(seen) == nSteps - 1

    for (s, H) in enumerate(seen)
        # after step s there are s+1 sites, enlarged by one new site
        @test size(H) == (2^(s + 2), 2^(s + 2))
        @test isdiag(H)

        # diagonal is the previous step's spectrum, each level repeated
        # for the two states of the incoming site
        F, _ = ExactEigen(vcat(flow[1:s]...), s + 1)
        @test diag(H) ≈ repeat(F.values, inner=2) atol=1e-10
    end
end

@testset "single-step flow never calls transform" begin
    calls = Ref(0)
    oneStep = MinceHamiltonian(ChainTerms(4), [4])
    res = RunIterDiag(oneStep, 2^4; transform=H -> (calls[] += 1; H))
    @test calls[] == 0
    F, _ = ExactEigen(ChainTerms(4), 4)
    @test res["energyPerSite"] ≈ F.values[1] / 4 atol=1e-10
end

@testset "constant shift H -> H + cI" begin
    c = 0.75
    shift = H -> H + c * I
    for (maxSize, label) in [(fullSize, "no truncation"), (8, "with truncation")]
        @testset "$label" begin
            base = RunIterDiag(flow, maxSize; corrDefs=CORR_DEFS)
            shifted = RunIterDiag(flow, maxSize; corrDefs=CORR_DEFS, transform=shift)

            # every call adds c to all levels, and the shift survives
            # every later rotation, so the final energy moves by c per call
            @test shifted["energyPerSite"] ≈ base["energyPerSite"] + c * (nSteps - 1) / L atol=1e-10

            # a uniform shift changes neither the eigenvectors nor which
            # states get truncated, so correlations are unchanged
            for key in keys(CORR_DEFS)
                @test shifted[key] ≈ base[key] atol=1e-8
            end
        end
    end
end

@testset "scaling H -> λH reproduces Σ_s λ^(n-s) H_s" begin
    # With no truncation every step is exact, so the final matrix is
    # the exact operator Σ_s λ^(nSteps-s) H_s written in some basis.
    for symmetries in (Char[], ['N'])
        for λ in (0.0, 0.5, 2.0)
            @testset "λ=$λ, symmetries=$symmetries" begin
                weighted = [(op, m, coupling * λ^(nSteps - s))
                            for (s, terms) in enumerate(flow) for (op, m, coupling) in terms]
                res = RunIterDiag(flow, fullSize; corrDefs=CORR_DEFS,
                                  symmetries=symmetries, transform=H -> λ * H)

                F, _ = ExactEigen(weighted, L)
                @test res["energyPerSite"] ≈ F.values[1] / L atol=1e-10

                # for λ = 0 only the last step survives and the ground
                # state is degenerate, so correlations are ill-defined
                if λ != 0
                    for (key, op) in CORR_DEFS
                        @test res[key] ≈ ExactExpectation(weighted, op, L) atol=1e-8
                    end
                end
            end
        end
    end
end

@testset "input stays within maxMaxSize under truncation" begin
    maxSize = 8
    seen = Matrix{Float64}[]
    RunIterDiag(ChainFlow(8), maxSize; maxMaxSize=maxSize,
                transform=H -> (push!(seen, copy(H)); H))
    @test length(seen) == length(ChainFlow(8)) - 1
    for H in seen
        @test size(H, 1) ≤ maxSize
        @test isdiag(H)
        levels = diag(H)[1:2:end]
        @test levels == diag(H)[2:2:end]
        @test issorted(levels)
    end
end

@testset "size-changing transform is rejected" begin
    shrink = H -> H[1:end-1, 1:end-1]
    @test_throws DimensionMismatch RunIterDiag(flow, fullSize; transform=shrink)
end
