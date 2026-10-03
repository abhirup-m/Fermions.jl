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


#=
Test suite for `IterDiag`, covering both its interface (argument checks,
output structure, accepted input forms) and its physics (agreement with
exact diagonalisation and with analytically solvable models).

Conventions assumed throughout, taken from the source:
  * spin orbitals are interleaved: site i has up = 2i-1, down = 2i,
    matching the magnetisation convention of BasisStates/QuantumNosForBasis;
  * energyPerSite = ground energy / (number of orbitals), not physical sites;
  * IterDiag mutates hamltFlow and every definition dict it is given, so all
    calls go through `Run`, which passes deep copies.

Without truncation (maxSize ≥ 2^orbitals) iterative diagonalisation is exact,
so most physics tests use that regime and compare against exact results.
With truncation the final Hamiltonian is P H P for a projector P, so results
must obey the variational bound, which the truncation tests check.
=#

const Term = Tuple{String, Vector{Int64}, Float64}
const CorrDict = Dict{String, Vector{Term}}
const VneDict = Dict{String, Vector{Int64}}
const MutInfoDict = Dict{String, NTuple{2, Vector{Int64}}}
const SpecDict = Dict{String, Dict{String, Vector{Term}}}

# ===========================================================================
# model builders
# ===========================================================================

# generic, non-repeating on-site potential: removes accidental degeneracies
GenericPotential(i) = 0.3 * cos(1.7 * i)

# open spinless chain: -t Σ (c†_i c_{i+1} + h.c.) + Σ ε_i n_i
function FreeChain(L::Int64; t::Float64=1.0, pot::Function=GenericPotential)
    terms = Term[]
    for i in 1:L-1
        push!(terms, ("+-", [i, i+1], -t))
        push!(terms, ("+-", [i+1, i], -t))
    end
    for i in 1:L
        pot(i) ≠ 0 && push!(terms, ("n", [i], pot(i)))
    end
    return terms
end

up(i) = 2i - 1
dn(i) = 2i

function SpinfulHop(i::Int64, j::Int64, t::Float64)
    terms = Term[]
    for σ in (up, dn)
        push!(terms, ("+-", [σ(i), σ(j)], -t))
        push!(terms, ("+-", [σ(j), σ(i)], -t))
    end
    return terms
end

# open Hubbard chain, written particle-hole symmetrically:
# H = -t Σ hops + U Σ n↑n↓ + Σ (ε_i - U/2 ∓ field) n_iσ
function HubbardChain(L::Int64; t::Float64=1.0, U::Float64=4.0,
                      pot::Function=i -> 0.0, field::Float64=0.0)
    terms = Term[]
    for i in 1:L-1
        append!(terms, SpinfulHop(i, i+1, t))
    end
    for i in 1:L
        push!(terms, ("n", [up(i)], pot(i) - U/2 - field))
        push!(terms, ("n", [dn(i)], pot(i) - U/2 + field))
        push!(terms, ("nn", [up(i), dn(i)], U))
    end
    return terms
end

# one physical site (two orbitals) per step
HubbardFlow(terms, L) = MinceHamiltonian(terms, 2:2:2L)

# S_i · S_j in fermionic form
function SpinSpin(i::Int64, j::Int64; scale::Float64=1.0)
    return Term[
        ("nn", [up(i), up(j)],  0.25scale),
        ("nn", [up(i), dn(j)], -0.25scale),
        ("nn", [dn(i), up(j)], -0.25scale),
        ("nn", [dn(i), dn(j)],  0.25scale),
        ("+-+-", [up(i), dn(i), dn(j), up(j)], 0.5scale),   # S+_i S-_j
        ("+-+-", [dn(i), up(i), up(j), dn(j)], 0.5scale),   # S-_i S+_j
    ]
end

NumberOp(orbitals) = Term[("n", [k], 1.0) for k in orbitals]
MagzOp(L) = vcat(Term[("n", [up(i)], 1.0) for i in 1:L], Term[("n", [dn(i)], -1.0) for i in 1:L])

AndersonImpurity(U) = Term[("n", [1], -U/2), ("n", [2], -U/2), ("nn", [1, 2], U)]

# ===========================================================================
# reference solvers
# ===========================================================================

function ExactSpectrum(terms, n::Int64; totOccReq=Int64[], magzReq=Int64[])
    basis = BasisStates(n; totOccReq=totOccReq, magzReq=magzReq)
    F = eigen(Hermitian(OperatorMatrix(basis, terms)))
    return F, basis
end

function ExactGroundState(terms, n::Int64; kwargs...)
    F, basis = ExactSpectrum(terms, n; kwargs...)
    return F.values[1], F.vectors[:, 1], basis
end

Expect(ψ, basis, op) = ψ' * OperatorMatrix(basis, op) * ψ

# entanglement of the first k orbitals, from the exact state vector.
# BasisStates orders site 1 as the most significant bit.
function LeadingEntropy(ψ::Vector{Float64}, n::Int64, k::Int64)
    M = reshape(ψ, 2^(n - k), 2^k)
    p = filter(>(1e-14), eigvals(Symmetric(transpose(M) * M)))
    return -sum(p .* log.(p))
end

# single-particle matrix of a quadratic, number-conserving Hamiltonian
function SingleParticleMatrix(terms, L::Int64)
    h = zeros(L, L)
    for (op, m, v) in terms
        if op == "+-"
            h[m[1], m[2]] += v
        elseif op == "n"
            h[m[1], m[1]] += v
        else
            error("non-quadratic term $op")
        end
    end
    return h
end

# free-fermion ground state: energy, ⟨c†_i c_j⟩, single-particle eigensystem
function FreeGroundState(terms, L::Int64; numParticles=nothing)
    F = eigen(Symmetric(SingleParticleMatrix(terms, L)))
    occ = isnothing(numParticles) ? findall(<(0), F.values) : 1:numParticles
    G = F.vectors[:, occ] * F.vectors[:, occ]'
    return sum(F.values[occ]), G, F
end

binaryEntropy(x) = (x > 1e-14 ? -x * log(x) : 0.0) + (1 - x > 1e-14 ? -(1 - x) * log(1 - x) : 0.0)
GaussianEntropy(G, A) = sum(binaryEntropy, clamp.(eigvals(Symmetric(G[A, A])), 0, 1))

# merge spectral coefficients (weight, pole) that sit at the same pole
function AggregatePoles(coeffs; wtol=1e-10, ptol=1e-7)
    out = Tuple{Float64, Float64}[]          # (pole, weight)
    for (w, p) in sort(filter(c -> abs(c[1]) > wtol, coeffs), by=last)
        if !isempty(out) && abs(out[end][1] - p) < ptol
            out[end] = (out[end][1], out[end][2] + w)
        else
            push!(out, (p, w))
        end
    end
    return out
end

# ===========================================================================
# runner
# ===========================================================================

function Run(flow, maxSize::Int64;
             corr=CorrDict(), vne=VneDict(), mutInfo=MutInfoDict(),
             specFunc=SpecDict(), kwargs...)
    return IterDiag(deepcopy(flow), maxSize;
                    correlationDefDict=deepcopy(corr),
                    vneDefDict=deepcopy(vne),
                    mutInfoDefDict=deepcopy(mutInfo),
                    specFuncDefDict=deepcopy(specFunc),
                    silent=true, kwargs...)
end

# ===========================================================================
# tests
# ===========================================================================

# ---------------------------------------------------------------------------
@testset "output structure" begin
    L = 4
    flow = MinceHamiltonian(FreeChain(L), 2:L)

    bare = Run(flow, 2^L)
    @test bare isa Dict{String, Any}
    @test Set(keys(bare)) == Set(["energyPerSite", "exitCode"])
    @test bare["energyPerSite"] isa Float64
    @test bare["exitCode"] == 0

    res = Run(flow, 2^L;
              corr=CorrDict("n1" => [("n", [1], 1.0)]),
              vne=VneDict("S12" => [1, 2]),
              mutInfo=MutInfoDict("I13" => ([1], [3])))

    # only requested quantities survive; the internal VNE / projector
    # entries with random names must be cleaned up
    @test Set(keys(res)) == Set(["energyPerSite", "exitCode", "n1", "S12", "I13"])
    @test res["exitCode"] == 0
    for key in ["energyPerSite", "n1", "S12", "I13"]
        @test res[key] isa Float64
        @test isfinite(res[key])
    end

    mktempdir() do dir
        spec = Run(flow, 2^L; dataDir=dir,
                   specFunc=SpecDict("A1" => Dict("create" => Term[("+", [1], 1.0)],
                                                  "destroy" => Term[("-", [1], 1.0)])))
        @test Set(keys(spec)) == Set(["energyPerSite", "exitCode", "A1",
                                      "specFuncOperators", "savePaths"])
        @test length(spec["savePaths"]) == length(flow) + 1   # + metadata
        @test all(isfile, spec["savePaths"])
        @test length(spec["A1"]) == length(flow)
        @test Set(keys(spec["specFuncOperators"]["A1"])) == Set(["create", "destroy"])
        @test length(spec["specFuncOperators"]["A1"]["create"]) == length(flow)
    end
end

# ---------------------------------------------------------------------------
@testset "argument validation" begin
    flow = MinceHamiltonian(FreeChain(4), 2:4)

    @test_throws AssertionError Run(flow, 16; maxMaxSize=8)
    @test_throws AssertionError Run(flow, 16; symmetries=['X'])
    @test_throws AssertionError Run(flow, 16; occReq=(o, N) -> true)               # needs 'N'
    @test_throws AssertionError Run(flow, 16; magzReq=(m, N) -> true)              # needs 'S'
    @test_throws AssertionError Run(flow, 16; corr=CorrDict("energyPerSite" => [("n", [1], 1.0)]))
    @test_throws AssertionError Run(flow, 16; corr=CorrDict("x" => [("n", [1], 1.0)]),
                                              vne=VneDict("x" => [1]))
    @test_throws AssertionError Run(flow, 16; vne=VneDict("S" => [2, 1]))
    @test_throws AssertionError Run(flow, 16; mutInfo=MutInfoDict("I" => ([3, 1], [2])))
    @test_throws AssertionError Run(flow, 16;
        specFunc=SpecDict("A" => Dict("create" => Term[("+", [1], 1.0)])))

    # maxMaxSize equal to maxSize is allowed
    @test Run(flow, 16; maxMaxSize=16)["exitCode"] == 0
end

# ---------------------------------------------------------------------------
@testset "input forms" begin
    L = 6
    terms = FreeChain(L)
    corr = CorrDict(
        "n1"    => [("n", [1], 1.0)],
        "hop14" => [("+-", [1, 4], 1.0), ("+-", [4, 1], 1.0)],
        "nn36"  => [("nn", [3, 6], 1.0)],
    )
    ref = Run(MinceHamiltonian(terms, 2:L), 2^L; corr=corr)

    @testset "partition $p" for p in ([2, 4, 6], [3, 6], [2, 3, 6], [6])
        res = Run(MinceHamiltonian(terms, p), 2^L; corr=corr)
        for key in ["energyPerSite"; collect(keys(corr))]
            @test res[key] ≈ ref[key] atol=1e-10
        end
    end

    @testset "unsorted members / reordered operators" begin
        # c†_{i+1} c_i  =  -c_i c†_{i+1}
        rewritten = Term[]
        for (op, m, v) in terms
            if op == "+-" && m[1] > m[2]
                push!(rewritten, ("-+", reverse(m), -v))
            else
                push!(rewritten, (op, m, v))
            end
        end
        res = Run(MinceHamiltonian(rewritten, 2:L), 2^L; corr=corr)
        for key in ["energyPerSite"; collect(keys(corr))]
            @test res[key] ≈ ref[key] atol=1e-10
        end

        # the same rewriting applied to the correlation definition
        corrRewritten = CorrDict("hop14" => [("+-", [1, 4], 1.0), ("-+", [1, 4], -1.0)])
        @test Run(MinceHamiltonian(terms, 2:L), 2^L; corr=corrRewritten)["hop14"] ≈ ref["hop14"] atol=1e-10
    end

    @testset "'h' operator" begin
        # ε n = ε - ε h, so swapping n for h shifts the energy by Σε
        swapped = [op == "n" ? ("h", m, -v) : (op, m, v) for (op, m, v) in terms]
        constant = sum(v for (op, _, v) in terms if op == "n")
        res = Run(MinceHamiltonian(swapped, 2:L), 2^L; corr=corr)
        @test res["energyPerSite"] * L ≈ ref["energyPerSite"] * L - constant atol=1e-10
        for key in keys(corr)
            @test res[key] ≈ ref[key] atol=1e-8
        end
    end
end

# ---------------------------------------------------------------------------
@testset "free fermions" begin

    @testset "uniform chain energy (analytic)" begin
        L = 8
        res = Run(MinceHamiltonian(FreeChain(L; pot=i -> 0.0), 2:L), 2^L)
        εk = [-2cos(k * π / (L + 1)) for k in 1:L]
        @test res["energyPerSite"] ≈ sum(filter(<(0), εk)) / L atol=1e-10
    end

    L = 6
    terms = FreeChain(L)
    flow = MinceHamiltonian(terms, 2:L)
    E0, G, sp = FreeGroundState(terms, L)

    @testset "energy and one-body correlations" begin
        corr = CorrDict("n$i" => [("n", [i], 1.0)] for i in 1:L)
        corr["Ntot"] = NumberOp(1:L)
        corr["hop14"] = [("+-", [1, 4], 1.0), ("+-", [4, 1], 1.0)]   # Jordan-Wigner string across 3 steps
        corr["hop25"] = [("+-", [2, 5], 1.0), ("+-", [5, 2], 1.0)]
        corr["Htot"] = terms
        res = Run(flow, 2^L; corr=corr)

        @test res["energyPerSite"] ≈ E0 / L atol=1e-10
        @test res["Htot"] ≈ E0 atol=1e-10
        @test res["Ntot"] ≈ count(<(0), sp.values) atol=1e-10
        for i in 1:L
            @test res["n$i"] ≈ G[i, i] atol=1e-10
        end
        @test res["hop14"] ≈ 2G[1, 4] atol=1e-10
        @test res["hop25"] ≈ 2G[2, 5] atol=1e-10
    end

    @testset "two-body correlations obey Wick's theorem" begin
        sitePairs = [(1, 3), (2, 5), (1, 6)]
        corr = CorrDict("nn$i$j" => [("nn", [i, j], 1.0)] for (i, j) in sitePairs)
        corr["four"] = [("+-+-", [1, 2, 4, 3], 1.0)]
        res = Run(flow, 2^L; corr=corr)
        for (i, j) in sitePairs
            @test res["nn$i$j"] ≈ G[i, i] * G[j, j] - G[i, j]^2 atol=1e-10
        end
        # c†1 c2 c†4 c3 = c†1 (δ24 - c†4 c2) c3 = -c†1 c†4 c2 c3 for distinct sites;
        # ⟨c†a c†b c_c c_d⟩ = G_ad G_bc - G_ac G_bd  with (a,b,c,d) = (1,4,2,3)
        @test res["four"] ≈ -(G[1, 3] * G[4, 2] - G[1, 2] * G[4, 3]) atol=1e-10
    end

    @testset "entanglement matches Gaussian formula" begin
        subsystems = Dict("S1" => [1], "S12" => [1, 2], "S23" => [2, 3], "S24" => [2, 4],
                          "S123" => [1, 2, 3], "S456" => [4, 5, 6])
        mutual = Dict("I1_3" => ([1], [3]), "I12_45" => ([1, 2], [4, 5]), "I2_3" => ([2], [3]))
        res = Run(flow, 2^L; vne=VneDict(subsystems), mutInfo=MutInfoDict(mutual))

        @test res["exitCode"] == 0
        for (name, A) in subsystems
            @test res[name] ≈ GaussianEntropy(G, A) atol=1e-8
        end
        @test res["S456"] ≈ res["S123"] atol=1e-8         # complements of a pure state
        for (name, (A, B)) in mutual
            expected = GaussianEntropy(G, A) + GaussianEntropy(G, B) - GaussianEntropy(G, sort([A; B]))
            @test res[name] ≈ expected atol=1e-8
            @test res[name] ≥ -1e-10
        end
    end

    @testset "fixed particle number via corrOccReq (k=$k)" for k in 1:3
        corr = CorrDict("n$i" => [("n", [i], 1.0)] for i in 1:L)
        corr["Ntot"] = NumberOp(1:L)
        corr["Htot"] = terms
        res = Run(flow, 2^L; corr=corr, symmetries=['N'], corrOccReq=(o, N) -> o == k)

        Ek, Gk, _ = FreeGroundState(terms, L; numParticles=k)
        @test res["Ntot"] ≈ k atol=1e-10
        @test res["Htot"] ≈ Ek atol=1e-10
        for i in 1:L
            @test res["n$i"] ≈ Gk[i, i] atol=1e-10
        end
        # energyPerSite always reports the global ground state
        @test res["energyPerSite"] ≈ E0 / L atol=1e-10
    end
end

# ---------------------------------------------------------------------------
@testset "Hubbard dimer (analytic)" begin
    t, U = 1.0, 3.0
    flow = HubbardFlow(HubbardChain(2; t=t, U=U), 2)
    root = sqrt(U^2 + 16t^2)
    E0 = -U/2 - root/2                       # half-filled singlet, p-h symmetric form
    d = 1/4 - U / (4root)                    # ⟨n↑n↓⟩ per site (Hellmann–Feynman)
    S1 = -2d * log(d) - 2(1/2 - d) * log(1/2 - d)

    corr = CorrDict(
        "double1" => [("nn", [1, 2], 1.0)],
        "double2" => [("nn", [3, 4], 1.0)],
        "SS"      => SpinSpin(1, 2),
        "Ntot"    => NumberOp(1:4),
        "Sz"      => MagzOp(2),
        "n1up"    => [("n", [1], 1.0)],
    )
    res = Run(flow, 16; corr=corr,
              vne=VneDict("S1" => [1, 2], "S2" => [3, 4], "Sall" => [1, 2, 3, 4]),
              mutInfo=MutInfoDict("I" => ([1, 2], [3, 4])))

    @test res["energyPerSite"] ≈ E0 / 4 atol=1e-10
    @test res["Ntot"] ≈ 2 atol=1e-10
    @test res["Sz"] ≈ 0 atol=1e-10
    @test res["n1up"] ≈ 0.5 atol=1e-10
    @test res["double1"] ≈ d atol=1e-10
    @test res["double2"] ≈ d atol=1e-10
    @test res["SS"] ≈ -0.75 * (1 - 2d) atol=1e-10      # singlet: ⟨S_tot²⟩ = 0
    @test res["S1"] ≈ S1 atol=1e-8
    @test res["S2"] ≈ S1 atol=1e-8
    @test res["Sall"] ≈ 0 atol=1e-8
    @test res["I"] ≈ 2S1 atol=1e-8                     # pure bipartite state
    @test res["exitCode"] == 0

    @testset "limits" begin
        # U = 0: free electrons, d = 1/4
        r0 = Run(HubbardFlow(HubbardChain(2; U=0.0), 2), 16; corr=CorrDict("d" => [("nn", [1, 2], 1.0)]))
        @test r0["energyPerSite"] ≈ -2.0 / 4 atol=1e-10      # two electrons in the bonding level
        @test r0["d"] ≈ 0.25 atol=1e-10
        # large U: local moments, Heisenberg singlet
        rU = Run(HubbardFlow(HubbardChain(2; U=200.0), 2), 16; corr=CorrDict("SS" => SpinSpin(1, 2)))
        @test rU["SS"] ≈ -0.75 atol=1e-3
    end
end

# ---------------------------------------------------------------------------
@testset "Kondo impurity" begin
    U, J = 2.0, 1.0

    @testset "two-site singlet (analytic)" begin
        # step 1: impurity alone; step 2: exchange with one bath site.
        # The spin-flip terms span both steps.
        flow = [AndersonImpurity(U), SpinSpin(1, 2; scale=J)]
        res = Run(flow, 16;
                  corr=CorrDict("SS" => SpinSpin(1, 2), "nd" => NumberOp([1, 2]),
                                "Sz" => MagzOp(2)),
                  vne=VneDict("Simp" => [1, 2]))
        @test res["energyPerSite"] ≈ (-U/2 - 3J/4) / 4 atol=1e-10
        @test res["SS"] ≈ -0.75 atol=1e-10
        @test res["nd"] ≈ 1 atol=1e-10
        @test res["Sz"] ≈ 0 atol=1e-10
        @test res["Simp"] ≈ log(2) atol=1e-8           # maximally entangled spin singlet
    end

    @testset "impurity + 3-site bath vs exact diagonalisation" begin
        nOrb = 8
        exchange = SpinSpin(1, 2; scale=J)
        bath = vcat(SpinfulHop(2, 3, 1.0), SpinfulHop(3, 4, 1.0),
                    Term[("n", [up(3)], 0.1), ("n", [dn(3)], 0.1)])
        flow = [AndersonImpurity(U), exchange, bath[findall(x -> maximum(x[2]) ≤ 6, bath)],
                bath[findall(x -> maximum(x[2]) > 6, bath)]]
        allTerms = vcat(flow...)
        E, ψ, basis = ExactGroundState(allTerms, nOrb)

        corr = CorrDict("SS" => SpinSpin(1, 2), "SS13" => SpinSpin(1, 3), "Htot" => allTerms)
        res = Run(flow, 2^nOrb; corr=corr)
        @test res["energyPerSite"] ≈ E / nOrb atol=1e-10
        @test res["Htot"] ≈ E atol=1e-10

        # the ground state may be degenerate (e.g. spin multiplets); spin
        # correlations are only compared if it is not
        F, _ = ExactSpectrum(allTerms, nOrb)
        if F.values[2] - F.values[1] > 1e-8
            @test res["SS"] ≈ Expect(ψ, basis, SpinSpin(1, 2)) atol=1e-8
            @test res["SS13"] ≈ Expect(ψ, basis, SpinSpin(1, 3)) atol=1e-8
        end

        # spin-flip terms written in sorted normal order
        reorderedExchange = [t for t in exchange if t[1] == "nn"]
        append!(reorderedExchange, Term[("+--+", [1, 2, 3, 4], -0.5J),   # c†1 c2 c†4 c3
                                        ("-++-", [1, 2, 3, 4], -0.5J)])  # c†2 c1 c†3 c4
        flow2 = [flow[1], reorderedExchange, flow[3], flow[4]]
        @test Run(flow2, 2^nOrb)["energyPerSite"] ≈ E / nOrb atol=1e-10
    end
end

# ---------------------------------------------------------------------------
@testset "symmetry sectors" begin
    L = 3
    nOrb = 2L
    terms = HubbardChain(L; U=3.0, pot=GenericPotential, field=0.05)
    flow = HubbardFlow(terms, L)
    E, ψ, basis = ExactGroundState(terms, nOrb)
    corr = CorrDict("SS12" => SpinSpin(1, 2), "double2" => [("nn", [3, 4], 1.0)],
                    "hop13up" => [("+-", [1, 5], 1.0), ("+-", [5, 1], 1.0)],
                    "Ntot" => NumberOp(1:nOrb), "Sz" => MagzOp(L), "Htot" => terms)

    @testset "symmetries = $syms" for syms in (Char[], ['N'], ['S'], ['N', 'S'])
        res = Run(flow, 2^nOrb; corr=corr, symmetries=syms)
        @test res["energyPerSite"] ≈ E / nOrb atol=1e-10
        for (name, op) in corr
            @test res[name] ≈ Expect(ψ, basis, op) atol=1e-8
        end
    end

    @testset "sector N=$N, Sz=$M" for (N, M) in ((3, 1), (3, -1), (2, 0), (4, 0))
        Esec, ψsec, bsec = ExactGroundState(terms, nOrb; totOccReq=N, magzReq=M)
        res = Run(flow, 2^nOrb; corr=corr, symmetries=['N', 'S'],
                  corrOccReq=(o, n) -> o == N, corrMagzReq=(m, n) -> m == M)
        @test res["Ntot"] ≈ N atol=1e-10
        @test res["Sz"] ≈ M atol=1e-10
        @test res["Htot"] ≈ Esec atol=1e-10
        @test res["SS12"] ≈ Expect(ψsec, bsec, SpinSpin(1, 2)) atol=1e-8
    end
end

# ---------------------------------------------------------------------------
@testset "truncation" begin
    L = 4
    nOrb = 2L
    terms = HubbardChain(L; U=2.0, pot=GenericPotential, field=0.05)
    flow = HubbardFlow(terms, L)
    E, _, _ = ExactGroundState(terms, nOrb)
    corr = CorrDict("Htot" => terms, "Ntot" => NumberOp(1:nOrb))

    @testset "maxSize = $maxSize" for maxSize in (16, 32, 64, 128)
        res = Run(flow, maxSize; corr=corr)
        # final Hamiltonian is P H P: variational upper bound on the true energy
        @test res["energyPerSite"] * nOrb ≥ E - 1e-10
        # the reported energy is the expectation value of the full H in the reported state
        @test res["Htot"] ≈ res["energyPerSite"] * nOrb atol=1e-8
        @test isapprox(res["Ntot"], round(res["Ntot"]); atol=1e-8)
    end

    @testset "no truncation is exact" begin
        @test Run(flow, 2^nOrb)["energyPerSite"] ≈ E / nOrb atol=1e-10
    end

    @testset "degenTol keeps degenerate multiplets intact" begin
        # SU(2)-symmetric model: many exact degeneracies at the cut
        su2 = HubbardChain(L; U=2.0, pot=GenericPotential)
        Esu2, _, _ = ExactGroundState(su2, nOrb)
        res = Run(HubbardFlow(su2, L), 16; corr=CorrDict("Htot" => su2))
        @test res["energyPerSite"] * nOrb ≥ Esu2 - 1e-10
        @test res["Htot"] ≈ res["energyPerSite"] * nOrb atol=1e-8
    end

    @testset "occReq keeps the target sector" begin
        half = nOrb ÷ 2
        Ehalf, _, _ = ExactGroundState(terms, nOrb; totOccReq=half)
        for maxSize in (16, 64)
            res = Run(flow, maxSize; corr=corr, symmetries=['N'],
                      occReq=(o, n) -> o == n ÷ 2, corrOccReq=(o, n) -> o == n ÷ 2)
            @test res["Ntot"] ≈ half atol=1e-10
            @test res["Htot"] ≥ Ehalf - 1e-10
        end
        res = Run(flow, 2^nOrb; corr=corr, symmetries=['N'], corrOccReq=(o, n) -> o == n ÷ 2)
        @test res["Htot"] ≈ Ehalf atol=1e-10
    end
end

# ---------------------------------------------------------------------------
@testset "spectral functions" begin
    probe(i) = Dict("create" => Term[("+", [i], 1.0)], "destroy" => Term[("-", [i], 1.0)])

    @testset "free chain: poles at ε_k with weight |φ_k(i)|²" begin
        L = 6
        terms = FreeChain(L)
        flow = MinceHamiltonian(terms, 2:L)
        mktempdir() do dir
            res = Run(flow, 2^L; dataDir=dir, specFunc=SpecDict("A1" => probe(1), "A3" => probe(3)))
            @test length(res["A1"]) == length(flow)

            # every step (untruncated, non-degenerate ground state) obeys the
            # sum rule ⟨{c, c†}⟩ = 1, split as hole weight ⟨n⟩ and particle weight 1-⟨n⟩
            for s in 1:length(flow)
                coeffs = res["A1"][s]
                @test !isempty(coeffs)
                @test all(c -> c[1] ≥ -1e-12, coeffs)
                _, Gs, _ = FreeGroundState(vcat(flow[1:s]...), s + 1)
                @test sum(first, coeffs) ≈ 1 atol=1e-8
                @test sum(c[1] for c in coeffs if c[2] < 0; init=0.0) ≈ Gs[1, 1] atol=1e-8
            end

            _, _, sp = FreeGroundState(terms, L)
            for (name, i) in (("A1", 1), ("A3", 3))
                poles = AggregatePoles(res[name][end])
                expected = [(sp.values[k], sp.vectors[i, k]^2) for k in 1:L if sp.vectors[i, k]^2 > 1e-10]
                @test length(poles) == length(expected)
                for ((p, w), (pe, we)) in zip(poles, sort(expected, by=first))
                    @test p ≈ pe atol=1e-8
                    @test w ≈ we atol=1e-8
                end
            end
        end
    end

    @testset "Hubbard dimer: sum rule and particle-hole symmetry" begin
        flow = HubbardFlow(HubbardChain(2; U=3.0), 2)
        mktempdir() do dir
            res = Run(flow, 16; dataDir=dir, specFunc=SpecDict("Aup" => probe(1)))
            coeffs = res["Aup"][end]
            @test sum(first, coeffs) ≈ 1 atol=1e-8
            @test sum(c[1] for c in coeffs if c[2] > 0) ≈ 0.5 atol=1e-8
            poles = AggregatePoles(coeffs)
            particle = [(p, w) for (p, w) in poles if p > 0]
            hole = sort([(-p, w) for (p, w) in poles if p < 0], by=first)
            @test length(particle) == length(hole)
            for ((p1, w1), (p2, w2)) in zip(particle, hole)
                @test p1 ≈ p2 atol=1e-8
                @test w1 ≈ w2 atol=1e-8
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Suspected bugs found while writing these tests. They are marked broken so
# the suite stays green; Julia reports "Unexpected Pass" once they are fixed,
# at which point change @test_broken to @test.
@testset "known issues" begin

    # save=true without spectral functions indexes `savePaths`, which is
    # `nothing` because SetupDataWrite only runs when specFuncNames is non-empty.
    mktempdir() do dir
        @test_broken begin
            Run(MinceHamiltonian(FreeChain(4), 2:4), 16; save=true, dataDir=dir)
            true
        end
    end

    # QuantumNosForBasis assigns spin by position *within the new sites*
    # ((-1)^(i+1) with i = 1, 2, ...), so a step that adds a lone down orbital
    # (an even site) is labelled as spin up. The blocks used by Diagonalise
    # are then wrong and the spectrum is wrong.
    let terms = HubbardChain(2; U=3.0, field=0.05)
        E, _, _ = ExactGroundState(terms, 4)
        # steps add orbitals {1,2}, then {3}, then {4}
        @test_broken Run(MinceHamiltonian(terms, 2:4), 16; symmetries=['S'])["energyPerSite"] ≈ E / 4 atol=1e-10
    end

    # With symmetries ['N','S'] and only magzReq, CombineRequirements applies
    # magzReq to q[1], which is the occupancy, not the magnetisation.
    let req = CombineRequirements(nothing, (m, N) -> m == 0)
        @test_broken req((2, 0), 4)          # (N, Sz) = (2, 0) should satisfy Sz == 0
    end
end
