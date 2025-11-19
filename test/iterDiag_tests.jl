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
    @test CheckTrivial("++--", [1,1,2,3]) == true
    # non-trivial example: distinct sites
    @test CheckTrivial("+-", [1,2]) == false

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

    # UpdateOldOperators: construct a small toy set of ops and check resulting hamltMatrix shape
    # Setup operators for 2-site system
    basis_small = BasisStates(2)
    ops = Dict{Tuple{String, Vector{Int64}}, Matrix{Float64}}()
    ops[("+",[1])] = OperatorMatrix(basis_small, [("+",[1],1.0)])
    ops = CreateDNH(ops, 1)
    ops[("+",[2])] = OperatorMatrix(basis_small, [("+",[2],1.0)])
    ops = CreateDNH(ops, 2)
    # rotation matrix = identity of size 4, eigVals of length 4
    eigenV = [0.0, 0.5, 1.0, 2.0]
    identityEnv = Diagonal(Bool[true])  # trivial env
    newBasket = [("+",[1]), ("n",[1]), ("h",[1]), ("+",[2]), ("n",[2]), ("h",[2])]
    corrOperatorDict = Dict("A" => nothing, "B" => (OperatorMatrix(basis_small, [("n",[1],1.0)])))
    hamltMatrix, ops2, baz, corrOut = UpdateOldOperators(eigenV, identityEnv, newBasket, ops, Matrix{Float64}(I,4,4), Matrix{Float64}(I,4,4), corrOperatorDict)
    @test size(hamltMatrix,1) == 4  # kron(diagm(eigVals), identityEnv) shape
    # rotated and enlarged operators should be present in ops2
    @test all(k in keys(ops2) for k in keys(newBasket))
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

# ---- Physics-level tests: Free fermion chain (tight-binding), compare exact vs IterDiag ----
@testset "Physics regression: free fermion tight-binding chain (exact vs IterDiag)" begin
    # small chain N=4, nearest neighbour hopping -t(c†_i c_{i+1} + h.c.)
    # We'll create hamltFlow by MinceHamiltonian with partitions that add 1 or 2 sites per step.
    N = 4
    t = 1.0
    # construct full Hamiltonian as list of terms
    ham_full = Tuple{String, Vector{Int64}, Float64}[]
    for i in 1:N-1
        push!(ham_full, ("+-", [i, i+1], -t))
        push!(ham_full, ("+-", [i+1, i], -t))
    end

    # partition into 4 single-site additions (add one site per step)
    hamflow = MinceHamiltonian(ham_full, collect(1:N))

    # correlation to compute: number operator at site 1
    corrDef = Dict("n1" => [("n",[1],1.0)])

    # Run IterDiag in a temporary directory with very large maxSize to emulate exact diagonalization
    mktempdir_local(dir -> begin
        results = IterDiag(hamflow, 2^N; # maxSize big enough to avoid truncation (use full Hilbert-space)
                           symmetries = Char[], # no symmetry for simplicity
                           correlationDefDict = corrDef,
                           quantumNoReq = nothing,
                           corrQuantumNoReq = nothing,
                           degenTol = 1e-10,
                           dataDir = dir,
                           silent = true,
                           specFuncNames = String[],
                           maxMaxSize = 0,
                           calculateThroughout = false,
                          )
        @test haskey(results, "energyPerSite")
        gs_energy_per_site = results["energyPerSite"]
        # Now compute exact spectrum via OperatorMatrix & eigen
        basis = BasisStates(N)
        Hmat = OperatorMatrix(basis, ham_full)
        F = eigen(Hermitian(Hmat))
        # ground-state energy intensive check:
        Egs_exact = minimum(F.values)
        @test isapprox_rel(Egs_exact / N, gs_energy_per_site; atol=1e-10)
        # ground-state correlation: compute using ground state eigenvector from F
        gs_index = argmin(F.values)
        gs_vec = F.vectors[:, gs_index]
        # expectation of n_1 via matrix
        n1_mat = OperatorMatrix(basis, [("n",[1], 1.0)])
        exp_n1_exact = gs_vec' * n1_mat * gs_vec
        # compare with IterDiag result (stored in results["n1"])
        @test isapprox_rel(results["n1"], exp_n1_exact; atol=1e-10)
    end)
end

# ---- Entanglement / Reduced density matrix / VonNEntropy tests ----
@testset "ReducedDM and VonNEntropy consistency tests (small free-fermion states)" begin
    # We'll use a 4-site chain and compute entanglement entropy of first two sites.
    N = 4
    # simple 2-particle state: Slater determinant with particles on sites 1 and 3 (a product basis state)
    basis = BasisStates(N)
    # create a pure state that is superposition of two configurations to have non-zero entanglement
    # for small test, take equally weighted superposition of |1001> and |0110>
    cfg1 = BitVector([1,0,0,1])
    cfg2 = BitVector([0,1,1,0])
    state = Dict{BitVector, Float64}()
    state[cfg1] = 1/sqrt(2)
    state[cfg2] = 1/sqrt(2)
    # compute reduced DM for subsystem sites [1,2]
    rd = ReducedDM(state, [1,2])
    # VonNEntropy via matrix form
    S1 = VonNEntropy(rd)
    # Via state-based routine
    S2 = VonNEntropy(state, [1,2])
    @test isapprox_rel(S1, S2; atol=1e-10)
    # von Neumann entropy should be positive and ≤ ln(4) for 2 qubits
    @test S1 ≥ 0
    @test S1 ≤ log(4) + 1e-12
end

# ---- Spectral function / SpectralCoefficients smoke tests (small) ----
@testset "SpectralCoefficients and SpecFunc smoke tests" begin
    # Use a tiny system with known excitations: N=2 single particle levels; basis of 2 sites; simple spectrum
    N = 2
    basis = BasisStates(N)
    # construct a toy eigVecs/eigVals: use OperatorMatrix eigenvectors for H with one-particle hopping
    ham = [("+-",[1,2], -1.0), ("+-",[2,1], -1.0)]
    H = OperatorMatrix(basis, ham)
    F = eigen(Hermitian(H))
    # convert eigVecs to vectors (columns)
    eigVecs = [collect(vec) for vec in eachcol(F.vectors)]
    eigVals = F.values
    # probe operator choose number operator on site1 as create/destroy (not physical but a test)
    probes = Dict("create" => OperatorMatrix(basis, [("n",[1],1.0)]),
                  "destroy" => OperatorMatrix(basis, [("n",[1],1.0)]))
    # compute spectral coefficients
    coeffs = SpectralCoefficients(eigVecs, eigVals, probes)
    @test isa(coeffs, Vector{NTuple{2, Float64}})
    # now compute SpecFunc from the coefficients on a small frequency grid
    freq = collect(range(-3, stop=3, length=41))
    sf = SpecFunc(coeffs, freq, 0.1; normalise=true)
    @test length(sf) == length(freq)
    @test all(x -> x ≥ -1e-14, sf)  # spec func should be non-negative (up to tiny numerical noise)
end

# ---- IterSpecFunc / IterSpectralCoeffs integration test (smoke) ----
@testset "IterSpectralCoeffs / IterSpecFunc smoke integration" begin
    # We'll do a tiny IterDiag run that writes spec operator state and then read it back via IterSpectralCoeffs.
    # Build a 2-site hopping model and request spectral function on a single operator (e.g., n1)
    N = 2
    ham_full = [("+-",[1,2], -1.0), ("+-",[2,1], -1.0)]
    hamflow = MinceHamiltonian(ham_full, collect(1:N))
    corrDef = Dict("n1" => [("n",[1],1.0)])
    mktempdir_local(dir -> begin
        # Run IterDiag with specFuncDefDict so saved files exist
        out = IterDiag(hamflow, 2^N; symmetries = Char[], correlationDefDict = corrDef,
                       quantumNoReq = nothing, corrQuantumNoReq = nothing,
                       degenTol = 1e-10, dataDir = dir, silent = true,
                       specFuncNames = ["n1"], maxMaxSize = 0, calculateThroughout = false)
        # if specFuncNames not empty, IterDiag returns (results, savePaths, specFuncOperators)
        results = out[1]
        savePaths = out[2]
        specOps = out[3]
        # Consolidate spec operators and call IterSpectralCoeffs
        # specOps is mapping corrName -> vector of matrices; choose the non-nothing entries
        corrVec = specOps["n1"]
        # get frequency grid
        freq = collect(range(-3.0, stop=3.0, length=31))
        # compute IterSpectralCoeffs directly
        coeffs_iter = IterSpectralCoeffs(savePaths, corrVec; degenTol=1e-10, silent=true)
        @test isa(coeffs_iter, Vector{NTuple{2, Float64}})
        # compute integrated spectral function via IterSpecFunc (standDev small)
        sf = IterSpecFunc(savePaths, corrVec, freq, 0.1; silent=true)
        @test length(sf) == length(freq)
    end)
end

# ---- UpdateRequirements operator-flow extension tests ----
@testset "UpdateRequirements operator-flow extension" begin
    # Construct operator A that spans sites 1..4 across flow; ensure UpdateRequirements(operator, newSitesFlow) works
    # Create a fictitious operator spanning sites [1,2,3,4]
    operator = [("+ - + -", [1,2,3,4], 1.0)] rescue [("+-+-",[1,2,3,4], 1.0)]
    # Build newSitesFlow: adding sites one by one
    newSitesFlow = [[1], [2], [3], [4]]
    create, retain = UpdateRequirements(operator, newSitesFlow)
    @test length(create) == length(newSitesFlow)
    @test length(retain) == length(newSitesFlow)
end

# ---- MinceHamiltonian edge cases ----
@testset "MinceHamiltonian edge cases" begin
    # Hamiltonian with single-term acting across entire chain; partition into two pieces
    ham = [("+-",[1,4], -1.0)]
    hamflow = MinceHamiltonian(ham, [2,4])
    # Since max index 4 belongs to second subspace, hamflow[2] should contain the term
    @test length(hamflow[1]) == 0
    @test length(hamflow[2]) == 1
end

# ---- Final note tests that ensure no major functions crash under random small inputs ----
@testset "Smoke tests: no crash on random small flows" begin
    for trial in 1:6
        N = rand(2:5)
        # random nearest neighbor hopping with random couplings
        ham = Tuple{String, Vector{Int64}, Float64}[]
        for i in 1:N-1
            push!(ham, ("+-", [i, i+1], randn()))
            push!(ham, ("+-", [i+1, i], randn()))
        end
        # random partition
        parts = sort(unique([rand(1:N) for _ in 1:rand(1:3)]))
        hamflow = MinceHamiltonian(ham, parts)
        # run a minimal IterDiag with small maxSize but non-zero to ensure code path exercised
        mktempdir_local(dir -> begin
            try
                _ = IterDiag(hamflow, 2^min(N,3); symmetries=Char[], correlationDefDict=Dict{String,Any}(),
                             quantumNoReq=nothing, corrQuantumNoReq=nothing, degenTol=1e-10, dataDir=dir,
                             silent=true, specFuncNames=String[], maxMaxSize=0, calculateThroughout=false)
                @test true  # no crash
            catch err
                # if any function not supported for this random input, mark as failure
                @test false "IterDiag crashed on random input: $err"
            end
        end)
    end
end

println("All tests defined in test_iterdiag.jl. Run with Julia's test runner.")
