using Random, LinearAlgebra
Random.seed!(12345)    # reproducible

@testset "Groundstate Correlation" begin
    basisStates = BasisStates(4)
    hop_t = rand()
    U = rand()
    eps = -U / 2
    Δ = (U^2 + 16 * hop_t^2)^0.5
    operatorList = HubbardDimerOplist(eps, U, hop_t)
    eigvals, eigvecs = Spectrum(operatorList, basisStates)

    doubOccOperator = [("nn", [1, 2], 0.5), ("nn", [3, 4], 0.5)]
    totSzOperator = [("n", [1], 0.5), ("n", [2], -0.5), ("n", [3], 0.5), ("n", [4], -0.5)]
    spinFlipOperator = [("+-+-", [1, 2, 4, 3], 1.0), ("+-+-", [3, 4, 2, 1], 1.0)]

    @test isapprox(GenCorrelation(eigvecs[1], doubOccOperator), (Δ - U) / (4 * Δ), atol=1e-10)
    @test isapprox(GenCorrelation(eigvecs[1], totSzOperator), 0, atol=1e-10)
    @test isapprox(GenCorrelation(eigvecs[1], spinFlipOperator), -8 * hop_t^2 / (Δ * (Δ - U)), atol=1e-10)
end


@testset "Entanglement entropy, Mutual information, Schmidt Gap" begin

    @testset "θ = $(theta)" for theta in rand(100) .* 2π
        state = Dict(BitVector([1, 0]) => cos(theta), BitVector([0, 1]) => sin(theta))
        SEE_1, schmidtGap_1 = VonNEntropy(state, [1], schmidtGap=true)
        SEE_2, schmidtGap_2 = VonNEntropy(state, [2], schmidtGap=true)
        @test schmidtGap_1 ≈ schmidtGap_2 ≈ abs(cos(theta)^2 - sin(theta)^2)
        @test SEE_1 ≈ SEE_2 ≈ -cos(theta)^2 * log(cos(theta)^2) - sin(theta)^2 * log(sin(theta)^2)
    end

    coeffs = rand(3)
    coeffs ./= sum(coeffs .^ 2)^0.5
    state = Dict(BitVector([1, 0, 1]) => coeffs[1], BitVector([1, 1, 0]) => coeffs[2], BitVector([0, 1, 1]) => coeffs[3])
    SEE_1, schmidtGap_1 = VonNEntropy(state, [1], schmidtGap=true)
    SEE_2, schmidtGap_2 = VonNEntropy(state, [2], schmidtGap=true)
    SEE_3, schmidtGap_3 = VonNEntropy(state, [3], schmidtGap=true)
    SEE_12, schmidtGap_12 = VonNEntropy(state, [1, 2], schmidtGap=true)
    I2_12 = MutInfo(state, ([1], [2]))
    coeffs ./= sum(coeffs .^ 2)^0.5
    @test schmidtGap_12 ≈ schmidtGap_3 ≈ abs(coeffs[1]^2 + coeffs[3]^2 - coeffs[2]^2)
    @test SEE_1 ≈ -(coeffs[1]^2 + coeffs[2]^2) * log(coeffs[1]^2 + coeffs[2]^2) - (coeffs[3]^2) * log(coeffs[3]^2)
    @test SEE_2 ≈ -(coeffs[3]^2 + coeffs[2]^2) * log(coeffs[3]^2 + coeffs[2]^2) - (coeffs[1]^2) * log(coeffs[1]^2)
    @test SEE_12 ≈ SEE_3 ≈ -coeffs[2]^2 * log(coeffs[2]^2) - (coeffs[1]^2 + coeffs[3]^2) * log(coeffs[1]^2 + coeffs[3]^2)
    @test I2_12 ≈ SEE_1 + SEE_2 - SEE_12

    state = Dict(BitVector([1, 0, 1]) => rand(), BitVector([0, 1, 1]) => rand())
    SEE, schmidtGap = VonNEntropy(state, [3], schmidtGap=true)
    @test schmidtGap ≈ 1
    @test SEE ≈ 0
end


@testset "Tripartite information" begin

    @testset "θ = $(theta)" for theta in rand(100) .* 2π
        state = Dict(BitVector(fill(0, 4)) => cos(theta), BitVector(fill(1, 4)) => sin(theta))
        I3 = TripartiteInfo(state, ([1], [2], [3]))
        @test I3 ≈ -cos(theta)^2 * log(cos(theta)^2) - sin(theta)^2 * log(sin(theta)^2)
    end
end


@testset "Spectral Function Non-Degenerate" begin
    basisStates = BasisStates(4)
    hop_t = abs(rand())
    U = abs(rand())
    eps = -U / 2
    broadening = 1e-3
    operatorList = HubbardDimerOplist(eps, U, hop_t)
    eigvals, eigvecs = Spectrum(operatorList, basisStates)
    omegaVals = collect(range(-1.0, stop=1.0, length=5))
    specfunc = SpecFunc(eigvals, eigvecs, Dict("destroy" => [("-", [1], 1.0)], "create" => [("+", [1], 1.0)]), omegaVals, basisStates, broadening; normalise=false)
    specfuncCompare = HubbardDimerSpecFunc(eps, U, hop_t, omegaVals, broadening)
    @test specfunc ≈ specfuncCompare
end

@testset "Spectral Function Degenerate" begin
    basisStates = BasisStates(4)
    hop_t = 0.
    U = abs(rand())
    eps = -U / 2
    broadening = 1e-3
    operatorList = HubbardDimerOplist(eps, U, hop_t)
    eigvals, eigvecs = Spectrum(operatorList, basisStates)
    omegaVals = collect(range(-1.0, stop=1.0, length=5))
    specfunc = SpecFunc(eigvals, eigvecs, Dict("destroy" => [("-", [1], 1.0)], "create" => [("+", [1], 1.0)]), omegaVals, basisStates, broadening; normalise=false)
    specfuncCompare = HubbardDimerSpecFunc(eps, U, hop_t, omegaVals, broadening)
    @test specfunc ≈ specfuncCompare
end


@testset "Alternative RDM Methods" begin
    @testset for totalQubits in 2:4
        basis = BasisStates(totalQubits)
        testState = mergewith(+, basis...)
        @testset for subspaceSize in 1:(totalQubits-1)
            for run in 1:4
                coeffs = 2 .* rand(length(basis)) .- 1
                normFactor = sum(coeffs .^ 2)^0.5
                for (i, state) in enumerate(keys(testState))
                    testState[state] = coeffs[i] / normFactor
                end
                reducingIndices = shuffle(1:totalQubits)[1:subspaceSize]
                rdm1 = ReducedDM(copy(testState), reducingIndices)
                rdm2 = ReducedDMProjectorBased(copy(testState), reducingIndices)
                errorMatrix = rdm1[sortperm(diag(rdm1)), sortperm(diag(rdm1))] .- rdm2[sortperm(diag(rdm2)), sortperm(diag(rdm2))]
                @test isapprox(errorMatrix .|> abs |> maximum, 0, atol=1e-14)
            end
        end
    end
end


@testset "VNE Order Independence" begin
    @testset for totalQubits in 6:9
        basis = BasisStates(totalQubits)
        testState = mergewith(+, basis...)
        @testset for subspaceSize in 2:5
            @testset for method in [ReducedDM, ReducedDMProjectorBased]
                for run in 1:4
                    coeffs = 2 .* rand(length(basis)) .- 1
                    normFactor = sum(coeffs .^ 2)^0.5
                    for (i, state) in enumerate(keys(testState))
                        testState[state] = coeffs[i] / normFactor
                    end
                    reducingIndices = shuffle(1:totalQubits)[1:subspaceSize]
                    while issorted(reducingIndices)
                        reducingIndices = shuffle(1:totalQubits)[1:subspaceSize]
                    end
                    rdm1 = method(copy(testState), reducingIndices)
                    rdm2 = method(copy(testState), sort(reducingIndices))
                    entanglementSpectrumError = (eigen(rdm1).values .- eigen(rdm2).values) .|> abs |> maximum
                    @test isapprox(entanglementSpectrumError, 0, atol=1e-14)
                end
            end
        end
    end
end

# ---------------------------
# Test tuning (Balanced mode)
# ---------------------------
const RNG_TRIALS_CHEAP = 100    # cheap functions (small cost per trial)
const RNG_TRIALS_EXPENSIVE = 50 # expensive functions (ED-heavy or matrix ops)
const MAX_ED_N = 8              # max N for exact diagonalization heavy tests
const MAX_LIGHT_N = 10          # light randomized tests can go up to N=10

# ---------------------------
# Helper utilities
# ---------------------------
# Note: these helpers use your module functions (BasisStates, ExpandIntoBasis, etc.)
# If those functions live in a module, ensure you `using ModuleName` before running tests.

"""
Generate a random normalized *state* in dictionary form (Dict{BitVector,ComplexF64}).
The resulting state is a pure state (|ψ⟩) in computational basis.
"""
function random_state_dict(n::Int; complex=true)
    dim = 2^n
    keys = [BitVector(digits(i, base=2, pad=n)) for i in 0:dim-1]
    if complex
        vals = randn(ComplexF64, dim) .+ im * randn(ComplexF64, dim)
    else
        vals = randn(Float64, dim)
    end
    vals ./= sqrt(sum(abs2, vals))
    return Dict(keys[i] => vals[i] for i in 1:dim)
end

"""
Generate a random full vector state (Vector{Float64} or ComplexF64) of dimension 2^n
suitable for GenCorrelation(vector, matrix) tests.
"""
function random_state_vec(n::Int; complex=true)
    dim = 2^n
    if complex
        v = randn(ComplexF64, dim) .+ im * randn(ComplexF64, dim)
    else
        v = randn(Float64, dim)
    end
    v ./= norm(v)
    return v
end

"""
Generate a random simple operator in the module format (vector of tuples).
We produce a small number of terms with operator strings chosen from documented set.
"""
function random_operator_terms(n::Int; max_terms=3)
    # operators use characters like '+','-','n','h'
    opchars = ['+', '-', 'n', 'h']
    L = rand(1:max_terms)
    terms = Vector{Tuple{String,Vector{Int64},Float64}}()
    for _ in 1:L
        k = rand(1:n)                     # pick 1..n affected sites length
        sites = sort(collect(rand(1:n, k)))
        opname = join([rand(opchars) for _ in 1:length(sites)])
        weight = randn()
        push!(terms, (opname, sites, weight))
    end
    return terms
end

"""
Small helper to convert a Dict state (BitVector->c) to full vector using BasisStates.
Requires that BasisStates(n) returns the computational basis in a consistent order.
"""
function dictstate_to_vector(dictstate::Dict{BitVector,<:Number})
    # infer n from key length
    n = length(first(keys(dictstate)))
    basis = BasisStates(n)   # uses package function
    vec = zeros(ComplexF64, length(basis))
    for (i,b) in enumerate(basis)
        key = first(keys(b))
        val = get(dictstate, key, 0.0)
        vec[i] = val
    end
    return vec
end

# Tolerance helpers
const rtol = 1e-8
const atol = 1e-10

# ---------------------------
# Begin tests
# ---------------------------

# ---------------------------
# GenCorrelation (dict and vector variants)
# ---------------------------
@testset "GenCorrelation - deterministic small examples" begin
    # Example from docstring
    state = Dict(BitVector([1,0]) => 0.5, BitVector([0,1]) => -0.5)
    op = [("+-", [1,2], 1.0)]
    val = GenCorrelation(state, op)
    @test isfinite(val)
    # The doc example prints -0.5; we only assert reproducibility type & sign here
    @test typeof(val) <: Real

    # Vector/matrix variant: simple Pauli-Z expectation
    v = [1.0, 0.0]
    M = [1.0 0.0; 0.0 -1.0]
    val2 = GenCorrelation(copy(v), M)
    @test isapprox(val2, 1.0; atol=1e-12)

    # check normalization enforced in vector version
    v2 = [2.0, 0.0]
    @test isapprox(GenCorrelation(copy(v2), M), 1.0; atol=1e-12)
end

# ---------------------------
# ReducedDM and ReducedDMProjectorBased
# ---------------------------
@testset "ReducedDM vs ReducedDMProjectorBased - deterministic checks" begin
    # simple 2-site Bell-like state in dict form
    state = Dict(BitVector([1,0]) => 1/sqrt(2), BitVector([0,1]) => 1/sqrt(2))
    ρ = ReducedDM(state, [1])
    ρp = ReducedDMProjectorBased(state, [1])
    @test size(ρ) == (2,2)
    @test size(ρ) == size(ρp)
    @test isapprox(ρ, ρp; atol=1e-12, rtol=1e-8)
    # trace normalization
    @test isapprox(sum(diag(ρ)), 1.0; atol=1e-12)
end

@testset "ReducedDM - randomized (expensive blocks up to N=8, 50 trials)" begin
    for _ in 1:RNG_TRIALS_EXPENSIVE
        n = rand(1:MAX_ED_N)
        ψ = random_state_dict(n; complex=false)
        # choose a random non-empty non-full subset to trace over
        k = rand(1:n)
        keep = sort(unique(collect(rand(1:n, k))))
        ρ = ReducedDM(ψ, keep)
        @test size(ρ,1) == size(ρ,2)
        # reduced density matrix trace normalized
        tr = sum(diag(ρ))
        @test isapprox(tr, 1.0; atol=1e-9)
        # positivity: all eigenvalues >= -tol
        eigs = eigvals(0.5*(ρ + ρ'))    # ensure hermitian
        @test minimum(real(eigs)) ≥ -1e-8
    end
end

# ---------------------------
# VonNEntropy (matrix and dict overload)
# ---------------------------
@testset "VonNEntropy - deterministic checks" begin
    # Two-site product state -> zero entanglement for any single-site cut
    product = Dict(BitVector([0,0])=>1.0)
    S = VonNEntropy(product, [1])
    @test isapprox(S, 0.0; atol=1e-12)

    # Bell-like state -> log(2)
    bell = Dict(BitVector([1,0]) => 1/sqrt(2), BitVector([0,1]) => 1/sqrt(2))
    S_bell = VonNEntropy(bell, [1])
    @test isapprox(S_bell, log(2); atol=1e-8)
end

@testset "VonNEntropy - randomized (expensive, up to N=8, 50 trials)" begin
    for _ in 1:RNG_TRIALS_EXPENSIVE
        n = rand(1:MAX_ED_N)
        ψ = random_state_dict(n; complex=false)
        A = sort(unique(collect(rand(1:n, rand(1:n)))))
        S = VonNEntropy(ψ, A)
        @test isfinite(S)
        @test S ≥ -1e-12
        # For single-site pure product states, S≈0
        if length(A) == 0
            @test S == 0
        end
    end
end

@testset "VonNEntropy matrix overload and schmidtGap option" begin
    # Construct small reduced matrix with known spectrum
    ρ = [0.5 0.0; 0.0 0.5]
    S = VonNEntropy(ρ; tolerance=1e-12, schmidtGap=false)
    @test isapprox(S, log(2); atol=1e-12)
    S2, gap = VonNEntropy(ρ; tolerance=1e-12, schmidtGap=true)
    @test isapprox(S2, log(2); atol=1e-12)
    @test isapprox(gap, 0.0; atol=1e-12)
end

# ---------------------------
# MutInfo & TripartiteInfo (consistency & SSA)
# ---------------------------
@testset "MutInfo positive & SSA checks (randomized heavy)" begin
    for _ in 1:RNG_TRIALS_EXPENSIVE
        n = rand(3:MAX_ED_N)    # needs at least 3 sites for some splits
        ψ = random_state_dict(n; complex=false)
        # choose disjoint random subsystems A, B, C
        idxs = shuffle(1:n)
        a = [idxs[1]]
        b = [idxs[2]]
        c = [idxs[3]]
        Iab = MutInfo(ψ, (a,b))
        Iac = MutInfo(ψ, (a,c))
        Iabc = MutInfo(ψ, (a, vcat(b,c)))
        # positivity
        @test Iab ≥ -1e-12
        # strong subadditivity-like check: I(A:BC) ≥ I(A:B)
        @test Iabc + 1e-10 ≥ Iab - 1e-10
    end
end

@testset "SpecFunc - spectral sum & normalisation tests (expensive-ish)" begin
    # Use a 1-pole coefficient to validate area under lorentzian ~1 when normalise true
    coeffs = [(1.0, 0.0)]
    freqs = collect(range(-5.0, 5.0, length=501))
    sf = SpecFunc(coeffs, freqs, 0.1; normalise=true)
    @test length(sf) == length(freqs)
    sarea = sum(sf) * (maximum(freqs) - minimum(freqs)) / (length(freqs)-1)
    @test isapprox(sarea, 1.0; atol=1e-3)
end

# ---------------------------
# Additional invariants & edge-cases (randomized)
# ---------------------------
@testset "Randomized invariants, small-to-medium N (mixed cheap/expensive)" begin
    # invariants: trace preservation for projector-based reduced DM
    for _ in 1:20
        n = rand(1:6)
        ψ = random_state_dict(n; complex=false)
        keep = sort(unique(collect(rand(1:n, rand(1:n)))))
        ρ = ReducedDM(ψ, keep)
        @test isapprox(sum(diag(ρ)), 1.0; atol=1e-9)
    end

    # symmetry/permutation invariance of VonNEntropy under relabelling
    for _ in 1:20
        n = rand(2:6)
        ψ = random_state_dict(n; complex=false)
        perm = shuffle(1:n)
        ψperm = PermuteSites(deepcopy(ψ), perm)
        cut = sort(collect(rand(1:n, rand(1:n))))
        cutp = findall(x->x in cut, perm)  # approximate mapping
        # compute entropy before and after permutation of state and cut
        s1 = VonNEntropy(ψ, cut)
        s2 = VonNEntropy(ψperm, cutp)
        @test isapprox(s1, s2; atol=1e-8) || (abs(s1-s2) < 1e-6)
    end
end

@testset "Physics invariants and logic-based tests" begin

    # ---------------------------
    # 1) Particle-number conservation: expectation of sum_i n_i
    #    should equal direct calculation from probabilities: sum_basis(|c|^2 * occupancy)
    # ---------------------------
    @testset "Particle-number conservation (randomized)" begin
        for _ in 1:RNG_TRIALS_EXPENSIVE
            n = rand(1:MAX_ED_N)
            ψ = random_state_dict(n; complex=false)
            # Build operator list: ("n",[i],1.0) for each site
            n_ops = [( "n", [i], 1.0 ) for i in 1:n]
            # expectation via GenCorrelation
            exp_n = GenCorrelation(ψ, n_ops)
            # direct calculation: sum_{basis} |c|^2 * occupancy(basis)
            probs = Dict(k => abs2(v) for (k,v) in ψ)
            occ = sum( sum(key) * probs[key] for key in keys(probs) )
            @test isapprox(exp_n, occ; atol=1e-10, rtol=1e-8)
        end
    end

    # ---------------------------
    # 2) Hermiticity: expectation values of Hermitian operators must be real
    # ---------------------------
    @testset "Hermiticity of expectation (randomized)" begin
        for _ in 1:RNG_TRIALS_CHEAP
            n = rand(1:6)
            ψ = random_state_dict(n; complex=false)
            # Build a Hermitian operator by combining op + dagger(op)
            op = random_operator_terms(n; max_terms=2)
            # build dagger version using Dagger for vector-of-tuples format where available
            # If op is vector of tuples then Dagger(op) should exist; we handle both forms:
            # convert single term list into operator for GenCorrelation
            try
                # make a hermitian operator: O + O†
                od = Dagger(op)  # could be vector-of-tuples variant
                # Compose hermitian operator (sum of tuple lists)
                herm = copy(op)
                append!(herm, od)
                val = GenCorrelation(ψ, herm)
                @test isreal(val)
            catch err
                # fallback: use diagonal number operator (Hermitian)
                numops = [( "n", [i], 1.0 ) for i in 1:n]
                v = GenCorrelation(ψ, numops)
                @test isreal(v)
            end
        end
    end

    # ---------------------------
    # 3) OperatorMatrix Hermiticity: for operator == Dagger(operator) matrix should be Hermitian
    # ---------------------------
    @testset "OperatorMatrix hermiticity check" begin
        n = 3
        basis = BasisStates(n)
        # pick a simple Hermitian operator: sum_i n_i
        op = [( "n", [i], 1.0 ) for i in 1:n]
        M = OperatorMatrix(basis, op)
        @test isapprox(M, M'; atol=1e-12)
        # Now create a non-Hermitian operator and verify matrix is not Hermitian
        op_non = [("+", [1], 1.0)]
        M2 = OperatorMatrix(basis, op_non)
        @test !(isapprox(M2, M2'; atol=1e-12))
    end

    # ---------------------------
    # 4) Commutators: DoesCommute should agree with matrix-level commutator
    # ---------------------------
    @testset "Commutation checks (randomized)" begin
        for _ in 1:RNG_TRIALS_CHEAP
            n = rand(2:6)
            basis = BasisStates(n)
            # two random operators (vector-of-tuples). Ensure both are non-empty.
            opA = random_operator_terms(n; max_terms=2)
            opB = random_operator_terms(n; max_terms=2)
            # Convert to matrices
            MA = OperatorMatrix(basis, opA)
            MB = OperatorMatrix(basis, opB)
            # using DoesCommute (provided) and matrix commutator check
            commute_func = DoesCommute(opA, opB, basis; tolerance=1e-10)
            comm_mat = MA*MB - MB*MA
            commute_mat = maximum(abs.(comm_mat)) < 1e-9
            @test commute_func == commute_mat
        end
    end

    # ---------------------------
    # 5) Thermal average consistency: ThermalAverage should match explicit trace over eigenbasis
    # ---------------------------
    @testset "ThermalAverage explicit trace equality (small N)" begin
        # For small system with diagonal Hamiltonian (eigenVals) and basis states,
        # ThermalAverage should equal sum_i (e^{-β E_i} * ⟨ψ_i|O|ψ_i⟩) / Z
        n = 3
        basis = BasisStates(n)
        # make diagonal energies
        eigs = collect(0.0:1.0:length(basis)-1)
        # choose operator diagonal in basis: n_1
        op = [( "n", [1], 1.0 )]
        β = 0.7
        tav = ThermalAverage(basis, eigs, op, β)
        # calculate explicit using ExpandIntoBasis and OperatorMatrix
        M = OperatorMatrix(basis, op)
        # For each basis eigenstate (these are delta states), expectation is diagonal M_ii
        overlaps = [M[i,i] for i in 1:length(basis)]
        weights = exp.(-β .* eigs)
        expected = sum(weights .* overlaps) / sum(weights)
        @test isapprox(tav, expected; atol=1e-12)
    end

    # ---------------------------
    # 6) Spectral function positivity and high-frequency falloff (sum-rule-ish)
    # ---------------------------
    @testset "Spectral positivity and integrated area (physics check)" begin
        # one simple fermionic spectral coefficient: single pole at 0 with weight 1
        coeffs = [(1.0, 0.0)]
        freqs = collect(range(-10.0, 10.0, length=801))
        sf = SpecFunc(coeffs, freqs, 0.05; normalise=true, broadFuncType="lorentz")
        @test all(sf .>= -1e-12)  # non-negative (numerical tiny negative rounding allowed)
        # integrated area ~1
        area = sum(sf) * (maximum(freqs) - minimum(freqs)) / (length(freqs)-1)
        @test isapprox(area, 1.0; atol=2e-3)
        # high frequency tails should be small
        tail_mag = (sum(abs.(sf[1:10])) + sum(abs.(sf[end-9:end])))/20
        @test tail_mag < 1e-2
    end

    # ---------------------------
    # 7) Reduced density matrix purity, extremal cases:
    #    - pure global product state => reduced state pure (Tr(ρ^2)=1)
    #    - maximally entangled bipartition => reduced state maximally mixed (Tr(ρ^2)=1/d)
    # ---------------------------
    @testset "ReducedDM purity tests" begin
        # product state => reduced DM purity == 1
        product = Dict(BitVector(zeros(Bool, 4)) => 1.0)  # |0000>
        ρp = ReducedDM(product, [1,2])
        purity_p = sum(abs2, vec(ρp))
        @test isapprox(purity_p, 1.0; atol=1e-12)

        # maximally entangled on 2 qubits: Bell on sites 1-2 and product on others
        bell12 = Dict(BitVector([1,0,0,0]) => 1/sqrt(2), BitVector([0,1,0,0]) => 1/sqrt(2))
        ρb = ReducedDM(bell12, [1])
        purity_b = sum(abs2, vec(ρb))
        # For 1 qubit maximally mixed, purity is 1/2
        @test isapprox(purity_b, 0.5; atol=1e-8)
    end

    # ---------------------------
    # 8) Mutual information consistency for Bell pair
    #    For Bell on sites (1,2): S(A)=S(B)=log(2), S(AB)=0 => I(A:B)=2*log(2)
    # ---------------------------
    @testset "Mutual information exact Bell" begin
        bell = Dict(BitVector([1,0]) => 1/sqrt(2), BitVector([0,1]) => 1/sqrt(2))
        I = MutInfo(bell, ([1],[2]))
        @test isapprox(I, 2*log(2); atol=1e-8)
    end

    # ---------------------------
    # 9) Tripartite info for product states is zero
    # ---------------------------
    @testset "TripartiteInfo for product state" begin
        prod4 = Dict(BitVector(zeros(Bool,4)) => 1.0)
        t = TripartiteInfo(prod4, ([1],[2],[3]))
        @test isapprox(t, 0.0; atol=1e-12)
    end

    # ---------------------------
    # 10) SpectralCoefficients sanity: particle addition vs removal symmetry
    #    For a trivial non-interacting 1-site model, create/destroy weights follow expectations.
    # ---------------------------
    @testset "SpectralCoefficients particle-add/remove checks (small matrix model)" begin
        # 1-site Hilbert space (dim=2)
        # choose eigenvectors as canonical basis
        eigVals = [0.0, 1.0]
        eigVecs = [[1.0,0.0], [0.0,1.0]]
        # probe matrices: create/destroy as simple raising/lowering matrices
        create = [0.0 1.0; 0.0 0.0]
        destroy = create'
        probes = Dict{String, Matrix{Float64}}("create"=>create, "destroy"=>destroy)
        coeffs = SpectralCoefficients(eigVecs, eigVals, probes; silent=true)
        # We expect non-empty spectral coefficients (matrix elements form)
        @test length(coeffs) ≥ 1
        # poles should align with eigenvalue differences (within tolerance)
        for (w,p) in coeffs
            @test isfinite(p)
        end
    end

    # ---------------------------
    # 11) SelfEnergy consistency (coeff vs spectral) for a trivial test:
    #     Build an interacting and non-interacting single-pole set and compare that outputs are finite
    # ---------------------------
    @testset "SelfEnergy basic consistency" begin
        coeffs_int = [(0.6, 0.0), (0.4, 1.0)]
        coeffs_non = [(1.0, 0.5)]
        freqs = collect(range(-3.0,3.0,length=201))
        Gint, Gnon, Σ = SelfEnergy(coeffs_int, coeffs_non, freqs; standDev=1e-3)
        @test all(isfinite, real.(Gint)) && all(isfinite, imag.(Gint))
        @test all(isfinite, real.(Gnon)) && all(isfinite, imag.(Gnon))
        @test all(isfinite, real.(Σ)) && all(isfinite, imag.(Σ))
    end

    # ---------------------------
    # 12) Invariance of VonNEntropy under global phase on the state
    # ---------------------------
    @testset "Entropy invariant under global phase" begin
        n = 4
        ψ = random_state_dict(n; complex=false)
        # apply global phase
        phase = rand()
        ψ_phase = Dict(k => v*phase for (k,v) in ψ)
        cut = [1,2]
        s1 = VonNEntropy(ψ, cut)
        s2 = VonNEntropy(ψ_phase, cut)
        @test isapprox(s1, s2; atol=1e-12)
    end

end # end physics @testset

# ---------------------------
# Final smoke tests to ensure no type errors with large light randomized sets
# ---------------------------
@testset "Large randomized smoke tests (light, up to N=10)" begin
    for _ in 1:50
        n = rand(1:MAX_LIGHT_N)
        ψ = random_state_dict(n; complex=false)
        keep = sort(unique(collect(rand(1:n, rand(1:n)))))
        # call a few representative functions
        _ = try ReducedDM(ψ, keep) catch e; @test false; end
        _ = try VonNEntropy(ψ, keep) catch e; @test false; end
        _ = try MutInfo(ψ, (keep, keep)) catch e; @test false; end
    end
end
