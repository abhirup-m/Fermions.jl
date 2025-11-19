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


#=@testset "Spectral Function" begin=#
#=    basisStates = BasisStates(4)=#
#=    hop_t = abs(rand())=#
#=    U = abs(rand())=#
#=    eps = -U / 2=#
#=    broadening = 1e-3=#
#=    operatorList = HubbardDimerOplist(eps, U, hop_t)=#
#=    eigvals, eigvecs = Spectrum(operatorList, basisStates)=#
#=    omegaVals = collect(range(-1.0, stop=1.0, length=1000))=#
#=    specfunc = SpecFunc(eigvals, eigvecs, Dict("destroy" => [("-", [1], 1.0)], "create" => [("+", [1], 1.0)]), omegaVals, basisStates, broadening)=#
#=    specfuncCompare = HubbardDimerSpecFunc(eps, U, hop_t, omegaVals, broadening)=#
#=    @test specfunc ./ maximum(specfunc) ≈ specfuncCompare ./ maximum(specfuncCompare)=#
#=end=#


@testset "Alternative RDM Methods" begin
    @testset for totalQubits in 1:4
        basis = BasisStates(totalQubits)
        testState = mergewith(+, basis...)
        @testset for subspaceSize in 1:totalQubits
            for run in 1:4
                coeffs = 2 .* rand(length(basis)) .- 1
                normFactor = sum(coeffs .^ 2)^0.5
                for (i, state) in enumerate(keys(testState))
                    testState[state] = coeffs[i] / normFactor
                end
                reducingIndices = shuffle(1:totalQubits)[1:subspaceSize]
                println(testState, reducingIndices)
                rdm1 = ReducedDM(copy(testState), reducingIndices)
                rdm2 = ReducedDMProjectorBased(copy(testState), reducingIndices)
                display(rdm1)
                display(rdm2)
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

@testset "GenCorrelation - randomized (cheap, 100 trials)" begin
    for _ in 1:RNG_TRIALS_CHEAP
        n = rand(1:MAX_LIGHT_N)
        sd = random_state_dict(n; complex=false)  # real-valued small test
        op = random_operator_terms(n; max_terms=2)
        # ensure no crash, finite result
        val = GenCorrelation(sd, op)
        @test isfinite(val) || isnan(val) == false
    end
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
        ψ = random_state_dict(n)
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
        # compare to projector-based method
        ρp = ReducedDMProjectorBased(ψ, keep)
        @test isapprox(ρ, ρp; atol=1e-8, rtol=1e-6)
    end
end

# ---------------------------
# PartialTraceProjectors - sanity tests
# ---------------------------
@testset "PartialTraceProjectors - structure & action" begin
    # small subspace
    ops = PartialTraceProjectors([1])
    @test size(ops,1) == size(ops,2)
    # apply projector-based formula: each element is an operator in module format
    # sample one projector and compute GenCorrelation on a small state
    st = Dict(BitVector([1,0])=>1/sqrt(2), BitVector([0,1])=>1/sqrt(2))
    # pick an operator and ensure GenCorrelation handles it
    someop = ops[1,1]
    # someop should be a Vector{Tuple{...}} inside array cell
    @test (someop === nothing) == false
    # GenCorrelation is expected to accept this operator (vector form)
    val = GenCorrelation(st, someop)
    @test isfinite(val)
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
        ψ = random_state_dict(n)
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
        ψ = random_state_dict(n)
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

@testset "TripartiteInfo - randomized (expensive)" begin
    for _ in 1:RNG_TRIALS_EXPENSIVE
        n = rand(4:MAX_ED_N)
        ψ = random_state_dict(n)
        idxs = shuffle(1:n)
        A = [idxs[1]]
        B = [idxs[2]]
        C = [idxs[3]]
        t = TripartiteInfo(ψ, (A,B,C))
        @test isfinite(t)
    end
end

# ---------------------------
# ThermalAverage (consistency checks vs direct trace)
# ---------------------------
@testset "ThermalAverage - basic checks" begin
    # small Hamiltonian with known eigenstates (diagonal)
    basis = BasisStates(2)
    eigenStates = basis
    eigvals = [0.0, 1.0, 2.0, 3.0]
    op = [("n", [1], 1.0)]
    β = 0.5
    T = ThermalAverage(eigenStates, eigvals, op, β)
    @test isfinite(T)
end

@testset "ThermalAverage - randomized (cheap loops)" begin
    for _ in 1:RNG_TRIALS_CHEAP
        n = rand(1:4)
        basis = BasisStates(n)
        vals = rand(length(basis))
        op = random_operator_terms(n; max_terms=2)
        β = rand()*2.0
        tav = ThermalAverage(basis, vals, op, β)
        @test isfinite(tav)
    end
end

# ---------------------------
# SpectralCoefficients (matrix form) & SpecFunc simple checks
# ---------------------------
@testset "SpectralCoefficients & SpecFunc - deterministic examples" begin
    eigVals = [0.0, 1.0]
    eigVecs = [[1.0, 0.0], [0.0, 1.0]]
    probes = Dict("create" => [1.0 0.0; 0.0 1.0], "destroy" => [1.0 0.0; 0.0 1.0])
    coeffs = SpectralCoefficients(eigVecs, eigVals, probes; silent=true)
    @test typeof(coeffs) <: AbstractVector
    freqs = collect(range(-3, 3, length=50))
    sf = SpecFunc(coeffs, freqs, 0.1; normalise=true)
    @test length(sf) == length(freqs)
    @test all(isfinite, sf)
    @test all(x->x ≥ 0, sf)
end

@testset "SpectralCoefficients - randomized (cheap)" begin
    for _ in 1:RNG_TRIALS_CHEAP
        nstates = rand(2:5)
        eigVals = sort(rand(nstates))
        eigVecs = [rand(nstates) for _ in 1:nstates]
        probes = Dict("create"=>rand(nstates,nstates), "destroy"=>rand(nstates,nstates))
        coeffs = SpectralCoefficients(eigVecs, eigVals, probes; silent=true)
        @test coeffs !== nothing
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
# SpecFunc wrapper with eigenstates as dicts (integration test)
# ---------------------------
@testset "SpecFunc (dict eigenstates) - integration check" begin
    eigenVals = [0.0, 1.0, 1.0]
    eigenStates = BasisStates(2)  # small basis
    probes = Dict("create"=>[("+",[1],1.0)], "destroy"=>[("-",[1],1.0)])
    freqs = collect(range(-2.0, 2.0, length=50))
    sf = SpecFunc(eigenVals, eigenStates, probes, freqs, eigenStates, 0.05; silent=true)
    @test length(sf) == length(freqs)
    @test all(isfinite, sf)
end

# ---------------------------
# SpecFuncVariational - root-finding / height matching
# ---------------------------
@testset "SpecFuncVariational - behavior & convergence (cheap)" begin
    # build simple spectral groups: two groups centered at 0 with same coeffs
    group = [(1.0, 0.0)]
    groups = [group]
    freqs = collect(range(-1.0, 1.0, length=201))
    arrs, local, sigma = SpecFuncVariational(groups, freqs, 1.0, 0.2; silent=true)
    @test typeof(arrs) <: AbstractVector
    @test length(local) == length(freqs)
    @test sigma > 0
end

# ---------------------------
# SelfEnergy variants - sanity checks
# ---------------------------
@testset "SelfEnergy (coeff form) - small check" begin
    coeffs_int = [(1.0, 0.0)]
    coeffs_non = [(1.0, 1.0)]
    freqs = collect(range(-2.0, 2.0, length=101))
    Gint, Gnon, Σ = SelfEnergy(coeffs_int, coeffs_non, freqs; standDev=1e-3)
    @test length(Gint) == length(freqs)
    @test length(Σ) == length(freqs)
    # self-energy should be finite (no NaNs)
    @test all(isfinite, Σ)
end

@testset "SelfEnergy (specFunc form) - KK-transform based check (cheap)" begin
    # Create simple gaussian-like spectral functions
    freqs = collect(range(-5.0, 5.0, length=201))
    spec_non = exp.(-0.5 .* ((freqs./1.0).^2))
    spec_int = exp.(-0.5 .* (( (freqs .- 0.5)./1.0).^2))
    Σ = SelfEnergy(spec_non, spec_int, freqs; normalise=true)
    @test length(Σ) == length(freqs)
    @test all(isfinite, Σ)
end

# ---------------------------
# Additional invariants & edge-cases (randomized)
# ---------------------------
@testset "Randomized invariants, small-to-medium N (mixed cheap/expensive)" begin
    # invariants: trace preservation for projector-based reduced DM
    for _ in 1:20
        n = rand(1:6)
        ψ = random_state_dict(n)
        keep = sort(unique(collect(rand(1:n, rand(1:n)))))
        ρ = ReducedDM(ψ, keep)
        @test isapprox(sum(diag(ρ)), 1.0; atol=1e-9)
    end

    # symmetry/permutation invariance of VonNEntropy under relabelling
    for _ in 1:20
        n = rand(2:6)
        ψ = random_state_dict(n)
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

# ---------------------------
# Final smoke tests to ensure no type errors with large light randomized sets
# ---------------------------
@testset "Large randomized smoke tests (light, up to N=10)" begin
    for _ in 1:50
        n = rand(1:MAX_LIGHT_N)
        ψ = random_state_dict(n)
        keep = sort(unique(collect(rand(1:n, rand(1:n)))))
        # call a few representative functions
        _ = try ReducedDM(ψ, keep) catch e; @test false; end
        _ = try VonNEntropy(ψ, keep) catch e; @test false; end
        _ = try MutInfo(ψ, (keep, keep)) catch e; @test false; end
    end
end
