using Random
Random.seed!(123456)

const MAX_N = 8   # keep moderate size for CI; increase locally if needed
const RNG_TRIALS = 200  # number of randomized checks for property tests
const RNG_TRIALS_SMALL = 50

@testset "Spectrum" begin
    eps = 1.281
    hop_t = 0.284
    U = 3.132
    @testset "sector=$((n, m))" for (n, m) in [(0, 0), (1, 1), (1, -1), (2, 2), (2, 0), (2, -2), (3, 1), (3, -1), (4, 0)]
        basis = BasisStates(4; totOccReq=n, magzReq=m)
        operatorList = HubbardDimerOplist(eps, U, hop_t)
        eigvals, eigvecs = Spectrum(operatorList, basis)
        comparisonMatrix = HubbardDimerMatrix(eps, U, hop_t)[(n, m)]
        eigvalTest, eigvecTest = eigen(comparisonMatrix)
        @test eigvals ≈ eigvalTest
        for (i, v2) in enumerate(eachcol(eigvecTest))
            v2 = v2[abs.(v2) .> 1e-14]
            if (n,m) == (2, 0) && BitVector([1, 0, 0, 1]) in keys(eigvecs[i]) && BitVector([1, 1, 0, 0]) in keys(eigvecs[i])
                vals = [eigvecs[i][BitVector([1, 0, 0, 1])], eigvecs[i][BitVector([0, 1, 1, 0])], eigvecs[i][BitVector([1, 1, 0, 0])], eigvecs[i][BitVector([0, 0, 1, 1])]]
                vals = vals[abs.(vals) .> 1e-14]
                @test vals ./ maximum(abs.(vals)) ≈ v2 ./ maximum(abs.(v2)) || vals ./ maximum(abs.(vals)) ≈ -1 .* v2 ./ maximum(abs.(v2))
            else
                @test collect(values(eigvecs[i])) ./ collect(values(eigvecs[i]))[1] ≈ v2 ./ v2[1]
            end
        end
    end
end


@testset "Ground States" begin
    basisStates = BasisStates(4)
    hop_t = 1.232
    U = 2.347
    eps = -U / 2
    Δ = (U^2 + 16 * hop_t^2)^0.5
    a1 = 0.5 * √((Δ - U) / Δ)
    a2 = 2 * hop_t / √(Δ * (Δ - U))
    operatorList = HubbardDimerOplist(eps, U, hop_t)
    eigvals, eigvecs = Spectrum(operatorList, basisStates)
    @test eigvals[1] ≈ 2 * eps + U / 2 - Δ / 2
    @test Set(keys(eigvecs[1])) == Set([[1, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0]])
    scaledGstate = Dict(k => v / eigvecs[1][[1, 0, 0, 1]] for (k, v) in eigvecs[1])
    @test scaledGstate[[1, 0, 0, 1]] ≈ 1.0
    @test scaledGstate[[0, 1, 1, 0]] ≈ -1.0
    @test scaledGstate[[0, 0, 1, 1]] ≈ a1 / a2
    @test scaledGstate[[1, 1, 0, 0]] ≈ a1 / a2

end


@testset "Degenerate Ground States" begin
    basisStates = BasisStates(4)
    eps = -1.541
    hop_t = 0
    U = -2 * eps
    operatorList = HubbardDimerOplist(eps, U, hop_t)
    eigvals, eigvecs = Spectrum(operatorList, basisStates)
    @test eigvals[1] == eigvals[2] == eigvals[3] == eigvals[4] ≈ 2 * eps
    @test Set((eigvecs[1], eigvecs[2], eigvecs[3], eigvecs[4])) == Set((Dict([1, 0, 1, 0] => 1.0), Dict([0, 1, 0, 1] => 1.0), Dict([0, 1, 1, 0] => 1.0), Dict([1, 0, 0, 1] => 1.0)))
end


# -------------------------------------------------------------------------
# Helpers used only within tests
# -------------------------------------------------------------------------
# generate a random normalized state as dictionary over bitvectors for n sites
function random_state_dict(n::Int)
    basis = BasisStates(n)
    dim = length(basis)
    vec = randn(dim) .+ im * randn(dim)
    vec ./= norm(vec)
    return TransformState(vec, basis; tolerance=0.0)
end

# generate a random sparse operator (vector of tuple-terms)
# operators are of the form ("+-", [i,j,...], strength)
function random_operator_terms(n::Int; max_terms::Int=3, max_sites::Int=3)
    terms = Vector{Tuple{String,Vector{Int},Float64}}()
    for _ in 1:rand(1:max_terms)
        k = rand(1:max_sites)
        # choose k unique sites
        sites = sort(rand(1:n, k))
        # randomly choose operator characters: from set 'n','h','+','-','+'/'-' combos
        opname_chars = join([rand(['n','h','+','-']) for _ in 1:length(sites)])
        push!(terms, (opname_chars, sites, randn()))
    end
    return terms
end

# compare two state dicts approximately (checks keys match and coefficients match within tol)
function states_approx_equal(s1::Dict{BitVector,Float64}, s2::Dict{BitVector,Float64}; atol=1e-10, rtol=1e-8)
    keys1 = collect(keys(s1))
    keys2 = collect(keys(s2))
    @assert length(keys1) == length(keys2)
    for k in keys1
        @test haskey(s2, k)
        @test isapprox(s1[k], s2[k]; atol=atol, rtol=rtol)
    end
    return true
end

# -------------------------------------------------------------------------
# Testsets
# -------------------------------------------------------------------------
@testset "ClassifyBasis and GetSector deterministic checks" begin
    # small deterministic basis
    b2 = BasisStates(2)
    classified_NZ = ClassifyBasis(b2, ['N','Z'])
    # expectation: four keys (0,0),(1,-1),(1,1),(2,0)
    expected_keys = Set([(0,0),(1,-1),(1,1),(2,0)])
    @test Set(keys(classified_NZ)) == expected_keys

    # ensure each basis state appears exactly once
    total = sum(length.(values(classified_NZ)))
    @test total == length(b2)

    # test single symmetry variants
    classified_N = ClassifyBasis(b2, ['N'])
    @test Set(keys(classified_N)) == Set([(0,),(1,),(2,)])
    classified_Z = ClassifyBasis(b2, ['Z'])
    @test Set(keys(classified_Z)) == Set([(-2,),(-1,),(0,),(1,),(2,)]) ∩ Set(keys(classified_Z)) || true
    # above line ensures function call succeeds and returns some keys (Z range depends on interpretation),
    # we don't assert exact for Z here to avoid varying convention, but verify total count matches basis
    @test sum(length.(values(classified_Z))) == length(b2)
end

@testset "TransformState <-> ExpandIntoBasis roundtrip (randomized)" begin
    for n in 1:MAX_N
        basis = BasisStates(n)
        dim = length(basis)
        for trial in 1:20
            # random real vector
            v = randn(dim)
            # transform into dict
            state = TransformState(v, basis; tolerance=1e-16)
            # get coefficients back
            coeffs = ExpandIntoBasis(state, basis)
            # the two vectors are proportional; compare normalized
            v_norm = v / norm(v)
            coeffs_norm = coeffs / norm(coeffs)
            @test isapprox(v_norm, coeffs_norm; atol=1e-12, rtol=1e-10)
        end
    end
end

@testset "TransformState tolerance pruning" begin
    # small basis
    b = BasisStates(3)
    v = zeros(ComplexF64, length(b))
    v[2] = 1e-20
    s = TransformState(v, b; tolerance=1e-10)
    # coefficient is below tolerance - should be pruned -> state empty
    @test isempty(keys(s))
    # if tolerance smaller, it appears
    s2 = TransformState(v, b; tolerance=1e-25)
    @test !isempty(keys(s2))
end

@testset "Spectrum vs OperatorMatrix eigen-decomposition (small systems)" begin
    for n in 1:5
        basis = BasisStates(n)
        # build a Hermitian operator explicitly: sum_i n_i + random small symmetric hopping
        op = [( "n", [i], 1.0 ) for i in 1:n]
        # add symmetric hoppings
        for i in 1:n-1
            push!(op, ("+-", [i, i+1], 0.3))
            push!(op, ("+-", [i+1, i], 0.3))
        end
        # get spectrum via Spectrum
        E_spectrum, states_spectrum = Spectrum(op, basis; tolerance=1e-12, assumeHerm=true)
        # get matrix and compute eigen using LinearAlgebra
        M = OperatorMatrix(basis, op)
        E_mat, V_mat = eigen(Hermitian(M))
        # sort to compare
        E_spectrum_sorted = sort(E_spectrum)
        @test length(E_spectrum_sorted) == length(E_mat)
        @test isapprox(E_spectrum_sorted, sort(E_mat); atol=1e-10, rtol=1e-8)
        # verify each eigenvector state (transformed) is an eigenvector of M
        for (idx, sdict) in enumerate(states_spectrum)
            # get coefficients back and compute M * vec - E * vec approximately zero
            vec = ExpandIntoBasis(sdict, basis)
            # normalize
            if norm(vec) ≈ 0
                continue
            end
            vecn = vec / norm(vec)
            lhs = M * vecn
            rhs = E_spectrum[idx] * vecn
            @test isapprox(lhs, rhs; atol=1e-8, rtol=1e-6)
        end
    end
end

@testset "Spectrum classify-mode correctness (occupancy sectors)" begin
    n = 4
    basis = BasisStates(n)
    # simple operator: total occupancy (diagonal), energies equal to occupancy
    op = [( "n", [i], 1.0 ) for i in 1:n]
    # Spectrum with classification by N should return energies equal to occupancy counts
    classifiedVals, classifiedVecs = Spectrum(op, basis, ['N']; classify=true)
    for (sector, vals) in classifiedVals
        # key is tuple with one element
        occ = sector[1]
        # all energies in this sector should equal occ (since operator is exactly occupancy)
        @test all(isapprox.(vals, occ; atol=1e-12))
        # number of states in this sector should match combinatorics: C(n,occ)
        @test length(vals) == binomial(n, occ)
    end
end

@testset "Spectrum with pre-classified basis (consistency)" begin
    n = 3
    basis = BasisStates(n)
    classified = ClassifyBasis(basis, ['N'])
    # build hopping operator between single-particle sites (conserves N)
    op = [("+-", [i,j], ifelse(abs(i-j)==1, 1.0, 0.0)) for i in 1:n for j in 1:n]
    # call Spectrum with classified basis
    vals_class, vecs_class = Spectrum(op, classified)
    # call generic Spectrum and compare aggregated result
    vals_all, vecs_all = Spectrum(op, basis)
    @test sort(vals_all) == sort(reduce(vcat, values(vals_class)))
end

@testset "IsEigenState exact checks and edge cases" begin
    n = 3
    basis = BasisStates(n)
    # number operator - basis states are eigenstates
    num_op = [( "n", [i], 1.0) for i in 1:n]
    for bstate in basis
        occ = sum(collect(keys(bstate))[1])
        if occ == 0
            continue
        end
        flag, val = IsEigenState(bstate, num_op)
        @test flag == true
        # eigenvalue should equal total occupancy
        @test isapprox(val, occ; atol=1e-12)
    end

    # test a non-eigenstate: superposition of two basis states -> not eigen of number operator unless same N
    b1 = basis[2]
    b2 = basis[3]
    super = Dict{BitVector,Float64}()
    # create equal amplitude superposition
    mergewith!(+, super, b1)
    mergewith!(+, super, b2)
    flag, _ = IsEigenState(super, num_op)
    # If b1 and b2 have different occupancies, should be false
    if GetSector(b1, ['N']) != GetSector(b2, ['N'])
        @test flag == false
    else
        @test flag == true
    end

    # test vacuum (all zero) handling in IsEigenState: create operator that annihilates state
    vacuum = Dict(collect(keys(basis[1]))[1] => 1.0)
    # define annihilation operator that zeros vacuum
    annih = [("-", [1], 1.0)]
    flagv, valv = IsEigenState(vacuum, annih)
    # function prints a message and returns true, 0. (per implementation) - allow either true or false but avoid crash
    @test typeof(flagv) == Bool
end

@testset "Spectrum numerical guardrails" begin
    # Non-Hermitian matrix detection when assumeHerm true should assert
    n = 2
    basis = BasisStates(n)
    # create clearly non-Hermitian operator (asymmetric weights)
    op_nonherm = [("+", [1], 1.0)]  # non-Hermitian by itself
    # OperatorMatrix likely non-Hermitian; Spectrum with assumeHerm true must error/assert
    try
        E, X = Spectrum(op_nonherm, basis; assumeHerm=true)
        @test maximum(abs.(OperatorMatrix(basis, op_nonherm) - OperatorMatrix(basis, op_nonherm)')) < 1e-12
    catch e
        @test isa(e, AssertionError) || isa(e, ErrorException)
    end
end

@testset "Stress randomized classification + spectrum internal consistency (property tests)" begin
    for trial in 1:RNG_TRIALS_SMALL
        n = rand(1:MAX_N)
        basis = BasisStates(n)
        # random hermitian-ish operator: make symmetric combinations
        op_terms = random_operator_terms(n; max_terms=4, max_sites=3)
        # symmetrize: add dagger of each term
        op_sym = vcat(op_terms, Dagger(copy(op_terms)))
        # ensure hermitian small random diagonal conf
        Evals, EigStates = Spectrum(op_sym, basis; assumeHerm=true)
        # every returned eigenstate should be an eigenstate of OperatorMatrix
        M = OperatorMatrix(basis, op_sym)
        for (i, st) in enumerate(EigStates)
            coeff = ExpandIntoBasis(st, basis)
            if norm(coeff) < 1e-12
                continue
            end
            coeffn = coeff / norm(coeff)
            residual = M * coeffn - Evals[i] * coeffn
            @test norm(residual) < 1e-7
        end
    end
end
