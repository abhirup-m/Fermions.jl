using Random, LinearAlgebra

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


@testset "Spectral Function" begin
    basisStates = BasisStates(4)
    hop_t = abs(rand())
    U = abs(rand())
    eps = -U / 2
    broadening = 1e-3
    operatorList = HubbardDimerOplist(eps, U, hop_t)
    eigvals, eigvecs = Spectrum(operatorList, basisStates)
    omegaVals = collect(range(-10.0, stop=10.0, length=1000))
    specfunc = SpecFunc(eigvals, eigvecs, Dict("destroy" => [("-", [1], 1.0)], "create" => [("+", [1], 1.0)]), omegaVals, basisStates, broadening)
    specfuncCompare = HubbardDimerSpecFunc(eps, U, hop_t, omegaVals, broadening)
    @test specfunc ./ maximum(specfunc) ≈ specfuncCompare ./ maximum(specfuncCompare)
end


@testset "Alternative RDM Methods" begin
    @testset for totalQubits in 1:8
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
