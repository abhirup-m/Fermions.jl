using Fermions, Plots

t1 = 1.
halfSize = 50
basis = BasisStates1p(2 * halfSize)
p = plot(legend=false)
for Δ in 0.7:0.1:1.3
    hamiltonian = SSHModel((t1, t1 * Δ), halfSize; joinEnds=false)
    E, X = Spectrum(hamiltonian, basis)
    scatter!(p, repeat([Δ], E |> unique |> length), E |> unique)
end
display(p)

Δ = 3.
tolerance = 1e-6
hamiltonian = SSHModel((t1, t1 * Δ), halfSize; joinEnds=false)
E, X = Spectrum(hamiltonian, basis)
zeroModes = findall(Ei -> abs(Ei) < tolerance, E)
numOperators = [[("n", [i], 1.)] for i in 1:2*halfSize]
p = plot()
for mode in zeroModes
    coeffs = [GenCorrelation(X[mode], ni) for ni in numOperators]
    println(mode, coeffs)
    plot!(p, coeffs)
end
display(p)
