using Fermions, Plots

### site indexing
### 1   2   3    4   5   6   7   8   9  10  ...
### du  dd  A1u A1d B1u B1d A2u A2d B2u B2d ...
hop_t = 1.
J = 1.
sites = 30

channel1Up = 3:4:(2 + 4 * sites)
channel2Up = 5:4:(4 + 4 * sites)

hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
for up in [3, 5]
    push!(hamiltonian, ("nn", [1, up], 0.25 * J))
    push!(hamiltonian, ("nn", [1, up+1], -0.25 * J))
    push!(hamiltonian, ("nn", [2, up], -0.25 * J))
    push!(hamiltonian, ("nn", [2, up+1], 0.25 * J))
    push!(hamiltonian, ("+-+-", [1, 2, up+1, up], 0.5 * J))
    push!(hamiltonian, ("+-+-", [2, 1, up, up+1], 0.5 * J))
end

for up1 in channel1Up[1:end-1]
    for up in [up1, up1 + 2]
        push!(hamiltonian, ("+-", [up, up+4], -hop_t))
        push!(hamiltonian, ("+-", [up+1, up+5], -hop_t))
        push!(hamiltonian, ("+-", [up+4, up], -hop_t))
        push!(hamiltonian, ("+-", [up+5, up], -hop_t))
    end
end

hamiltonianFamily = MinceHamiltonian(hamiltonian, 6:2:(2 + 4*sites))
Ad = Dict("create" => [("+", [1,], 1.0), ("+", [2], 1.0)])
Ad_1 = Dict("create" => [("+-+", [1,2,3], 1.0), ("+-+", [2,1,4], 1.0)])
Ad_2 = Dict("create" => [("+-+", [1,2,5], 1.0), ("+-+", [2,1,6], 1.0)])
Ad_12 = Dict("create" => [("+-+", [1,2,5], 1.0), ("+-+", [2,1,6], 1.0), ("+-+", [1,2,5], 1.0), ("+-+", [2,1,6], 1.0)])
for A in [Ad, Ad_1, Ad_2, Ad_12]
    A["destroy"] = Dagger(copy(A["create"]))
end

results = IterDiag(hamiltonianFamily, 
                   1000; 
                   specFuncDefDict=Dict(
                                        "Ad" => Ad,
                                        "Ad_1" => Ad_1,
                                        "Ad_2" => Ad_2,
                                        "Ad_12" => Ad_12,
                                        ),
                  )
ω = collect(-10:0.01:10)
p = plot()
for name in ["Ad", "Ad_1", "Ad_2", "Ad_12"]
    A = SpecFunc(vcat(results[name]...), ω, 0.1)
    plot!(p, ω, A)
end
display(p)
