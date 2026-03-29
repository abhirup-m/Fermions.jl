using Plots, Fermions

N = 3200
Ek = 10 .^ collect(range(-5, stop=1, length=N))
Jk = 10 .^ collect(range(-1, stop=-2, length=N))
timeDef = Dict{String, Number}("start" => 0., "stop" => 1000000., "numSteps" => 1000)
timeResults = zeros(timeDef["numSteps"])
len = 40
bins = [range(1, step=step, stop=1 + step * len) for step in [1, 2, 4, 8, 16, 32, 64]]
M = 500
initState = Dict{BitVector, Float64}(vcat([1, 0, 0, 1], repeat([0, 0], 3)) => 1.)
operator = [("n", [1], 1.0), ("n", [2], -1.0)]
@showprogress for bin in bins
    ham = Tuple{String, Vector{Int64}, Float64}[]
    for (i, ki) in enumerate(bin)
        for (j, kj) in enumerate(bin)
            J = Jk[maximum((ki, kj))]
            push!(ham, ("n+-", [1, 1 + 2 * i, 1 + 2 * j], J/4))
            push!(ham, ("n+-", [1, 2 + 2 * i, 2 + 2 * j], -J/4))
            push!(ham, ("n+-", [2, 1 + 2 * i, 1 + 2 * j], -J/4))
            push!(ham, ("n+-", [2, 2 + 2 * i, 2 + 2 * j], J/4))
            push!(ham, ("+-+-", [1, 2, 2 + 2 * i, 1 + 2 * j], J/2))
            push!(ham, ("+-+-", [2, 1, 1 + 2 * i, 2 + 2 * j], J/2))
        end
        push!(ham, ("n", [1 + 2 * i,], Ek[ki]))
        push!(ham, ("n", [2 + 2 * i,], Ek[ki]))
    end
    hamFlow = MinceHamiltonian(ham, 10:2:2*(1+length(bin)))
    global timeResults += TimeEvolve(initState, operator, hamFlow, M, timeDef)
end
timeResults .*= GenCorrelation(initState,  operator) / timeResults[1]
plot(timeResults)
