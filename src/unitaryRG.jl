using ProgressMeter

function Renormalisation(
        operatorDef::Function,
        numLegs::Int,
        occupied::Vector{Int64},
        vacant::Vector{Int64},
        inside::Vector{Int64},
    )
    occupied = collect(range(2 * minimum(occupied) - 1, 2 * maximum(occupied)))
    vacant = collect(range(2 * minimum(vacant) - 1, 2 * maximum(vacant)))
    println(occupied)
    println(vacant)
    allSites = collect(1:2*maximum(inside))
    onShell = vcat(occupied, vacant)
    spins = Dict(site => isodd(site) ? 1 : -1 for site in allSites)
    renormalisation = Dict()
    @showprogress for right in Iterators.product([allSites for _ in 1:numLegs]...)
        if intersect(right, occupied) |> isempty && intersect(right, vacant) |> isempty
            continue
        end
        operator, factorRight = operatorDef(right, spins)
        if factorRight == 0
            continue
        end
        rightOperator, rightMembers, rightSign = Simplify(copy(operator), collect(right))
        factorRight *= rightSign
        if count(p -> p[1] ∈ vcat(vacant) && p[2] ∈ "+-", zip(rightMembers, rightOperator)) |> iseven
            if count(p -> p[1] ∈ vcat(occupied) && p[2] ∈ "+-", zip(rightMembers, rightOperator)) |> iseven
                continue
            end
        end
        for left in Iterators.product([allSites for _ in 1:numLegs]...)
            if intersect(left, occupied) |> isempty && intersect(left, vacant) |> isempty
                continue
            end
            operator, factorLeft = operatorDef(left, spins)
            if factorLeft == 0
                continue
            end
            leftOperator, leftMembers, leftSign = Simplify(copy(operator), collect(left))
            factorLeft *= leftSign
            if count(p -> p[1] ∈ vcat(vacant) && p[2] ∈ "+-", zip(leftMembers, leftOperator)) |> iseven
                if count(p -> p[1] ∈ vcat(occupied) && p[2] ∈ "+-", zip(leftMembers, leftOperator)) |> iseven
                    continue
                end
            end
            combinedOperator = Product([(join(leftOperator), leftMembers, factorLeft)], [(join(rightOperator), rightMembers, factorRight)])
            @assert length(combinedOperator) == 1
            operator, members, factor = combinedOperator[1]
            operator, members, sign1 = Simplify([ch for ch in operator], members)
            ioms = true
            for (t, m) in zip(operator, members)
                if m ∈ vacant && t ≠ 'h'
                    ioms = false
                    break
                end
                if m ∈ occupied && t ≠ 'n'
                    ioms = false
                    break
                end
            end
            if !ioms
                continue
            end
            operator = operator[findall(∉(onShell), members)]
            members = members[findall(∉(onShell), members)]
            if isempty(operator)
                continue
            end
            operator, members, sign2 = OrganiseOperator(join(operator), members)
            operator = join(operator)
            factor = factor * sign1 * sign2
            if factor ≠ 0
                if haskey(renormalisation, (operator, members))
                    renormalisation[(operator, members)] += factor
                else
                    renormalisation[(operator, members)] = factor
                end
            end
        end
    end
    renormalisation = Dict(k => v for (k,v) in renormalisation if v ≠ 0)
    found_h = true
    while found_h
        found_h = false
        for ((op, mem), v) in renormalisation
            h_loc = findfirst(==('h'), op)
            if !isnothing(h_loc)
                found_h = true
                minusKey = [ch for ch in op]
                minusKey[h_loc] = 'n'
                minusKey = join(minusKey)
                oneKey = op[1:h_loc-1] * op[h_loc+1:end]
                oneMem = vcat(mem[1:h_loc-1], mem[h_loc+1:end])
                for (key, val) in [((oneKey, oneMem), v), ((minusKey, mem), -v)]
                    if haskey(renormalisation, key)
                        renormalisation[key] += val
                    else
                        renormalisation[key] = val
                    end
                end
                delete!(renormalisation, (op, mem))
            end
        end
    end
    return renormalisation
end
export Renormalisation
