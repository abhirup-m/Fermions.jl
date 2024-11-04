function KondoModel(
        numBathSites::Int64,
        hop_t::Float64,
        kondoJ::Float64;
        globalField::Float64=0.,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    # intra-bath hopping
    for site in 1:(numBathSites-1)
        push!(hamiltonian, ("+-",  [1 + 2 * site, 3 + 2 * site], -hop_t)) # c^†_{j,up} c_{j+1,up}
        push!(hamiltonian, ("+-",  [3 + 2 * site, 1 + 2 * site], -hop_t)) # c^†_{j+1,up} c_{j,up}
        push!(hamiltonian, ("+-",  [2 + 2 * site, 4 + 2 * site], -hop_t)) # c^†_{j,dn} c_{j+1,dn}
        push!(hamiltonian, ("+-",  [4 + 2 * site, 2 + 2 * site], -hop_t)) # c^†_{j+1,dn} c_{j,dn}
    end

    # kondo terms
    push!(hamiltonian, ("nn",  [1, 3], kondoJ/4)) # n_{d up, n_{0 up}
    push!(hamiltonian, ("nn",  [1, 4], -kondoJ/4)) # n_{d up, n_{0 down}
    push!(hamiltonian, ("nn",  [2, 3], -kondoJ/4)) # n_{d down, n_{0 up}
    push!(hamiltonian, ("nn",  [2, 4], kondoJ/4)) # n_{d down, n_{0 down}
    push!(hamiltonian, ("+-+-",  [1, 2, 4, 3], kondoJ/2)) # S_d^+ S_0^-
    push!(hamiltonian, ("+-+-",  [2, 1, 3, 4], kondoJ/2)) # S_d^- S_0^+

    # global magnetic field (to lift any trivial degeneracy)
    if globalField ≠ 0
        for site in 0:numBathSites
            push!(hamiltonian, ("n",  [1 + 2 * site], globalField))
            push!(hamiltonian, ("n",  [2 + 2 * site], -globalField))
        end
    end

    return hamiltonian
end
export KondoModel


function KondoModel(
        dispersion::Vector{Float64},
        kondoJ::Float64;
        globalField::Float64=0.,
        cavityIndices::Vector{Int64},
    )
    numBathSites = length(dispersion)
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    # kinetic energy
    for site in 1:(numBathSites-1)
        push!(hamiltonian, ("n",  [1 + 2 * site], dispersion[site])) # up spin
        push!(hamiltonian, ("n",  [2 + 2 * site], dispersion[site])) # down spin
    end

    # kondo terms
    for indices in Iterators.product(1:numBathSites, 1:numBathSites)
        if any(∈(cavityIndices), indices)
            kondoJ_indices = 0
        else
            kondoJ_indices = kondoJ
        end
        up1, up2 = 2 .* indices .+ 1
        down1, down2 = (up1, up2) .+ 1
        push!(hamiltonian, ("n+-",  [1, up1, up2], kondoJ_indices / 4)) # n_{d up, n_{0 up}
        push!(hamiltonian, ("n+-",  [1, down1, down2], -kondoJ_indices / 4)) # n_{d up, n_{0 down}
        push!(hamiltonian, ("n+-",  [2, up1, up2], -kondoJ_indices / 4)) # n_{d down, n_{0 up}
        push!(hamiltonian, ("n+-",  [2, down1, down2], kondoJ_indices / 4)) # n_{d down, n_{0 down}
        push!(hamiltonian, ("+-+-",  [1, 2, down1, up2], kondoJ_indices / 2)) # S_d^+ S_0^-
        push!(hamiltonian, ("+-+-",  [2, 1, up1, down2], kondoJ_indices / 2)) # S_d^- S_0^+
    end

    # global magnetic field (to lift any trivial degeneracy)
    if globalField ≠ 0
        for site in 0:numBathSites
            push!(hamiltonian, ("n",  [1 + 2 * site], globalField))
            push!(hamiltonian, ("n",  [2 + 2 * site], -globalField))
        end
    end

    return hamiltonian
end
export KondoModel
