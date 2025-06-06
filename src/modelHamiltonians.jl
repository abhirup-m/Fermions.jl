function KondoModel(
        numBathSites::Int64,
        hop_t::Float64,
        kondoJ::Float64;
        globalField::Float64=0.,
        couplingTolerance::Float64=1e-15,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    # intra-bath hopping
    if abs(hop_t) > couplingTolerance
        for site in 1:(numBathSites-1)
            push!(hamiltonian, ("+-",  [1 + 2 * site, 3 + 2 * site], -hop_t)) # c^†_{j,up} c_{j+1,up}
            push!(hamiltonian, ("+-",  [3 + 2 * site, 1 + 2 * site], -hop_t)) # c^†_{j+1,up} c_{j,up}
            push!(hamiltonian, ("+-",  [2 + 2 * site, 4 + 2 * site], -hop_t)) # c^†_{j,dn} c_{j+1,dn}
            push!(hamiltonian, ("+-",  [4 + 2 * site, 2 + 2 * site], -hop_t)) # c^†_{j+1,dn} c_{j,dn}
        end
    end

    # kondo terms
    if abs(kondoJ) > couplingTolerance
        push!(hamiltonian, ("nn",  [1, 3], kondoJ/4)) # n_{d up, n_{0 up}
        push!(hamiltonian, ("nn",  [1, 4], -kondoJ/4)) # n_{d up, n_{0 down}
        push!(hamiltonian, ("nn",  [2, 3], -kondoJ/4)) # n_{d down, n_{0 up}
        push!(hamiltonian, ("nn",  [2, 4], kondoJ/4)) # n_{d down, n_{0 down}
        push!(hamiltonian, ("+-+-",  [1, 2, 4, 3], kondoJ/2)) # S_d^+ S_0^-
        push!(hamiltonian, ("+-+-",  [2, 1, 3, 4], kondoJ/2)) # S_d^- S_0^+
    end

    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in 0:numBathSites
            push!(hamiltonian, ("n",  [1 + 2 * site], globalField/2))
            push!(hamiltonian, ("n",  [2 + 2 * site], -globalField/2))
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export KondoModel


function KondoModel(
        numBathSites::Int64,
        hop_t::Float64,
        kondoJ::Float64,
        bathInt::Float64;
        globalField::Float64=0.,
        couplingTolerance::Float64=1e-15,
    )
    hamiltonian = KondoModel(numBathSites, hop_t, kondoJ; 
                             globalField=globalField,
                             couplingTolerance=couplingTolerance
                            )

    if abs(bathInt) > couplingTolerance
        push!(hamiltonian, ("n", [3], -bathInt / 2)) # 
        push!(hamiltonian, ("n", [4], -bathInt / 2)) # 
        push!(hamiltonian, ("nn", [3, 4], bathInt)) # 
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export KondoModel


function KondoModel(
        dispersion::Vector{Float64},
        kondoJ::Float64;
        globalField::Float64=0.,
        impurityField::Float64=0.,
        cavityIndices::Vector{Int64}=Int64[],
        couplingTolerance::Float64=1e-15,
    )
    numBathSites = length(dispersion)
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    # kinetic energy
    for site in 1:numBathSites
        if abs(dispersion[site]) < couplingTolerance
            continue
        end
        push!(hamiltonian, ("n",  [1 + 2 * site], dispersion[site])) # up spin
        push!(hamiltonian, ("n",  [2 + 2 * site], dispersion[site])) # down spin
    end

    # kondo terms
    if abs(kondoJ) > couplingTolerance
        for indices in Iterators.product(1:numBathSites, 1:numBathSites)
            if any(∈(cavityIndices), indices)
                continue
            end
            up1, up2 = 2 .* indices .+ 1
            down1, down2 = (up1, up2) .+ 1
            push!(hamiltonian, ("n+-",  [1, up1, up2], kondoJ / 4)) # n_{d up, n_{0 up}
            push!(hamiltonian, ("n+-",  [1, down1, down2], -kondoJ / 4)) # n_{d up, n_{0 down}
            push!(hamiltonian, ("n+-",  [2, up1, up2], -kondoJ / 4)) # n_{d down, n_{0 up}
            push!(hamiltonian, ("n+-",  [2, down1, down2], kondoJ / 4)) # n_{d down, n_{0 down}
            push!(hamiltonian, ("+-+-",  [1, 2, down1, up2], kondoJ / 2)) # S_d^+ S_0^-
            push!(hamiltonian, ("+-+-",  [2, 1, up1, down2], kondoJ / 2)) # S_d^- S_0^+
        end
    end

    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in 0:numBathSites
            push!(hamiltonian, ("n",  [1 + 2 * site], globalField/2))
            push!(hamiltonian, ("n",  [2 + 2 * site], -globalField/2))
        end
    end

    # impurity magnetic field (to lift any trivial local degeneracy)
    if abs(impurityField) > couplingTolerance
        push!(hamiltonian, ("n",  [1], impurityField/2))
        push!(hamiltonian, ("n",  [2], -impurityField/2))
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export KondoModel


function KondoModel(
        dispersion::Vector{Float64},
        kondoJ::Matrix{Float64};
        globalField::Float64=0.,
        impurityField::Float64=0.,
        couplingTolerance::Float64=1e-15,
        cavityIndices::Vector{Int64}=Int64[],
    )
    numBathSites = length(dispersion)
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    # kinetic energy
    for site in 1:numBathSites
        if abs(dispersion[site]) < couplingTolerance
            continue
        end
        push!(hamiltonian, ("n",  [1 + 2 * site], dispersion[site])) # up spin
        push!(hamiltonian, ("n",  [2 + 2 * site], dispersion[site])) # down spin
    end

    # kondo terms
    for indices in Iterators.product(1:numBathSites, 1:numBathSites)
        if any(∈(cavityIndices), indices)
            kondoJ_indices = 0
        else
            kondoJ_indices = kondoJ[indices...]
        end
        if abs(kondoJ_indices) < couplingTolerance
            continue
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
    if abs(globalField) > couplingTolerance
        for site in 0:numBathSites
            push!(hamiltonian, ("n",  [1 + 2 * site], globalField/2))
            push!(hamiltonian, ("n",  [2 + 2 * site], -globalField/2))
        end
    end

    # impurity magnetic field (to lift any trivial local degeneracy)
    if abs(impurityField) > couplingTolerance
        push!(hamiltonian, ("n",  [1], impurityField/2))
        push!(hamiltonian, ("n",  [2], -impurityField/2))
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export KondoModel


function KondoModel(
        dispersionArray::Vector{Vector{Float64}},
        kondoJ::Vector{Dict{NTuple{2, Int64}, Float64}};
        globalField::Float64=0.,
        couplingTolerance::Float64=1e-15,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    @assert dispersionArray .|> length |> allequal
    numBathSites = length(dispersionArray[1])
    upIndices = [1 .+ 2 * channel .+ 2 * length(kondoJ) .* (0:(numBathSites-1)) for channel in 1:length(kondoJ)]

    # bath kinetic energy
    for (channel, dispersion) in enumerate(dispersionArray)
        append!(hamiltonian, [("n", [upIndices[channel][i]], Ek) for (i, Ek) in enumerate(dispersion)])
        append!(hamiltonian, [("n", [upIndices[channel][i]+1], Ek) for (i, Ek) in enumerate(dispersion)])
    end

    # kondo terms
    for channel in 1:length(kondoJ)
        for ((i, j), J) in kondoJ[channel]
            if abs(J) > couplingTolerance
                upSite_i = upIndices[channel][i]
                upSite_j = upIndices[channel][j]
                push!(hamiltonian, ("n+-",  [1, upSite_i, upSite_j], J/4)) # n_{d up, n_{0 up}
                push!(hamiltonian, ("n+-",  [1, upSite_i+1, upSite_j+1], -J/4)) # n_{d up, n_{0 down}
                push!(hamiltonian, ("n+-",  [2, upSite_i, upSite_j], -J/4)) # n_{d down, n_{0 up}
                push!(hamiltonian, ("n+-",  [2, upSite_i+1, upSite_j+1], J/4)) # n_{d down, n_{0 down}
                push!(hamiltonian, ("+-+-",  [1, 2, upSite_i+1, upSite_j], J/2)) # S_d^+ S_0^-
                push!(hamiltonian, ("+-+-",  [2, 1, upSite_i, upSite_j+1], J/2)) # S_d^- S_0^+
            end
        end
    end

    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in 1:(1 + numBathSites * length(kondoJ))
            push!(hamiltonian, ("n",  [2 * site - 1], globalField/2))
            push!(hamiltonian, ("n",  [2 * site], -globalField/2))
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export KondoModel


function KondoModel(
        dispersion::Vector{Float64},
        kondoJ::Float64,
        bathInt::Float64;
        bathIntLegs::Int64=4,
        globalField::Float64=0.,
        impurityField::Float64=0.,
        couplingTolerance::Float64=1e-15,
        cavityIndices::Vector{Int64}=Int64[],
    )
    numBathSites = length(dispersion)
    hamiltonian = KondoModel(dispersion, kondoJ; globalField=globalField, 
                             cavityIndices=cavityIndices, couplingTolerance=couplingTolerance)

    if abs(bathInt) > couplingTolerance
        for indices in Iterators.product(repeat([1:numBathSites], 4)...)
            if length(unique(indices)) > bathIntLegs
                continue
            end
            up1, up2, up3, up4 = 2 .* indices .+ 1
            down1, down2, down3, down4 = (up1, up2, up3, up4) .+ 1
            push!(hamiltonian, ("+-+-",  [up1, up2, up3, up4], -bathInt / 2)) # 
            push!(hamiltonian, ("+-+-",  [down1, down2, down3, down4], -bathInt / 2)) # 
            push!(hamiltonian, ("+-+-",  [up1, up2, down3, down4], bathInt / 2)) # 
            push!(hamiltonian, ("+-+-",  [down1, down2, up3, up4], bathInt / 2)) # 
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export KondoModel


function KondoModel(
        dispersion::Vector{Float64},
        kondoJ::Matrix{Float64},
        k_indices::Vector{Int64},
        bathIntFunc::Function;
        bathIntLegs::Int64=4,
        globalField::Float64=0.,
        impurityField::Float64=0.,
        couplingTolerance::Float64=1e-15,
        cavityIndices::Vector{Int64}=Int64[],
    )
    @assert length(k_indices) == length(dispersion)

    numBathSites = length(dispersion)
    hamiltonian = KondoModel(dispersion, kondoJ; globalField=globalField, 
                             impurityField=impurityField, cavityIndices=cavityIndices, 
                             couplingTolerance=couplingTolerance
                            )
    for indices in Iterators.product(repeat([1:numBathSites], 4)...)
        bathIntVal = bathIntFunc([k_indices[i] for i in indices])
        if length(unique(indices)) > bathIntLegs || abs(bathIntVal) < couplingTolerance
            continue
        end
        up1, up2, up3, up4 = 2 .* indices .+ 1
        down1, down2, down3, down4 = (up1, up2, up3, up4) .+ 1
        push!(hamiltonian, ("+-+-",  [up1, up2, up3, up4], -bathIntVal / 2)) # 
        push!(hamiltonian, ("+-+-",  [down1, down2, down3, down4], -bathIntVal / 2)) # 
        push!(hamiltonian, ("+-+-",  [up1, up2, down3, down4], bathIntVal / 2)) # 
        push!(hamiltonian, ("+-+-",  [down1, down2, up3, up4], bathIntVal / 2)) # 
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export KondoModel


function  SiamRealSpace(
        hop_t::Float64,
        numBathSites::Int64,
        hybridisation::Float64,
        impOnsite::Float64,
        impCorr::Float64;
        globalField::Float64=0.,
        couplingTolerance::Float64=1e-15,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    # intra-bath hopping
    if abs(hop_t) > couplingTolerance
        for site in 1:(numBathSites-1)
            push!(hamiltonian, ("+-",  [1 + 2 * site, 3 + 2 * site], -hop_t)) # c^†_{j,up} c_{j+1,up}
            push!(hamiltonian, ("+-",  [3 + 2 * site, 1 + 2 * site], -hop_t)) # c^†_{j+1,up} c_{j,up}
            push!(hamiltonian, ("+-",  [2 + 2 * site, 4 + 2 * site], -hop_t)) # c^†_{j,dn} c_{j+1,dn}
            push!(hamiltonian, ("+-",  [4 + 2 * site, 2 + 2 * site], -hop_t)) # c^†_{j+1,dn} c_{j,dn}
        end
    end

    # hybridisation terms
    if abs(hybridisation) > couplingTolerance
        push!(hamiltonian, ("+-",  [1, 3], hybridisation)) # c^†_{d,up} c_{0,up}
        push!(hamiltonian, ("+-",  [3, 1], hybridisation)) # c^†_{0,up} c_{d,up}
        push!(hamiltonian, ("+-",  [2, 4], hybridisation)) # c^†_{d,dn} c_{0,dn}
        push!(hamiltonian, ("+-",  [4, 2], hybridisation)) # c^†_{0,dn} c_{d,dn}
    end

    # impurity local terms
    push!(hamiltonian, ("n",  [1], impOnsite)) # Ed nup
    push!(hamiltonian, ("n",  [2], impOnsite)) # Ed ndown
    push!(hamiltonian, ("nn",  [1, 2], impCorr)) # U nup ndown

    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in 0:numBathSites
            push!(hamiltonian, ("n",  [1 + 2 * site], globalField/2))
            push!(hamiltonian, ("n",  [2 + 2 * site], -globalField/2))
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"
    return hamiltonian
end
export SiamRealSpace


function SiamKSpace(
        dispersion::Vector{Float64},
        hybridisation::Float64,
        impOnsite::Float64,
        impCorr::Float64;
        globalField::Float64=0.,
        cavityIndices::Vector{Int64}=Int64[],
        couplingTolerance::Float64=1e-15,
    )
    numBathSites = length(dispersion)
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    # kinetic energy
    for site in 1:numBathSites
        push!(hamiltonian, ("n",  [1 + 2 * site], dispersion[site])) # up spin
        push!(hamiltonian, ("n",  [2 + 2 * site], dispersion[site])) # down spin
    end

    push!(hamiltonian, ("n",  [1], impOnsite)) # Ed nup
    push!(hamiltonian, ("n",  [2], impOnsite)) # Ed ndown
    push!(hamiltonian, ("nn",  [1, 2], impCorr)) # U nup ndown

    # hybridisation
    for site in 1:numBathSites
        if site ∈(cavityIndices)
            continue
        end
        if abs(hybridisation) < couplingTolerance
            continue
        end
        up = 2 * site + 1
        down = up + 1
        push!(hamiltonian, ("+-",  [1, up], hybridisation)) 
        push!(hamiltonian, ("+-",  [up, 1], hybridisation))
        push!(hamiltonian, ("+-",  [2, down], hybridisation))
        push!(hamiltonian, ("+-",  [down, 2], hybridisation))
    end

    # global magnetic field (to lift any trivial degeneracy)
    if globalField ≠ 0
        for site in 0:numBathSites
            push!(hamiltonian, ("n",  [1 + 2 * site], globalField/2))
            push!(hamiltonian, ("n",  [2 + 2 * site], -globalField/2))
        end
    end

    return hamiltonian
end
export SiamKSpace


function SiamKSpace(
        dispersion::Vector{Float64},
        hybridisation::Vector{Float64},
        impOnsite::Float64,
        impCorr::Float64;
        globalField::Float64=0.,
        impurityField::Float64=0.,
        cavityIndices::Vector{Int64}=Int64[],
        couplingTolerance::Float64=1e-15,
    )
    numBathSites = length(dispersion)
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    # kinetic energy
    for site in 1:numBathSites
        push!(hamiltonian, ("n",  [1 + 2 * site], dispersion[site])) # up spin
        push!(hamiltonian, ("n",  [2 + 2 * site], dispersion[site])) # down spin
    end

    push!(hamiltonian, ("n",  [1], impOnsite)) # Ed nup
    push!(hamiltonian, ("n",  [2], impOnsite)) # Ed ndown
    push!(hamiltonian, ("nn",  [1, 2], impCorr)) # U nup ndown

    # hybridisation
    for site in 1:numBathSites
        if site ∈(cavityIndices)
            continue
        end
        if abs(hybridisation[site]) < couplingTolerance
            continue
        end
        up = 2 * site + 1
        down = up + 1
        push!(hamiltonian, ("+-",  [1, up], hybridisation[site])) 
        push!(hamiltonian, ("+-",  [up, 1], hybridisation[site]))
        push!(hamiltonian, ("+-",  [2, down], hybridisation[site]))
        push!(hamiltonian, ("+-",  [down, 2], hybridisation[site]))
    end

    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in 0:numBathSites
            push!(hamiltonian, ("n",  [1 + 2 * site], globalField/2))
            push!(hamiltonian, ("n",  [2 + 2 * site], -globalField/2))
        end
    end

    # impurity magnetic field (to lift any trivial local degeneracy)
    if abs(impurityField) > couplingTolerance
        push!(hamiltonian, ("n",  [1], impurityField/2))
        push!(hamiltonian, ("n",  [2], -impurityField/2))
    end

    return hamiltonian
end
export SiamKSpace


function SiamKondoKSpace(
        dispersion::Vector{Float64},
        hybridisation::Float64,
        kondoJ::Float64,
        impOnsite::Float64,
        impCorr::Float64;
        globalField::Float64=0.,
        cavityIndices::Vector{Int64}=Int64[],
        couplingTolerance::Float64=1e-15,
    )
    hamiltonian = SiamKSpace(dispersion, hybridisation, impOnsite, impCorr; globalField=globalField, cavityIndices=cavityIndices)

    numBathSites = length(dispersion)
    # kondo terms
    for indices in Iterators.product(1:numBathSites, 1:numBathSites)
        if any(∈(cavityIndices), indices)
            kondoJ_indices = 0
            continue
        else
            kondoJ_indices = kondoJ
        end
        if abs(kondoJ) < couplingTolerance
            continue
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
    return hamiltonian

    
end
export SiamKondoKSpace


function Dispersion(
        numStates::Int64, 
        lowerCutOff::Float64,
        upperCutoff::Float64,
        discretisation::String;
        phSymmetry::Bool=true,
    )

    @assert abs(lowerCutOff) ≤ abs(upperCutoff)
    @assert discretisation == "log" || discretisation == "lin"
    if discretisation == "log"
        @assert lowerCutOff > 0
    end

    dispersion = zeros(numStates)
    if discretisation == "log"
        if phSymmetry
            if numStates % 2 == 0
                dispersion[1:2:end] .= 10. .^ range(log10(lowerCutOff), stop=log10(upperCutoff), length=div(numStates, 2))
                dispersion[2:2:end] .= -1 .* dispersion[1:2:end]
            else
                dispersion[3:2:end] .= 10. .^ range(log10(lowerCutOff), stop=log10(upperCutoff), length=div(numStates - 1, 2))
                dispersion[2:2:end] .= -1 .* dispersion[3:2:end]
            end
        else
            if numStates % 2 == 0
                dispersion .= 10. .^ range(log10(lowerCutOff), stop=log10(upperCutoff), length=numStates)
            else
                dispersion[2:end] .= 10. .^ range(log10(lowerCutOff), stop=log10(upperCutoff), length=numStates-1)
            end
        end
    else
        if phSymmetry
            if numStates % 2 == 0
                dispersion[1:2:end] = range(abs(lowerCutOff), stop=abs(upperCutoff), length=div(numStates, 2)) 
                dispersion[2:2:end] .= -1 .* dispersion[1:2:end]
            else
                @assert dispersion[1] == 0
                dispersion[1:2:end] = range(abs(lowerCutOff), stop=abs(upperCutoff), length=div(numStates+1, 2)) 
                dispersion[2:2:end] .= -1 .* dispersion[3:2:end]
            end
        else
            dispersion = collect(range(abs(lowerCutOff), stop=abs(upperCutoff), length=numStates))
        end
    end
    return dispersion
end
export Dispersion


function J1J2Model(
        J2J1Ratio::Float64,
        numSites::Int64
    )
    J1 = 1.
    J2 = J1 * J2J1Ratio
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    for site in 1:numSites-1
        # J1 terms
        push!(hamiltonian, ("nn", [site, site+1], J1))
        push!(hamiltonian, ("n", [site], -J1/2))
        push!(hamiltonian, ("n", [site+1], -J1/2))
        push!(hamiltonian, ("+-", [site, site+1], J1/2))   
        push!(hamiltonian, ("+-", [site+1, site], J1/2))   

        if site == numSites-1
            continue
        end

        # J2 terms
        push!(hamiltonian, ("nn", [site, site+2], J2))
        push!(hamiltonian, ("n", [site], -J2/2))
        push!(hamiltonian, ("n", [site+2], -J2/2))
        push!(hamiltonian, ("+-", [site, site+2], J2/2))   
        push!(hamiltonian, ("+-", [site+2, site], J2/2))   
    end
    return hamiltonian
end
export J1J2Model


function CollatzModel(
        numSites::Int64,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    for site in 2:2:numSites
        push!(hamiltonian, ("+-", [div(site, 2), site], 1.))
    end
    for site in 1:2:div(numSites-1, 3)
        push!(hamiltonian, ("+-", [site * 3 + 1, site], 1.))
    end
    return hamiltonian
end
export CollatzModel


function SSHModel(
        hopPair::NTuple{2, Float64},
        numSitesPerSubl::Int64;
        joinEnds::Bool=false,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    for i in 1:(2*numSitesPerSubl-1)
        hop = isodd(i) ? hopPair[1] : hopPair[2]
        push!(hamiltonian, ("+-", [i, i + 1], -hop))
        push!(hamiltonian, ("+-", [i + 1, i], -hop))
    end
    if joinEnds
        push!(hamiltonian, ("+-", [2 * numSitesPerSubl, 1], -hopPair[2]))
        push!(hamiltonian, ("+-", [1, 2 * numSitesPerSubl], -hopPair[2]))
    end
    return hamiltonian
end
export SSHModel


function SSHModel(
        hopPairX::NTuple{2, Float64},
        hopPairY::NTuple{2, Float64},
        numSitesPerSubl::NTuple{2, Int64};
        joinEnds::NTuple{2, Bool}=(false, false),
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    for i in 1:2*numSitesPerSubl[2] # y
        for j in 1:(2 * numSitesPerSubl[1] - 1) # x
            hop = isodd(j) ? hopPairX[1] : hopPairX[2]
            position = (i - 1) * 2 * numSitesPerSubl[1] + j
            push!(hamiltonian, ("+-", [position, position + 1], -hop))
            push!(hamiltonian, ("+-", [position + 1, position], -hop))
        end
        if joinEnds[1] && numSitesPerSubl[1] > 1
            push!(hamiltonian, ("+-", [i * 2 * numSitesPerSubl[1], (i - 1) * 2 * numSitesPerSubl[1] + 1], -hopPairX[2]))
            push!(hamiltonian, ("+-", [(i - 1) * 2 * numSitesPerSubl[1] + 1, i * 2 * numSitesPerSubl[1]], -hopPairX[2]))
        end
    end
    for i in 1:2*numSitesPerSubl[1] # x
        for j in 1:(2 * numSitesPerSubl[2] - 1) # y
            hop = isodd(j) ? hopPairY[1] : hopPairY[2]
            position = (j - 1) * 2 * numSitesPerSubl[1] + i
            push!(hamiltonian, ("+-", [position, position + 2 * numSitesPerSubl[1]], -hop))
            push!(hamiltonian, ("+-", [position + 2 * numSitesPerSubl[1], position], -hop))
        end
        if joinEnds[2] && numSitesPerSubl[2] > 1
            push!(hamiltonian, ("+-", [2 * numSitesPerSubl[1] * (2 * numSitesPerSubl[2] - 1) + i, i], -hopPairY[2]))
            push!(hamiltonian, ("+-", [i, 2 * numSitesPerSubl[1] * (2 * numSitesPerSubl[2] - 1) + i], -hopPairY[2]))
        end
    end
    return hamiltonian
end
export SSHModel


"""
Multichannel Kondo model Hamiltonian. Index 1 is impurity site.
If numBathSites is 5, the indices 
`1 + numBathSites + 1` to `1 + 2 * numBathSites` is the second,
and so on. Number of channels is determined by the length of the
kondo coupling vector `[J1, J2, ...]`.
"""
function KondoModel(
        numBathSites::Int64,
        hop_t::Float64,
        kondoJ::Vector{Float64};
        globalField::Float64=0.,
        couplingTolerance::Float64=1e-15,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    kondoJEnlarged = [Dict((1, 1) => J) for J in kondoJ]
    return KondoModel(numBathSites, hop_t, kondoJEnlarged; globalField=globalField, couplingTolerance=couplingTolerance)

end
export KondoModel


function KondoModel(
        numBathSites::Int64,
        hop_t::Float64,
        kondoJ::Vector{Dict{NTuple{2, Int64}, Float64}};
        globalField::Float64=0.,
        couplingTolerance::Float64=1e-15,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    upIndices = [1 .+ 2 * channel .+ 2 * length(kondoJ) .* (0:(numBathSites-1)) for channel in 1:length(kondoJ)]

    # intra-bath hopping
    if abs(hop_t) > couplingTolerance
        for site_i in 1:(numBathSites-1)
            for channel in 1:length(kondoJ)
                upSite_i = upIndices[channel][site_i]
                upSite_ip1 = upIndices[channel][site_i + 1]
                push!(hamiltonian, ("+-",  [upSite_i, upSite_ip1], -hop_t)) # c^†_{j,up} c_{j+1,up}
                push!(hamiltonian, ("+-",  [upSite_ip1, upSite_i], -hop_t)) # c^†_{j+1,up} c_{j,up}
                push!(hamiltonian, ("+-",  [upSite_i + 1, upSite_ip1 + 1], -hop_t)) # c^†_{j,dn} c_{j+1,dn}
                push!(hamiltonian, ("+-",  [upSite_ip1 + 1, upSite_i + 1], -hop_t)) # c^†_{j+1,dn} c_{j,dn}
            end
        end
    end

    # kondo terms
    for channel in 1:length(kondoJ)
        for ((i, j), J) in kondoJ[channel]
            if abs(J) > couplingTolerance
                upSite_i = upIndices[channel][i]
                upSite_j = upIndices[channel][j]
                push!(hamiltonian, ("n+-",  [1, upSite_i, upSite_j], J/4)) # n_{d up, n_{0 up}
                push!(hamiltonian, ("n+-",  [1, upSite_i+1, upSite_j+1], -J/4)) # n_{d up, n_{0 down}
                push!(hamiltonian, ("n+-",  [2, upSite_i, upSite_j], -J/4)) # n_{d down, n_{0 up}
                push!(hamiltonian, ("n+-",  [2, upSite_i+1, upSite_j+1], J/4)) # n_{d down, n_{0 down}
                push!(hamiltonian, ("+-+-",  [1, 2, upSite_i+1, upSite_j], J/2)) # S_d^+ S_0^-
                push!(hamiltonian, ("+-+-",  [2, 1, upSite_i, upSite_j+1], J/2)) # S_d^- S_0^+
            end
        end
    end

    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in 1:(1 + numBathSites * length(kondoJ))
            push!(hamiltonian, ("n",  [2 * site - 1], globalField/2))
            push!(hamiltonian, ("n",  [2 * site], -globalField/2))
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end
export KondoModel


function TightBinding(
        numSites::Int64;
        joinEnds::Bool=true,
        spinLess::Bool=false,
    )
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]
    if spinLess
        for site in 1:numSites-1
            append!(hamiltonian, [("+-", [site, site+1], -1.0), ("+-", [site+1, site], -1.0)])
        end
        if joinEnds
            append!(hamiltonian, [("+-", [1, numSites], -1.0), ("+-", [numSites, 1], -1.0)])
        end
    else
        for site in 1:2:2*numSites-2
            append!(hamiltonian, [("+-", [site, site+2], -1.0), ("+-", [site+2, site], -1.0)])
            append!(hamiltonian, [("+-", [site + 1, site + 3], -1.0), ("+-", [site + 3, site + 1], -1.0)])
        end
        if joinEnds
            append!(hamiltonian, [("+-", [1, 2 * numSites - 1], -1.0), ("+-", [2 * numSites - 1, 1], -1.0)])
            append!(hamiltonian, [("+-", [2, 2 * numSites], -1.0), ("+-", [2 * numSites, 2], -1.0)])
        end
    end
end
export TightBinding

function KondoModel2D(
        kondoJ::Array{Float64, 2},
        sortedIndices::Vector{Int64},
        hop_t::Number;
        couplingTolerance::Float64=1e-15,
        globalField::Float64=0.,
    )
    dimension = Int(size(kondoJ)[1]^0.5)
    hamiltonian = Tuple{String, Vector{Int64}, Float64}[]

    # intra-bath hopping
    if abs(hop_t) > couplingTolerance
        for p1 in sortedIndices[2:end]
            for p2 in [p1 + 1, p1 + dimension]
                if p2 ∉ sortedIndices || p2 == sortedIndices[1]
                    continue
                end
                sites = findall(∈([p1, p2]), sortedIndices)
                println(sites)
                push!(hamiltonian, ("+-",  [2 * sites[1] - 1, 2 * sites[2] - 1], -hop_t)) # c^†_{j,up} c_{j+1,up}
                push!(hamiltonian, ("+-",  [2 * sites[2] - 1, 2 * sites[1] - 1], -hop_t)) # c^†_{j+1,up} c_{j,up}
                push!(hamiltonian, ("+-",  [2 * sites[1], 2 * sites[2]], -hop_t)) # c^†_{j,dn} c_{j+1,dn}
                push!(hamiltonian, ("+-",  [2 * sites[2], 2 * sites[1]], -hop_t)) # c^†_{j+1,dn} c_{j,dn}
            end
        end
    end

    # kondo terms
    for p1 in sortedIndices[2:end]
        for p2 in sortedIndices[2:end]
            coupling = kondoJ[p1, p2]
            if abs(coupling) > couplingTolerance
                sites = findall(∈([p1, p2]), sortedIndices)
                if p1 == p2
                    sites = repeat(sites, 2)
                end
                push!(hamiltonian, ("n+-",  [1, 2 * sites[1] - 1, 2 * sites[2] - 1], coupling/4)) # n_{d up, n_{0 up}
                push!(hamiltonian, ("n+-",  [1, 2 * sites[1], 2 * sites[2]], -coupling/4)) # n_{d up, n_{0 down}
                push!(hamiltonian, ("n+-",  [2, 2 * sites[1] - 1, 2 * sites[2] - 1], -coupling/4)) # n_{d down, n_{0 up}
                push!(hamiltonian, ("n+-",  [2, 2 * sites[1], 2 * sites[2]], coupling/4)) # n_{d down, n_{0 down}
                push!(hamiltonian, ("+-+-",  [1, 2, 2 * sites[1], 2 * sites[2] - 1], coupling/2)) # S_d^+ S_0^-
                push!(hamiltonian, ("+-+-",  [2, 1, 2 * sites[1] - 1, 2 * sites[2]], coupling/2)) # S_d^- S_0^+
            end
        end
    end

    # global magnetic field (to lift any trivial degeneracy)
    if abs(globalField) > couplingTolerance
        for site in eachindex(sortedIndices)
            push!(hamiltonian, ("n",  [2 * site - 1], globalField/2))
            push!(hamiltonian, ("n",  [2 * site], -globalField/2))
        end
    end

    @assert !isempty(hamiltonian) "Hamiltonian is empty!"

    return hamiltonian
end

export KondoModel2D
