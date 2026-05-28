using Fermions
include("../src/unitaryRG.jl")

function hamJ(sites, spins)
   factor = 1
   s1,s2 = sites
   if spins[s1] == 1 || spins[s2] == -1
       factor = 0
   end
   return (['+', '-', '+', '-'], [1, 2, sites...], factor)
end

function hamW(sites, spins)
   factor = 1
   s1,s2,s3,s4 = sites
   if spins[s1] ≠ spins[s2] || spins[s3] ≠ spins[s4]
       factor = 0
   end
   if spins[s1] ≠ spins[s3]
       factor = -1
   end
   return (['+', '-', '+', '-'], sites, factor)
end

occupied = [3, 4]
vacant = [5, 6]
inside = [7, 8, 9, 10]
allSites = vcat(occupied, vacant, inside)
spins = Dict(i => isodd(i) ? 1 : -1 for i in allSites)
r = Renormalisation(Dict("J" => hamJ, "W" => hamW), Dict("J" => 2, "W" => 4), occupied, vacant, allSites, spins; allowed=[("W", "J")])
println("----")
for ri in r
    println(ri)
end

#=("+-+-", [1, 2, 8, 10]) => 4.0=#
#=("+--+", [1, 2, 7, 9]) => 4.0=#
#=("+-n", [1, 2, 7]) => -4.0=#
#=("+--+", [1, 2, 8, 10]) => -4.0=#
#=("+-n", [1, 2, 9]) => -4.0=#
#=("+-", [1, 2]) => -12.0=#
#=("+-+-", [1, 2, 7, 9]) => -4.0=#
#=("+-n", [1, 2, 8]) => 4.0=#
#=("+-n", [1, 2, 10]) => 4.0=#
#
#=("+-+-", [1, 2, 8, 10]) => -4.0=#
#=("+--+", [1, 2, 7, 9]) => -4.0=#
#=("+-n", [1, 2, 7]) => 4.0=#
#=("+--+", [1, 2, 8, 10]) => 4.0=#
#=("+-n", [1, 2, 9]) => 4.0=#
#=("+-", [1, 2]) => -12.0=#
#=("+-+-", [1, 2, 7, 9]) => 4.0=#
#=("+-n", [1, 2, 8]) => -4.0=#
#=("+-n", [1, 2, 10]) => -4.0=#
