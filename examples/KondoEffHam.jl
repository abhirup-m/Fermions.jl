using Fermions
include("../src/eigen.jl")

hop_t = 1.
basisStates = BasisStates(6)
H0 = [
      ("nn", [1, 3], 1/4),
      ("nn", [1, 4], -1/4),
      ("nn", [2, 3], -1/4),
      ("nn", [2, 4], 1/4),
      ("+-+-", [1, 2, 4, 3], 1/2),
      ("+-+-", [2, 1, 3, 4], 1/2),
     ]
perturbation = [
                ("+-", [3, 5], -hop_t),
                ("+-", [5, 3], -hop_t),
                ("+-", [4, 6], -hop_t),
                ("+-", [6, 4], -hop_t),
                ("+-", [7, 5], -hop_t),
                ("+-", [5, 7], -hop_t),
                ("+-", [8, 6], -hop_t),
                ("+-", [6, 8], -hop_t)
               ]
quantumNumbers = Dict(
                      "n1u" => [("n", [5], 1.0)], 
                      "n1d" => [("n", [6], 1.0)], 
                      "n2u" => [("n", [7], 1.0)], 
                      "n2d" => [("n", [8], 1.0)], 
                     )
s2, s4 = EffHamiltonian(H0, perturbation, quantumNumbers)
display(s2)
display(s4)
