######## BASIC EXAMPLE of USING FERMIONS.JL ########
# Constructs a 5-site toy model of the 1D Hubbard model.
# Diagonalises it and calculates certain VEVs
####################################################

using Fermions

hop_t = 1.
hubbard_U = 2.
num_sites = 5

kinetic_energy_OBC_up = vcat([("+-", [i, i+2], -hop_t) for i in 1:2:2*(num_sites-1)], [("+-", [i+2, i], -hop_t) for i in 1:2:2*(num_sites-1)])
kinetic_energy_OBC_down = vcat([("+-", [i, i+2], -hop_t) for i in 2:2:2*(num_sites-1)], [("+-", [i+2, i], -hop_t) for i in 2:2:2*(num_sites-1)])
hubbard_term = [("nn", [i, i+1], hubbard_U) for i in 1:2:2*num_sites]
hamiltonian = vcat(kinetic_energy_OBC_up, kinetic_energy_OBC_down, hubbard_term)
basis = BasisStates(2 * num_sites)
E, V = Spectrum(hamiltonian, basis)
display(E)
display(V)
