using Fermions, Plots

freqVals = collect(-50:0.01:50)
broadening_sb = 0.05
broadening_delta = 1e2
broadening_fl = 0.1
A_sidebands = exp.(-broadening_sb .* (freqVals .- 20).^2) + exp.(-broadening_sb .* (freqVals .+ 20).^2)
A_sidebands ./= sum(A_sidebands) * (maximum(freqVals) - minimum(freqVals)) / (length(freqVals) - 1)
A_fermLiq = exp.(-broadening_fl .* (freqVals .- 30).^2) + exp.(-broadening_fl .* (freqVals .- 1).^2) + exp.(-broadening_fl .* (freqVals).^2) + exp.(-broadening_fl .* (freqVals .+ 1).^2) + exp.(-broadening_fl .* (freqVals .+ 30).^2)
A_fermLiq ./= sum(A_fermLiq) * (maximum(freqVals) - minimum(freqVals)) / (length(freqVals) - 1)
A_fermigas = exp.(-0.01 .* (freqVals).^2)
A_fermigas ./= sum(A_fermigas) * (maximum(freqVals) - minimum(freqVals)) / (length(freqVals) - 1)
selfEnergy = SelfEnergy(A_fermigas, A_fermLiq, freqVals)
#=p = plot(freqVals, [A_fermLiq, A_fermigas])=#
p = plot(freqVals, [real(selfEnergy), imag(selfEnergy)], ylims=(-10, 10), xlims=(-30, 30), labels=["R" "I"])
display(p)
