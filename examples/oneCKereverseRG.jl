using Fermions

operator = [("+-+-", [1, 2, 4, 3], 1.0)]
operatorFlow = [operator]
global numEnt = 2
@time for step in 1:3
    unitary = unitaries1CK(1.0, numEnt, "ph")
    global numEnt += 2
    push!(operatorFlow, Product(Product(unitary, operatorFlow[end]), unitary))
    println(length(operatorFlow[end]))
end
