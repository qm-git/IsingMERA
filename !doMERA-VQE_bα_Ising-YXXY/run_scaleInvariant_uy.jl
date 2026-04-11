include("./quan_fgθ-bα-VQE.jl")

# assign transverse field strength
Bz = -1.0

# assign MERA layer structure 
fold = [10]
nLayers=10

@assert sum(fold) == nLayers

e_gs = groundEnergyDensity_QIsing(Bz)

sample = 100
δe_initial_list = []
δe_final_list = []
θL_list = []
for s ∈ 1:sample
    θlength = vcat(ones(Int, length(fold)) * 4, ones(Int, length(fold)) * 2) 
    θL_full, θL, info, alloc, prealloc = Ising_bαMERA_setup(Bz, fold, nLayers, rand_amp=0.2, θlength=θlength)
    f,gL = fg_Ising(θL, info, alloc, prealloc)
    f_MERA, uwθ_final, f_array_final, g_array_final, α_array_final, fnoκ_array_final = L_BFGS_RR(fg_Ising!, θL, info, alloc,prealloc; f_target=e_gs ,eps=1e-9, iterMin = 10, iterMax = 1000, prnt=false)
    append!(δe_initial_list, f - e_gs)
    append!(δe_final_list, f_MERA - e_gs)
    push!(θL_list, uwθ_final)
    println(s,"th completed")
end

using Statistics

filename = "./results_Bz"*string(Bz)*"_L"*string(3*2^nLayers)*"_T"*string(nLayers)*"_fold"*string(fold[1])
for i ∈ 2:length(fold)
    global filename *= "-"*string(fold[i])
end
filename *= "_samples"*string(sample)*"_uyYXXY.txt"
open(filename, "w") do io
    println(io, "---------- Summary ----------\n",sample, " samples")
    println(io, "\nInitial δe:\n\tMean=", mean(δe_initial_list), "; Std=", std(δe_initial_list))
    fmin,idx = findmin(δe_final_list)
    println(io, "\nFinal δe:\n\tBest=", fmin, ";\n\tMean=", mean(δe_final_list), "; Std=", std(δe_final_list))
    sa = sum([sum(abs.(θL)) for θL ∈ θL_list[idx]])
    println(io, "\n\n---------- Best δe ----------\n\nGate angles (π unit):\n")
    l = 1
    for f ∈ 1:length(fold)
        if fold[f] == 1
            println(io, "layer "*string(f)*": ", θL_list[idx][f], " (unitary, yyYXXY type),  ", θL_list[idx][f+length(fold)], " (isometry, YXXY type)")
        elseif fold[f] >= 1
            println(io, "layer "*string(f)*"-"*string(f+fold[f]-1)*": ", θL_list[idx][f], " (unitary, yyYXXY type),  ", θL_list[idx][f+length(fold)], " (isometry, YXXY type)")
        end
        l += fold[f]
    end
    println(io, "\n\nsum|angles|=",sa)
    
    anglemin, idx = findmin([sum([sum(abs.(θL)) for θL ∈ θL_list[idx]]) for idx ∈ 1:sample])
    println(io, "\n\n------ Best ∑|angles| -------\n\nδe - δe_best=", δe_final_list[idx]-fmin, "; min ∑|angles| - (∑|angles|_bestEnergy)=", anglemin-sa)
    println(io, "\nGate angles (π unit):\n")
    l = 1
    for f ∈ 1:length(fold)
        if fold[f] == 1
            println(io, "layer "*string(f)*": ", θL_list[idx][f], " (unitary, yyYXXY type),  ", θL_list[idx][f+length(fold)], " (isometry, YXXY type)")
        elseif fold[f] >= 1
            println(io, "layer "*string(f)*"-"*string(f+fold[f]-1)*": ", θL_list[idx][f], " (unitary, yyYXXY type),  ", θL_list[idx][f+length(fold)], " (isometry, YXXY type)")
        end
        l += fold[f]
    end

    println(io, "\n\n------ Quantum circuit structure (time arrow starts from right to left) ------ ")
    println(io, "\n\tyyYXXY type:")
    println(io, 
    "\n—— Ry(θ1) —— · ————————————————— · ——       qubit 1")
    println(io, 
    "             |  YX(θ3)  XY(θ4)   |          ")
    println(io, 
    "—— Ry(θ2) —— · ————————————————— · ——       qubit 2")

    println(io, "\n\n\tYXXY type:")
    println(io, 
    "\n—— · ————————————————— · ——       qubit 1")
    println(io, 
    "   |  YX(θ1)  XY(θ2)   |          ")
    println(io, 
    "—— · ————————————————— · ——       qubit 2")
end