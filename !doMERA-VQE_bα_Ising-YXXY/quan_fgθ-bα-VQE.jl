#-------------------------------------------------------------------------
#
# homogeneous binary MERA
# Causal cone sampling for energy expectation and their partial derivative
#
#    |0>  |0>  |0> 
#     |    |    | 
#    |||||||||||||
#     |    |    |
#    -------------  <----- random path draw {1/2,1/2}
#     |    |    |
#    |||||||||||||
#     |    |    |
#    -------------  <----- random path draw {1/2,1/2}
#     |    |    |
#    |||||||||||||
#     |    |    |
#     |    |    |
#    |^^^^^^^^^^^|
#    |___________|  <----- measurement
#     
#-------------------------------------------------------------------------

# Data storaged
#  1,    2,    1
# uθ1,  uθ2,  uθ3
# 
# Layout, from l=1 to l=4
# uθ1,uθ2,uθ2,uθ3
# wθ1,wθ2,wθ2,wθ3
#
# u -> u1,u2
# w -> w1,w2,w3
include("./core_quantumKit.jl")
include("./core_contractionKit.jl")

function construct_fullΘL(θL::Vector{Vector{Float64}}, fold::Vector{Int64}, nLayers::Int64)
    @assert sum(fold) == nLayers && length(θL) == 2*length(fold)
    θL_full = Vector{Vector{Float64}}(undef, 5*nLayers)
    idx = 0
    for i ∈ eachindex(fold)
        for j ∈ 1:fold[i]
            θL_full[j+idx] = copy(θL[i]) # u1
            θL_full[j+idx+nLayers] = copy(θL[i]) # u2
            θL_full[j+idx+2*nLayers] = copy(θL[i+length(fold)]) # w1
            θL_full[j+idx+3*nLayers] = copy(θL[i+length(fold)]) # w2
            θL_full[j+idx+4*nLayers] = copy(θL[i+length(fold)]) # w3
        end
        idx += fold[i]
    end
    return θL_full
end
function construct_fullΘL!(θL_full::Vector{Vector{Float64}}, θL::Vector{Vector{Float64}}, fold::Vector{Int64}, nLayers::Int64)
    @assert sum(fold) == nLayers && length(θL) == 2*length(fold)
    idx = 0
    for i ∈ eachindex(fold)
        for j ∈ 1:fold[i]
            copyto!(θL_full[j+idx], θL[i]) # u1
            copyto!(θL_full[j+idx+nLayers], θL[i]) # u2
            copyto!(θL_full[j+idx+2*nLayers], θL[i+length(fold)]) # w1
            copyto!(θL_full[j+idx+3*nLayers], θL[i+length(fold)]) # w2
            copyto!(θL_full[j+idx+4*nLayers], θL[i+length(fold)]) # w3
        end
        idx += fold[i]
    end
    return nothing
end
function construct_fullΘL_type2(θL::Vector{Vector{Float64}}, fold::Vector{Int64}, nLayers::Int64)
    @assert sum(fold) == nLayers && length(θL) == 2*length(fold)+1
    θL_full = Vector{Vector{Float64}}(undef, 5*nLayers)
    idx = 0
    for i ∈ 1:length(fold)
        for j ∈ 1:fold[i]
            θL_full[j+idx] = copy(θL[i]) # u1
            θL_full[j+idx+nLayers] = copy(θL[i]) # u2
            θL_full[j+idx+2*nLayers] = copy(θL[i+length(fold)]) # w1
            θL_full[j+idx+3*nLayers] = copy(θL[i+length(fold)]) # w2
            if i == length(fold) && j == fold[i]
                θL_full[j+idx+4*nLayers] = copy(θL[i+length(fold)+1]) # w_top
            else
                θL_full[j+idx+4*nLayers] = copy(θL[i+length(fold)]) # w3
            end
        end
        idx += fold[i]
    end
    return θL_full
end
function construct_fullΘL_type2!(θL_full::Vector{Vector{Float64}}, θL::Vector{Vector{Float64}}, fold::Vector{Int64}, nLayers::Int64)
    @assert sum(fold) == nLayers && length(θL) == 2*length(fold)+1
    idx = 0
    for i ∈ 1:length(fold)
        for j ∈ 1:fold[i]
            copyto!(θL_full[j+idx], θL[i]) # u1
            copyto!(θL_full[j+idx+nLayers], θL[i]) # u2
            copyto!(θL_full[j+idx+2*nLayers], θL[i+length(fold)]) # w1
            copyto!(θL_full[j+idx+3*nLayers], θL[i+length(fold)]) # w2
            if i == length(fold) && j == fold[i]
                copyto!(θL_full[j+idx+4*nLayers], θL[i+length(fold)+1]) # w_top
            else
                copyto!(θL_full[j+idx+4*nLayers], θL[i+length(fold)]) # w3
            end
        end
        idx += fold[i]
    end
    return nothing
end

function constructTensors(θL_full::Vector{Vector{Float64}}, nLayers::Int64)
    @assert length(θL_full) == 5*nLayers
    ξL = get_ξL(θL_full)

    u1 = Vector{Array{ComplexF64,4}}(undef, nLayers) 
    u2 = Vector{Array{ComplexF64,4}}(undef, nLayers) 

    w1 = Vector{Array{ComplexF64,3}}(undef, nLayers) 
    w2 = Vector{Array{ComplexF64,3}}(undef, nLayers) 
    w3 = Vector{Array{ComplexF64,3}}(undef, nLayers) 

    for l ∈ 1:nLayers
        u1[l] = quanReshape(ξL[l],(2,2, 2,2))
        u2[l] = quanReshape(ξL[l+nLayers],(2,2, 2,2))
        ## isometry w
        #    3  |0⟩
        #    |   | 
        #   |||||||
        #    |   |
        #    1   2
        if l < nLayers
            w1[l] = quanReshape(ξL[l+2*nLayers][:,[1,3]], (2,2, 2))
            w2[l] = quanReshape(ξL[l+3*nLayers][:,[1,3]], (2,2, 2))
            w3[l] = quanReshape(ξL[l+4*nLayers][:,[1,3]], (2,2, 2))
        else
            w1[l] = quanReshape(ξL[l+2*nLayers][:,[1]], (2,2, 1))
            w2[l] = quanReshape(ξL[l+3*nLayers][:,[1]], (2,2, 1))
            w3[l] = quanReshape(ξL[l+4*nLayers][:,[1]], (2,2, 1))
        end
    end
    return u1,u2,w1,w2,w3
end
function constructTensors!(u1::Vector{Array{ComplexF64,4}},u2::Vector{Array{ComplexF64,4}},w1::Vector{Array{ComplexF64,3}},w2::Vector{Array{ComplexF64,3}},w3::Vector{Array{ComplexF64,3}}, 
        ξL::AbstractVector{Matrix{ComplexF64}}, θL_full::Vector{Vector{Float64}}, nLayers::Int64)
    @assert length(θL_full) == 5*nLayers
    get_ξL!(ξL, θL_full)

    for l ∈ 1:nLayers
        quanReshapeMtoH!(u1[l], ξL[l],(2,2, 2,2))
        quanReshapeMtoH!(u2[l], ξL[l+nLayers],(2,2, 2,2))
        ## isometry w
        #    3  |0⟩
        #    |   | 
        #   |||||||
        #    |   |
        #    1   2
        if l < nLayers
            quanReshapeMtoΛ!(w1[l], view(ξL[l+2*nLayers], :,[1,3]), (2,2, 2))
            quanReshapeMtoΛ!(w2[l], view(ξL[l+3*nLayers], :,[1,3]), (2,2, 2))
            quanReshapeMtoΛ!(w3[l], view(ξL[l+4*nLayers], :,[1,3]), (2,2, 2))
        else
            quanReshapeMtoΛ!(w1[l], view(ξL[l+2*nLayers], :,[1]), (2,2, 1))
            quanReshapeMtoΛ!(w2[l], view(ξL[l+3*nLayers], :,[1]), (2,2, 1))
            quanReshapeMtoΛ!(w3[l], view(ξL[l+4*nLayers], :,[1]), (2,2, 1))
        end
    end
    return nothing
end
function constructTensors_type2(θL_full::Vector{Vector{Float64}}, nLayers::Int64)
    @assert length(θL_full) == 5*nLayers
    ξL = get_ξL(θL_full)

    u1 = Vector{Array{ComplexF64,4}}(undef, nLayers) 
    u2 = Vector{Array{ComplexF64,4}}(undef, nLayers) 

    w1 = Vector{Array{ComplexF64,3}}(undef, nLayers) 
    w2 = Vector{Array{ComplexF64,3}}(undef, nLayers) 
    w3 = Vector{Array{ComplexF64,3}}(undef, nLayers) 

    for l ∈ 1:nLayers
        u1[l] = quanReshape(ξL[l],(2,2, 2,2))
        u2[l] = quanReshape(ξL[l+nLayers],(2,2, 2,2))
        ## isometry w
        #    3  |0⟩
        #    |   | 
        #   |||||||
        #    |   |
        #    1   2
        w1[l] = quanReshape(ξL[l+2*nLayers][:,[1,3]], (2,2, 2))
        w2[l] = quanReshape(ξL[l+3*nLayers][:,[1,3]], (2,2, 2))
        if l < nLayers
            w3[l] = quanReshape(ξL[l+4*nLayers][:,[1,3]], (2,2, 2))
        else
            w3[l] = quanReshape(ξL[l+4*nLayers][:,[1]], (2,2, 1)) # w_top
        end
    end
    return u1,u2,w1,w2,w3
end
function constructTensors_type2!(u1::Vector{Array{ComplexF64,4}},u2::Vector{Array{ComplexF64,4}},w1::Vector{Array{ComplexF64,3}},w2::Vector{Array{ComplexF64,3}},w3::Vector{Array{ComplexF64,3}}, 
        ξL::AbstractVector{Matrix{ComplexF64}}, θL_full::Vector{Vector{Float64}}, nLayers::Int64)

    @assert length(θL_full) == 5*nLayers
    get_ξL!(ξL, θL_full)

    for l ∈ 1:nLayers
        quanReshapeMtoH!(u1[l], ξL[l],(2,2, 2,2))
        quanReshapeMtoH!(u2[l], ξL[l+nLayers],(2,2, 2,2))
        ## isometry w
        #    3  |0⟩
        #    |   | 
        #   |||||||
        #    |   |
        #    1   2
        quanReshapeMtoΛ!(w1[l], view(ξL[l+2*nLayers], :,[1,3]), (2,2, 2))
        quanReshapeMtoΛ!(w2[l], view(ξL[l+3*nLayers], :,[1,3]), (2,2, 2))
        if l < nLayers
            quanReshapeMtoΛ!(w3[l], view(ξL[l+4*nLayers], :,[1,3]), (2,2, 2))
        else
            quanReshapeMtoΛ!(w3[l], view(ξL[l+4*nLayers], :,[1]), (2,2, 1)) # w_top
        end
    end
    return nothing
end

function Ising_bαMERA_setup(Bz::Real, fold::Vector{Int}, nLayers::Int; J::Real = -1.0, θL::Union{Vector{Vector{Float64}}, Nothing}=nothing, rand_generator::Char='G', rand_amp::Float64=0.1, θlength::Vector{Int}=ones(Int,2*length(fold))*2)
    @assert nLayers ≥ 1

    if θL === nothing
        if rand_generator=='G'
            θL = [randn(L)*rand_amp for L ∈ θlength]
        elseif rand_generator=='U'
            θL = [(rand(L).-0.5)*rand_amp for L ∈ θlength]
        end
    end

    θL_full = construct_fullΘL(θL, fold, nLayers)
    ξL = get_ξL(θL_full)
    u1,u2,w1,w2,w3 = constructTensors(θL_full, nLayers)

    info = (fold, nLayers, Bz,J)
    alloc = (u1,u2, w1,w2,w3, ξL, [length(i) for i ∈ θL])
    prealloc = preAllocation(u1,u2,w1,w2,w3, nLayers)

    return θL_full, θL, info, alloc, prealloc
end
function Ising_bαMERA_type2_setup(Bz::Real, fold::Vector{Int}, nLayers::Int; J::Real = -1.0, θL::Union{Vector{Vector{Float64}}, Nothing}=nothing, rand_generator::Char='G', rand_amp::Float64=0.1, θlength::Vector{Int}=vcat(ones(Int,2*length(fold))*2,[1]))
    @assert nLayers ≥ 2

    if θL === nothing
        if rand_generator=='G'
            θL = [randn(L)*rand_amp for L ∈ θlength]
        elseif rand_generator=='U'
            θL = [(rand(L).-0.5)*rand_amp for L ∈ θlength]
        end
    end

    θL_full = construct_fullΘL_type2(θL, fold, nLayers)
    ξL = get_ξL(θL_full)
    u1,u2,w1,w2,w3 = constructTensors_type2(θL_full, nLayers)

    info = (fold, nLayers, Bz,J)
    alloc = (u1,u2, w1,w2,w3, ξL, [length(i) for i ∈ θL])
    prealloc = preAllocation_type2(u1,u2,w1,w2,w3, nLayers)

    return θL_full, θL, info, alloc, prealloc
end

function f_Ising(θL_full::Vector{Vector{Float64}}, info::Tuple, alloc::Tuple, prealloc::Tuple)::Float64
    _, nLayers, Bz,J = info 

    u1,u2, w1,w2,w3, ξL, _ = alloc
    constructTensors!(u1,u2,w1,w2,w3, ξL, θL_full, nLayers)
    return IsingEnergy(u1, u2, w1, w2, w3, nLayers, prealloc, Bz, J=J)
end

function g_Ising(θL_full::Vector{Vector{Float64}}, info::Tuple, alloc::Tuple, prealloc::Tuple)::Vector{Vector{Float64}}
    fold, nLayers, _,_ = info 
    _,_, _,_,_, _, θLength = alloc

    gL = [zeros(i) for i ∈ θLength]
    idx = 0
    for i ∈ 1:length(fold)
        for j ∈ 1:fold[i]
            #-----------------------------
            # gL[i] .+= gL_full[j+idx]  # u1
            ig = j+idx
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i][jg] += (f_p - f_m) * 0.5
            end
            
            #-----------------------------
            # gL[i] .+= gL_full[j+idx+nLayers]  # u2
            ig = j+idx+nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i][jg] += (f_p - f_m) * 0.5
            end

            #-----------------------------
            # gL[i+length(fold)] .+= gL_full[j+idx+2*nLayers] # w1
            ig = j+idx+2*nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
            end

            #-----------------------------
            # gL[i+length(fold)] .+= gL_full[j+idx+3*nLayers] # w2
            ig = j+idx+3*nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
            end

            #-----------------------------
            # gL[i+length(fold)] .+= gL_full[j+idx+4*nLayers] # w3
            ig = j+idx+4*nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
            end
        end
        idx += fold[i]
    end

    return gL
end


function fg_Ising(θL::Vector{Vector{Float64}}, info::Tuple, alloc::Tuple, prealloc::Tuple)
    fold, nLayers, _,_ = info 
    θL_full = construct_fullΘL(θL, fold, nLayers)
    return f_Ising(θL_full, info, alloc, prealloc), g_Ising(θL_full, info, alloc, prealloc)
end

function fg_Ising!(θL::Vector{Vector{Float64}},gL::Vector{Vector{Float64}}, info::Tuple, alloc::Tuple, prealloc::Tuple)
    fold, nLayers, _,_ = info 
    # construct_fullΘL!(θL_full, θL, fold, nLayers)
    θL_full = construct_fullΘL(θL, fold, nLayers)
    f = f_Ising(θL_full, info, alloc, prealloc)

    for g ∈ gL 
        fill!(g, 0.0)
    end
    idx = 0
    for i ∈ 1:length(fold)
        for j ∈ 1:fold[i]
            #-----------------------------
            # gL[i] .+= gL_full[j+idx]  # u1
            ig = j+idx
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i][jg] += (f_p - f_m) * 0.5
            end
            
            #-----------------------------
            # gL[i] .+= gL_full[j+idx+nLayers]  # u2
            ig = j+idx+nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i][jg] += (f_p - f_m) * 0.5
            end

            #-----------------------------
            # gL[i+length(fold)] .+= gL_full[j+idx+2*nLayers] # w1
            ig = j+idx+2*nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
            end

            #-----------------------------
            # gL[i+length(fold)] .+= gL_full[j+idx+3*nLayers] # w2
            ig = j+idx+3*nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
            end

            #-----------------------------
            # gL[i+length(fold)] .+= gL_full[j+idx+4*nLayers] # w3
            ig = j+idx+4*nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
            end
        end
        idx += fold[i]
    end

    return f, f
end

function f_Ising_type2(θL_full::Vector{Vector{Float64}}, info::Tuple, alloc::Tuple, prealloc::Tuple)::Float64
    _, nLayers, Bz,J = info 

    u1,u2, w1,w2,w3, ξL, _ = alloc
    constructTensors_type2!(u1,u2,w1,w2,w3, ξL, θL_full, nLayers)
    return IsingEnergy_type2(u1, u2, w1, w2, w3, nLayers, prealloc, Bz, J=J)
end

function g_Ising_type2(θL_full::Vector{Vector{Float64}}, info::Tuple, alloc::Tuple, prealloc::Tuple)::Vector{Vector{Float64}}
    fold, nLayers, _,_ = info 
    _,_, _,_,_, _, θLength = alloc

    gL = [zeros(i) for i ∈ θLength]
    idx = 0
    for i ∈ 1:length(fold)
        for j ∈ 1:fold[i]
            #-----------------------------
            # gL[i] .+= gL_full[j+idx]  # u1
            ig = j+idx
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i][jg] += (f_p - f_m) * 0.5
            end
            
            #-----------------------------
            # gL[i] .+= gL_full[j+idx+nLayers]  # u2
            ig = j+idx+nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i][jg] += (f_p - f_m) * 0.5
            end

            #-----------------------------
            # gL[i+length(fold)] .+= gL_full[j+idx+2*nLayers] # w1
            ig = j+idx+2*nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
            end

            #-----------------------------
            # gL[i+length(fold)] .+= gL_full[j+idx+3*nLayers] # w2
            ig = j+idx+3*nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
            end

            if i == length(fold) && j == fold[i]
                #-----------------------------
                # gL[i+length(fold)+1] = gL_full[j+idx+4*nLayers] # w_top
                ig = j+idx+4*nLayers
                for jg ∈ 1:length(θL_full[ig])
                    θL_full[ig][jg] += 0.5
                    f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                    θL_full[ig][jg] -= 1.0
                    f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                    θL_full[ig][jg] += 0.5

                    gL[i+length(fold)+1][jg] += (f_p - f_m) * 0.5
                end
            else
                #-----------------------------
                # gL[i+length(fold)] .+= gL_full[j+idx+4*nLayers] # w3
                ig = j+idx+4*nLayers
                for jg ∈ 1:length(θL_full[ig])
                    θL_full[ig][jg] += 0.5
                    f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                    θL_full[ig][jg] -= 1.0
                    f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                    θL_full[ig][jg] += 0.5

                    gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
                end
            end
        end
        idx += fold[i]
    end

    return gL
end

function fg_Ising_type2(θL::Vector{Vector{Float64}}, info::Tuple, alloc::Tuple, prealloc::Tuple)
    fold, nLayers, _,_ = info 
    θL_full = construct_fullΘL_type2(θL, fold, nLayers)
    return f_Ising_type2(θL_full, info, alloc, prealloc), g_Ising_type2(θL_full, info, alloc, prealloc)
end

function fg_Ising_type2!(θL::Vector{Vector{Float64}},gL::Vector{Vector{Float64}}, info::Tuple, alloc::Tuple, prealloc::Tuple)
    fold, nLayers, _,_ = info 
    # construct_fullΘL_type2!(θL_full, θL, fold, nLayers)
    θL_full = construct_fullΘL_type2(θL, fold, nLayers)
    f = f_Ising_type2(θL_full, info, alloc, prealloc)

    for g ∈ gL 
        fill!(g, 0.0)
    end
    idx = 0
    for i ∈ 1:length(fold)
        for j ∈ 1:fold[i]
            #-----------------------------
            # gL[i] .+= gL_full[j+idx]  # u1
            ig = j+idx
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i][jg] += (f_p - f_m) * 0.5
            end
            
            #-----------------------------
            # gL[i] .+= gL_full[j+idx+nLayers]  # u2
            ig = j+idx+nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i][jg] += (f_p - f_m) * 0.5
            end

            #-----------------------------
            # gL[i+length(fold)] .+= gL_full[j+idx+2*nLayers] # w1
            ig = j+idx+2*nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
            end

            #-----------------------------
            # gL[i+length(fold)] .+= gL_full[j+idx+3*nLayers] # w2
            ig = j+idx+3*nLayers
            for jg ∈ 1:length(θL_full[ig])
                θL_full[ig][jg] += 0.5
                f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] -= 1.0
                f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                θL_full[ig][jg] += 0.5

                gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
            end

            if i == length(fold) && j == fold[i]
                #-----------------------------
                # gL[i+length(fold)+1] = gL_full[j+idx+4*nLayers] # w_top
                ig = j+idx+4*nLayers
                for jg ∈ 1:length(θL_full[ig])
                    θL_full[ig][jg] += 0.5
                    f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                    θL_full[ig][jg] -= 1.0
                    f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                    θL_full[ig][jg] += 0.5

                    gL[i+length(fold)+1][jg] += (f_p - f_m) * 0.5
                end
            else
                #-----------------------------
                # gL[i+length(fold)] .+= gL_full[j+idx+4*nLayers] # w3
                ig = j+idx+4*nLayers
                for jg ∈ 1:length(θL_full[ig])
                    θL_full[ig][jg] += 0.5
                    f_p = f_Ising_type2(θL_full, info, alloc, prealloc)
                    θL_full[ig][jg] -= 1.0
                    f_m = f_Ising_type2(θL_full, info, alloc, prealloc)
                    θL_full[ig][jg] += 0.5

                    gL[i+length(fold)][jg] += (f_p - f_m) * 0.5
                end
            end
        end
        idx += fold[i]
    end

    return f, f
end


include("./core_optimKitR.jl")
include("./quanIsing.jl")
