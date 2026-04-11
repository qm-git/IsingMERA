#-------------------------------------------------------------------------
#
# homogeneous binary MERA
# Causal cone sampling for energy expectation and their partial derivative
#
#
#-------------------------------------------------------------------------

using LinearAlgebra
using TensorOperations

#-------------------------------------------------------------------------
function setTopRhoTwo(w_top::Array{ComplexF64,3})::Array{ComplexF64,4}
    @tensor TopRhoTwo[-1,-2,-3,-4] := (conj(w_top[-1,-2,1]) * w_top[-3,-4,1] + conj(w_top[-2,-1,1]) * w_top[-4,-3,1]) / 2
    return TopRhoTwo
end
function setTopRhoTwo!(TopRhoTwo::Array{ComplexF64,4}, w_top::Array{ComplexF64,3})
    @tensor TopRhoTwo[-1,-2,-3,-4] = (conj(w_top[-1,-2,1]) * w_top[-3,-4,1] + conj(w_top[-2,-1,1]) * w_top[-4,-3,1]) / 2
    return nothing
end

function setTopRhoThree()::Array{ComplexF64,6}
    return copy(reshape([1.0+0.0im],(1,1,1,1,1,1)))
end
function setTopRhoThree!(TopRhoThree::Array{ComplexF64,6})
    copyto!(TopRhoThree,reshape([1.0+0.0im],(1,1,1,1,1,1)))
    return nothing
end

#-------------------------------------------------------------------------
# Transition Top Layer (T), Causal Cone: 3 sites (Bottom) <-> 2 sites (Top)
function RhoDescend(TopRhoTwo::Array{ComplexF64,4}, w0::Array{ComplexF64,3}, w1::Array{ComplexF64,3}, u0::Array{ComplexF64,4}, u1::Array{ComplexF64,4})::Array{ComplexF64,6}
    @tensor rhoThreeOut[-1,-2,-3,-4,-5,-6] := 0.5 * (TopRhoTwo[7,8,1,2] * w0[6,3,1] * w1[4,5,2] * u0[-4,-5,3,4] * u1[-6,11,5,6] * conj(w0[10,12,7]) * conj(w1[13,9,8]) * conj(u0[-1,-2,12,13]) * conj(u1[-3,11,9,10]) + # L.
                                                TopRhoTwo[7,8,1,2] * w0[6,3,1] * w1[4,5,2] * u0[11,-4,3,4] * u1[-5,-6,5,6] * conj(w0[13,9,7]) * conj(w1[10,12,8]) * conj(u0[11,-1,9,10]) * conj(u1[-2,-3,12,13]) # R
                                                ) 
    return rhoThreeOut
end
function RhoDescend!(rhoThreeOut::Array{ComplexF64,6}, TopRhoTwo::Array{ComplexF64,4}, w0::Array{ComplexF64,3}, w1::Array{ComplexF64,3}, u0::Array{ComplexF64,4}, u1::Array{ComplexF64,4})
    @tensor rhoThreeOut[-1,-2,-3,-4,-5,-6] = 0.5 * (TopRhoTwo[7,8,1,2] * w0[6,3,1] * w1[4,5,2] * u0[-4,-5,3,4] * u1[-6,11,5,6] * conj(w0[10,12,7]) * conj(w1[13,9,8]) * conj(u0[-1,-2,12,13]) * conj(u1[-3,11,9,10]) + # L.
                                                TopRhoTwo[7,8,1,2] * w0[6,3,1] * w1[4,5,2] * u0[11,-4,3,4] * u1[-5,-6,5,6] * conj(w0[13,9,7]) * conj(w1[10,12,8]) * conj(u0[11,-1,9,10]) * conj(u1[-2,-3,12,13]) # R
                                                ) 
    return nothing
end

#-------------------------------------------------------------------------
"Time Cost: (2χ^8 + 2χ^5) + χ^8; Space Cost: χ^6 + χ^7" 
function preContract1st(rhoThree::Array{ComplexF64,6}, w1::Array{ComplexF64,3},w2::Array{ComplexF64,3},w3::Array{ComplexF64,3})
    @tensor r2w2conjw[-1,-2,-3,-4,-5,-6] := rhoThree[6,-2,4,5,-5,3] * w1[1,-4,5] * w3[-6,2,3] * conj(w1[1,-1,6]) * conj(w3[-3,2,4])
    @tensor r3w2conjw[-1,-2,-3,-4,-5,-6,-7] := r2w2conjw[-1,-2,-3,-4,1,-7] * w2[-5,-6,1]

    return r2w2conjw::Array{ComplexF64,6}, r3w2conjw::Array{ComplexF64,7}
end
function preContract1st!(rhoThree::Array{ComplexF64,6}, w1::Array{ComplexF64,3},w2::Array{ComplexF64,3},w3::Array{ComplexF64,3}, r2w2conjw::Array{ComplexF64,6}, r3w2conjw::Array{ComplexF64,7})
    @tensor r2w2conjw[-1,-2,-3,-4,-5,-6] = rhoThree[6,-2,4,5,-5,3] * w1[1,-4,5] * w3[-6,2,3] * conj(w1[1,-1,6]) * conj(w3[-3,2,4])
    @tensor r3w2conjw[-1,-2,-3,-4,-5,-6,-7] = r2w2conjw[-1,-2,-3,-4,1,-7] * w2[-5,-6,1]

    return nothing
end
"Time Cost: 2χ^9 + 2χ^9; Space Cost: χ^6 + χ^6" 
function preContract2nd(r3w2conjw::Array{ComplexF64,7}, w2::Array{ComplexF64,3}, u1::Array{ComplexF64,4},u2::Array{ComplexF64,4})
    @tensor r3w1u1conju3conjw_L[-1,-2,-3,-4,-5,-6] := r3w2conjw[-1,5,4,-4,-5,3,2] * u2[-6,6,3,2] * conj(u2[-3,6,1,4]) * conj(w2[-2,1,5])
    @tensor r3w1u1conju3conjw_R[-1,-2,-3,-4,-5,-6] := r3w2conjw[4,5,-3,2,3,-5,-6] * u1[6,-4,2,3] * conj(u1[6,-1,4,1]) * conj(w2[1,-2,5])
    
    return r3w1u1conju3conjw_L::Array{ComplexF64,6}, r3w1u1conju3conjw_R::Array{ComplexF64,6}
end 
function preContract2nd!(r3w2conjw::Array{ComplexF64,7}, w2::Array{ComplexF64,3}, u1::Array{ComplexF64,4},u2::Array{ComplexF64,4}, r3w1u1conju3conjw_L::Array{ComplexF64,6}, r3w1u1conju3conjw_R::Array{ComplexF64,6})
    @tensor r3w1u1conju3conjw_L[-1,-2,-3,-4,-5,-6] = r3w2conjw[-1,5,4,-4,-5,3,2] * u2[-6,6,3,2] * conj(u2[-3,6,1,4]) * conj(w2[-2,1,5])
    @tensor r3w1u1conju3conjw_R[-1,-2,-3,-4,-5,-6] = r3w2conjw[4,5,-3,2,3,-5,-6] * u1[6,-4,2,3] * conj(u1[6,-1,4,1]) * conj(w2[1,-2,5])
    
    return nothing
end 

"Time Cost: 2χ^8 + 2χ^8; Space Cost: χ^6"
function RhoDescend_withPre(r3w1u1conju3conjw_L::Array{ComplexF64,6}, r3w1u1conju3conjw_R::Array{ComplexF64,6}, u1::Array{ComplexF64,4},u2::Array{ComplexF64,4})
    @tensor rhoThreeOut[-1,-2,-3,-4,-5,-6] := 
        (r3w1u1conju3conjw_L[1,2,-3,3,4,-6] * u1[-4,-5,3,4] * conj(u1[-1,-2,1,2]) + # L
        r3w1u1conju3conjw_R[-1,1,2,-4,3,4] * u2[-5,-6,3,4] * conj(u2[-2,-3,1,2]))/2 # R

    return rhoThreeOut::Array{ComplexF64,6}
end
function RhoDescend_withPre!(r3w1u1conju3conjw_L::Array{ComplexF64,6}, r3w1u1conju3conjw_R::Array{ComplexF64,6}, u1::Array{ComplexF64,4},u2::Array{ComplexF64,4}, rhoThreeOut::Array{ComplexF64,6})
    @tensor rhoThreeOut[-1,-2,-3,-4,-5,-6] = 
        (r3w1u1conju3conjw_L[1,2,-3,3,4,-6] * u1[-4,-5,3,4] * conj(u1[-1,-2,1,2]) + # L
        r3w1u1conju3conjw_R[-1,1,2,-4,3,4] * u2[-5,-6,3,4] * conj(u2[-2,-3,1,2]))/2 # R

    return nothing
end

#-------------------------------------------------------------------------
function preAllocation(u1::Vector{Array{ComplexF64,4}}, u2::Vector{Array{ComplexF64,4}}, w1::Vector{Array{ComplexF64,3}}, w2::Vector{Array{ComplexF64,3}}, w3::Vector{Array{ComplexF64,3}}, nLayers::Int)

    rhoThree = Vector{Array{ComplexF64,6}}(undef, nLayers) 
    rhoThree[nLayers] = setTopRhoThree()

    r2w2conjw = Vector{Array{ComplexF64,6}}(undef,nLayers)
    r3w2conjw = Vector{Array{ComplexF64,7}}(undef,nLayers)
    r3w1u1conju3conjw_L = Vector{Array{ComplexF64,6}}(undef,nLayers)
    r3w1u1conju3conjw_R = Vector{Array{ComplexF64,6}}(undef,nLayers)

    for level ∈ nLayers:-1:1
        r2w2conjw[level], r3w2conjw[level] = preContract1st(rhoThree[level], w1[level], w2[level], w3[level])
        r3w1u1conju3conjw_L[level], r3w1u1conju3conjw_R[level] = preContract2nd(r3w2conjw[level], w2[level], u1[level], u2[level])
        if level > 1
            rhoThree[level-1] = RhoDescend_withPre(r3w1u1conju3conjw_L[level], r3w1u1conju3conjw_R[level], u1[level], u2[level])
        end
    end

    prealloc = (rhoThree, r2w2conjw, r3w2conjw, r3w1u1conju3conjw_L, r3w1u1conju3conjw_R)    

    return prealloc
end
#-------------------------------------------------------------------------
function preAllocation_type2(u1::Vector{Array{ComplexF64,4}}, u2::Vector{Array{ComplexF64,4}}, w1::Vector{Array{ComplexF64,3}}, w2::Vector{Array{ComplexF64,3}}, w3::Vector{Array{ComplexF64,3}}, nLayers::Int)
    @assert nLayers == length(u1) == length(u2) == length(w1) == length(w2) == length(w3)
    rhoThree = Vector{Array{ComplexF64,6}}(undef, nLayers-1) 
    # Transition layer (nLayers-th)
    rhoTop = setTopRhoTwo(w3[nLayers])
    # Normal layer (nLayers-1 th)
    rhoThree[nLayers-1] = RhoDescend(rhoTop, w1[nLayers], w2[nLayers], u1[nLayers], u2[nLayers])

    r2w2conjw = Vector{Array{ComplexF64,6}}(undef,nLayers-1)
    r3w2conjw = Vector{Array{ComplexF64,7}}(undef,nLayers-1)
    r3w1u1conju3conjw_L = Vector{Array{ComplexF64,6}}(undef,nLayers-1)
    r3w1u1conju3conjw_R = Vector{Array{ComplexF64,6}}(undef,nLayers-1)

    for level ∈ nLayers-1:-1:1
        r2w2conjw[level], r3w2conjw[level] = preContract1st(rhoThree[level], w1[level], w2[level], w3[level])
        r3w1u1conju3conjw_L[level], r3w1u1conju3conjw_R[level] = preContract2nd(r3w2conjw[level], w2[level], u1[level], u2[level])
        if level > 1
            rhoThree[level-1] = RhoDescend_withPre(r3w1u1conju3conjw_L[level], r3w1u1conju3conjw_R[level], u1[level], u2[level])
        end
    end

    prealloc = (rhoThree, r2w2conjw, r3w2conjw, r3w1u1conju3conjw_L, r3w1u1conju3conjw_R)    

    return prealloc
end

#-------------------------------------------------------------------------
function physicalRho_fourSites_averageChannel(u1::Vector{Array{ComplexF64,4}}, u2::Vector{Array{ComplexF64,4}}, w1::Vector{Array{ComplexF64,3}}, w2::Vector{Array{ComplexF64,3}}, w3::Vector{Array{ComplexF64,3}}, nLayers::Int, prealloc::Tuple)
    rhoThree, r2w2conjw, r3w2conjw, r3w1u1conju3conjw_L, r3w1u1conju3conjw_R = prealloc
    setTopRhoThree!(rhoThree[nLayers])
    for level ∈ nLayers:-1:1
        preContract1st!(rhoThree[level], w1[level], w2[level], w3[level], r2w2conjw[level], r3w2conjw[level])
        preContract2nd!(r3w2conjw[level], w2[level], u1[level], u2[level], r3w1u1conju3conjw_L[level], r3w1u1conju3conjw_R[level])
        if level > 1 
            RhoDescend_withPre!(r3w1u1conju3conjw_L[level], r3w1u1conju3conjw_R[level], u1[level], u2[level], rhoThree[level-1])
        end
    end

    ## density operator (rhoThree)
    #    1   2   3
    #    |   |   |
    #   |||||||||||
    #    |   |   |
    #    4   5   6

    ## isometry w
    #      3 
    #      | 
    #   |||||||
    #    |   |
    #    1   2

    # May be not optimal
    @tensor rhoFour[-1,-2,-3,-4, -5,-6,-7,-8] := rhoThree[1][2,7,5,3,8,6] * w1[1][1,13,3] * w2[1][14,15,8] * w3[1][16,4,6] * conj(w1[1][1,9,2]) * conj(w2[1][10,11,7]) * conj(w3[1][12,4,5]) * 
                                                                        u1[1][-5,-6,13,14] * u2[1][-7,-8,15,16] * conj(u1[1][-1,-2,9,10]) * conj(u2[1][-3,-4,11,12])
    return rhoFour::Array{ComplexF64,8}
end

#-------------------------------------------------------------------------
const sigmaZ_real = [1.0 0.0; 0.0 -1] 
const rotationGateforMeasureX_real = [0.7071067811865476 0.7071067811865476;
                                     -0.7071067811865476 0.7071067811865476]

const zMap = [1   1   1   1;
            1   1   1  -1;
            1   1  -1   1;
            1   1  -1  -1;
            1  -1   1   1;
            1  -1   1  -1;
            1  -1  -1   1;
            1  -1  -1  -1;
            -1   1   1   1;
            -1   1   1  -1;
            -1   1  -1   1;
            -1   1  -1  -1;
            -1  -1   1   1;
            -1  -1   1  -1;
            -1  -1  -1   1;
            -1  -1  -1  -1]
const zzMap = [1   1   1;
            1   1  -1;
            1  -1  -1;
            1  -1   1;
            -1  -1   1;
            -1  -1  -1;
            -1   1  -1;
            -1   1   1;
            -1   1   1;
            -1   1  -1;
            -1  -1  -1;
            -1  -1   1;
            1  -1   1;
            1  -1  -1;
            1   1  -1;
            1   1   1]

# For L = 3*2^nLayers
function IsingEnergy(u1::Vector{Array{ComplexF64,4}}, u2::Vector{Array{ComplexF64,4}}, w1::Vector{Array{ComplexF64,3}}, w2::Vector{Array{ComplexF64,3}}, w3::Vector{Array{ComplexF64,3}}, nLayers::Int, prealloc::Tuple,
    Bz::Real; J::Real=-1.0)::Float64

    rhoFour = physicalRho_fourSites_averageChannel(u1,u2, w1,w2,w3, nLayers, prealloc)
    @assert size(rhoFour) == (2,2,2,2, 2,2,2,2)
    diagRho = zeros(2^4)
    for i ∈ 1:2, j ∈ 1:2, k ∈ 1:2, l ∈ 1:2
        diagRho[(i-1)*8+(j-1)*4+(k-1)*2+l] = rhoFour[i,j,k,l, i,j,k,l]
    end
    s = sum(diagRho)
    for i ∈ eachindex(diagRho)
        diagRho[i] /= s
    end
    # To measure Z, |0> -> +1 while |1> -> -1 
    # diagRho[1] --> |0000>
    # diagRho[2] --> |0001>
    # ...
    # diagRho[15] --> |1110>
    # diagRho[16] --> |1111>
    zMean = 0.0
    for i ∈ 1:16
        zMean += real(diagRho[i]) * (zMap[i,1] + 3*zMap[i,2] + 3*zMap[i,3] + zMap[i,4])/8 # weight is very important!
    end

    # To measure X-basis, should add Ry(-pi/2)
    @tensor rhoFour_Xbasis[-1,-2,-3,-4, -5,-6,-7,-8] := rhoFour[1,2,3,4,5,6,7,8] * rotationGateforMeasureX_real[-5,5] * rotationGateforMeasureX_real[-6,6] * rotationGateforMeasureX_real[-7,7] * rotationGateforMeasureX_real[-8,8] * 
                                                            rotationGateforMeasureX_real[-1,1] * rotationGateforMeasureX_real[-2,2] * rotationGateforMeasureX_real[-3,3] * rotationGateforMeasureX_real[-4,4]
    @assert size(rhoFour_Xbasis) == (2,2,2,2, 2,2,2,2)
    diagRho_X = zeros(2^4)
    for i ∈ 1:2, j ∈ 1:2, k ∈ 1:2, l ∈ 1:2
        diagRho_X[(i-1)*8+(j-1)*4+(k-1)*2+l] = rhoFour_Xbasis[i,j,k,l, i,j,k,l]
    end
    s = sum(diagRho_X)
    for i ∈ eachindex(diagRho_X)
        diagRho_X[i] /= s
    end
    xxMean = 0.0
    for i ∈ 1:16
        xxMean += real(diagRho_X[i]) * (zzMap[i,1] + 2*zzMap[i,2] + zzMap[i,3])/4 # weight is very important!
    end

    return J * xxMean + Bz * zMean
end 

# For L = 2*2^nLayers, and nLayers >= 2
# w3[end] is w_top
# The nLayer-th layer is transition Layer; Causal Cone: 3 sites (Bottom) <-> 2 sites (Top)
function IsingEnergy_type2(u1::Vector{Array{ComplexF64,4}}, u2::Vector{Array{ComplexF64,4}}, w1::Vector{Array{ComplexF64,3}}, w2::Vector{Array{ComplexF64,3}}, w3::Vector{Array{ComplexF64,3}}, nLayers::Int, prealloc::Tuple,
    Bz::Real; J::Real=-1.0)::Float64

    rhoThree, r2w2conjw, r3w2conjw, r3w1u1conju3conjw_L, r3w1u1conju3conjw_R = prealloc
    @assert length(rhoThree) == nLayers-1

    # Transition layer (nLayer-th)
    rhoTop = setTopRhoTwo(w3[end])
    RhoDescend!(rhoThree[end], rhoTop, w1[end], w2[end], u1[end], u2[end])
    
    # Normal layer
    for level ∈ nLayers-1:-1:1
        preContract1st!(rhoThree[level], w1[level], w2[level], w3[level], r2w2conjw[level], r3w2conjw[level])
        preContract2nd!(r3w2conjw[level], w2[level], u1[level], u2[level], r3w1u1conju3conjw_L[level], r3w1u1conju3conjw_R[level])
        if level > 1 
            RhoDescend_withPre!(r3w1u1conju3conjw_L[level], r3w1u1conju3conjw_R[level], u1[level], u2[level], rhoThree[level-1])
        end
    end

    # May be not optimal
    @tensor rhoFour[-1,-2,-3,-4, -5,-6,-7,-8] := rhoThree[1][2,7,5,3,8,6] * w1[1][1,13,3] * w2[1][14,15,8] * w3[1][16,4,6] * conj(w1[1][1,9,2]) * conj(w2[1][10,11,7]) * conj(w3[1][12,4,5]) * 
                                                                        u1[1][-5,-6,13,14] * u2[1][-7,-8,15,16] * conj(u1[1][-1,-2,9,10]) * conj(u2[1][-3,-4,11,12])

    @assert size(rhoFour) == (2,2,2,2, 2,2,2,2)
    diagRho = zeros(2^4)
    for i ∈ 1:2, j ∈ 1:2, k ∈ 1:2, l ∈ 1:2
        diagRho[(i-1)*8+(j-1)*4+(k-1)*2+l] = rhoFour[i,j,k,l, i,j,k,l]
    end
    s = sum(diagRho)
    for i ∈ eachindex(diagRho)
        diagRho[i] /= s
    end
    # To measure Z, |0> -> +1 while |1> -> -1 
    # diagRho[1] --> |0000>
    # diagRho[2] --> |0001>
    # ...
    # diagRho[15] --> |1110>
    # diagRho[16] --> |1111>
    zMean = 0.0
    for i ∈ 1:16
        zMean += real(diagRho[i]) * (zMap[i,1] + 3*zMap[i,2] + 3*zMap[i,3] + zMap[i,4])/8 # weight is very important!
    end

    # To measure X-basis, should add Ry(-pi/2)
    @tensor rhoFour_Xbasis[-1,-2,-3,-4, -5,-6,-7,-8] := rhoFour[1,2,3,4,5,6,7,8] * rotationGateforMeasureX_real[-5,5] * rotationGateforMeasureX_real[-6,6] * rotationGateforMeasureX_real[-7,7] * rotationGateforMeasureX_real[-8,8] * 
                                                            rotationGateforMeasureX_real[-1,1] * rotationGateforMeasureX_real[-2,2] * rotationGateforMeasureX_real[-3,3] * rotationGateforMeasureX_real[-4,4]
    @assert size(rhoFour_Xbasis) == (2,2,2,2, 2,2,2,2)
    diagRho_X = zeros(2^4)
    for i ∈ 1:2, j ∈ 1:2, k ∈ 1:2, l ∈ 1:2
        diagRho_X[(i-1)*8+(j-1)*4+(k-1)*2+l] = rhoFour_Xbasis[i,j,k,l, i,j,k,l]
    end
    s = sum(diagRho_X)
    for i ∈ eachindex(diagRho_X)
        diagRho_X[i] /= s
    end
    xxMean = 0.0
    for i ∈ 1:16
        xxMean += real(diagRho_X[i]) * (zzMap[i,1] + 2*zzMap[i,2] + zzMap[i,3])/4 # weight is very important!
    end

    return J * xxMean + Bz * zMean
end 