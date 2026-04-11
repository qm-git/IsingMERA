# -*- coding: utf-8 -*-
# quanIsing.jl
#
# by Qiang Miao - last modified 07mm/17dd/2024

using QuadGK
using TensorOperations

#-------------------------------------------------------------------------
"""
Define two site quantum Ising model Hamiltonian \\
If field == 'Z', H = J * XX + g * Z; elseif == 'X', H = J * ZZ + g * X\\
If blocked, q0 = 2 and chi = 2^q0 = 4. Otherwise, q0 = 1 and chi = 2\\
Note energy shift for use with Evenbly-Vidal MERA
"""
function HamQuanIsing(g::Real, shifted::Bool, twoSites::Bool, qORχ::Char; J::Real=-1.0, blocked::Bool=false, field::Char='Z')
    
    pauli_x = deepcopy(sigmaX)
    pauli_z = deepcopy(sigmaZ)
    I2 = deepcopy(sigmaI)
    I4 = kron(I2,I2)

    if (field == 'Z')
        h0 = J* ⊗([pauli_x,pauli_x]) + 0.5* g *( ⊗([pauli_z,I2]) + ⊗([I2,pauli_z]) )
    elseif (field == 'X')
        h0 = J* ⊗([pauli_z,pauli_z]) + 0.5* g *( ⊗([pauli_x,I2]) + ⊗([I2,pauli_x]) )
    else
        println("undefined")
    end

    if (blocked == false)
        χ0 = 2; q0 = 1; @assert χ0 == 2^q0
        if (twoSites == true)
            if (shifted == false)
                h = quanReshape(h0,(χ0,χ0,χ0,χ0))
                shift = 0.0
            else
                # shift energy spectrum by largest eigenvalue
                shift = eigmax(h0)
                h = quanReshape(h0,(χ0,χ0,χ0,χ0)) - shift*(quanReshape(I4,(χ0,χ0,χ0,χ0)))
            end
            if (qORχ=='χ')
                return h::Array{Complex{Float64},4}, χ0::Int, shift::Float64
            else
                return h::Array{Complex{Float64},4}, q0::Int, shift::Float64
            end
        else # threeSites == true
            h0 = 0.5*(⊗([h0,I2]) + ⊗([I2,h0]))
            if (shifted == false)
                h = quanReshape(h0,(χ0,χ0,χ0,χ0,χ0,χ0))
                shift = 0.0
            else
                # shift energy spectrum by largest eigenvalue
                shift = eigmax(h0); 
                h = quanReshape(h0,(χ0,χ0,χ0,χ0,χ0,χ0)) - shift*(quanReshape(⊗([I2,I2,I2]),(χ0,χ0,χ0,χ0,χ0,χ0)))
            end
            if (qORχ=='χ')
                return h::Array{Complex{Float64},6}, χ0::Int, shift::Float64
            else
                return h::Array{Complex{Float64},6}, q0::Int, shift::Float64
            end
        end
    else
        # compute blocked Hamiltonian
        u_tilde = quanReshape(I4,(2,2,4))
        h0 = quanReshape(h0,(2,2,2,2))
        @tensor h_half[-1,-2] := conj(u_tilde[1,2,-1]) * h0[1,2,3,4] * u_tilde[3,4,-2]
        @tensor h_middle[-1,-2,-3,-4] := conj(u_tilde[1,2,-1])*u_tilde[1,3,-3]*h0[2,5,3,6]*conj(u_tilde[5,4,-2])*u_tilde[6,4,-4]
        @tensor h_lhalf[-1,-2,-3,-4] := 0.5* h_half[-1,-3] * I4[-2,-4]
        @tensor h_rhalf[-1,-2,-3,-4] := 0.5* I4[-1,-3] * h_half[-2,-4]
        h0 = h_lhalf + h_middle + h_rhalf

        χ0 = 4; q0 = 2; @assert χ0 == 2^q0
        # Normilize h as a single-site energy density operator
        h0 = 0.5 * quanReshape(h0, (χ0^2,χ0^2))
        if (twoSites == true)
            if (shifted == false)
                h = quanReshape(h0,(χ0,χ0,χ0,χ0))
                shift = 0.0
            else
                # shift energy spectrum by largest eigenvalue
                shift = eigmax(h0)
                h = quanReshape(h0,(χ0,χ0,χ0,χ0)) - shift*(quanReshape(⊗([I4,I4]),(χ0,χ0,χ0,χ0)))
            end
            if (qORχ=='χ')
                return h::Array{Complex{Float64},4}, χ0::Int, shift::Float64
            else
                return h::Array{Complex{Float64},4}, q0::Int, shift::Float64
            end
        else # threeSites == true
            h0 = 0.5*(⊗([h0,I4]) + ⊗([I4,h0]))
            if (shifted == false)
                h = quanReshape(h0,(χ0,χ0,χ0,χ0,χ0,χ0))
                shift = 0.0
            else
                # shift energy spectrum by largest eigenvalue
                shift = eigmax(h0); 
                h = quanReshape(h0,(χ0,χ0,χ0,χ0,χ0,χ0)) - shift*(quanReshape(⊗([I4,I4,I4]),(χ0,χ0,χ0,χ0,χ0,χ0)))
            end
            if (qORχ=='χ')
                return h::Array{Complex{Float64},6}, χ0::Int, shift::Float64
            else
                return h::Array{Complex{Float64},6}, q0::Int, shift::Float64
            end
        end
    end

end

#-------------------------------------------------------------------------
"Give ground energy density E_gs/L for quantum Ising model in thermodynamic limit (L -> ∞)"
function groundEnergyDensity_QIsing(g::Real; J::Real=-1.0)::Float64
    groundEnergyDensity,_ = quadgk(k -> √(J^2 + g^2 + 2.0*(g*J)*cos(k)), 0, 2*π)
    groundEnergyDensity = - groundEnergyDensity / (2*π) 

    return groundEnergyDensity
end
function groundEnergyDensity_QIsing(g::Real, N::Int; J::Real=-1.0)::Float64
    "has LOWER energy density than the thermodynamic limit case"
    if (iseven(N) == true)
        groundEnergyDensity = [√(J^2 + g^2 + 2.0*(g*J)*cos(π*(2*k+1)/N)) for k ∈ 0:N-1] # APBC
    else
        # groundEnergyDensity = [√(J^2 + g^2 + 2.0*(g*J)*cos(π*(2*k)/N)) for k ∈ 0:N-1] # PBC # mightbe wrong
        @assert 0 == 1
    end
    groundEnergyDensity = - sum(groundEnergyDensity) / N 

    return groundEnergyDensity
end

#-------------------------------------------------------------------------
"Give theoretical predictions for ⟨sigmaZ⟩"
function groundState_SigmaZ(g::Real; J::Real=-1.0)::Float64
    # Physical Review A, 66, 032110 (2002), Eq.(35)
    if J == 0
        return 1.0
    else
        g = g/abs(J)
        if abs(g) != 1
            groundstateSigmaZ,_ = quadgk(k -> (g + cos(k)) / √(1 + g^2 + 2*g*cos(k)), 0, 2*π)
        elseif g == 1
            groundstateSigmaZ = 4 # quadgk(k -> √(1 + cos(k)) / (√2), 0, 2*π)
        else
            groundstateSigmaZ = -4 # quadgk(k -> √(1 - cos(k)) / (-√2), 0, 2*π)
        end
        groundstateSigmaZ = - groundstateSigmaZ / (2*π)
        return groundstateSigmaZ
    end
end
function groundState_SigmaZ(g::Real, N::Int; J::Real=-1.0)::Float64
    if J == 0
        return 1.0
    else
        g = g/abs(J)
        if iseven(N) == true
            groundstateSigmaZ = [(g + cos((2*k-1)*π/N)) / √(1 + g^2 + 2*g*cos((2*k-1)*π/N)) for k ∈ 1:N//2]
            groundstateSigmaZ = - 2 * sum(groundstateSigmaZ) / N
        else
            @assert 1 == 0 
        end
        return groundstateSigmaZ
    end
end

"Give (spontaneous symmetry breaking) abs(⟨sigmaX⟩)"
function groundState_SigmaX(g::Real; J::Real=-1.0)::Float64
    # Physical Review A, 66, 032110 (2002), Eq.(34) 
    if J == 0
        return 0.0
    else
        g = g / J
        if abs(g) ≥ 1
            return 0.0
        else
            return (1-g^2)^(1/8)
        end
    end
end
function groundState_SigmaX(g::Real; J::Real=-1.0)::Float64
    return 0.0
end

function groundState_SigmaY(g::Real, N::Int; J::Real=-1.0)::Float64
    return 0.0
end
function groundState_SigmaY(g::Real, N::Int; J::Real=-1.0)::Float64
    return 0.0
end

function groundState_SigmaXX(g::Real; J::Real=-1.0)::Float64
    # Physical Review A, 66, 032110 (2002), Eq.(21) 
    if J == 0
        return 0.0
    else
        g = g / J
        # for H = - \sum_i (X_i X_i+1 + g* Z_i)
        if abs(g) != 1
            groundstateSigmaXX,_ = quadgk(k -> (g*cos(k) + 1) / √(1 + g^2 + 2*g*cos(k)), 0, 2*π)
        else
            groundstateSigmaXX = 4 # quadgk(k -> √(1 + cos(k)) / √2, 0, 2*π) 
        end
        groundstateSigmaXX = groundstateSigmaXX / (2*π)

        if J < 0
            return groundstateSigmaXX
        else
            return - groundstateSigmaXX
        end
    end
end
function groundState_SigmaXX(g::Real, N::Int; J::Real=-1.0)::Float64
    if J == 0
        return 0.0
    else
        g = g / J
        # for H = - \sum_i (X_i X_i+1 + g* Z_i)
        if iseven(N) == true
            groundstateSigmaXX = [(g*cos((2*k-1)*π/N) + 1) / √(1 + g^2 + 2*g*cos((2*k-1)*π/N)) for k ∈ 1:N//2]
            groundstateSigmaXX = 2 * sum(groundstateSigmaXX) / N
        else
            @assert 1 == 0 
        end

        if J < 0
            return groundstateSigmaXX
        else
            return - groundstateSigmaXX
        end
    end
end

function groundState_SigmaYY(g::Real; J::Real=-1.0)::Float64
    # Physical Review A, 66, 032110 (2002), Eq.(22) 
    if J == 0
        return 0.0
    else
        g = g / J
        # for H = - \sum_i (X_i X_i+1 + g* Z_i)
        if abs(g) != 1
            groundstateSigmaYY,_ = quadgk(k -> (g*cos(k) + cos(2*k)) / √(1 + g^2 + 2*g*cos(k)), 0, 2*π)
        else
            groundstateSigmaYY = -4/3 # quadgk(k -> √(1 + cos(k)) * (2*cos(k) - 1) / √2, 0, 2*π) 
        end
        groundstateSigmaYY = groundstateSigmaYY / (2*π)

        if J < 0
            return groundstateSigmaYY
        else
            return - groundstateSigmaYY
        end
    end
end
function groundState_SigmaYY(g::Real, N::Int; J::Real=-1.0)::Float64
    if J == 0
        return 0.0
    else
        g = g / J
        # for H = - \sum_i (X_i X_i+1 + g* Z_i)
        if iseven(N) == true
            groundstateSigmaYY = [(g*cos((2*k-1)*π/N) + cos(2*(2*k-1)*π/N)) / √(1 + g^2 + 2*g*cos((2*k-1)*π/N)) for k ∈ 1:N//2] 
            groundstateSigmaYY = 2 * sum(groundstateSigmaYY) / N
        else
            @assert 1 == 0 
        end

        if J < 0
            return groundstateSigmaYY
        else
            return - groundstateSigmaYY
        end
    end
end

function groundState_SigmaZZ(g::Real; J::Real=-1.0)::Float64
    # Physical Review A, 66, 032110 (2002), Eq.(23) 
    if J == 0
        return 1.0
    else
        return (groundState_SigmaZ(g))^2 - groundState_SigmaYY(g,J=J)*groundState_SigmaXX(g,J=J)
    end
end
function groundState_SigmaZZ(g::Real, N::Int; J::Real=-1.0)::Float64
    if J == 0
        return 1.0
    else
        return (groundState_SigmaZ(g, N))^2 - groundState_SigmaYY(g, N, J=J)*groundState_SigmaXX(g, N, J=J)
    end
end

# function groundState_SigmaXZ(g::Real; J::Real=-1.0)::Float64
#     # Physical Review A, 66, 032110 (2002), Eq.(22) 
#     if J == 0
#         return 0.0
#     else
#         g = g / J
#         # for H = - \sum_i (X_i X_i+1 + g* Z_i)
#         if abs(g) != 1
#             groundEnergySigmaYY,_ = quadgk(k -> (g*cos(k) + cos(2*k)) / √(1 + g^2 + 2*g*cos(k)), 0, 2*π)
#         else
#             groundEnergySigmaYY,_ = quadgk(k -> √(1 + cos(k))*(2*cos(k) - 1) / √2, 0, 2*π) 
#         end
#         groundEnergySigmaYY = groundEnergySigmaYY / (2*π)

#         if J < 0
#             return groundEnergySigmaYY
#         else
#             return - groundEnergySigmaYY
#         end
#     end
# end

#-------------------------------------------------------------------------
"Give core function for correlation matrix"
function groundState_coreG(g::Real, l::Int; J::Real=-1.0)::Float64
    # Pierre Pfeuty 1970, Annals of Physics, The one-dimensional Ising model with a transverse field (in the same fold)
    if J != 0
        g = g/abs(J)
        if abs(g) != 1 # cannot converge if abs(|g|-1) < 1e-7
            coreG,_ = quadgk(k -> (g*cos(k*l) + cos(k*(l+1))) / √(1 + g^2 + 2*g*cos(k)), 0, 2*π)
        elseif g == 1
            coreG = 4 * (-1)^l / (2*l + 1)
        else
            coreG = -4 / (2*l + 1)
        end
        if J > 0
            coreG = - coreG 
        end
        return coreG/ (2*π)
    end
end
function groundState_coreG(g::Real, N::Int, l::Int; J::Real=-1.0)::Float64
    if J != 0
        g = g/abs(J)
        if iseven(N) == true
            s = 0
            for j ∈ 1:N//2
                k = (2*j-1)*π/N
                s += (g*cos(k*l) + cos(k*(l+1))) / √(1 + g^2 + 2*g*cos(k))
            end
            if J > 0
                return - 2 * s / N
            else
                return 2 * s / N
            end
        else
            @assert 1 == 0 
        end
    end
end

using LinearAlgebra
function groundState_SigmaX0Xn(n::Int, g::Real; J::Real=-1.0)::Float64
    @assert n >= 1 
    coreD_dn = zeros(Float64, 2*n-1)
    for i ∈ -n:n-2
        coreD_dn[i+(n+1)] = groundState_coreG(g, i)
    end
    coreD = zeros(Float64, n,n)
    for i ∈ 1:n
        for j ∈ 1:n
            dn = i-1 - j
            coreD[i,j] = coreD_dn[dn+(n+1)]
        end
    end
    return det(coreD)
end
function groundState_SigmaX0Xn(n::Int, g::Real, N::Int; J::Real=-1.0)::Float64
    @assert n >= 1
    coreD_dn = zeros(Float64, 2*n-1)
    for i ∈ -n:n-2
        coreD_dn[i+(n+1)] = groundState_coreG(g,N, i)
    end
    coreD = zeros(Float64, n,n)
    for i ∈ 1:n
        for j ∈ 1:n
            dn = i-1 - j
            coreD[i,j] = coreD_dn[dn+(n+1)]
        end
    end
    return det(coreD)
end

function groundState_SigmaY0Yn(n::Int, g::Real; J::Real=-1.0)::Float64
    @assert n >= 1
    coreD_dn = zeros(Float64, 2*n-1)
    for i ∈ -n+2:n
        coreD_dn[i+(n-1)] = groundState_coreG(g, i)
    end
    coreD = zeros(Float64, n,n)
    for i ∈ 1:n
        for j ∈ 1:n
            dn = i+1 - j
            coreD[i,j] = coreD_dn[dn+(n-1)]
        end
    end
    return det(coreD)
end
function groundState_SigmaY0Yn(n::Int, g::Real, N::Int; J::Real=-1.0)::Float64
    @assert n >= 1
    coreD_dn = zeros(Float64, 2*n-1)
    for i ∈ -n+2:n
        coreD_dn[i+(n-1)] = groundState_coreG(g,N, i)
    end
    coreD = zeros(Float64, n,n)
    for i ∈ 1:n
        for j ∈ 1:n
            dn = i+1 - j
            coreD[i,j] = coreD_dn[dn+(n-1)]
        end
    end
    return det(coreD)
end

function groundState_SigmaZ0Zn(n::Int, g::Real; J::Real=-1.0)::Float64
    @assert n >= 0
    Z = groundState_SigmaZ(g)
    return Z^2 - groundState_coreG(g, n) * groundState_coreG(g,-n)
end
function groundState_SigmaZ0Zn(n::Int, g::Real, N::Int; J::Real=-1.0)::Float64
    @assert n >= 0
    Z = groundState_SigmaZ(g, N)
    return Z^2 - groundState_coreG(g,N, n) * groundState_coreG(g,N,-n)
end

function entanglementEntropy(n::Int, g::Real; which::Int=2)
    coreG_positive_dn = zeros(Float64, n-1)
    coreG_negative_dn = zeros(Float64, n-1)
    for i ∈ 1:n-1
        coreG_positive_dn[i] = groundState_coreG(g, i)
        coreG_negative_dn[i] = groundState_coreG(g,-i)
    end
    coreG = zeros(Float64, n,n)
    diag = groundState_coreG(g,0)
    for i ∈ 1:n
        coreG[i,i] = diag
        for j ∈ i+1:n
            coreG[i,j] = coreG_positive_dn[j-i]
            coreG[j,i] = coreG_negative_dn[j-i]
        end
    end
    ν = svdvals(coreG)
    if which == 1
        return sum(binary_vNentropy.((1 .+ ν) ./ 2))::Float64
    elseif which == 2
        return sum(binary_renyiEntropy.((1 .+ ν) ./ 2))::Float64
    else
        return sum(binary_vNentropy.((1 .+ ν) ./ 2))::Float64, sum(binary_renyiEntropy.((1 .+ ν) ./ 2))::Float64
    end
end
function entanglementEntropy(n::Int, g::Real, N::Int; which::Int=2)
    coreG_positive_dn = zeros(Float64, n-1)
    coreG_negative_dn = zeros(Float64, n-1)
    for i ∈ 1:n-1
        coreG_positive_dn[i] = groundState_coreG(g,N, i)
        coreG_negative_dn[i] = groundState_coreG(g,N,-i)
    end
    coreG = zeros(Float64, n,n)
    diag = groundState_coreG(g,N,0)
    for i ∈ 1:n
        coreG[i,i] = diag
        for j ∈ i+1:n
            coreG[i,j] = coreG_positive_dn[j-i]
            coreG[j,i] = coreG_negative_dn[j-i]
        end
    end
    ν = svdvals(coreG)
    if which == 1
        return sum(binary_vNentropy.((1 .+ ν) ./ 2))::Float64
    elseif which == 2
        return sum(binary_renyiEntropy.((1 .+ ν) ./ 2))::Float64
    else
        return sum(binary_vNentropy.((1 .+ ν) ./ 2))::Float64, sum(binary_renyiEntropy.((1 .+ ν) ./ 2))::Float64
    end
end
function binary_vNentropy(x::Float64; base=2)::Float64
    if 0<x<1
        return - x*log(base, x) - (1-x)*log(base, 1-x)
    else
        return 0
    end
end
function binary_renyiEntropy(x::Float64; base=2)::Float64
    if 0<x<1
        return - log(base, x^2 + (1-x)^2)
    else
        return 0
    end
end

function self_test()
    println("E(",-1,",",0,") = ", groundEnergyDensity_QIsing(0,-1))    
    println("E(",-1,",",0.5,") = ", groundEnergyDensity_QIsing(0.5,-1))    
    println("E(",-1,",",1,") = ", groundEnergyDensity_QIsing(1,-1))    
    println("E(",-1,",",1.5,") = ", groundEnergyDensity_QIsing(1.5,-1))   

    h, h, q0, shift = HamQuanIsing(0.5, true, true, 'q'; J = -1, blocked=true, field='X')
    println("q0 = ", q0, ", shift = ", shift, ", h = ", h)
end

#self_test()
    