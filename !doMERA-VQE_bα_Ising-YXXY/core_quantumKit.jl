# -*- coding: utf-8 -*-
# myPackage/kit/core_quantumKit.jl
# add myCopyto!(), which is safer than copyto!() but might be a little bit slower. To be tested!
# by Qiang Miao - last modified 05mm/12dd/2023

# add yyYXXY gate 01mm/18dd/2026

using LinearAlgebra

# |0> = [1; 0; 0; 0 ...]
# |1> = [0; 1; 0; 0 ...]
# |2> = [0; 0; 0; 1 ...]

sigmaX = [0.0im 1.0; 1.0 0.0]   # Pauli matrix \sigma_x
sigmaY = [0.0 -1im; 1im 0.0]    # Pauli matrix \sigma_y
sigmaZ = [1.0 0.0im; 0.0 -1]    # Pauli matrix \sigma_z
sigmaI = [1.0 0.0im; 0.0 1.0]   

const halfSqrt2 = 0.7071067811865475244

spin1_x = halfSqrt2 * [0.0 1.0 0.0;
                       1.0 0.0 1.0;
                       0.0 1.0 0.0im]
spin1_y = halfSqrt2 * [0.0 -1im 0.0;
                       1im 0.0 -1im;
                       0.0 1im 0.0]
spin1_z = [1 0.0 0.0im;
           0 0.0 0.0;
           0 0.0 -1.0]
spin1_raising = [0 √2 0;
              0 0 √2;
              0 0 0]
spin1_lowering = [0 0 0;
               √2 0 0;
               0 √2 0]
"""
spin1_x4 = [0  1  1. 0;
            1  0  0im 1;
            1  0  0  1;
            0  1  1  0] / 2
spin1_y4 = [0  -1im -1im 0;
            1im  0. 0 -1im;
            1im  0  0  -1im;
            0  1im  1im  0] / 2
spin1_z4 = [1.0 0.0 0im 0.0;
           0.0 0.0 0.0 0.0;
           0.0 0.0 0.0 0.0;
           0.0 0.0 0.0 -1.]
Ξ = [0 0. 0. 0;
     0 1. -1 0;
     0 -1 1. 0im;
     0 0  0  0] / 2
"""
spin1_x4 = halfSqrt2 * [0.0 1.0 0.0 0.0;
                        1.0 0.0 1.0 0.0;
                        0.0 1.0 0.0im 0;
                        0.0 0.0 0.0 0.0]
spin1_y4 = halfSqrt2 * [0.0 -1im 0.0 0.0;
                        1im 0.0 -1im 0.0;
                         0.0 1im 0.0 0.0;
                         0.0 0.0 0.0 0.0]
spin1_z4 = [1.0 0.0 0im 0.0;
            0.0 0.0 0.0 0.0;
            0.0 0.0 -1. 0.0;
            0.0 0.0 0.0 0.0]
Ξ = [0 0 0 0;
     0 0 0 0;
     0 0 0 0im;
     0 0 0 1.0]
spin1_raising4 = [0 √2 0 0;
                  0 0 √2 0;
                  0 0 0 0;
                  0 0 0 0]
spin1_lowering4 = [0 0 0 0;
                  √2 0 0 0;
                  0 √2 0 0;
                  0 0 0 0]

#H_new = U' * H * U, with U = [1 0 0 0; 0 1/sqrt(2) 1/sqrt(2) 0; 0 0 0 1; 0 1/sqrt(2) -1/sqrt(2) 0]


Hadamard = halfSqrt2 * [1 1; 1 -1]

const halfSqrt3 = 0.86602540378443865

spin3half_x = [0.0im  halfSqrt3  0.0  0.0;
        halfSqrt3  0.0  1.0  0.0;
        0.0  1.0  0.0  halfSqrt3;
        0.0  0.0  halfSqrt3  0.0]
spin3half_y = [0.0  -halfSqrt3*1.0im  0.0  0.0;
        halfSqrt3*1.0im  0.0  -1.0im  0.0;
        0.0  1.0im  0.0  -halfSqrt3*1.0im;
        0.0  0.0  halfSqrt3*1.0im  0.0]
spin3half_z = [1.5  0.0  0.0  0.0im;
        0.0  0.5  0.0  0.0;
        0.0  0.0  -0.5  0.0;
        0.0  0.0  0.0  -1.5]
I4 = Matrix{ComplexF64}(I,4,4)

function randHermitian(χ::Int,siteNum::Int;traceless=true,normOne=true)::Array{ComplexF64}
    @assert χ ≥ 2
    A = rand(ComplexF64, (χ^siteNum,χ^siteNum))
    H = A + A'
    if traceless == true
        H = H - tr(H) * Matrix{ComplexF64}(I,χ^siteNum,χ^siteNum) / χ^siteNum
        if normOne == true
            H = H / sqrt(tr(H*H'))
        end
    else
        if normOne == true
            d = sqrt(tr(H*H')-(tr(H))^2/χ^siteNum)
            H = H / d
        end
    end
    @assert ishermitian(H)
    shapeH::Tuple = Tuple(repeat([χ],2*siteNum))
    return quanReshape(H,shapeH)
end

function GeneralizedIsoHeisenberg(χ::Int,siteNum::Int;normOne=true)::Array{ComplexF64}
    @assert χ ≥ 2
    H = zeros(ComplexF64, (χ^siteNum,χ^siteNum))
    for i ∈ 1:χ^2-1
        axpy!(1, ⊗([GeneralizedGellMann(i,χ) for _ ∈ 1:siteNum]), H)
    end
    if normOne == true
        lmul!(1/sqrt(2^siteNum * (χ^2-1)), H)
        @assert isapprox(tr(H*H),1,atol=1e-13)
    end
    @assert ishermitian(H)
    @assert isapprox(tr(H),0,atol=1e-15)
    shapeH::Tuple = Tuple(repeat([χ],2*siteNum))
    return quanReshape(H,shapeH)
end

#----------------------------------------------------------------------------------------------------------------------------------------------
# Generalized Gell-Mann matrices (Hermitian)
function E_jk(j::Int,k::Int,d::Int)::Matrix{ComplexF64}
    Ejk = zeros(ComplexF64,d,d)
    Ejk[j,k] = 1.0
    return Ejk
end
function h_kd(k::Int,d::Int)::Matrix{ComplexF64}
    @assert 1 ≤ k ≤ d
    if k == 1
        return Matrix{ComplexF64}(I,d,d)
    elseif k == d
        return √(2/(d*(d-1))) * ⊕(Matrix{ComplexF64}(I,d-1,d-1),1-d)
    else
        return ⊕(h_kd(k,d-1),0)
    end
end
function GeneralizedGellMann(k::Int,j::Int,d::Int)::Matrix{ComplexF64}
    @assert 1 ≤ k ≤ d  && 1 ≤ j ≤ d && d ≥ 2
    if k < j 
        return E_jk(k,j,d) + E_jk(j,k,d)
    elseif k > j
        return -1im * (E_jk(j,k,d) - E_jk(k,j,d))
    elseif k == j 
        return h_kd(k,d)
    end
end
"Identity if i = 0; otherwise, return Hermitian and traceless generalized Gell-Mann matrices acting on qudits, with tr(GGM_i * GGM_j) = 2 * δ_i,j."
function GeneralizedGellMann(i::Int,d::Int)::Matrix{ComplexF64}
    @assert 0 ≤ i ≤ d^2 - 1

    # 1 ≤ k ≤ d  && 1 ≤ j ≤ d
    # if k > j
    #     i = (k-1)^2 + 2*(j-1) + 1
    # else
    #     i = (j-1)^2 + 2*(k-1)
    # end

    fs::Int = floor(sqrt(i))
    res::Int = i - fs^2
    if iseven(res)
        k = Int(res/2)+1
        j = fs + 1
    else
        j = Int((res-1)/2) + 1
        k = fs + 1
    end
    
    return GeneralizedGellMann(k,j,d)
end
# test GeneralizedGellMann()
# for i in 1:255
#     for j in 1:255
#         a=tr(GeneralizedGellMann(i,16)*GeneralizedGellMann(j,16))
#         b = ==(i,j)
#         @assert isapprox(a,2*b, atol = 1e-15)
#     end
# end

"Direct sum of matrices"
function ⊕(A::Number,B::Matrix)::Matrix
    Bi, Bj = size(B)
    directSum::Matrix = zeros(typeof(A*B[1]), 1+Bi, 1+Bj)
    directSum[1,1] = A[1,1]
    @inbounds begin
        for j ∈ 1:Bj
            for i ∈ 1:Bi
                directSum[1 + i, 1 + j] = B[i,j]
            end
        end
    end
    return directSum
end
function ⊕(A::Matrix,B::Number)::Matrix
    Ai, Aj = size(A)
    directSum::Matrix = zeros(typeof(A[1]*B), Ai+1, Aj+1)
    @inbounds begin
        for j ∈ 1:Aj
            for i ∈ 1:Ai
                directSum[i,j] = A[i,j]
            end
        end
    end
    directSum[Ai + 1, Aj + 1] = B
    return directSum
end
function ⊕(A::Matrix,B::Matrix)::Matrix
    Ai, Aj = size(A)
    Bi, Bj = size(B)
    directSum::Matrix = zeros(typeof(A[1]*B[1]), Ai+Bi, Aj+Bj)
    @inbounds begin
        for j ∈ 1:Aj
            for i ∈ 1:Ai
                directSum[i,j] = A[i,j]
            end
        end
    end
    @inbounds begin
        for j ∈ 1:Bj
            for i ∈ 1:Bi
                directSum[Ai + i, Aj + j] = B[i,j]
            end
        end
    end
    return directSum
end

#----------------------------------------------------------------------------------------------------------------------------------------------
"Entanglement entropy / α-order Renyi entropy"
function getEntropy(ρ::AbstractArray; α::Int=1, Log2::Bool=false, err::Float64=1.0e-15)::Float64
    @assert α >= 1
    if length(size(ρ)) != 2
        vol = prod(size(ρ))
        l = Int(sqrt(vol))
        @assert l^2 == vol
        ρ = quanReshape(ρ, (l,l))
    end

    @assert size(ρ)[1] == size(ρ)[2]
    @assert isapprox(1.0,tr(ρ))
    @assert isapprox(ρ,ρ') #ishermitian(ρ)
    @assert real(tr(ρ*ρ)) <= 1+err && abs(imag(tr(ρ*ρ))) <= err
    if α == 1 
        if Log2 == false
            return real(-tr(ρ*log(ρ)))
        else
            return real(-tr(ρ*log2(ρ)))
        end
    else
        if Log2 == false
            return real(log(tr(ρ^α))/ (1 - α))
        else
            return real(log2(tr(ρ^α))/ (1 - α))
        end
    end
end

#----------------------------------------------------------------------------------------------------------------------------------------------
"Rotation gate around x : Rx(arg) = e^(-1im*arg*sigma_x/2)"
function Rx(x::Float64)::Array{ComplexF64,2}
    x = x/2
    sinθhalf, cosθhalf = sincospi(x) # sincospi(x) Computes sin(πx) and cos(πx)
    Rxθ = [cosθhalf -1im*sinθhalf; -1im*sinθhalf cosθhalf]
    return Rxθ
end

"Rotation gate around y : Ry(arg) = e^(-1im*arg*sigma_y/2)"
function Ry(x::Float64)::Array{ComplexF64,2}
    x = x/2
    sinθhalf, cosθhalf = sincospi(x)
    Ryθ = [cosθhalf -sinθhalf; sinθhalf cosθhalf]
    return Ryθ
end

"Rotation gate around z : Rz(arg) = e^(-1im*arg*sigma_z/2)" # sigma_z/2 = s^z for spin-1/2
function Rz(x::Float64)::Array{ComplexF64,2}
    x = x/2
    Rzθ = [cispi(-x) 0; 0 cispi(x)] # cispi(x) Computes exp(iπx) more accurately
    return Rzθ
end

#----------------------------------------------------------------------------------------------------------------------------------------------
"Two-qubit gate XX : XX(arg) = e^(-1im/2*arg* sigma_x ⊗ sigma_x), which is maximally entangling at x = 0.5"
function XX(x::Float64)::Matrix{ComplexF64}
    x = x/2
    sinθhalf, cosθhalf = sincospi(x) # sincospi(x) Computes sin(πx) and cos(πx)
    XXθ = [cosθhalf 0 0 -1im*sinθhalf; 0 cosθhalf -1im*sinθhalf 0; 0 -1im*sinθhalf cosθhalf 0; -1im*sinθhalf 0 0 cosθhalf]
    return XXθ
end

"Two-qubit gate YY : YY(arg) = e^(-1im/2*arg* sigma_y ⊗ sigma_y), which is maximally entangling at x = 0.5"
function  YY(x::Float64)::Matrix{ComplexF64}
    x = x/2
    sinθhalf, cosθhalf = sincospi(x) # sincospi(x) Computes sin(πx) and cos(πx)
    YYθ = [cosθhalf 0 0 1im*sinθhalf; 0 cosθhalf -1im*sinθhalf 0; 0 -1im*sinθhalf cosθhalf 0; 1im*sinθhalf 0 0 cosθhalf]
    return YYθ
end

"Two-qubit gate ZZ : ZZ(arg) = e^(-1im/2*arg* sigma_z ⊗ sigma_z)"
function ZZ(x::Float64)::Matrix{ComplexF64}
    x = x/2
    m = cispi(-x) # cispi(x) Computes exp(iπx) more accurately
    p = cispi(x) 
    ZZθ = [m 0 0 0; 0 p 0 0; 0 0 p 0; 0 0 0 m]
    return ZZθ
end

"Two-qubit gate XY : XY(arg) = e^(-1im/2*arg* sigma_x ⊗ sigma_y)"
function XY(x::Float64; float::Bool=false)
    x = x/2
    sinθhalf, cosθhalf = sincospi(x) # sincospi(x) Computes sin(πx) and cos(πx)
    if float == true
        XYθ = [cosθhalf 0 0 -sinθhalf; 0 cosθhalf sinθhalf 0; 0 -sinθhalf cosθhalf 0; sinθhalf 0 0 cosθhalf]
        return XYθ::Matrix{Float64}
    else
        XYθ = [cosθhalf 0 0 -sinθhalf; 0 cosθhalf sinθhalf 0; 0 -sinθhalf cosθhalf 0; sinθhalf 0 0+0.0im cosθhalf]
        return XYθ::Matrix{ComplexF64} 
    end
end

"Two-qubit gate YX : YX(arg) = e^(-1im/2*arg* sigma_y ⊗ sigma_x)"
function YX(x::Float64; float::Bool=false)
    x = x/2
    sinθhalf, cosθhalf = sincospi(x) # sincospi(x) Computes sin(πx) and cos(πx)
    if float == true
        YXθ = [cosθhalf 0 0 -sinθhalf; 0 cosθhalf -sinθhalf 0; 0 sinθhalf cosθhalf 0; sinθhalf 0 0 cosθhalf]
        return YXθ::Matrix{Float64}
    else
        YXθ = [cosθhalf 0 0 -sinθhalf; 0 cosθhalf -sinθhalf 0; 0 sinθhalf cosθhalf 0; sinθhalf 0 0+0.0im cosθhalf]
        return YXθ::Matrix{ComplexF64}
    end
end

function YXXY(x1::Float64, x2::Float64; float::Bool=false)
    xp = (x1+x2)/2; xm = (x2-x1)/2
    sinθPhalf, cosθPhalf = sincospi(xp) # sincospi(x) Computes sin(πx) and cos(πx)
    sinθMhalf, cosθMhalf = sincospi(xm)
    if float == true
        YXθ = [cosθPhalf 0 0 -sinθPhalf; 0 cosθMhalf sinθMhalf 0; 0 -sinθMhalf cosθMhalf 0; sinθPhalf 0 0 cosθPhalf]
        return YXθ::Matrix{Float64}
    else
        YXθ = [cosθPhalf 0 0 -sinθPhalf; 0 cosθMhalf sinθMhalf 0; 0 -sinθMhalf cosθMhalf 0; sinθPhalf 0 0+0.0im cosθPhalf]
        return YXθ::Matrix{ComplexF64}
    end
end

#----------------------------------------------------------------------------------------------------------------------------------------------
"""
argList = [var1, var2, ..., var_n]
var1 \\otimes var2 \\otimes ... var_n\\
example: IXYZ = np.kron(np.kron(np.kron(np.eye(2),sX),sY),sZ)
              = kron([sI,sX,sY,sZ])
"""
function ⊗(argList::AbstractVector{Matrix{ComplexF64}})::Matrix{ComplexF64}
    if (length(argList)==1)
        return argList[1]
    elseif (length(argList)==2)
        return kron(argList[1],argList[2])   #simplification for special case
    else
        Kron = [1.0+0.0im]
        for i ∈ eachindex(argList)
            Kron = kron(Kron,argList[i])
        end
        return Kron
    end
end
function ⊗(argList::AbstractVector{Matrix{Float64}})::Matrix{Float64}
    if (length(argList)==1)
        return argList[1]
    elseif (length(argList)==2)
        return kron(argList[1],argList[2])   #simplification for special case
    else
        Kron = [1.0]
        for i ∈ eachindex(argList)
            Kron = kron(Kron,argList[i])
        end
        return Kron
    end
end
function ⊗(argList::AbstractVector{Matrix{Int}})::Matrix{Int}
    if (length(argList)==1)
        return argList[1]
    elseif (length(argList)==2)
        return kron(argList[1],argList[2])   #simplification for special case
    else
        Kron = [1]
        for i ∈ eachindex(argList)
            Kron = kron(Kron,argList[i])
        end
        return Kron
    end
end
function ⊗(argList::AbstractVector{Vector{ComplexF64}})::Vector{ComplexF64}
    if (length(argList)==1)
        return argList[1]
    elseif (length(argList)==2)
        return kron(argList[1],argList[2])   #simplification for special case
    else
        Kron = [1]
        for i ∈ eachindex(argList)
            Kron = kron(Kron,argList[i])
        end
        return Kron
    end
end
function ⊗(argList::AbstractVector{Vector{Float64}})::Vector{Float64}
    if (length(argList)==1)
        return argList[1]
    elseif (length(argList)==2)
        return kron(argList[1],argList[2])   #simplification for special case
    else
        Kron = [1]
        for i ∈ eachindex(argList)
            Kron = kron(Kron,argList[i])
        end
        return Kron
    end
end
function ⊗(argList::AbstractVector{Vector{Int}})::Vector{Int}
    if (length(argList)==1)
        return argList[1]
    elseif (length(argList)==2)
        return kron(argList[1],argList[2])   #simplification for special case
    else
        Kron = [1]
        for i ∈ eachindex(argList)
            Kron = kron(Kron,argList[i])
        end
        return Kron
    end
end
function ⊗(argList::AbstractVector{Array{Int}})::Array{Int} # for mixed vector of vectors and matrices
    if (length(argList)==1)
        return argList[1]
    elseif (length(argList)==2)
        return kron(argList[1],argList[2])   #simplification for special case
    else
        Kron = [1]
        for i ∈ eachindex(argList)
            Kron = kron(Kron,argList[i])
        end
        return Kron
    end
end

function otimes!(C::Matrix{ComplexF64},argList::AbstractVector{Matrix{ComplexF64}}) 
    if length(argList)==1
        copyto!(C,argList[1])
    elseif length(argList)==2
        otimes!(C,argList[1],argList[2])   #simplification for special case
    elseif length(argList)==3
        otimes!(C,argList[1],argList[2],argList[3])
    elseif length(argList)==4
        otimes!(C,argList[1],argList[2],argList[3],argList[4])
    elseif length(argList)==5
        otimes!(C,argList[1],argList[2],argList[3],argList[4],argList[5])
    elseif length(argList)==6
        otimes!(C,argList[1],argList[2],argList[3],argList[4],argList[5],argList[6])
    else
        Kron = [1.0+0.0im]
        for i ∈ 1:length(argList)-1
            Kron = kron(Kron,argList[i])
        end
        kron!(C, Kron, argList[end])
    end
    return nothing
end

function otimes!(C::Matrix{ComplexF64},A::Matrix{ComplexF64},B::Matrix{ComplexF64})
    Ai,Aj = size(A)
    Bi,Bj = size(B)
    @assert size(C) == (Ai*Bi,Aj*Bj)
    # a_tmp::ComplexF64 = 0.0im
    @inbounds begin # faster with @inbounds but dangerous sine bounds checking is disabled
        for aj ∈ 1:Aj
            aBj = (aj-1)*Bj
            for ai ∈ 1:Ai
                aBi = (ai-1)*Bi
                a_tmp = A[ai,aj]
                for bj ∈ 1:Bj
                    for bi ∈ 1:Bi
                        C[aBi + bi, aBj + bj] = a_tmp * B[bi,bj]
                    end
                end
            end
        end    
    end
    
    return nothing
end
function otimes!(D::Matrix{ComplexF64},A::Matrix{ComplexF64},B::Matrix{ComplexF64},C::Matrix{ComplexF64})
    Ai,Aj = size(A)
    Bi,Bj = size(B)
    Ci,Cj = size(C)
    @assert size(D) == (Ai*Bi*Ci, Aj*Bj*Cj)
    # a_tmp::ComplexF64 = 0.0im; b_tmp::ComplexF64 = 0.0im
    @inbounds begin # faster with @inbounds but dangerous sine bounds checking is disabled
        for aj ∈ 1:Aj
            aBCj = (aj-1)*Bj*Cj
            for ai ∈ 1:Ai
                aBCi = (ai-1)*Bi*Ci
                a_tmp = A[ai,aj]
                for bj ∈ 1:Bj
                    bCj = aBCj + (bj-1)*Cj
                    for bi ∈ 1:Bi
                        bCi = aBCi + (bi-1)*Ci
                        b_tmp = a_tmp * B[bi,bj]
                        for cj ∈ 1:Cj
                            for ci ∈ 1:Ci
                                D[bCi + ci, bCj + cj] = b_tmp * C[ci,cj]
                            end
                        end
                    end
                end
            end
        end    
    end
    
    return nothing
end
function otimes!(E::Matrix{ComplexF64},A::Matrix{ComplexF64},B::Matrix{ComplexF64},C::Matrix{ComplexF64},D::Matrix{ComplexF64})
    Ai,Aj = size(A)
    Bi,Bj = size(B)
    Ci,Cj = size(C)
    Di,Dj = size(D)
    @assert size(E) == (Ai*Bi*Ci*Di, Aj*Bj*Cj*Dj)
    # a_tmp::ComplexF64 = 0.0im; b_tmp::ComplexF64 = 0.0im; c_tmp::ComplexF64 = 0.0im
    @inbounds begin # faster with @inbounds but dangerous sine bounds checking is disabled
        for aj ∈ 1:Aj
            aBCDj = (aj-1)*Bj*Cj*Dj
            for ai ∈ 1:Ai
                a_tmp = A[ai,aj]
                aBCDi = (ai-1)*Bi*Ci*Di
                for bj ∈ 1:Bj
                    bCDj = aBCDj + (bj-1)*Cj*Dj
                    for bi ∈ 1:Bi
                        bCDi = aBCDi + (bi-1)*Ci*Di
                        b_tmp = a_tmp * B[bi,bj]
                        for cj ∈ 1:Cj
                            cDj = bCDj + (cj-1)*Dj
                            for ci ∈ 1:Ci
                                cDi = bCDi + (ci-1)*Di
                                c_tmp = b_tmp * C[ci,cj]
                                for dj ∈ 1:Dj
                                    for di ∈ 1:Di
                                        E[cDi + di, cDj + dj] = c_tmp * D[di,dj]
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end    
    end
    
    return nothing
end
function otimes!(F::Matrix{ComplexF64},A::Matrix{ComplexF64},B::Matrix{ComplexF64},C::Matrix{ComplexF64},D::Matrix{ComplexF64},E::Matrix{ComplexF64})
    Ai,Aj = size(A)
    Bi,Bj = size(B)
    Ci,Cj = size(C)
    Di,Dj = size(D)
    Ei,Ej = size(E)
    @assert size(F) == (Ai*Bi*Ci*Di*Ei, Aj*Bj*Cj*Dj*Ej)

    @inbounds begin # faster with @inbounds but dangerous sine bounds checking is disabled
        for aj ∈ 1:Aj
            aBCDEj = (aj-1)*Bj*Cj*Dj*Ej
            for ai ∈ 1:Ai
                a_tmp = A[ai,aj]
                aBCDEi = (ai-1)*Bi*Ci*Di*Ei
                for bj ∈ 1:Bj
                    bCDEj = aBCDEj + (bj-1)*Cj*Dj*Ej
                    for bi ∈ 1:Bi
                        bCDEi = aBCDEi + (bi-1)*Ci*Di*Ei
                        b_tmp = a_tmp * B[bi,bj]
                        for cj ∈ 1:Cj
                            cDEj = bCDEj + (cj-1)*Dj*Ej
                            for ci ∈ 1:Ci
                                cDEi = bCDEi + (ci-1)*Di*Ei
                                c_tmp = b_tmp * C[ci,cj]
                                for dj ∈ 1:Dj
                                    dEj = cDEj + (dj-1)*Ej
                                    for di ∈ 1:Di
                                        dEi = cDEi + (di-1)*Ei
                                        d_tmp = c_tmp * D[di,dj]
                                        for ej ∈ 1:Ej
                                            for ei ∈ 1:Ei
                                                F[dEi + ei, dEj + ej] = d_tmp * E[ei,ej]
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end    
    end
    
    return nothing
end
function otimes!(G::Matrix{ComplexF64},A::Matrix{ComplexF64},B::Matrix{ComplexF64},C::Matrix{ComplexF64},D::Matrix{ComplexF64},E::Matrix{ComplexF64},F::Matrix{ComplexF64})
    Ai,Aj = size(A)
    Bi,Bj = size(B)
    Ci,Cj = size(C)
    Di,Dj = size(D)
    Ei,Ej = size(E)
    Fi,Fj = size(F)
    @assert size(G) == (Ai*Bi*Ci*Di*Ei*Fi, Aj*Bj*Cj*Dj*Ej*Fj)

    @inbounds begin # faster with @inbounds but dangerous sine bounds checking is disabled
        for aj ∈ 1:Aj
            aBCDEFj = (aj-1)*Bj*Cj*Dj*Ej*Fj
            for ai ∈ 1:Ai
                a_tmp = A[ai,aj]
                aBCDEFi = (ai-1)*Bi*Ci*Di*Ei*Fi
                for bj ∈ 1:Bj
                    bCDEFj = aBCDEFj + (bj-1)*Cj*Dj*Ej*Fj
                    for bi ∈ 1:Bi
                        bCDEFi = aBCDEFi + (bi-1)*Ci*Di*Ei*Fi
                        b_tmp = a_tmp * B[bi,bj]
                        for cj ∈ 1:Cj
                            cDEFj = bCDEFj + (cj-1)*Dj*Ej*Fj
                            for ci ∈ 1:Ci
                                cDEFi = bCDEFi + (ci-1)*Di*Ei*Fi
                                c_tmp = b_tmp * C[ci,cj]
                                for dj ∈ 1:Dj
                                    dEFj = cDEFj + (dj-1)*Ej*Fj
                                    for di ∈ 1:Di
                                        dEFi = cDEFi + (di-1)*Ei*Fi
                                        d_tmp = c_tmp * D[di,dj]
                                        for ej ∈ 1:Ej
                                            eFj = dEFj + (ej-1)*Fj
                                            for ei ∈ 1:Ei
                                                eFi = dEFi + (ei-1)*Fi
                                                e_tmp = d_tmp * E[ei,ej]
                                                for fj ∈ 1:Fj
                                                    for fi ∈ 1:Fi
                                                        G[eFi + fi, eFj + fj] = e_tmp * F[fi,fj]
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end    
    end
    
    return nothing
end
#----------------------------------------------------------------------------------------------------------------------------------------------
## Julia's reshape() and vec() is not consistent with quantum convention of kron()
"Convert vector to matrix in row-major, a lazy wrapper type for a transpose view of Matrix{Type}"
function vec2matrix(x::AbstractVector, shape::Tuple)::AbstractMatrix
    return transpose(reshape(x, reverse(shape)))
end
"Convert matrix to vector in row-major, the fastest inverse operation of vec2matrix(), using transpose() which creates a view"
function matrix2vec(x::AbstractMatrix)::AbstractVector
    return vec(transpose(x))
end
"Convert array to vector in row-major, using PermutedDimsArray() which creates a view"
function ravel(x::AbstractArray)::AbstractVector
    index = collect(ndims(x):-1:1)
    return vec(PermutedDimsArray(x,index))
end

"Reshape array following quantum convention"
function quanReshape(x::AbstractArray, shape::Tuple)
    index = collect(ndims(x):-1:1)
    invShape = reverse(shape)
    indexShape = collect(length(shape):-1:1)
    return permutedims(reshape(PermutedDimsArray(x,index),invShape),indexShape)
end
function quanReshape(x::AbstractArray, shape::AbstractVector)
    index = collect(ndims(x):-1:1)
    invShape = Tuple(reverse(shape))
    indexShape = collect(length(shape):-1:1)
    return permutedims(reshape(PermutedDimsArray(x,index),invShape),indexShape)
end
"2nd version of quanReshape()"
function quanReshape02(x::AbstractArray, shape::Tuple)
    index = collect(ndims(x):-1:1)
    invShape = reverse(shape)
    indexShape = collect(length(shape):-1:1)
    return permutedims(reshape(permutedims(x,index),invShape),indexShape)
end
function quanReshape02!(dest::AbstractArray, x::AbstractArray, shape::Tuple)
    index = collect(ndims(x):-1:1)
    invShape = reverse(shape)
    indexShape = collect(length(shape):-1:1)
    return permutedims!(dest,reshape(permutedims(x,index),invShape),indexShape)
end
function quanReshape02!(dest::AbstractArray, temp::AbstractMatrix, x::AbstractMatrix, shape::Tuple)
    index = collect(ndims(x):-1:1)
    invShape = reverse(shape)
    indexShape = collect(length(shape):-1:1)
    return permutedims!(dest,reshape(permutedims!(temp, x, index),invShape),indexShape)
end
"3rd version of quanReshape()"
function quanReshape03(x::AbstractArray, shape::Tuple)::AbstractArray
    index = collect(ndims(x):-1:1)
    invShape = reverse(shape)
    indexShape = collect(length(shape):-1:1)
    return PermutedDimsArray(reshape(PermutedDimsArray(x,index),invShape),indexShape)
end
function quanReshape03(x::AbstractArray, shape::AbstractVector)::AbstractArray
    index = collect(ndims(x):-1:1)
    invShape = Tuple(reverse(shape))
    indexShape = collect(length(shape):-1:1)
    return PermutedDimsArray(reshape(PermutedDimsArray(x,index),invShape),indexShape)
end

function quanReshapeMtoH!(Y::Array{ComplexF64,4},X::Matrix{ComplexF64}, shape::Tuple)
    @assert size(Y) == shape && size(X) == (shape[1]*shape[2],shape[3]*shape[4])
    @inbounds begin
        for l ∈ 1:shape[4]
            for k ∈ 1:shape[3]
                xj = (k-1)*shape[4] + l

                for j ∈ 1:shape[2]
                    for i ∈ 1:shape[1]
                        xi = (i-1)*shape[2] + j
                        Y[i,j,k,l] = X[xi,xj]
                    end
                end

            end
        end
    end
    return nothing
end
function quanReshapeMtoH_mul!(Y::Array{ComplexF64,4},X1::Matrix{ComplexF64},X2::Matrix{ComplexF64}, shape::Tuple)
    @assert size(Y) == shape && size(X1)[1] == shape[1]*shape[2] && size(X2)[2] == shape[3]*shape[4] && size(X1)[2] == size(X2)[1]
    @inbounds begin
        for l ∈ 1:shape[4]
            for k ∈ 1:shape[3]
                xj = (k-1)*shape[4] + l

                for j ∈ 1:shape[2]
                    for i ∈ 1:shape[1]
                        xi = (i-1)*shape[2] + j
                        s::ComplexF64=0.0im
                        for m ∈ 1:size(X1)[2]
                            s += X1[xi,m] * X2[m,xj]
                        end
                        Y[i,j,k,l] = s#X[xi,xj]
                    end
                end

            end
        end
    end
    return nothing
end

function quanReshapeMtoΛ!(Y::Array{ComplexF64,3},X::AbstractMatrix{ComplexF64}, shape::Tuple) 
    @assert size(Y) == shape && size(X) == (shape[1]*shape[2],shape[3])
    @inbounds begin
        for k ∈ 1:shape[3]
            for j ∈ 1:shape[2]
                for i ∈ 1:shape[1]
                    xi = (i-1)*shape[2] + j
                    Y[i,j,k] = X[xi,k]
                end
            end
        end
    end
    return nothing
end
# for ternary MERA's w
function quanReshapeMtoΛ!(Y::Array{ComplexF64,4},X::Matrix{ComplexF64}, shape::Tuple) 
    @assert size(Y) == shape && size(X) == (shape[1]*shape[2]*shape[3],shape[4])
    @inbounds begin
        for k ∈ 1:shape[4]
            for j ∈ 1:shape[3]
                for i ∈ 1:shape[2]
                    for h ∈ 1:shape[1]
                        xi = (h-1)*shape[2]*shape[3] + (i-1)*shape[3] + j
                        Y[h,i,j,k] = X[xi,k]
                    end
                end
            end
        end
    end
    return nothing
end

function quanReshapeMtoΛ_mul!(Y::Array{ComplexF64,3},X1::Matrix{ComplexF64},X2::Matrix{ComplexF64}, shape::Tuple) 
    @assert size(Y) == shape && size(X1)[1] == shape[1]*shape[2] && size(X2)[2] == shape[3] && size(X1)[2] == size(X2)[1]
    @inbounds begin
        for k ∈ 1:shape[3]
            for j ∈ 1:shape[2]
                for i ∈ 1:shape[1]
                    xi = (i-1)*shape[2] + j
                    s::ComplexF64=0.0im
                    for m ∈ 1:size(X1)[2]
                        s += X1[xi,m] * X2[m,k]
                    end
                    Y[i,j,k] = s#X[xi,k]
                end
            end
        end
    end
    return nothing
end
# for ternary MERA's w
function quanReshapeMtoΛ_mul!(Y::Array{ComplexF64,4},X1::Matrix{ComplexF64},X2::Matrix{ComplexF64}, shape::Tuple) 
    @assert size(Y) == shape && size(X1)[1] == shape[1]*shape[2]*shape[3] && size(X2)[2] == shape[4] && size(X1)[2] == size(X2)[1]
    @inbounds begin
        for l ∈ 1:shape[4]
            for k ∈ 1:shape[3]
                for j ∈ 1:shape[2]
                    for i ∈ 1:shape[1]
                        xi = (i-1)*shape[2]*shape[3] + (j-1)*shape[3] + k
                        s::ComplexF64=0.0im
                        for m ∈ 1:size(X1)[2]
                            s += X1[xi,m] * X2[m,l]
                        end
                        Y[i,j,k,l] = s#X[xi,l]
                    end
                end
            end
        end
    end
    return nothing
end

"""ambiguous for ternary MERA'w"""
function quanReshapeHΛtoM!(Y::Matrix{ComplexF64},X::Array{ComplexF64,4}, shape::Tuple)
    @assert size(Y) == shape && prod(size(X)) == shape[1]*shape[2]
    @inbounds begin
        for l ∈ 1:size(X)[4]
            for k ∈ 1:size(X)[3]
                xj = (k-1)*size(X)[4] + l

                for j ∈ 1:size(X)[2]
                    for i ∈ 1:size(X)[1]
                        xi = (i-1)*size(X)[2] + j
                        Y[xi,xj] = X[i,j,k,l]
                    end
                end

            end
        end
    end
    return nothing
end
function quanReshapeHΛtoM!(Y::Matrix{ComplexF64},X::Array{ComplexF64,3}, shape::Tuple)
    @assert size(Y) == shape && prod(size(X)) == shape[1]*shape[2]
    @inbounds begin
        for k ∈ 1:size(X)[3]
            for j ∈ 1:size(X)[2]
                for i ∈ 1:size(X)[1]
                    xi = (i-1)*size(X)[2] + j
                    Y[xi,k] = X[i,j,k]
                end
            end
        end
    end
    return nothing
end

function quanReshapeHΛtoM_transpose!(Y::Matrix{ComplexF64},X::Array{ComplexF64,4}, shape::Tuple)
    @assert size(Y) == shape && prod(size(X)) == shape[2]*shape[1]
    @inbounds begin
        for l ∈ 1:size(X)[4]
            for k ∈ 1:size(X)[3]
                xj = (k-1)*size(X)[4] + l

                for j ∈ 1:size(X)[2]
                    for i ∈ 1:size(X)[1]
                        xi = (i-1)*size(X)[2] + j
                        Y[xj,xi] = X[i,j,k,l] # transpose
                    end
                end

            end
        end
    end
    return nothing
end
function quanReshapeHΛtoM_transpose!(Y::Matrix{ComplexF64},X::Array{ComplexF64,3}, shape::Tuple)
    @assert size(Y) == shape && prod(size(X)) == shape[2]*shape[1]
    @inbounds begin
        for k ∈ 1:size(X)[3]
            for j ∈ 1:size(X)[2]
                for i ∈ 1:size(X)[1]
                    xi = (i-1)*size(X)[2] + j
                    Y[k,xi] = X[i,j,k] # transpose
                end
            end
        end
    end
    return nothing
end

"""
Drift a matrix by vitual shape as below (in row-major)

    |^|       quanReshape    1 ——|^|—— 3   drift    2 ——|^|—— 4   quanReshape         |^|
1 ——|m|—— 2  ------------->      |m|      ------->      |m|      ------------->  1' ——|m|—— 2'
    |_|                      2 ——|_|—— 4            1 ——|_|—— 3                       |_|

"""
function quanDrift(X::Matrix{ComplexF64}, shape::Tuple)::Matrix{ComplexF64}
    @assert length(shape) == 4 && prod(size(X)) == shape[1]*shape[2]*shape[3]*shape[4]
    p::Int = 0; q::Int = 0; m::Int = 0; n::Int = 0
    Y::Matrix{ComplexF64} = similar(X)
    @inbounds begin
        for k ∈ 1:shape[3]
            for l ∈ 1:shape[4]
                q = q + 1
                n = (l-1)*shape[3] + k

                # Seting p and m as inner loop can be faster, since Julia uses the column-major order
                p = 0
                for i ∈ 1:shape[1]
                    for j ∈ 1:shape[2]
                        p = p + 1
                        m = (j-1)*shape[1] + i
                        Y[m,n] = X[p,q]
                    end
                end
            end
        end
    end
    return Y
end
#---
function quanDrift!(Y::Matrix{ComplexF64}, X::Matrix{ComplexF64}, shape::Tuple)
    @assert length(shape) == 4 && sizeof(X) == sizeof(Y) && prod(size(X)) == shape[1]*shape[2]*shape[3]*shape[4]
    p::Int = 0; q::Int = 0; m::Int = 0; n::Int = 0
    #Y::Matrix{ComplexF64} = similar(X)
    @inbounds begin
        for k ∈ 1:shape[3]
            for l ∈ 1:shape[4]
                q = q + 1
                n = (l-1)*shape[3] + k

                # Seting p and m as inner loop can be faster, since Julia uses the column-major order
                p = 0
                for i ∈ 1:shape[1]
                    for j ∈ 1:shape[2]
                        p = p + 1
                        m = (j-1)*shape[1] + i
                        Y[m,n] = X[p,q]
                    end
                end
            end
        end
    end
    return nothing
end
function quanDrift01(m::Matrix{ComplexF64}, shape::Tuple)::Matrix{ComplexF64}
    @assert length(shape) == 4
    return permutedims(reshape(PermutedDimsArray(reshape(transpose(m),reverse(shape)),(2,1,4,3)),(shape[3]*shape[2],:)))
end
function quanDrift02(m::Matrix{ComplexF64}, shape::Tuple)::Matrix{ComplexF64}
    @assert length(shape) == 4
    return permutedims(reshape(permutedims(reshape(transpose(m),reverse(shape)),(2,1,4,3)),(shape[3]*shape[2],:)))
end


function mulMy!(Z::AbstractMatrix,X::AbstractMatrix,Y::AbstractMatrix) # slower than mul!() if X/Y is large
    mX,nX = size(X); mY,nY = size(Y); mZ,nZ = size(Z)
    @assert nX == mY && mZ == mX && nZ == nY
    @inbounds begin
        for j ∈ 1:nZ
            z_index = (j-1)*mZ
            for i ∈ 1:mZ
                s = zero(eltype(Z))
                for k ∈ 1:nX
                    s += X[i,k]*Y[k+z_index]
                end
                Z[i+z_index] = s
            end
        end
    end 
    return nothing
end

#----------------------------------------------------------------------------------------------------------------------------------------------
# CNOT
# index     index 
# 1 —— · —— 3   control qubit
#      |  
# 2 —— + —— 4   targeted qubit
CNOT = [1 0 0 0;
        0 1 0 0;
        0 0 0 1;
        0 0 1 0]
# CNOT_flip
# index     index 
# 1 —— + —— 3
#      |  
# 2 —— . —— 4
CNOT_flip = [1 0 0 0;
             0 0 0 1;
             0 0 1 0;
             0 1 0 0]

SWAP = [1 0 0 0;
        0 0 1 0;
        0 1 0 0;
        0 0 0 1]

# offDiag = kron(sigmaX,sigmaX)
OffDiag = [0 0 0 1;
           0 0 1 0;
           0 1 0 0;
           1 0 0 0]

setBellstate = CNOT * kron(Hadamard,sigmaI); # 00 --> 00+11, 01 --> 01+10, 10 --> 00-11, 11 --> 01-10
setSinglet = setBellstate * OffDiag # if given |00> state
#=
# super SWAP
# a = np.eye(8)
# (a.reshape(2,2,2,2,2,2).transpose(2,1,0,3,4,5)).reshape(8,8)
array([[1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1.]])

# a = np.eye(16)
# (a.reshape(2,2,2,2,2,2,2,2).transpose(3,2,1,0,4,5,6,7)).reshape(16,16)
array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
=#


function get_ξL(θL::Vector{Vector{Float64}})::Vector{Matrix{ComplexF64}}
    ξL = Vector{Matrix{ComplexF64}}(undef, 0)
    for i ∈ 1:length(θL)
        if length(θL[i]) == 1
            push!(ξL,setXY(θL[i]))
        elseif length(θL[i]) == 2
            push!(ξL,setYXXY(θL[i]))
        elseif length(θL[i]) == 4
            push!(ξL,setyyYXXY(θL[i]))

        elseif length(θL[i]) == 3
            push!(ξL,setzzXX(θL[i]))

        elseif length(θL[i]) == 7
            push!(ξL,setXX_n(θL[i]))
        elseif length(θL[i]) == 11
            push!(ξL,setXX_x(θL[i]))

        elseif length(θL[i]) == 8
            push!(ξL,setYYXX_n(θL[i]))
        elseif length(θL[i]) == 12
            push!(ξL,setYYXX_x(θL[i]))
        elseif length(θL[i]) == 14
            push!(ξL,setYYXX_H(θL[i]))

        elseif length(θL[i]) == 9
            push!(ξL,setCAN_n(θL[i]))
        elseif length(θL[i]) == 13
            push!(ξL,setCAN_x(θL[i]))
        elseif length(θL[i]) == 15
            push!(ξL,setCAN_H(θL[i]))
        end
    end
    return ξL
end
function get_ξL!(ξL::AbstractVector{Matrix{ComplexF64}},θL::AbstractVector{Vector{Float64}})
    for i ∈ 1:length(θL)

        if length(θL[i]) == 1
            setXY!(ξL[i], θL[i][1])
        elseif length(θL[i]) == 2
            setYXXY!(ξL[i], θL[i][1], θL[i][2])
        elseif length(θL[i]) == 4
            setyyYXXY!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4])

        # XX type
        elseif length(θL[i]) == 3
            setzzXX!(ξL[i], θL[i][1], θL[i][2], θL[i][3])
        elseif length(θL[i]) == 7
            setXX_n!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4], θL[i][5], θL[i][6], θL[i][7])
        elseif length(θL[i]) == 11
            setXX_x!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4], θL[i][5], θL[i][6], θL[i][7], θL[i][8], θL[i][9], 
                θL[i][10], θL[i][11])

        # YYXX type
        # elseif length(θL[i]) == 4
        #     setzzYYXX!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4])
        elseif length(θL[i]) == 8
            setYYXX_n!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4], θL[i][5], θL[i][6], θL[i][7], θL[i][8])
        elseif length(θL[i]) == 12
            setYYXX_x!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4], θL[i][5], θL[i][6], θL[i][7], θL[i][8], θL[i][9], 
                θL[i][10], θL[i][11], θL[i][12])
        elseif length(θL[i]) == 14
            setYYXX_H!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4], θL[i][5], θL[i][6], θL[i][7], θL[i][8], θL[i][9], 
                θL[i][10], θL[i][11], θL[i][12], θL[i][13], θL[i][14])

        # CAN type
        elseif length(θL[i]) == 5
            #setzzCAN!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4], θL[i][5])
        elseif length(θL[i]) == 9
            setCAN_n!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4], θL[i][5], θL[i][6], θL[i][7], θL[i][8], θL[i][9])
        elseif length(θL[i]) == 13
            setCAN_x!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4], θL[i][5], θL[i][6], θL[i][7], θL[i][8], θL[i][9], 
                θL[i][10], θL[i][11], θL[i][12], θL[i][13])
        elseif length(θL[i]) == 15
            setCAN_H!(ξL[i], θL[i][1], θL[i][2], θL[i][3], θL[i][4], θL[i][5], θL[i][6], θL[i][7], θL[i][8], θL[i][9], 
                θL[i][10], θL[i][11], θL[i][12], θL[i][13], θL[i][14], θL[i][15])
        end
    end
    return nothing
end


#----------------------------------------------------------------------------------------------------------------------------------------------
## XY-scheme (native gates + free Rz gate on DQC's device)
"""
two qubit block (argList = [arg1])

                             | quantum circuit (from right to left)
 —— · ——————————— · ——       | |psi>
    |   XY(arg1)  |          |
 —— · ——————————— · ——       | |psi>

eyeYXXY([0., 0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setXY(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    # square type       
    # 1 ————— 3  
    #    | |       
    # 2 ————— 4
    #
    return XY(Args[1])
end
eyeXY = [0.]
function setXY!(X::Matrix{ComplexF64},x::Float64)
    x = x/2
    sinθhalf, cosθhalf = sincospi(x) # sincospi(x) Computes sin(πx) and cos(πx)

    X[1,1] = cosθhalf;
    X[1,2] = 0;
    X[1,3] = 0; 
    X[1,4] = -sinθhalf; 
    
    X[2,1] = 0;
    X[2,2] = cosθhalf;
    X[2,3] = sinθhalf;
    X[2,4] = 0; 
    
    X[3,1] = 0;
    X[3,2] = -sinθhalf;
    X[3,3] = cosθhalf;
    X[3,4] = 0; 
    
    X[4,1] = sinθhalf;
    X[4,2] = 0;
    X[4,3] = 0+0.0im;
    X[4,4] = cosθhalf;

    return nothing
end
function traceXY(env::Matrix{ComplexF64}, x::Float64)::ComplexF64
    x = x/2
    sinθhalf, cosθhalf = sincospi(x) 

    X11 = cosθhalf;
    X14 = -sinθhalf; 
    
    X22 = cosθhalf;
    X23 = sinθhalf;

    X32 = -sinθhalf;
    X33 = cosθhalf;

    X41 = sinθhalf;
    X44 = cosθhalf;

    trEnvTwoQubitGate = env[1,1]*X11                               + env[1,4]*X41 + 
                                       env[2,2]*X22 + env[2,3]*X32                + 
                                       env[3,2]*X23 + env[3,3]*X33                + 
                        env[4,1]*X14                               + env[4,4]*X44

    return trEnvTwoQubitGate::ComplexF64
end
## YXXY-scheme (native gates + free Rz gate on DQC's device)
"""
two qubit block (argList = [arg1, arg2])

                                      | quantum circuit (from right to left)
 —— · ———————————————————— · ——       | |psi>
    |  YX(arg1)  XY(arg2)  |          |
 —— · ———————————————————— · ——       | |psi>

eyeYXXY([0., 0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setYXXY(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    # square type       
    # 1 ————— 3  
    #    | |       
    # 2 ————— 4
    #
    return YXXY(Args[1], Args[2])
end
eyeYXXY = [0., 0.]
function setYXXY!(X::Matrix{ComplexF64},x1::Float64,x2::Float64)

    xp = (x1+x2)/2; xm = (x2-x1)/2
    sinθPhalf, cosθPhalf = sincospi(xp) # sincospi(x) Computes sin(πx) and cos(πx)
    sinθMhalf, cosθMhalf = sincospi(xm)

    X[1,1] = cosθPhalf;
    X[1,2] = 0;
    X[1,3] = 0; 
    X[1,4] = -sinθPhalf; 
    
    X[2,1] = 0;
    X[2,2] = cosθMhalf;
    X[2,3] = sinθMhalf;
    X[2,4] = 0; 
    
    X[3,1] = 0;
    X[3,2] = -sinθMhalf;
    X[3,3] = cosθMhalf;
    X[3,4] = 0; 
    
    X[4,1] = sinθPhalf;
    X[4,2] = 0;
    X[4,3] = 0+0.0im;
    X[4,4] = cosθPhalf;

    return nothing
end
function traceYXXY(env::Matrix{ComplexF64}, x1::Float64,x2::Float64)::ComplexF64

    xp = (x1+x2)/2; xm = (x2-x1)/2
    sinθPhalf, cosθPhalf = sincospi(xp) 
    sinθMhalf, cosθMhalf = sincospi(xm)

    X11 = cosθPhalf;
    X14 = -sinθPhalf; 
    
    X22 = cosθMhalf;
    X23 = sinθMhalf;

    X32 = -sinθMhalf;
    X33 = cosθMhalf;

    X41 = sinθPhalf;
    X44 = cosθPhalf;

    trEnvTwoQubitGate = env[1,1]*X11                               + env[1,4]*X41 + 
                                       env[2,2]*X22 + env[2,3]*X32                + 
                                       env[3,2]*X23 + env[3,3]*X33                + 
                        env[4,1]*X14                               + env[4,4]*X44

    return trEnvTwoQubitGate::ComplexF64
end

"""
two qubit block of n type (argList = [y1, y2, arg1, arg2])

                                               | quantum circuit (from right to left)
—— Ry(y1) —— · ———————————————————— · ——       | |psi>
             |  YX(arg1)  XY(arg2)  |          |
—— Ry(y2) —— · ———————————————————— · ——       | |psi>

eyeYXXY([0., 0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setyyYXXY(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    # square type       
    # 1 ——————— 3  
    #      | |       
    # 2 ——————— 4
    #
    return kron(Ry(Args[1]), Ry(Args[2])) * YXXY(Args[3], Args[4])
end
eyeyyYXXY = [0., 0., 0., 0.]
function setyyYXXY!(X::Matrix{ComplexF64},y1::Float64,y2::Float64, x1::Float64,x2::Float64)

    sinθ1half, cosθ1half = sincospi(y1/2)
    sinθ2half, cosθ2half = sincospi(y2/2)

    xp = (x1+x2)/2; xm = (x2-x1)/2
    sinθPhalf, cosθPhalf = sincospi(xp) # sincospi(x) Computes sin(πx) and cos(πx)
    sinθMhalf, cosθMhalf = sincospi(xm)

    X[1,1] = cosθ1half*cosθ2half*cosθPhalf + sinθ1half*sinθ2half*sinθPhalf;
    X[1,2] = -cosθ1half*sinθ2half*cosθMhalf + sinθ1half*cosθ2half*sinθMhalf;
    X[1,3] = -sinθ1half*cosθ2half*cosθMhalf - cosθ1half*sinθ2half*sinθMhalf; 
    X[1,4] = sinθ1half*sinθ2half*cosθPhalf - cosθ1half*cosθ2half*sinθPhalf; 
    
    X[2,1] = cosθ1half*sinθ2half*cosθPhalf - sinθ1half*cosθ2half*sinθPhalf;
    X[2,2] = cosθ1half*cosθ2half*cosθMhalf + sinθ1half*sinθ2half*sinθMhalf;
    X[2,3] = -sinθ1half*sinθ2half*cosθMhalf + cosθ1half*cosθ2half*sinθMhalf;
    X[2,4] = -sinθ1half*cosθ2half*cosθPhalf - cosθ1half*sinθ2half*sinθPhalf; 
    
    X[3,1] = sinθ1half*cosθ2half*cosθPhalf - cosθ1half*sinθ2half*sinθPhalf;
    X[3,2] = -sinθ1half*sinθ2half*cosθMhalf - cosθ1half*cosθ2half*sinθMhalf;
    X[3,3] = cosθ1half*cosθ2half*cosθMhalf - sinθ1half*sinθ2half*sinθMhalf;
    X[3,4] = -cosθ1half*sinθ2half*cosθPhalf - sinθ1half*cosθ2half*sinθPhalf; 
    
    X[4,1] = sinθ1half*sinθ2half*cosθPhalf + cosθ1half*cosθ2half*sinθPhalf;
    X[4,2] = sinθ1half*cosθ2half*cosθMhalf - cosθ1half*sinθ2half*sinθMhalf;
    X[4,3] = cosθ1half*sinθ2half*cosθMhalf + sinθ1half*cosθ2half*sinθMhalf;
    X[4,4] = cosθ1half*cosθ2half*cosθPhalf - sinθ1half*sinθ2half*sinθPhalf;

    return nothing
end
function traceyyYXXY(env::Matrix{ComplexF64}, y1::Float64,y2::Float64, x1::Float64,x2::Float64)::ComplexF64

    sinθ1half, cosθ1half = sincospi(y1/2)
    sinθ2half, cosθ2half = sincospi(y2/2)

    xp = (x1+x2)/2; xm = (x2-x1)/2
    sinθPhalf, cosθPhalf = sincospi(xp) # sincospi(x) Computes sin(πx) and cos(πx)
    sinθMhalf, cosθMhalf = sincospi(xm)

    X11 = cosθ1half*cosθ2half*cosθPhalf + sinθ1half*sinθ2half*sinθPhalf;
    X12 = -cosθ1half*sinθ2half*cosθMhalf + sinθ1half*cosθ2half*sinθMhalf;
    X13 = -sinθ1half*cosθ2half*cosθMhalf - cosθ1half*sinθ2half*sinθMhalf; 
    X14 = sinθ1half*sinθ2half*cosθPhalf - cosθ1half*cosθ2half*sinθPhalf; 
    
    X21 = cosθ1half*sinθ2half*cosθPhalf - sinθ1half*cosθ2half*sinθPhalf;
    X22 = cosθ1half*cosθ2half*cosθMhalf + sinθ1half*sinθ2half*sinθMhalf;
    X23 = -sinθ1half*sinθ2half*cosθMhalf + cosθ1half*cosθ2half*sinθMhalf;
    X24 = -sinθ1half*cosθ2half*cosθPhalf - cosθ1half*sinθ2half*sinθPhalf; 
    
    X31 = sinθ1half*cosθ2half*cosθPhalf - cosθ1half*sinθ2half*sinθPhalf;
    X32 = -sinθ1half*sinθ2half*cosθMhalf - cosθ1half*cosθ2half*sinθMhalf;
    X33 = cosθ1half*cosθ2half*cosθMhalf - sinθ1half*sinθ2half*sinθMhalf;
    X34 = -cosθ1half*sinθ2half*cosθPhalf - sinθ1half*cosθ2half*sinθPhalf; 
    
    X41 = sinθ1half*sinθ2half*cosθPhalf + cosθ1half*cosθ2half*sinθPhalf;
    X42 = sinθ1half*cosθ2half*cosθMhalf - cosθ1half*sinθ2half*sinθMhalf;
    X43 = cosθ1half*sinθ2half*cosθMhalf + sinθ1half*cosθ2half*sinθMhalf;
    X44 = cosθ1half*cosθ2half*cosθPhalf - sinθ1half*sinθ2half*sinθPhalf;

    trEnvTwoQubitGate = env[1,1]*X11 + env[1,2]*X21 + env[1,3]*X31 + env[1,4]*X41 + 
                        env[2,1]*X12 + env[2,2]*X22 + env[2,3]*X32 + env[2,4]*X42 + 
                        env[3,1]*X13 + env[3,2]*X23 + env[3,3]*X33 + env[3,4]*X43 + 
                        env[4,1]*X14 + env[4,2]*X24 + env[4,3]*X34 + env[4,4]*X44

    return trEnvTwoQubitGate::ComplexF64
end

#----------------------------------------------------------------------------------------------------------------------------------------------
## XX-scheme
"""
two qubit block of u type (argList = [arg1, arg2,arg3,arg4, arg5,arg6,arg7])

                                                                | quantum circuit (from right to left)
 —— · —————————— · —— Rz(arg2) —— Ry(arg3) —— Rz(arg4) ——       | |psi>
    |  XX(arg1)  |                                              |
 —— · —————————— · —— Rz(arg5) —— Ry(arg6) —— Rz(arg7) ——       | |psi>

setXX_u([0., 0.,0.,0., 0.,0.,0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setXX_u(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #   u type       
    # 1 -—————— 3  
    #    | |       
    # 2 -—————— 4
    #
    unitary = kron(ZYZ_gate(Args[2:4]), ZYZ_gate(Args[5:7]))

    return XX(Args[1]) * unitary
end
eyeXX_u = [0., 0.,0.,0., 0.,0.,0.]

#----------------------------------------------------------------------------------------------------------------------------------------------
## zzXX-scheme (native gates + free Rz gate on DQC's device)
"""
two qubit block of n type (argList = [arg1, arg4, arg9])

                                     | quantum circuit (from right to left)
 —— Rz(arg1) · —————————— · ——       | |psi>
             |  XX(arg9)  |          |
 —— Rz(arg4) · —————————— · ——       | |psi>

eyezzXX([0., 0., 0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setzzXX(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #   n type       
    # 1 -—————— 3  
    #      | |       
    # 2 -—————— 4
    #
    unitary0 = kron(Rz(Args[1]), Rz(Args[2]))

    return unitary0* XX(Args[3])
end
eyezzXX = [0., 0., 0.]
function setzzXX!(X::Matrix{ComplexF64},arg1::Float64,arg4::Float64, arg9::Float64)

    α1 = (arg1)/2; α2 = (arg4)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mα2half = cispi(-α2); pα2half = cispi(α2)

    φ = (0 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (0 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X[1,1] = cosφhalf * mα1half*mα2half; 
    X[1,2] = 0; 
    X[1,3] = 0; 
    X[1,4] = 1im*sinφhalf * mα1half*mα2half;
    
    X[2,1] = 0; 
    X[2,2] = cosϕhalf * mα1half*pα2half; 
    X[2,3] = -1im*sinϕhalf * mα1half*pα2half; 
    X[2,4] = 0;

    X[3,1] = 0; 
    X[3,2] = - 1im*sinϕhalf * mα2half*pα1half; 
    X[3,3] = cosϕhalf * mα2half*pα1half; 
    X[3,4] = 0;

    X[4,1] = 1im*sinφhalf * pα1half*pα2half; 
    X[4,2] = 0; 
    X[4,3] = 0; 
    X[4,4] = cosφhalf * pα1half*pα2half;

    return nothing
end

function tracezzXX(env::Matrix{ComplexF64},arg1::Float64, arg4::Float64, arg9::Float64)::ComplexF64

    α1 = (arg1)/2; α2 = (arg4)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mα2half = cispi(-α2); pα2half = cispi(α2)

    φ = (0 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (0 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X11 = cosφhalf * mα1half*mα2half;  
    X14 = 1im*sinφhalf * mα1half*mα2half;
    
    X22 = cosϕhalf * mα1half*pα2half; 
    X23 = -1im*sinϕhalf * mα1half*pα2half; 

    X32 = - 1im*sinϕhalf * mα2half*pα1half; 
    X33 = cosϕhalf * mα2half*pα1half; 

    X41 = 1im*sinφhalf * pα1half*pα2half; 
    X44 = cosφhalf * pα1half*pα2half;

    trEnvTwoQubitGate = env[1,1]*X11                               + env[1,4]*X41 + 
                                       env[2,2]*X22 + env[2,3]*X32                + 
                                       env[3,2]*X23 + env[3,3]*X33                + 
                        env[4,1]*X14                               + env[4,4]*X44

    return trEnvTwoQubitGate::ComplexF64
end

"""
two qubit block of n type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg7])

                                                                | quantum circuit (from right to left)
 —— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— · —————————— · ——       | |psi>
                                        |  XX(arg7)  |          |
 —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · —————————— · ——       | |psi>

setXX_n([0.,0.,0., 0.,0.,0., 0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setXX_n(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #   n type       
    # 1 -—————— 3  
    #      | |       
    # 2 -—————— 4
    #
    unitary = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))

    return unitary * XX(Args[7])
end
eyeXX_n = [0.,0.,0., 0.,0.,0., 0.]
function setXX_n!(X::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64, arg9::Float64)

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    ϕ = (arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X[1,1] = cosϕhalf * mα1half*mα2half*cosβ1half*cosβ2half - 1im*sinϕhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    X[1,2] = -cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X[1,3] = 1im*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X[1,4] = -1im*sinϕhalf * mα1half*mα2half*cosβ1half*cosβ2half + cosϕhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    X[2,1] = cosϕhalf * mα1half*pγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    X[2,2] = cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X[2,3] = -1im*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X[2,4] = -1im*sinϕhalf * mα1half*pγ2half*cosβ1half*sinβ2half - cosϕhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    X[3,1] = cosϕhalf * mα2half*pγ1half*cosβ2half*sinβ1half + 1im*sinϕhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    X[3,2] = -cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X[3,3] = 1im*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X[3,4] = -1im*sinϕhalf * mα2half*pγ1half*cosβ2half*sinβ1half - cosϕhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    X[4,1] = cosϕhalf * pγ1half*pγ2half*sinβ1half*sinβ2half - 1im*sinϕhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    X[4,2] = cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X[4,3] = -1im*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X[4,4] = -1im*sinϕhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + cosϕhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    return nothing
end

function traceXX_n(env::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64, arg9::Float64)::ComplexF64

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    ϕ = (arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X11 = cosϕhalf * mα1half*mα2half*cosβ1half*cosβ2half - 1im*sinϕhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    X12 = -cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X13 = 1im*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X14 = -1im*sinϕhalf * mα1half*mα2half*cosβ1half*cosβ2half + cosϕhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    X21 = cosϕhalf * mα1half*pγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    X22 = cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X23 = -1im*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X24 = -1im*sinϕhalf * mα1half*pγ2half*cosβ1half*sinβ2half - cosϕhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    X31 = cosϕhalf * mα2half*pγ1half*cosβ2half*sinβ1half + 1im*sinϕhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    X32 = -cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X33 = 1im*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X34 = -1im*sinϕhalf * mα2half*pγ1half*cosβ2half*sinβ1half - cosϕhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    X41 = cosϕhalf * pγ1half*pγ2half*sinβ1half*sinβ2half - 1im*sinϕhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    X42 = cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X43 = -1im*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X44 = -1im*sinϕhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + cosϕhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    trEnvTwoQubitGate = env[1,1]*X11 + env[1,2]*X21 + env[1,3]*X31 + env[1,4]*X41 + 
                        env[2,1]*X12 + env[2,2]*X22 + env[2,3]*X32 + env[2,4]*X42 + 
                        env[3,1]*X13 + env[3,2]*X23 + env[3,3]*X33 + env[3,4]*X43 + 
                        env[4,1]*X14 + env[4,2]*X24 + env[4,3]*X34 + env[4,4]*X44

    return trEnvTwoQubitGate::ComplexF64
end


function setMS_n(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #   n type       
    # 1 -—————— 3  
    #      | |       
    # 2 -—————— 4
    #
    unitary = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))

    return unitary * XX(0.5)
end


"""
two qubit block of x type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg9, arg10,arg11, arg13,arg14])

                                                                                          | quantum circuit (from right to left)
 —— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— · —————————— · —— Rz(arg10) —— Ry(arg11) ——       | |0>
                                        |  XX(arg9)  |                                    |
 —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · —————————— · —— Rz(arg13) —— Ry(arg14) ——       | |0>

setYYXX_x([0.,0.,0., 0.,0.,0., 0., 0.,0., 0.,0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setXX_x(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #     x type       
    # 1 ———————— 3  
    #      | |       
    # 2 ———————— 4
    #
    unitary0 = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))
    unitary1 = kron(ZY_gate(Args[8:9]), ZY_gate(Args[10:11]))

    return unitary0* XX(Args[7])* unitary1
end
eyeXX_x = [0.,0.,0., 0.,0.,0., 0., 0.,0., 0.,0.]
function setXX_x!(X::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64, arg9::Float64,
    arg10::Float64,arg11::Float64, arg13::Float64,arg14::Float64)

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    ϕ = (arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    L11 = cosϕhalf * mα1half*mα2half*cosβ1half*cosβ2half - 1im*sinϕhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    L12 = -cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L13 = 1im*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L14 = -1im*sinϕhalf * mα1half*mα2half*cosβ1half*cosβ2half + cosϕhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    L21 = cosϕhalf * mα1half*pγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    L22 = cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L23 = -1im*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L24 = -1im*sinϕhalf * mα1half*pγ2half*cosβ1half*sinβ2half - cosϕhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    L31 = cosϕhalf * mα2half*pγ1half*cosβ2half*sinβ1half + 1im*sinϕhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    L32 = -cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L33 = 1im*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L34 = -1im*sinϕhalf * mα2half*pγ1half*cosβ2half*sinβ1half - cosϕhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    L41 = cosϕhalf * pγ1half*pγ2half*sinβ1half*sinβ2half - 1im*sinϕhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    L42 = cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L43 = -1im*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L44 = -1im*sinϕhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + cosϕhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    α1 = (arg10)/2; α2 = (arg13)/2
    β1 = arg11/2; β2 = arg14/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    sinβ2half, cosβ2half = sincospi(β2)

    R11 = mα1half*mα2half*cosβ1half*cosβ2half; R12 = -mα1half*mα2half*cosβ1half*sinβ2half; R13 = -mα2half*mα1half*cosβ2half*sinβ1half; R14 = mα1half*mα2half*sinβ1half*sinβ2half;
    R21 = mα1half*pα2half*cosβ1half*sinβ2half; R22 = mα1half*pα2half*cosβ1half*cosβ2half; R23 = -mα1half*pα2half*sinβ1half*sinβ2half; R24 = -mα1half*pα2half*cosβ2half*sinβ1half;
    R31 = mα2half*pα1half*cosβ2half*sinβ1half; R32 = -mα2half*pα1half*sinβ1half*sinβ2half; R33 = mα2half*pα1half*cosβ1half*cosβ2half; R34 = -mα2half*pα1half*cosβ1half*sinβ2half;
    R41 = pα1half*pα2half*sinβ1half*sinβ2half; R42 = pα2half*pα1half*cosβ2half*sinβ1half; R43 = pα1half*pα2half*cosβ1half*sinβ2half; R44 = pα1half*pα2half*cosβ1half*cosβ2half;
    
    X[1,1] = L11*R11 + L12*R21 + L13*R31 + L14*R41;
    X[1,2] = L11*R12 + L12*R22 + L13*R32 + L14*R42;
    X[1,3] = L11*R13 + L12*R23 + L13*R33 + L14*R43;
    X[1,4] = L11*R14 + L12*R24 + L13*R34 + L14*R44;

    X[2,1] = L21*R11 + L22*R21 + L23*R31 + L24*R41;
    X[2,2] = L21*R12 + L22*R22 + L23*R32 + L24*R42;
    X[2,3] = L21*R13 + L22*R23 + L23*R33 + L24*R43;
    X[2,4] = L21*R14 + L22*R24 + L23*R34 + L24*R44;

    X[3,1] = L31*R11 + L32*R21 + L33*R31 + L34*R41;
    X[3,2] = L31*R12 + L32*R22 + L33*R32 + L34*R42;
    X[3,3] = L31*R13 + L32*R23 + L33*R33 + L34*R43;
    X[3,4] = L31*R14 + L32*R24 + L33*R34 + L34*R44;

    X[4,1] = L41*R11 + L42*R21 + L43*R31 + L44*R41;
    X[4,2] = L41*R12 + L42*R22 + L43*R32 + L44*R42;
    X[4,3] = L41*R13 + L42*R23 + L43*R33 + L44*R43;
    X[4,4] = L41*R14 + L42*R24 + L43*R34 + L44*R44;

    return nothing
end
function traceXX_x(env::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64, arg9::Float64,
    arg10::Float64,arg11::Float64, arg13::Float64,arg14::Float64)::ComplexF64

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    ϕ = (arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    L11 = cosϕhalf * mα1half*mα2half*cosβ1half*cosβ2half - 1im*sinϕhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    L12 = -cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L13 = 1im*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L14 = -1im*sinϕhalf * mα1half*mα2half*cosβ1half*cosβ2half + cosϕhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    L21 = cosϕhalf * mα1half*pγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    L22 = cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L23 = -1im*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L24 = -1im*sinϕhalf * mα1half*pγ2half*cosβ1half*sinβ2half - cosϕhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    L31 = cosϕhalf * mα2half*pγ1half*cosβ2half*sinβ1half + 1im*sinϕhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    L32 = -cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L33 = 1im*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L34 = -1im*sinϕhalf * mα2half*pγ1half*cosβ2half*sinβ1half - cosϕhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    L41 = cosϕhalf * pγ1half*pγ2half*sinβ1half*sinβ2half - 1im*sinϕhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    L42 = cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L43 = -1im*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L44 = -1im*sinϕhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + cosϕhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    α1 = (arg10)/2; α2 = (arg13)/2
    β1 = arg11/2; β2 = arg14/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    sinβ2half, cosβ2half = sincospi(β2)

    R11 = mα1half*mα2half*cosβ1half*cosβ2half; R12 = -mα1half*mα2half*cosβ1half*sinβ2half; R13 = -mα2half*mα1half*cosβ2half*sinβ1half; R14 = mα1half*mα2half*sinβ1half*sinβ2half;
    R21 = mα1half*pα2half*cosβ1half*sinβ2half; R22 = mα1half*pα2half*cosβ1half*cosβ2half; R23 = -mα1half*pα2half*sinβ1half*sinβ2half; R24 = -mα1half*pα2half*cosβ2half*sinβ1half;
    R31 = mα2half*pα1half*cosβ2half*sinβ1half; R32 = -mα2half*pα1half*sinβ1half*sinβ2half; R33 = mα2half*pα1half*cosβ1half*cosβ2half; R34 = -mα2half*pα1half*cosβ1half*sinβ2half;
    R41 = pα1half*pα2half*sinβ1half*sinβ2half; R42 = pα2half*pα1half*cosβ2half*sinβ1half; R43 = pα1half*pα2half*cosβ1half*sinβ2half; R44 = pα1half*pα2half*cosβ1half*cosβ2half;
    
    X11 = L11*R11 + L12*R21 + L13*R31 + L14*R41;
    X12 = L11*R12 + L12*R22 + L13*R32 + L14*R42;
    X13 = L11*R13 + L12*R23 + L13*R33 + L14*R43;
    X14 = L11*R14 + L12*R24 + L13*R34 + L14*R44;

    X21 = L21*R11 + L22*R21 + L23*R31 + L24*R41;
    X22 = L21*R12 + L22*R22 + L23*R32 + L24*R42;
    X23 = L21*R13 + L22*R23 + L23*R33 + L24*R43;
    X24 = L21*R14 + L22*R24 + L23*R34 + L24*R44;

    X31 = L31*R11 + L32*R21 + L33*R31 + L34*R41;
    X32 = L31*R12 + L32*R22 + L33*R32 + L34*R42;
    X33 = L31*R13 + L32*R23 + L33*R33 + L34*R43;
    X34 = L31*R14 + L32*R24 + L33*R34 + L34*R44;

    X41 = L41*R11 + L42*R21 + L43*R31 + L44*R41;
    X42 = L41*R12 + L42*R22 + L43*R32 + L44*R42;
    X43 = L41*R13 + L42*R23 + L43*R33 + L44*R43;
    X44 = L41*R14 + L42*R24 + L43*R34 + L44*R44;

    trEnvTwoQubitGate = env[1,1]*X11 + env[1,2]*X21 + env[1,3]*X31 + env[1,4]*X41 + 
                        env[2,1]*X12 + env[2,2]*X22 + env[2,3]*X32 + env[2,4]*X42 + 
                        env[3,1]*X13 + env[3,2]*X23 + env[3,3]*X33 + env[3,4]*X43 + 
                        env[4,1]*X14 + env[4,2]*X24 + env[4,3]*X34 + env[4,4]*X44

    return trEnvTwoQubitGate
end


"""
two qubit block of H type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg7, arg8,arg9,arg10, arg11,arg12,arg13])

                                                                                                       | quantum circuit (from right to left)
 —— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— · —————————— · —— Rz(arg8 ) —— Ry(arg9 ) —— Rz(arg10) ——       | |psi>
                                        |  XX(arg7)  |                                                 |
 —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · —————————— · —— Rz(arg11) —— Ry(arg12) —— Rz(arg13) ——       | |psi>

setXX_H([0.,0.,0., 0.,0.,0., 0., 0.,0.,0., 0.,0.,0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setXX_H(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #     H type       
    # 1 ————————— 3  
    #      | |       
    # 2 ————————— 4
    #
    unitary0 = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))
    unitary1 = kron(ZYZ_gate(Args[8:10]), ZYZ_gate(Args[11:13]))

    return unitary0* XX(Args[7]) * unitary1
end
eyeXX_H = [0.,0.,0., 0.,0.,0., 0., 0.,0.,0., 0.,0.,0.]

function setMS_H(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #     H type       
    # 1 ————————— 3  
    #      | |       
    # 2 ————————— 4
    #
    unitary0 = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))
    unitary1 = kron(ZYZ_gate(Args[7:9]), ZYZ_gate(Args[10:12]))

    return unitary0* XX(0.5) * unitary1
end

#----------------------------------------------------------------------------------------------------------------------------------------------
## YYXX-scheme (native gates + free Rz gate on DQC's device)
"""
two qubit block of n type (argList = [arg1, arg4, arg8,arg9])

                                              | quantum circuit (from right to left)
 —— Rz(arg1) · ——————————————————— · ——       | |psi>
             |  YY(arg8) XX(arg9)  |          |
 —— Rz(arg4) · ——————————————————— · ——       | |psi>

setYYXX_n([0., 0., 0.,0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setzzYYXX(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #   n type       
    # 1 -—————— 3  
    #      | |       
    # 2 -—————— 4
    #
    unitary0 = kron(Rz(Args[1]), Rz(Args[2]))

    return unitary0* YY(Args[3])* XX(Args[4])
end
eyezzYYXX = [0., 0., 0.,0.]
function setzzYYXX!(X::Matrix{ComplexF64},arg1::Float64,arg4::Float64, arg8::Float64,arg9::Float64)

    α1 = (arg1)/2; α2 = (arg4)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mα2half = cispi(-α2); pα2half = cispi(α2)

    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X[1,1] = cosφhalf * mα1half*mα2half; 
    X[1,2] = 0; 
    X[1,3] = 0; 
    X[1,4] = 1im*sinφhalf * mα1half*mα2half;
    
    X[2,1] = 0; 
    X[2,2] = cosϕhalf * mα1half*pα2half; 
    X[2,3] = -1im*sinϕhalf * mα1half*pα2half; 
    X[2,4] = 0;

    X[3,1] = 0; 
    X[3,2] = - 1im*sinϕhalf * mα2half*pα1half; 
    X[3,3] = cosϕhalf * mα2half*pα1half; 
    X[3,4] = 0;

    X[4,1] = 1im*sinφhalf * pα1half*pα2half; 
    X[4,2] = 0; 
    X[4,3] = 0; 
    X[4,4] = cosφhalf * pα1half*pα2half;

    return nothing
end

function tracezzYYXX(env::Matrix{ComplexF64},arg1::Float64, arg4::Float64, arg8::Float64,arg9::Float64)::ComplexF64

    α1 = (arg1)/2; α2 = (arg4)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mα2half = cispi(-α2); pα2half = cispi(α2)

    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X11 = cosφhalf * mα1half*mα2half;  
    X14 = 1im*sinφhalf * mα1half*mα2half;
    
    X22 = cosϕhalf * mα1half*pα2half; 
    X23 = -1im*sinϕhalf * mα1half*pα2half; 

    X32 = - 1im*sinϕhalf * mα2half*pα1half; 
    X33 = cosϕhalf * mα2half*pα1half; 

    X41 = 1im*sinφhalf * pα1half*pα2half; 
    X44 = cosφhalf * pα1half*pα2half;

    trEnvTwoQubitGate = env[1,1]*X11                               + env[1,4]*X41 + 
                                       env[2,2]*X22 + env[2,3]*X32                + 
                                       env[3,2]*X23 + env[3,3]*X33                + 
                        env[4,1]*X14                               + env[4,4]*X44

    return trEnvTwoQubitGate::ComplexF64
end

#----------------------------------------------------------------------------------------------------------------------------------------------
## YYXX-scheme (native gates on DQC's device)
"""
two qubit block of n type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg8,arg9])

                                                                         | quantum circuit (from right to left)
 —— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— · ——————————————————— · ——       | |psi>
                                        |  YY(arg8) XX(arg9)  |          |
 —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · ——————————————————— · ——       | |psi>

setYYXX_n([0.,0.,0., 0.,0.,0., 0.,0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setYYXX_n(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #   n type       
    # 1 -—————— 3  
    #      | |       
    # 2 -—————— 4
    #
    unitary0 = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))

    return unitary0* YY(Args[7])* XX(Args[8])
end
eyeYYXX_n = [0.,0.,0., 0.,0.,0., 0.,0.]
function setYYXX_n!(X::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64, arg8::Float64,arg9::Float64)

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X[1,1] = cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    X[1,2] = -cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X[1,3] = 1im*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X[1,4] = 1im*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    X[2,1] = cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    X[2,2] = cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X[2,3] = -1im*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X[2,4] = 1im*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    X[3,1] = cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    X[3,2] = -cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X[3,3] = 1im*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X[3,4] = 1im*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    X[4,1] = cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    X[4,2] = cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X[4,3] = -1im*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X[4,4] = 1im*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    return nothing
end

function traceYYXX_n(env::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64, arg8::Float64,arg9::Float64)::ComplexF64

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X11 = cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    X12 = -cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X13 = 1im*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X14 = 1im*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    X21 = cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    X22 = cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X23 = -1im*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X24 = 1im*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    X31 = cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    X32 = -cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X33 = 1im*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X34 = 1im*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    X41 = cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    X42 = cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X43 = -1im*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X44 = 1im*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    trEnvTwoQubitGate = env[1,1]*X11 + env[1,2]*X21 + env[1,3]*X31 + env[1,4]*X41 + 
                        env[2,1]*X12 + env[2,2]*X22 + env[2,3]*X32 + env[2,4]*X42 + 
                        env[3,1]*X13 + env[3,2]*X23 + env[3,3]*X33 + env[3,4]*X43 + 
                        env[4,1]*X14 + env[4,2]*X24 + env[4,3]*X34 + env[4,4]*X44

    return trEnvTwoQubitGate::ComplexF64
end

"""
two qubit block of x type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg8,arg9, arg10,arg11, arg13,arg14])

                                                                                                   | quantum circuit (from right to left)
 —— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— · ——————————————————— · —— Rz(arg10) —— Ry(arg11) ——       | |0>
                                        |  YY(arg8) XX(arg9)  |                                    |
 —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · ——————————————————— · —— Rz(arg13) —— Ry(arg14) ——       | |0>

setYYXX_x([0.,0.,0., 0.,0.,0., 0.,0., 0.,0., 0.,0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setYYXX_x(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #     x type       
    # 1 ———————— 3  
    #      | |       
    # 2 ———————— 4
    #
    unitary0 = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))
    unitary1 = kron(ZY_gate(Args[9:10]), ZY_gate(Args[11:12]))

    return unitary0* YY(Args[7])* XX(Args[8])* unitary1
end
eyeYYXX_x = [0.,0.,0., 0.,0.,0., 0.,0., 0.,0., 0.,0.]
function setYYXX_x!(X::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64, arg8::Float64,arg9::Float64,
    arg10::Float64,arg11::Float64, arg13::Float64,arg14::Float64)

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    L11 = cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    L12 = -cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L13 = 1im*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L14 = 1im*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    L21 = cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    L22 = cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L23 = -1im*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L24 = 1im*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    L31 = cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    L32 = -cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L33 = 1im*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L34 = 1im*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    L41 = cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    L42 = cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L43 = -1im*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L44 = 1im*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    α1 = (arg10)/2; α2 = (arg13)/2
    β1 = arg11/2; β2 = arg14/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    sinβ2half, cosβ2half = sincospi(β2)

    R11 = mα1half*mα2half*cosβ1half*cosβ2half; R12 = -mα1half*mα2half*cosβ1half*sinβ2half; R13 = -mα2half*mα1half*cosβ2half*sinβ1half; R14 = mα1half*mα2half*sinβ1half*sinβ2half;
    R21 = mα1half*pα2half*cosβ1half*sinβ2half; R22 = mα1half*pα2half*cosβ1half*cosβ2half; R23 = -mα1half*pα2half*sinβ1half*sinβ2half; R24 = -mα1half*pα2half*cosβ2half*sinβ1half;
    R31 = mα2half*pα1half*cosβ2half*sinβ1half; R32 = -mα2half*pα1half*sinβ1half*sinβ2half; R33 = mα2half*pα1half*cosβ1half*cosβ2half; R34 = -mα2half*pα1half*cosβ1half*sinβ2half;
    R41 = pα1half*pα2half*sinβ1half*sinβ2half; R42 = pα2half*pα1half*cosβ2half*sinβ1half; R43 = pα1half*pα2half*cosβ1half*sinβ2half; R44 = pα1half*pα2half*cosβ1half*cosβ2half;
    
    X[1,1] = L11*R11 + L12*R21 + L13*R31 + L14*R41;
    X[1,2] = L11*R12 + L12*R22 + L13*R32 + L14*R42;
    X[1,3] = L11*R13 + L12*R23 + L13*R33 + L14*R43;
    X[1,4] = L11*R14 + L12*R24 + L13*R34 + L14*R44;

    X[2,1] = L21*R11 + L22*R21 + L23*R31 + L24*R41;
    X[2,2] = L21*R12 + L22*R22 + L23*R32 + L24*R42;
    X[2,3] = L21*R13 + L22*R23 + L23*R33 + L24*R43;
    X[2,4] = L21*R14 + L22*R24 + L23*R34 + L24*R44;

    X[3,1] = L31*R11 + L32*R21 + L33*R31 + L34*R41;
    X[3,2] = L31*R12 + L32*R22 + L33*R32 + L34*R42;
    X[3,3] = L31*R13 + L32*R23 + L33*R33 + L34*R43;
    X[3,4] = L31*R14 + L32*R24 + L33*R34 + L34*R44;

    X[4,1] = L41*R11 + L42*R21 + L43*R31 + L44*R41;
    X[4,2] = L41*R12 + L42*R22 + L43*R32 + L44*R42;
    X[4,3] = L41*R13 + L42*R23 + L43*R33 + L44*R43;
    X[4,4] = L41*R14 + L42*R24 + L43*R34 + L44*R44;

    return nothing
end
function traceYYXX_x(env::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64, arg8::Float64,arg9::Float64,
    arg10::Float64,arg11::Float64, arg13::Float64,arg14::Float64)::ComplexF64

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    L11 = cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    L12 = -cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L13 = 1im*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L14 = 1im*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    L21 = cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    L22 = cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L23 = -1im*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L24 = 1im*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    L31 = cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    L32 = -cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L33 = 1im*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L34 = 1im*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    L41 = cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    L42 = cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L43 = -1im*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L44 = 1im*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    α1 = (arg10)/2; α2 = (arg13)/2
    β1 = arg11/2; β2 = arg14/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    sinβ2half, cosβ2half = sincospi(β2)

    R11 = mα1half*mα2half*cosβ1half*cosβ2half; R12 = -mα1half*mα2half*cosβ1half*sinβ2half; R13 = -mα2half*mα1half*cosβ2half*sinβ1half; R14 = mα1half*mα2half*sinβ1half*sinβ2half;
    R21 = mα1half*pα2half*cosβ1half*sinβ2half; R22 = mα1half*pα2half*cosβ1half*cosβ2half; R23 = -mα1half*pα2half*sinβ1half*sinβ2half; R24 = -mα1half*pα2half*cosβ2half*sinβ1half;
    R31 = mα2half*pα1half*cosβ2half*sinβ1half; R32 = -mα2half*pα1half*sinβ1half*sinβ2half; R33 = mα2half*pα1half*cosβ1half*cosβ2half; R34 = -mα2half*pα1half*cosβ1half*sinβ2half;
    R41 = pα1half*pα2half*sinβ1half*sinβ2half; R42 = pα2half*pα1half*cosβ2half*sinβ1half; R43 = pα1half*pα2half*cosβ1half*sinβ2half; R44 = pα1half*pα2half*cosβ1half*cosβ2half;
    
    X11 = L11*R11 + L12*R21 + L13*R31 + L14*R41;
    X12 = L11*R12 + L12*R22 + L13*R32 + L14*R42;
    X13 = L11*R13 + L12*R23 + L13*R33 + L14*R43;
    X14 = L11*R14 + L12*R24 + L13*R34 + L14*R44;

    X21 = L21*R11 + L22*R21 + L23*R31 + L24*R41;
    X22 = L21*R12 + L22*R22 + L23*R32 + L24*R42;
    X23 = L21*R13 + L22*R23 + L23*R33 + L24*R43;
    X24 = L21*R14 + L22*R24 + L23*R34 + L24*R44;

    X31 = L31*R11 + L32*R21 + L33*R31 + L34*R41;
    X32 = L31*R12 + L32*R22 + L33*R32 + L34*R42;
    X33 = L31*R13 + L32*R23 + L33*R33 + L34*R43;
    X34 = L31*R14 + L32*R24 + L33*R34 + L34*R44;

    X41 = L41*R11 + L42*R21 + L43*R31 + L44*R41;
    X42 = L41*R12 + L42*R22 + L43*R32 + L44*R42;
    X43 = L41*R13 + L42*R23 + L43*R33 + L44*R43;
    X44 = L41*R14 + L42*R24 + L43*R34 + L44*R44;

    trEnvTwoQubitGate = env[1,1]*X11 + env[1,2]*X21 + env[1,3]*X31 + env[1,4]*X41 + 
                        env[2,1]*X12 + env[2,2]*X22 + env[2,3]*X32 + env[2,4]*X42 + 
                        env[3,1]*X13 + env[3,2]*X23 + env[3,3]*X33 + env[3,4]*X43 + 
                        env[4,1]*X14 + env[4,2]*X24 + env[4,3]*X34 + env[4,4]*X44

    return trEnvTwoQubitGate
end

"""
two qubit block of H type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg8,arg9, arg10,arg11,arg12, arg13,arg14,arg15])

                                                                                                                | quantum circuit (from right to left)
 —— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— · ——————————————————— · —— Rz(arg10) —— Ry(arg11) —— Rz(arg12) ——       | |psi>
                                        |  YY(arg8) XX(arg9)  |                                                 |
 —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · ——————————————————— · —— Rz(arg13) —— Ry(arg14) —— Rz(arg15) ——       | |psi>

setYYXX_H([0.,0.,0., 0.,0.,0., 0.,0., 0.,0.,0., 0.,0.,0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setYYXX_H(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #     H type       
    # 1 ————————— 3  
    #      | |       
    # 2 ————————— 4
    #
    unitary0 = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))
    unitary1 = kron(ZYZ_gate(Args[9:11]), ZYZ_gate(Args[12:14]))

    return unitary0* YY(Args[7])* XX(Args[8])* unitary1
end
eyeYYXX_H = [0.,0.,0., 0.,0.,0., 0.,0., 0.,0.,0., 0.,0.,0.]
function setYYXX_H!(X::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64, arg8::Float64,arg9::Float64,
    arg10::Float64,arg11::Float64,arg12::Float64, arg13::Float64,arg14::Float64,arg15::Float64)

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    L11 = cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    L12 = -cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L13 = 1im*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L14 = 1im*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    L21 = cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    L22 = cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L23 = -1im*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L24 = 1im*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    L31 = cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    L32 = -cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L33 = 1im*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L34 = 1im*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    L41 = cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    L42 = cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L43 = -1im*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L44 = 1im*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    α1 = (arg10+arg12)/2; α2 = (arg13+arg15)/2
    β1 = arg11/2; β2 = arg14/2
    γ1 = (arg10-arg12)/2; γ2 = (arg13-arg15)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    R11 = mα1half*mα2half*cosβ1half*cosβ2half; R12 = -mα1half*mγ2half*cosβ1half*sinβ2half; R13 = -mα2half*mγ1half*cosβ2half*sinβ1half; R14 = mγ1half*mγ2half*sinβ1half*sinβ2half;
    R21 = mα1half*pγ2half*cosβ1half*sinβ2half; R22 = mα1half*pα2half*cosβ1half*cosβ2half; R23 = -mγ1half*pγ2half*sinβ1half*sinβ2half; R24 = -mγ1half*pα2half*cosβ2half*sinβ1half;
    R31 = mα2half*pγ1half*cosβ2half*sinβ1half; R32 = -mγ2half*pγ1half*sinβ1half*sinβ2half; R33 = mα2half*pα1half*cosβ1half*cosβ2half; R34 = -mγ2half*pα1half*cosβ1half*sinβ2half;
    R41 = pγ1half*pγ2half*sinβ1half*sinβ2half; R42 = pα2half*pγ1half*cosβ2half*sinβ1half; R43 = pα1half*pγ2half*cosβ1half*sinβ2half; R44 = pα1half*pα2half*cosβ1half*cosβ2half;
    
    X[1,1] = L11*R11 + L12*R21 + L13*R31 + L14*R41;
    X[1,2] = L11*R12 + L12*R22 + L13*R32 + L14*R42;
    X[1,3] = L11*R13 + L12*R23 + L13*R33 + L14*R43;
    X[1,4] = L11*R14 + L12*R24 + L13*R34 + L14*R44;

    X[2,1] = L21*R11 + L22*R21 + L23*R31 + L24*R41;
    X[2,2] = L21*R12 + L22*R22 + L23*R32 + L24*R42;
    X[2,3] = L21*R13 + L22*R23 + L23*R33 + L24*R43;
    X[2,4] = L21*R14 + L22*R24 + L23*R34 + L24*R44;

    X[3,1] = L31*R11 + L32*R21 + L33*R31 + L34*R41;
    X[3,2] = L31*R12 + L32*R22 + L33*R32 + L34*R42;
    X[3,3] = L31*R13 + L32*R23 + L33*R33 + L34*R43;
    X[3,4] = L31*R14 + L32*R24 + L33*R34 + L34*R44;

    X[4,1] = L41*R11 + L42*R21 + L43*R31 + L44*R41;
    X[4,2] = L41*R12 + L42*R22 + L43*R32 + L44*R42;
    X[4,3] = L41*R13 + L42*R23 + L43*R33 + L44*R43;
    X[4,4] = L41*R14 + L42*R24 + L43*R34 + L44*R44;

    return nothing
end
function traceYYXX_H(env::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64, arg8::Float64,arg9::Float64,
    arg10::Float64,arg11::Float64,arg12::Float64, arg13::Float64,arg14::Float64,arg15::Float64)::ComplexF64

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    L11 = cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    L12 = -cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L13 = 1im*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L14 = 1im*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    L21 = cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    L22 = cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L23 = -1im*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L24 = 1im*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    L31 = cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    L32 = -cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L33 = 1im*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L34 = 1im*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    L41 = cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    L42 = cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L43 = -1im*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L44 = 1im*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    α1 = (arg10+arg12)/2; α2 = (arg13+arg15)/2
    β1 = arg11/2; β2 = arg14/2
    γ1 = (arg10-arg12)/2; γ2 = (arg13-arg15)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    R11 = mα1half*mα2half*cosβ1half*cosβ2half; R12 = -mα1half*mγ2half*cosβ1half*sinβ2half; R13 = -mα2half*mγ1half*cosβ2half*sinβ1half; R14 = mγ1half*mγ2half*sinβ1half*sinβ2half;
    R21 = mα1half*pγ2half*cosβ1half*sinβ2half; R22 = mα1half*pα2half*cosβ1half*cosβ2half; R23 = -mγ1half*pγ2half*sinβ1half*sinβ2half; R24 = -mγ1half*pα2half*cosβ2half*sinβ1half;
    R31 = mα2half*pγ1half*cosβ2half*sinβ1half; R32 = -mγ2half*pγ1half*sinβ1half*sinβ2half; R33 = mα2half*pα1half*cosβ1half*cosβ2half; R34 = -mγ2half*pα1half*cosβ1half*sinβ2half;
    R41 = pγ1half*pγ2half*sinβ1half*sinβ2half; R42 = pα2half*pγ1half*cosβ2half*sinβ1half; R43 = pα1half*pγ2half*cosβ1half*sinβ2half; R44 = pα1half*pα2half*cosβ1half*cosβ2half;
    
    X11 = L11*R11 + L12*R21 + L13*R31 + L14*R41;
    X12 = L11*R12 + L12*R22 + L13*R32 + L14*R42;
    X13 = L11*R13 + L12*R23 + L13*R33 + L14*R43;
    X14 = L11*R14 + L12*R24 + L13*R34 + L14*R44;

    X21 = L21*R11 + L22*R21 + L23*R31 + L24*R41;
    X22 = L21*R12 + L22*R22 + L23*R32 + L24*R42;
    X23 = L21*R13 + L22*R23 + L23*R33 + L24*R43;
    X24 = L21*R14 + L22*R24 + L23*R34 + L24*R44;

    X31 = L31*R11 + L32*R21 + L33*R31 + L34*R41;
    X32 = L31*R12 + L32*R22 + L33*R32 + L34*R42;
    X33 = L31*R13 + L32*R23 + L33*R33 + L34*R43;
    X34 = L31*R14 + L32*R24 + L33*R34 + L34*R44;

    X41 = L41*R11 + L42*R21 + L43*R31 + L44*R41;
    X42 = L41*R12 + L42*R22 + L43*R32 + L44*R42;
    X43 = L41*R13 + L42*R23 + L43*R33 + L44*R43;
    X44 = L41*R14 + L42*R24 + L43*R34 + L44*R44;

    trEnvTwoQubitGate = env[1,1]*X11 + env[1,2]*X21 + env[1,3]*X31 + env[1,4]*X41 + 
                        env[2,1]*X12 + env[2,2]*X22 + env[2,3]*X32 + env[2,4]*X42 + 
                        env[3,1]*X13 + env[3,2]*X23 + env[3,3]*X33 + env[3,4]*X43 + 
                        env[4,1]*X14 + env[4,2]*X24 + env[4,3]*X34 + env[4,4]*X44

    return trEnvTwoQubitGate
end

#----------------------------------------------------------------------------------------------------------------------------------------------
## Canonical decomposition
"""
two qubit block of n type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg7,arg8,arg9])

                                                                                  | quantum circuit (from right to left)
 —— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— · ———————————————————————————— · ——       | |psi>
                                        |  ZZ(arg7) YY(arg8) XX(arg9)  |          |
 —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · ———————————————————————————— · ——       | |psi>

setCAN_n([0.,0.,0., 0.,0.,0., 0.,0.,0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setCAN_n(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #   n type       
    # 1 -—————— 3  
    #      | |       
    # 2 -—————— 4
    #
    unitary0 = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))

    return unitary0* ZZ(Args[7])* YY(Args[8])* XX(Args[9])
end
eyeCAN_n = [0.,0.,0., 0.,0.,0., 0.,0.,0.]
function setCAN_n!(X::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64,arg7::Float64,arg8::Float64,arg9::Float64)

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    θ = arg7/2
    mθhalf = cispi(-θ); pθhalf = cispi(θ)
    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X[1,1] = mθhalf*cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*mθhalf*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    X[1,2] = -pθhalf*cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*pθhalf*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X[1,3] = 1im*pθhalf*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - pθhalf*cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X[1,4] = 1im*mθhalf*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + mθhalf*cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    X[2,1] = mθhalf*cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*mθhalf*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    X[2,2] = pθhalf*cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*pθhalf*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X[2,3] = -1im*pθhalf*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - pθhalf*cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X[2,4] = 1im*mθhalf*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - mθhalf*cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    X[3,1] = mθhalf*cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*mθhalf*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    X[3,2] = -pθhalf*cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*pθhalf*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X[3,3] = 1im*pθhalf*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + pθhalf*cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X[3,4] = 1im*mθhalf*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - mθhalf*cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    X[4,1] = mθhalf*cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*mθhalf*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    X[4,2] = pθhalf*cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*pθhalf*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X[4,3] = -1im*pθhalf*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + pθhalf*cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X[4,4] = 1im*mθhalf*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + mθhalf*cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    return nothing
end
function setCAN_body!(X::Matrix{ComplexF64},arg7::Float64,arg8::Float64,arg9::Float64)

    θ = arg7/2
    mθhalf = cispi(-θ); pθhalf = cispi(θ)
    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X[1,1] = mθhalf*cosφhalf; X[1,2] = 0; X[1,3] = 0; X[1,4] = 1im*mθhalf*sinφhalf;
    X[2,1] = 0; X[2,2] = pθhalf*cosϕhalf; X[2,3] = -1im*pθhalf*sinϕhalf; X[2,4] = 0;
    X[3,1] = 0; X[3,2] = -1im*pθhalf*sinϕhalf; X[3,3] = pθhalf*cosϕhalf; X[3,4] = 0;
    X[4,1] = 1im*mθhalf*sinφhalf; X[4,2] = 0; X[4,3] = 0; X[4,4] = mθhalf*cosφhalf;

    return nothing
end
"""
                                              | quantum circuit (from right to left)
 —— Rz(arg1) —— Ry(arg2) —— Rz(arg3) ——       | |0>
                                              |
 —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) ——       | |0>

"""
function setZYZ_ZYZ!(X::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64)

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    X[1,1] = mα1half*mα2half*cosβ1half*cosβ2half; X[1,2] = -mα1half*mγ2half*cosβ1half*sinβ2half; X[1,3] = -mα2half*mγ1half*cosβ2half*sinβ1half; X[1,4] = mγ1half*mγ2half*sinβ1half*sinβ2half;
    X[2,1] = mα1half*pγ2half*cosβ1half*sinβ2half; X[2,2] = mα1half*pα2half*cosβ1half*cosβ2half; X[2,3] = -mγ1half*pγ2half*sinβ1half*sinβ2half; X[2,4] = -mγ1half*pα2half*cosβ2half*sinβ1half;
    X[3,1] = mα2half*pγ1half*cosβ2half*sinβ1half; X[3,2] = -mγ2half*pγ1half*sinβ1half*sinβ2half; X[3,3] = mα2half*pα1half*cosβ1half*cosβ2half; X[3,4] = -mγ2half*pα1half*cosβ1half*sinβ2half;
    X[4,1] = pγ1half*pγ2half*sinβ1half*sinβ2half; X[4,2] = pα2half*pγ1half*cosβ2half*sinβ1half; X[4,3] = pα1half*pγ2half*cosβ1half*sinβ2half; X[4,4] = pα1half*pα2half*cosβ1half*cosβ2half;
    
    return nothing
end

function traceCAN_n(env::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64,arg7::Float64,arg8::Float64,arg9::Float64)::ComplexF64

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    θ = arg7/2
    mθhalf = cispi(-θ); pθhalf = cispi(θ)
    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    X11 = mθhalf*cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*mθhalf*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    X12 = -pθhalf*cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*pθhalf*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X13 = 1im*pθhalf*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - pθhalf*cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    X14 = 1im*mθhalf*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + mθhalf*cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    X21 = mθhalf*cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*mθhalf*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    X22 = pθhalf*cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*pθhalf*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X23 = -1im*pθhalf*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - pθhalf*cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    X24 = 1im*mθhalf*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - mθhalf*cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    X31 = mθhalf*cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*mθhalf*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    X32 = -pθhalf*cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*pθhalf*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X33 = 1im*pθhalf*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + pθhalf*cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    X34 = 1im*mθhalf*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - mθhalf*cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    X41 = mθhalf*cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*mθhalf*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    X42 = pθhalf*cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*pθhalf*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X43 = -1im*pθhalf*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + pθhalf*cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    X44 = 1im*mθhalf*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + mθhalf*cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    trEnvTwoQubitGate = env[1,1]*X11 + env[1,2]*X21 + env[1,3]*X31 + env[1,4]*X41 + 
                        env[2,1]*X12 + env[2,2]*X22 + env[2,3]*X32 + env[2,4]*X42 + 
                        env[3,1]*X13 + env[3,2]*X23 + env[3,3]*X33 + env[3,4]*X43 + 
                        env[4,1]*X14 + env[4,2]*X24 + env[4,3]*X34 + env[4,4]*X44

    return trEnvTwoQubitGate::ComplexF64
end

"""
two qubit block of x type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg7,arg8,arg9, arg10,arg11, arg12,arg13])

                                                                                                            | quantum circuit (from right to left)
 —— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— · ———————————————————————————— · —— Rz(arg10) —— Ry(arg11) ——       | |0>
                                        |  ZZ(arg7) YY(arg8) XX(arg9)  |                                    |
 —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · ———————————————————————————— · —— Rz(arg13) —— Ry(arg14) ——       | |0>

setCAN_x([0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0., 0.,0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setCAN_x(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #     x type       
    # 1 ———————— 3  
    #      | |       
    # 2 ———————— 4
    #
    unitary0 = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))
    unitary1 = kron(ZY_gate(Args[10:11]), ZY_gate(Args[12:13]))

    return unitary0* ZZ(Args[7])* YY(Args[8])* XX(Args[9])* unitary1
end
eyeCAN_x = [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0., 0.,0.]
function setCAN_x!(X::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64,arg7::Float64,arg8::Float64,arg9::Float64,
    arg10::Float64,arg11::Float64, arg13::Float64,arg14::Float64)

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    θ = arg7/2
    mθhalf = cispi(-θ); pθhalf = cispi(θ)
    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    L11 = mθhalf*cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*mθhalf*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    L12 = -pθhalf*cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*pθhalf*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L13 = 1im*pθhalf*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - pθhalf*cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L14 = 1im*mθhalf*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + mθhalf*cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    L21 = mθhalf*cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*mθhalf*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    L22 = pθhalf*cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*pθhalf*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L23 = -1im*pθhalf*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - pθhalf*cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L24 = 1im*mθhalf*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - mθhalf*cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    L31 = mθhalf*cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*mθhalf*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    L32 = -pθhalf*cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*pθhalf*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L33 = 1im*pθhalf*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + pθhalf*cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L34 = 1im*mθhalf*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - mθhalf*cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    L41 = mθhalf*cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*mθhalf*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    L42 = pθhalf*cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*pθhalf*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L43 = -1im*pθhalf*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + pθhalf*cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L44 = 1im*mθhalf*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + mθhalf*cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    α1 = (arg10)/2; α2 = (arg13)/2
    β1 = arg11/2; β2 = arg14/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    sinβ2half, cosβ2half = sincospi(β2)

    R11 = mα1half*mα2half*cosβ1half*cosβ2half; R12 = -mα1half*mα2half*cosβ1half*sinβ2half; R13 = -mα2half*mα1half*cosβ2half*sinβ1half; R14 = mα1half*mα2half*sinβ1half*sinβ2half;
    R21 = mα1half*pα2half*cosβ1half*sinβ2half; R22 = mα1half*pα2half*cosβ1half*cosβ2half; R23 = -mα1half*pα2half*sinβ1half*sinβ2half; R24 = -mα1half*pα2half*cosβ2half*sinβ1half;
    R31 = mα2half*pα1half*cosβ2half*sinβ1half; R32 = -mα2half*pα1half*sinβ1half*sinβ2half; R33 = mα2half*pα1half*cosβ1half*cosβ2half; R34 = -mα2half*pα1half*cosβ1half*sinβ2half;
    R41 = pα1half*pα2half*sinβ1half*sinβ2half; R42 = pα2half*pα1half*cosβ2half*sinβ1half; R43 = pα1half*pα2half*cosβ1half*sinβ2half; R44 = pα1half*pα2half*cosβ1half*cosβ2half;
    
    X[1,1] = L11*R11 + L12*R21 + L13*R31 + L14*R41;
    X[1,2] = L11*R12 + L12*R22 + L13*R32 + L14*R42;
    X[1,3] = L11*R13 + L12*R23 + L13*R33 + L14*R43;
    X[1,4] = L11*R14 + L12*R24 + L13*R34 + L14*R44;

    X[2,1] = L21*R11 + L22*R21 + L23*R31 + L24*R41;
    X[2,2] = L21*R12 + L22*R22 + L23*R32 + L24*R42;
    X[2,3] = L21*R13 + L22*R23 + L23*R33 + L24*R43;
    X[2,4] = L21*R14 + L22*R24 + L23*R34 + L24*R44;

    X[3,1] = L31*R11 + L32*R21 + L33*R31 + L34*R41;
    X[3,2] = L31*R12 + L32*R22 + L33*R32 + L34*R42;
    X[3,3] = L31*R13 + L32*R23 + L33*R33 + L34*R43;
    X[3,4] = L31*R14 + L32*R24 + L33*R34 + L34*R44;

    X[4,1] = L41*R11 + L42*R21 + L43*R31 + L44*R41;
    X[4,2] = L41*R12 + L42*R22 + L43*R32 + L44*R42;
    X[4,3] = L41*R13 + L42*R23 + L43*R33 + L44*R43;
    X[4,4] = L41*R14 + L42*R24 + L43*R34 + L44*R44;

    return nothing
end
function traceCAN_x(env::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64,arg7::Float64,arg8::Float64,arg9::Float64,
    arg10::Float64,arg11::Float64, arg13::Float64,arg14::Float64)::ComplexF64

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    θ = arg7/2
    mθhalf = cispi(-θ); pθhalf = cispi(θ)
    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    L11 = mθhalf*cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*mθhalf*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    L12 = -pθhalf*cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*pθhalf*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L13 = 1im*pθhalf*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - pθhalf*cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L14 = 1im*mθhalf*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + mθhalf*cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    L21 = mθhalf*cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*mθhalf*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    L22 = pθhalf*cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*pθhalf*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L23 = -1im*pθhalf*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - pθhalf*cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L24 = 1im*mθhalf*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - mθhalf*cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    L31 = mθhalf*cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*mθhalf*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    L32 = -pθhalf*cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*pθhalf*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L33 = 1im*pθhalf*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + pθhalf*cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L34 = 1im*mθhalf*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - mθhalf*cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    L41 = mθhalf*cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*mθhalf*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    L42 = pθhalf*cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*pθhalf*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L43 = -1im*pθhalf*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + pθhalf*cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L44 = 1im*mθhalf*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + mθhalf*cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    α1 = (arg10)/2; α2 = (arg13)/2
    β1 = arg11/2; β2 = arg14/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    sinβ2half, cosβ2half = sincospi(β2)

    R11 = mα1half*mα2half*cosβ1half*cosβ2half; R12 = -mα1half*mα2half*cosβ1half*sinβ2half; R13 = -mα2half*mα1half*cosβ2half*sinβ1half; R14 = mα1half*mα2half*sinβ1half*sinβ2half;
    R21 = mα1half*pα2half*cosβ1half*sinβ2half; R22 = mα1half*pα2half*cosβ1half*cosβ2half; R23 = -mα1half*pα2half*sinβ1half*sinβ2half; R24 = -mα1half*pα2half*cosβ2half*sinβ1half;
    R31 = mα2half*pα1half*cosβ2half*sinβ1half; R32 = -mα2half*pα1half*sinβ1half*sinβ2half; R33 = mα2half*pα1half*cosβ1half*cosβ2half; R34 = -mα2half*pα1half*cosβ1half*sinβ2half;
    R41 = pα1half*pα2half*sinβ1half*sinβ2half; R42 = pα2half*pα1half*cosβ2half*sinβ1half; R43 = pα1half*pα2half*cosβ1half*sinβ2half; R44 = pα1half*pα2half*cosβ1half*cosβ2half;
    
    X11 = L11*R11 + L12*R21 + L13*R31 + L14*R41;
    X12 = L11*R12 + L12*R22 + L13*R32 + L14*R42;
    X13 = L11*R13 + L12*R23 + L13*R33 + L14*R43;
    X14 = L11*R14 + L12*R24 + L13*R34 + L14*R44;

    X21 = L21*R11 + L22*R21 + L23*R31 + L24*R41;
    X22 = L21*R12 + L22*R22 + L23*R32 + L24*R42;
    X23 = L21*R13 + L22*R23 + L23*R33 + L24*R43;
    X24 = L21*R14 + L22*R24 + L23*R34 + L24*R44;

    X31 = L31*R11 + L32*R21 + L33*R31 + L34*R41;
    X32 = L31*R12 + L32*R22 + L33*R32 + L34*R42;
    X33 = L31*R13 + L32*R23 + L33*R33 + L34*R43;
    X34 = L31*R14 + L32*R24 + L33*R34 + L34*R44;

    X41 = L41*R11 + L42*R21 + L43*R31 + L44*R41;
    X42 = L41*R12 + L42*R22 + L43*R32 + L44*R42;
    X43 = L41*R13 + L42*R23 + L43*R33 + L44*R43;
    X44 = L41*R14 + L42*R24 + L43*R34 + L44*R44;

    trEnvTwoQubitGate = env[1,1]*X11 + env[1,2]*X21 + env[1,3]*X31 + env[1,4]*X41 + 
                        env[2,1]*X12 + env[2,2]*X22 + env[2,3]*X32 + env[2,4]*X42 + 
                        env[3,1]*X13 + env[3,2]*X23 + env[3,3]*X33 + env[3,4]*X43 + 
                        env[4,1]*X14 + env[4,2]*X24 + env[4,3]*X34 + env[4,4]*X44

    return trEnvTwoQubitGate
end

"""
two qubit block of H type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg7,arg8,arg9, arg10,arg11,arg12, arg13,arg14,arg15])

                                                                                                                         | quantum circuit (from right to left)
 —— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— · ———————————————————————————— · —— Rz(arg10) —— Ry(arg11) —— Rz(arg12) ——       | |psi>
                                        |  ZZ(arg7) YY(arg8) XX(arg9)  |                                                 |
 —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · ———————————————————————————— · —— Rz(arg13) —— Ry(arg14) —— Rz(arg15) ——       | |psi>

setCAN_H([0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.]) = Matrix{ComplexF64}(I, 4,4) 
"""
function setCAN_H(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #     H type       
    # 1 ————————— 3  
    #      | |       
    # 2 ————————— 4
    #
    unitary0 = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))
    unitary1 = kron(ZYZ_gate(Args[10:12]), ZYZ_gate(Args[13:15]))

    return unitary0* ZZ(Args[7])* YY(Args[8])* XX(Args[9])* unitary1
end
eyeCAN_H = [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.]
function setCAN_H!(X::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64,arg7::Float64,arg8::Float64,arg9::Float64,
    arg10::Float64,arg11::Float64,arg12::Float64, arg13::Float64,arg14::Float64,arg15::Float64)

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    θ = arg7/2
    mθhalf = cispi(-θ); pθhalf = cispi(θ)
    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    L11 = mθhalf*cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*mθhalf*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    L12 = -pθhalf*cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*pθhalf*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L13 = 1im*pθhalf*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - pθhalf*cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L14 = 1im*mθhalf*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + mθhalf*cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    L21 = mθhalf*cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*mθhalf*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    L22 = pθhalf*cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*pθhalf*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L23 = -1im*pθhalf*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - pθhalf*cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L24 = 1im*mθhalf*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - mθhalf*cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    L31 = mθhalf*cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*mθhalf*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    L32 = -pθhalf*cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*pθhalf*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L33 = 1im*pθhalf*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + pθhalf*cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L34 = 1im*mθhalf*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - mθhalf*cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    L41 = mθhalf*cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*mθhalf*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    L42 = pθhalf*cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*pθhalf*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L43 = -1im*pθhalf*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + pθhalf*cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L44 = 1im*mθhalf*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + mθhalf*cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    α1 = (arg10+arg12)/2; α2 = (arg13+arg15)/2
    β1 = arg11/2; β2 = arg14/2
    γ1 = (arg10-arg12)/2; γ2 = (arg13-arg15)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    R11 = mα1half*mα2half*cosβ1half*cosβ2half; R12 = -mα1half*mγ2half*cosβ1half*sinβ2half; R13 = -mα2half*mγ1half*cosβ2half*sinβ1half; R14 = mγ1half*mγ2half*sinβ1half*sinβ2half;
    R21 = mα1half*pγ2half*cosβ1half*sinβ2half; R22 = mα1half*pα2half*cosβ1half*cosβ2half; R23 = -mγ1half*pγ2half*sinβ1half*sinβ2half; R24 = -mγ1half*pα2half*cosβ2half*sinβ1half;
    R31 = mα2half*pγ1half*cosβ2half*sinβ1half; R32 = -mγ2half*pγ1half*sinβ1half*sinβ2half; R33 = mα2half*pα1half*cosβ1half*cosβ2half; R34 = -mγ2half*pα1half*cosβ1half*sinβ2half;
    R41 = pγ1half*pγ2half*sinβ1half*sinβ2half; R42 = pα2half*pγ1half*cosβ2half*sinβ1half; R43 = pα1half*pγ2half*cosβ1half*sinβ2half; R44 = pα1half*pα2half*cosβ1half*cosβ2half;
    
    X[1,1] = L11*R11 + L12*R21 + L13*R31 + L14*R41;
    X[1,2] = L11*R12 + L12*R22 + L13*R32 + L14*R42;
    X[1,3] = L11*R13 + L12*R23 + L13*R33 + L14*R43;
    X[1,4] = L11*R14 + L12*R24 + L13*R34 + L14*R44;

    X[2,1] = L21*R11 + L22*R21 + L23*R31 + L24*R41;
    X[2,2] = L21*R12 + L22*R22 + L23*R32 + L24*R42;
    X[2,3] = L21*R13 + L22*R23 + L23*R33 + L24*R43;
    X[2,4] = L21*R14 + L22*R24 + L23*R34 + L24*R44;

    X[3,1] = L31*R11 + L32*R21 + L33*R31 + L34*R41;
    X[3,2] = L31*R12 + L32*R22 + L33*R32 + L34*R42;
    X[3,3] = L31*R13 + L32*R23 + L33*R33 + L34*R43;
    X[3,4] = L31*R14 + L32*R24 + L33*R34 + L34*R44;

    X[4,1] = L41*R11 + L42*R21 + L43*R31 + L44*R41;
    X[4,2] = L41*R12 + L42*R22 + L43*R32 + L44*R42;
    X[4,3] = L41*R13 + L42*R23 + L43*R33 + L44*R43;
    X[4,4] = L41*R14 + L42*R24 + L43*R34 + L44*R44;

    return nothing
end
function traceCAN_H(env::Matrix{ComplexF64},arg1::Float64,arg2::Float64,arg3::Float64,arg4::Float64,arg5::Float64,arg6::Float64,arg7::Float64,arg8::Float64,arg9::Float64,
    arg10::Float64,arg11::Float64,arg12::Float64, arg13::Float64,arg14::Float64,arg15::Float64)::ComplexF64

    α1 = (arg1+arg3)/2; α2 = (arg4+arg6)/2
    β1 = arg2/2; β2 = arg5/2
    γ1 = (arg1-arg3)/2; γ2 = (arg4-arg6)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    θ = arg7/2
    mθhalf = cispi(-θ); pθhalf = cispi(θ)
    φ = (arg8 - arg9)/2
    sinφhalf, cosφhalf = sincospi(φ)
    ϕ = (arg8 + arg9)/2
    sinϕhalf, cosϕhalf = sincospi(ϕ)

    L11 = mθhalf*cosφhalf * mα1half*mα2half*cosβ1half*cosβ2half + 1im*mθhalf*sinφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half; 
    L12 = -pθhalf*cosϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half + 1im*pθhalf*sinϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L13 = 1im*pθhalf*sinϕhalf * mα1half*mγ2half*cosβ1half*sinβ2half - pθhalf*cosϕhalf * mα2half*mγ1half*cosβ2half*sinβ1half; 
    L14 = 1im*mθhalf*sinφhalf * mα1half*mα2half*cosβ1half*cosβ2half + mθhalf*cosφhalf * mγ1half*mγ2half*sinβ1half*sinβ2half;
    
    L21 = mθhalf*cosφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - 1im*mθhalf*sinφhalf * mγ1half*pα2half*cosβ2half*sinβ1half; 
    L22 = pθhalf*cosϕhalf * mα1half*pα2half*cosβ1half*cosβ2half + 1im*pθhalf*sinϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L23 = -1im*pθhalf*sinϕhalf * mα1half*pα2half*cosβ1half*cosβ2half - pθhalf*cosϕhalf * mγ1half*pγ2half*sinβ1half*sinβ2half; 
    L24 = 1im*mθhalf*sinφhalf * mα1half*pγ2half*cosβ1half*sinβ2half - mθhalf*cosφhalf * mγ1half*pα2half*cosβ2half*sinβ1half;

    L31 = mθhalf*cosφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - 1im*mθhalf*sinφhalf * mγ2half*pα1half*cosβ1half*sinβ2half; 
    L32 = -pθhalf*cosϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half - 1im*pθhalf*sinϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L33 = 1im*pθhalf*sinϕhalf * mγ2half*pγ1half*sinβ1half*sinβ2half + pθhalf*cosϕhalf * mα2half*pα1half*cosβ1half*cosβ2half; 
    L34 = 1im*mθhalf*sinφhalf * mα2half*pγ1half*cosβ2half*sinβ1half - mθhalf*cosφhalf * mγ2half*pα1half*cosβ1half*sinβ2half;

    L41 = mθhalf*cosφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + 1im*mθhalf*sinφhalf * pα1half*pα2half*cosβ1half*cosβ2half; 
    L42 = pθhalf*cosϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half - 1im*pθhalf*sinϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L43 = -1im*pθhalf*sinϕhalf * pα2half*pγ1half*cosβ2half*sinβ1half + pθhalf*cosϕhalf * pα1half*pγ2half*cosβ1half*sinβ2half; 
    L44 = 1im*mθhalf*sinφhalf * pγ1half*pγ2half*sinβ1half*sinβ2half + mθhalf*cosφhalf * pα1half*pα2half*cosβ1half*cosβ2half;

    α1 = (arg10+arg12)/2; α2 = (arg13+arg15)/2
    β1 = arg11/2; β2 = arg14/2
    γ1 = (arg10-arg12)/2; γ2 = (arg13-arg15)/2

    mα1half = cispi(-α1); pα1half = cispi(α1)
    mγ1half = cispi(-γ1); pγ1half = cispi(γ1)
    sinβ1half, cosβ1half = sincospi(β1)

    mα2half = cispi(-α2); pα2half = cispi(α2)
    mγ2half = cispi(-γ2); pγ2half = cispi(γ2)
    sinβ2half, cosβ2half = sincospi(β2)

    R11 = mα1half*mα2half*cosβ1half*cosβ2half; R12 = -mα1half*mγ2half*cosβ1half*sinβ2half; R13 = -mα2half*mγ1half*cosβ2half*sinβ1half; R14 = mγ1half*mγ2half*sinβ1half*sinβ2half;
    R21 = mα1half*pγ2half*cosβ1half*sinβ2half; R22 = mα1half*pα2half*cosβ1half*cosβ2half; R23 = -mγ1half*pγ2half*sinβ1half*sinβ2half; R24 = -mγ1half*pα2half*cosβ2half*sinβ1half;
    R31 = mα2half*pγ1half*cosβ2half*sinβ1half; R32 = -mγ2half*pγ1half*sinβ1half*sinβ2half; R33 = mα2half*pα1half*cosβ1half*cosβ2half; R34 = -mγ2half*pα1half*cosβ1half*sinβ2half;
    R41 = pγ1half*pγ2half*sinβ1half*sinβ2half; R42 = pα2half*pγ1half*cosβ2half*sinβ1half; R43 = pα1half*pγ2half*cosβ1half*sinβ2half; R44 = pα1half*pα2half*cosβ1half*cosβ2half;
    
    X11 = L11*R11 + L12*R21 + L13*R31 + L14*R41;
    X12 = L11*R12 + L12*R22 + L13*R32 + L14*R42;
    X13 = L11*R13 + L12*R23 + L13*R33 + L14*R43;
    X14 = L11*R14 + L12*R24 + L13*R34 + L14*R44;

    X21 = L21*R11 + L22*R21 + L23*R31 + L24*R41;
    X22 = L21*R12 + L22*R22 + L23*R32 + L24*R42;
    X23 = L21*R13 + L22*R23 + L23*R33 + L24*R43;
    X24 = L21*R14 + L22*R24 + L23*R34 + L24*R44;

    X31 = L31*R11 + L32*R21 + L33*R31 + L34*R41;
    X32 = L31*R12 + L32*R22 + L33*R32 + L34*R42;
    X33 = L31*R13 + L32*R23 + L33*R33 + L34*R43;
    X34 = L31*R14 + L32*R24 + L33*R34 + L34*R44;

    X41 = L41*R11 + L42*R21 + L43*R31 + L44*R41;
    X42 = L41*R12 + L42*R22 + L43*R32 + L44*R42;
    X43 = L41*R13 + L42*R23 + L43*R33 + L44*R43;
    X44 = L41*R14 + L42*R24 + L43*R34 + L44*R44;

    trEnvTwoQubitGate = env[1,1]*X11 + env[1,2]*X21 + env[1,3]*X31 + env[1,4]*X41 + 
                        env[2,1]*X12 + env[2,2]*X22 + env[2,3]*X32 + env[2,4]*X42 + 
                        env[3,1]*X13 + env[3,2]*X23 + env[3,3]*X33 + env[3,4]*X43 + 
                        env[4,1]*X14 + env[4,2]*X24 + env[4,3]*X34 + env[4,4]*X44

    return trEnvTwoQubitGate
end

#----------------------------------------------------------------------------------------------------------------------------------------------
## CNOT decomposition
"""
two qubit block of u type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg7,arg8,arg9])

CNOT_flip          CNOT           CNOT_filp                                         | quantum circuit (from right to left)
—— + —————————————— · —— Rz(arg2) —— + —— Rz(arg4) —— Ry(arg5) —— Rz(arg6) ——       | |psi>
   |                |                |                                              |
—— · —— Ry(arg1) —— + —— Ry(arg3) —— · —— Rz(arg7) —— Ry(arg8) —— Rz(arg9) ——       | |psi>

setCNOT_u([0.5,-0.5,-0.5, 0.0,0.0,0.0, 0.0,0.5,0.0]) = (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4) 
setCNOT_u([0.,0.,0.,0.,0.,0.,0.,0.,0.]) = SWAP
"""
function setCNOT_u(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #   u type       
    # 1 -—————— 3  
    #    | |       
    # 2 -—————— 4
    #
    unitary = kron(ZYZ_gate(Args[4:6]), ZYZ_gate(Args[7:9]))

    return CNOT_flip* kron(sigmaI,Ry(Args[1]))* CNOT* kron(Rz(Args[2]), Ry(Args[3]))* CNOT_flip* unitary
end
eyeGate_u = [0.5,-0.5,-0.5, 0.0,0.0,-0.5, 0.0,0.0,0.5]

"""
two qubit block of n type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg7,arg8,arg9])

                                   CNOT_flip          CNOT           CNOT_filp      | quantum circuit (from right to left)
—— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— + —————————————— · —— Rz(arg8) —— + ——       | |psi>
                                       |                |                |          |
—— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · —— Ry(arg7) —— + —— Ry(arg9) —— · ——       | |psi>

setCNOT_n([0.0,0.0,0.0, 0.0,0.5,0.0, 0.5,-0.5,-0.5]) = (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4)
setCNOT_n([0.,0.,0.,0.,0.,0.,0.,0.,0.]) = SWAP
"""
function setCNOT_n(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #   n type       
    # 1 -—————— 3  
    #      | |       
    # 2 -—————— 4
    #
    unitary = kron(ZYZ_gate(Args[1:3]), ZYZ_gate(Args[4:6]))

    return unitary* CNOT_flip* kron(sigmaI,Ry(Args[7]))* CNOT* kron(Rz(Args[8]), Ry(Args[9]))* CNOT_flip
end
eyeGate_n = [0.0,0.0,-0.5, 0.0,0.0,0.5, 0.5,-0.5,-0.5]

"""
two qubit block of H type (argList = [arg1,arg2,arg3, arg4,arg5,arg6, arg7,arg8,arg9, arg10,arg11,arg12, arg13,arg14,arg15])

                                    CNOT_flip          CNOT           CNOT_filp                                            | quantum circuit (from right to left)
—— Rz(arg1) —— Ry(arg2) —— Rz(arg3) —— + —————————————— · —— Rz(arg8) —— + —— Rz(arg10) —— Ry(arg11) —— Rz(arg12) ——       | |psi>
                                       |                |                |                                                 |
—— Rz(arg4) —— Ry(arg5) —— Rz(arg6) —— · —— Ry(arg7) —— + —— Ry(arg9) —— · —— Rz(arg13) —— Ry(arg14) —— Rz(arg15) ——       | |psi>

setCNOT_H([0.0,0.0,-0.5, 0.0,0.0,0.5, 0.5,-0.5,-0.5, 0.0,0.0,0.0, 0.0,0.0,0.0]) = (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4)
setCNOT_H([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]) = SWAP
"""
function setCNOT_H(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    #     H type       
    # 1 ————————— 3  
    #      | |       
    # 2 ————————— 4
    #
    return setCNOT_n(Args[1:9]) * kron(ZYZ_gate(Args[10:12]),ZYZ_gate(Args[13:15]))
end
eyeGate_H = [0.0,0.0,-0.5, 0.0,0.0,0.5, 0.5,-0.5,-0.5, 0.0,0.0,0.0, 0.0,0.0,0.0]

"""
—— Rz(arg1) —— Ry(arg2) ——       | |0>
"""
function ZY_gate(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    @assert length(Args) == 2
    return Rz(Args[1])* Ry(Args[2])
end
"""
—— Rz(arg1) —— Ry(arg2) —— Rz(arg3) ——       | |psi>
"""
function ZYZ_gate(Args::AbstractVector{Float64})::Matrix{ComplexF64}
    @assert length(Args) == 3
    return Rz(Args[1])* Ry(Args[2])* Rz(Args[3])
end
function ZYZ_gate_SpinOne(Args::AbstractVector{Float64})::Matrix{ComplexF64} # [0,1.0,0]|0> = |2>; [0.,0.5,0.]|0> = 0.5|0> + 0.5|2> + √2/2 |1>
    return exp(-1im*π*(Args[1])*spin1_z) * exp(-1im*π*(Args[2])*spin1_y) * exp(-1im*π*(Args[3])*spin1_z)
end 
function ZYZ_gate_SpinOne4(Args::AbstractVector{Float64})::Matrix{ComplexF64} # [0,1.0,0]|0> = |2>; [0.,0.5,0.]|0> = 0.5|0> + 0.5|2> + √2/2 |1>
    return exp(-1im*π*(Args[1])*spin1_z4) * exp(-1im*π*(Args[2])*spin1_y4) * exp(-1im*π*(Args[3])*spin1_z4)
end 
function ZYZ_gate_Spin3half(Args::AbstractVector{Float64})::Matrix{ComplexF64} # [0,1.0,0]|0> = |3>
    return exp(-1im*π*(Args[1])*spin3half_z) * exp(-1im*π*(Args[2])*spin3half_y) * exp(-1im*π*(Args[3])*spin3half_z)
end

#----------------------------------------------------------------------------------------------------------------------------------------------
"q_output, q_input, LMR='M', return an Iso Matrix (only applicable for bi-MERA!)"
function setIsoTopLayer(q_output::Int,q_input::Int; LMR::Char='M')
    #
    #  output  <————  |psi/0>
    #  output  <————  |psi/0>  input
    #  output  <————  |psi/0>
    #  output  <————  |psi/0>
    #
    @assert q_output >= 0 && q_input >= 0
    if (q_output > 0)
        if (q_output == 4 && q_input == 3)
            if (LMR != 'R')
                psi_or_zero = [1,1,0,1] 
            else
                psi_or_zero = [1,0,1,1]
            end
        else
            if (q_input == 0)
                psi_or_zero = zeros(Int,q_output)
            elseif (q_input == q_output)
                psi_or_zero = ones(Int,q_output)
            else
                NumZeros = q_output - q_input # NumZeros should > 0 and q_output should be even
                if (LMR =='L')
                    psi_or_zero = reshape(append!(ones(Int,q_input), zeros(Int,NumZeros)),:,2)
                    psi_or_zero = vec(transpose(psi_or_zero)) 
                elseif (LMR =='R')
                    psi_or_zero = reshape(append!(zeros(Int,NumZeros), ones(Int,q_input)),:,2)
                    psi_or_zero = vec(transpose(psi_or_zero))
                else
                    Mpart = ones(Int,q_input)
                    Lpart = zeros(Int,NumZeros÷2)
                    Rpart = zeros(Int,(NumZeros+1)÷2)
                    psi_or_zero = cat(Lpart,Mpart,Rpart; dims=1) # Concatenation
                end
            end
        end
            
        return ⊗([sigmaI[:,1:i+1] for i ∈ psi_or_zero])::Matrix{ComplexF64}
    else# (q_output==0)
        @assert q_input == 0
        return Matrix{ComplexF64}(I,1,1)
    end
end

""" LR = 'R' means the MPS bond is on the right, and |0> sites are on the left"""
function setIsoTopLayer_MPS(q_output::Int,q_input::Int; LR::Char='R')
    #
    #  output  <————  |psi/0>
    #  output  <————  |psi/0>  input
    #  output  <————  |psi/0>
    #  output  <————  |psi/0>
    #
    @assert q_output >= 0 && q_input >= 0
    if (q_output > 0)
        psi_or_zero = [0 for _ ∈ 1:q_output]
        for i ∈ 1:q_input
            psi_or_zero[i] = 1     
        end
        if LR == 'R'
            reverse!(psi_or_zero)
        end
                    
        return ⊗([sigmaI[:,1:i+1] for i ∈ psi_or_zero])::Matrix{ComplexF64}
    else# (q_output==0)
        @assert q_input == 0
        return Matrix{ComplexF64}(I,1,1)
    end
end

"χ_output, χ_input, return an Iso Matrix"
function setIsoTopLayerχ(χ_output::Int,χ_input::Int)::Matrix{ComplexF64}
    #  output  <————  |psi/0>
    #  output  <————  |psi/0>  input
    #  output  <————  |psi/0>
    #  output  <————  |psi/0>
    #
    @assert χ_output >= χ_input && χ_input >= 1
    if (χ_output > 1)
        if (χ_input > 1)
            return Matrix{ComplexF64}(I,χ_output,χ_input)[:,1:χ_input]
        else
            return quanReshape(Matrix{ComplexF64}(I,χ_output,χ_input)[:,1:χ_input],(χ_output,χ_input))
        end
    else
        @assert χ_input == 1
        return Matrix{ComplexF64}(I,1,1)
    end
end
#=
Row-major order: C/C++, Python(default)
Column-major order: Fortran, MATLAB, Julia
=#

function myCopyto!(X::AbstractArray{Int64},Y::AbstractArray{Int64})
    @assert size(X) == size(Y)
    @inbounds begin
        for i ∈ eachindex(Y)
            X[i] = Y[i]
        end
    end
    return nothing
end
function myCopyto!(X::AbstractArray{Float64},Y::AbstractArray{Float64})
    @assert size(X) == size(Y)
    @inbounds begin
        for i ∈ eachindex(Y)
            X[i] = Y[i]
        end
    end
    return nothing
end
function myCopyto!(X::AbstractArray{ComplexF64},Y::AbstractArray{ComplexF64})
    @assert size(X) == size(Y)
    @inbounds begin
        for i ∈ eachindex(Y)
            X[i] = Y[i]
        end
    end
    return nothing
end

function self_test_quantumKit()
    println("\nsetIsoTopLayer(4,2,LMR='M'):\n",setIsoTopLayer(4,2,LMR='M'))
    println("\nsetIsoTopLayer(4,2,LMR='L'):\n",setIsoTopLayer(4,2,LMR='L'))
    println("\nsetIsoTopLayer(4,2,LMR='R'):\n",setIsoTopLayer(4,2,LMR='R'))

    println("\nsetTwoQB_u(eyeGate_u)-(-1)^4 =\n\t",(setTwoQB_u(eyeGate_u) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[1,:])
    println("\t",(setTwoQB_u(eyeGate_u) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[2,:])
    println("\t",(setTwoQB_u(eyeGate_u) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[3,:])
    println("\t",(setTwoQB_u(eyeGate_u) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[4,:])
    println("||⋅|| = ",norm(setTwoQB_u(eyeGate_u) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4)))

    println("\n||setTwoQB_n(eyeGate_n)-(-1)^4|| =\n\t",(setTwoQB_n(eyeGate_n) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[1,:])
    println("\t",(setTwoQB_n(eyeGate_n) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[2,:])
    println("\t",(setTwoQB_n(eyeGate_n) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[3,:])
    println("\t",(setTwoQB_n(eyeGate_n) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[4,:])
    println("||⋅|| =",norm(setTwoQB_n(eyeGate_n) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4)))
    
    println("\n||setTwoQB_H(eyeGate_H)-(-1)^4|| =\n\t",(setTwoQB_H(eyeGate_H) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[1,:])
    println("\t",(setTwoQB_H(eyeGate_H) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[2,:])
    println("\t",(setTwoQB_H(eyeGate_H) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[3,:])
    println("\t",(setTwoQB_H(eyeGate_H) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4))[4,:])
    println("||⋅|| = ",norm(setTwoQB_H(eyeGate_H) - (-1+0im)^(1/4)*Matrix{ComplexF64}(I, 4,4)))

    println("\n||setTwoQB_u(0)-SWAP|| = ",norm(setTwoQB_u(zeros(Float64,9)) - SWAP))
    println("||setTwoQB_n(0)-SWAP|| = ",norm(setTwoQB_n(zeros(Float64,9)) - SWAP))
    println("||setTwoQB_H(0)-SWAP|| = ",norm(setTwoQB_H(zeros(Float64,15)) - SWAP))
end

#self_test_quantumKit()
