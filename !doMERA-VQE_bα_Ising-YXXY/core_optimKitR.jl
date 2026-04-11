# -*- coding: utf-8 -*-
# core_optimKitR.jl
# by Qiang Miao - last modified 05mm/17dd/2022
# using wolfe condition 1 AND 2 to avoid negative-definite approximated Hessian matrix
# 

#---

#-------------------------------------------------------------------------
function g_metric(δ::Vector{Float64}, τ::Vector{Float64})::Float64
    return δ ⋅ τ
end
function λ0(x::Vector{Vector{Float64}},y::Vector{Vector{Float64}})::Float64
    s::Float64 = 0.0
    @inbounds begin
        for i ∈ 1:length(x)
            s += g_metric(x[i],y[i])
        end
    end
    return s
end
function λ1!(x::Vector{Vector{Float64}},c::Float64,y::Vector{Vector{Float64}})
    @inbounds begin
        for i ∈ 1:length(x)
            #@. x[i] = x[i] - c * y[i]
            axpy!(-c,y[i],x[i])
        end
    end
    return nothing
end

function λ2!(x::Vector{Vector{Float64}},c::Float64,y::Vector{Vector{Float64}})
    @inbounds begin
        for i ∈ 1:length(x)
            #@. x[i] = x[i] + c * y[i]
            axpy!(c,y[i],x[i])
        end
    end
    return nothing
end

function λ3!(x::Vector{Vector{Float64}},y::Vector{Vector{Float64}},z::Vector{Vector{Float64}})
    @inbounds begin
        for i ∈ 1:length(y)
            @. x[i] = y[i] - z[i]
        end
    end
    return nothing
end

function λ4!(z1::Vector{Vector{Float64}},z2::Vector{Vector{Float64}},y1::Vector{Vector{Float64}},y2::Vector{Vector{Float64}})
    @inbounds begin
        for i ∈ 1:length(z1)
            copyto!(z1[i],y1[i])
            copyto!(z2[i],y2[i])
        end
    end
    return nothing
end

#-------------------------------------------------------------------------
function m!(x::Vector{Vector{Float64}},c::Float64)
    @inbounds begin
        for i in 1:length(x)
            lmul!(c,x[i])
        end
    end
    return nothing
end
function neg!(x::Vector{Vector{Float64}},y::Vector{Vector{Float64}})
    @inbounds begin
        for i in 1:length(x)
            @. x[i] = - y[i]
        end
    end
    return nothing
end
function mapCopyTo!(x::Vector{Vector{Float64}},y::Vector{Vector{Float64}})
    @inbounds begin
        for i in 1:length(x)
            copyto!(x[i], y[i])
        end
    end
    return nothing
end

#-------------------------------------------------------------------------
"L-BFGS Real function in real variables (mod 2π), rewrite θL0"
function L_BFGS_RR(fg!,θL0::Vector{Vector{Float64}},info,prealloc1,prealloc2; f_target::Float64=0.,c1::Float64=0.1,c2::Float64=0.9,m::Int=9,eps::Float64=1e-12,αiM::Int=14,iterMin::Int = 100, iterMax::Int = 10000, 
    f_array::Vector{Float64} = Float64[], g_array::Vector{Float64} = Float64[], α_array::Vector{Float64} = Float64[], prnt::Bool=false, fnoκ_array::Vector{Float64} = Float64[])

    k::Int = 0; l::Int = 0; γ::Float64 = 1.0
    ∇L0::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)]
    ∇L::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)]
    f0,fnoκ0 = fg!(θL0,∇L0,info,prealloc1,prealloc2) # function f and its arguments θ in list format  
    norm_∇L::Float64 = √λ0(∇L0,∇L0)

    f_array = push!(f_array,f0-f_target); g_array = push!(g_array,norm_∇L); fnoκ_array = push!(fnoκ_array,fnoκ0-f_target); α_array = push!(α_array,0.0) # indicates none of iteration 
    println("input: f0-f_target:", f0-f_target, " |∇L|:", norm_∇L)

    δL::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)] 
    θL::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)] 
    sL::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)] 
    yL::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)] #计算速度好于deepcopy(θL0)或者deepcopy(∇L0)，也许因为他们内存地址不连续。但为何deepcopy的结果也不连续呢？

    sLL::Vector{Vector{Vector{Float64}}}=[deepcopy(sL) for _ ∈ 1:m+1]; yLL::Vector{Vector{Vector{Float64}}}=[deepcopy(yL) for _ ∈ 1:m+1]; ρL = Vector{Float64}(undef,0)
   
    while ((norm_∇L > eps || k < iterMin) && k < iterMax)

        neg!(δL,∇L0)
        μ = Vector{Float64}(undef,0)

        # two-loop recursion                                                                    # <--- bottleneck
        for i ∈ k-l:-1:1
            push!(μ, ρL[i]* λ0(sLL[i+m-(k-l)+1],δL))  # Adding element at the end
            λ1!(δL,μ[end],yLL[i+m-(k-l)+1]) #q = q - μ[end]*yLL[i]
        end
        m!(δL,γ)
        for i ∈ 1:k-l
            w = ρL[i] * λ0(yLL[i+m-(k-l)+1],δL)
            λ2!(δL,μ[k-l+1-i]-w,sLL[i+m-(k-l)+1])
        end 
        
        α,f,fnoκ = line_search!(∇L,θL,sL,f0,∇L0, fg!,θL0,info,prealloc1,prealloc2, δL,c1,c2; iterMax=αiM)
        #λ3 = (x,y,z) -> x = y - z
        λ3!(yL, ∇L,∇L0)                                                        

        gsy_temp = λ0(sL,yL)
        push!(ρL,1.0/gsy_temp) # may have NAN due to possible zero denominater
        γ = gsy_temp/ λ0(yL,yL)
        
        l = max(k-m,0)
        if (k>m)
             popfirst!(ρL) # Delete first element 
        end
       
        # shift sLL and yLL
        Threads.@threads for i ∈ 1:k-l
            λ4!(sLL[i+m-k+l],yLL[i+m-k+l],sLL[i+1+m-k+l],yLL[i+1+m-k+l])
        end
        mapCopyTo!(yLL[m+1],yL); mapCopyTo!(sLL[m+1],sL)
             
        f0 = f; mapCopyTo!(∇L0, ∇L); mapCopyTo!(θL0, θL); norm_∇L = √λ0(∇L0,∇L0) 
        f_array = push!(f_array,f-f_target); g_array = push!(g_array,norm_∇L); α_array = push!(α_array,α); fnoκ_array = push!(fnoκ_array,fnoκ-f_target)
        k = k + 1
        if (prnt==true)
            println("iter ",k, ", f(κ)-f_target=", f-f_target, ", |∇L|=", norm_∇L,  ", α_(",k-1 ,"-->",k,")=", α, ", f-f_target=", fnoκ-f_target)
        end

        μ = nothing  
    
        if (mod(k, 50) == 0)
            GC.gc()
        end
    end

    return f0::Float64, θL0, f_array, g_array, α_array, fnoκ_array
end

#---
function line_search!(∇L::Vector{Vector{Float64}},θL::Vector{Vector{Float64}},sL::Vector{Vector{Float64}},
    f0::Float64,∇L0, fg!,θL0,info,prealloc1,prealloc2, δL,c1::Float64,c2::Float64; α::Float64 = 1.0,iterMax::Int=14, wolfe_bool::Bool = false)
    
    wolfe0::Float64 = 0.0
    wolfe1::Float64 = 0.0 
    wolfe2::Float64 = 0.0
    iteration::Int = 0
    f::Float64 = 0.0; fnoκ::Float64 = 0.0

    #---------------------------
    while wolfe_bool == false                                                                 # <--- bottleneck

        λ5!(θL,sL,α,δL,θL0)
        #θL = θL0 + α * δL and mod by 2

        f,fnoκ = fg!(θL,∇L,info,prealloc1,prealloc2)
        wolfe0 = α * λ0(∇L0,δL)
        wolfe1 = f - f0 - c1 * wolfe0 # strictly descent (c1>0) might be harmful when trying to jump from local min

        wolfe2 = λ0(∇L,sL) - c2 * wolfe0

        if ((wolfe1 <= 0 && wolfe2 >= 0) || iteration >= iterMax)
            wolfe_bool = true
        else
            iteration = iteration + 1; α = α /2.0
        end
    end
    
    return α::Float64,f::Float64,fnoκ::Float64
end

function λ5!(θL::Vector{Vector{Float64}},sL::Vector{Vector{Float64}},α::Float64,δL::Vector{Vector{Float64}},θL0::Vector{Vector{Float64}})
    for i ∈ 1:length(θL0)
        copyto!(sL[i], α .* δL[i])
        for j ∈ 1:length(θL0[i])
            θL[i][j] = argRevise(θL0[i][j], sL[i][j])
        end
    end
    return nothing
end

function argRevise(θ0::Float64,s::Float64)::Float64
    tmp = mod(θ0+s,2)
    if tmp >= 1.0
        tmp -= 2.0
    end
    return tmp
end






# No penalty term
#-------------------------------------------------------------------------
"L-BFGS Real function in real variables (mod 2π), rewrite θL0"
function L_BFGS_RR0(fg!,θL0::Vector{Vector{Float64}},info,prealloc1,prealloc2; f_target::Float64=0.,c1::Float64=0.1,c2::Float64=0.9,m::Int=9,eps::Float64=1e-12,αiM::Int=14,iterMin::Int = 100, iterMax::Int = 10000, 
    f_array::Vector{Float64} = Float64[], g_array::Vector{Float64} = Float64[], α_array::Vector{Float64} = Float64[], prnt::Bool=false, fnoκ_array::Vector{Float64} = Float64[])

    k::Int = 0; l::Int = 0; γ::Float64 = 1.0
    ∇L0::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)]
    ∇L::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)]
    f0 = fg!(θL0,∇L0,info,prealloc1,prealloc2) # function f and its arguments θ in list format  
    norm_∇L::Float64 = √λ0(∇L0,∇L0)

    f_array = push!(f_array,f0-f_target); g_array = push!(g_array,norm_∇L); α_array = push!(α_array,0.0) # indicates none of iteration 
    println("input: f0-f_target:", f0-f_target, " |∇L|:", norm_∇L)

    δL::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)] 
    θL::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)] 
    sL::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)] 
    yL::Vector{Vector{Float64}} = [similar(θL0[i]) for i ∈ 1:length(θL0)] #计算速度好于deepcopy(θL0)或者deepcopy(∇L0)，也许因为他们内存地址不连续。但为何deepcopy的结果也不连续呢？

    sLL::Vector{Vector{Vector{Float64}}}=[deepcopy(sL) for _ ∈ 1:m+1]; yLL::Vector{Vector{Vector{Float64}}}=[deepcopy(yL) for _ ∈ 1:m+1]; ρL = Vector{Float64}(undef,0)
   
    while ((norm_∇L > eps || k < iterMin) && k < iterMax)

        neg!(δL,∇L0)
        μ = Vector{Float64}(undef,0)

        # two-loop recursion                                                                    # <--- bottleneck
        for i ∈ k-l:-1:1
            push!(μ, ρL[i]* λ0(sLL[i+m-(k-l)+1],δL))  # Adding element at the end
            λ1!(δL,μ[end],yLL[i+m-(k-l)+1]) #q = q - μ[end]*yLL[i]
        end
        m!(δL,γ)
        for i ∈ 1:k-l
            w = ρL[i] * λ0(yLL[i+m-(k-l)+1],δL)
            λ2!(δL,μ[k-l+1-i]-w,sLL[i+m-(k-l)+1])
        end 
        
        α,f = line_search0!(∇L,θL,sL,f0,∇L0, fg!,θL0,info,prealloc1,prealloc2, δL,c1,c2; iterMax=αiM)
        #λ3 = (x,y,z) -> x = y - z
        λ3!(yL, ∇L,∇L0)                                                        

        gsy_temp = λ0(sL,yL)
        push!(ρL,1.0/gsy_temp) # may have NAN due to possible zero denominater
        γ = gsy_temp/ λ0(yL,yL)
        
        l = max(k-m,0)
        if (k>m)
             popfirst!(ρL) # Delete first element 
        end
       
        # shift sLL and yLL
        Threads.@threads for i ∈ 1:k-l
            λ4!(sLL[i+m-k+l],yLL[i+m-k+l],sLL[i+1+m-k+l],yLL[i+1+m-k+l])
        end
        mapCopyTo!(yLL[m+1],yL); mapCopyTo!(sLL[m+1],sL)
             
        f0 = f; mapCopyTo!(∇L0, ∇L); mapCopyTo!(θL0, θL); norm_∇L = √λ0(∇L0,∇L0) 
        f_array = push!(f_array,f-f_target); g_array = push!(g_array,norm_∇L); α_array = push!(α_array,α)
        k = k + 1
        if (prnt==true)
            println("iter ",k, ", f-f_target=", f-f_target, ", |∇L|=", norm_∇L,  ", α_(",k-1 ,"-->",k,")=", α)
        end

        μ = nothing  
    
        if (mod(k, 50) == 0)
            GC.gc()
        end
    end

    return f0::Float64, θL0, f_array, g_array, α_array
end

#---
function line_search0!(∇L::Vector{Vector{Float64}},θL::Vector{Vector{Float64}},sL::Vector{Vector{Float64}},
    f0::Float64,∇L0, fg!,θL0,info,prealloc1,prealloc2, δL,c1::Float64,c2::Float64; α::Float64 = 1.0,iterMax::Int=14, wolfe_bool::Bool = false)
    
    wolfe0::Float64 = 0.0
    wolfe1::Float64 = 0.0 
    wolfe2::Float64 = 0.0
    iteration::Int = 0
    f::Float64 = 0.0

    #---------------------------
    while wolfe_bool == false                                                                 # <--- bottleneck

        λ5!(θL,sL,α,δL,θL0)
        #θL = θL0 + α * δL and mod by 2

        f = fg!(θL,∇L,info,prealloc1,prealloc2)
        wolfe0 = α * λ0(∇L0,δL)
        wolfe1 = f - f0 - c1 * wolfe0 # strictly descent (c1>0) might be harmful when trying to jump from local min

        wolfe2 = λ0(∇L,sL) - c2 * wolfe0

        if ((wolfe1 <= 0 && wolfe2 >= 0) || iteration >= iterMax)
            wolfe_bool = true
        else
            iteration = iteration + 1; α = α /2.0
        end
    end
    
    return α::Float64,f::Float64
end