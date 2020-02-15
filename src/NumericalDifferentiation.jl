using LinearAlgebra, SparseArrays
using NumericalIntegration: integrate, cumul_integrate
using LinearMaps, IterativeSolvers#, Preconditioners

abstract type DifferentiationMethod end

struct TotalVariation{T<:AbstractFloat} <: DifferentiationMethod
    ϵ::T
end
TotalVariation() = TotalVariation(1e-12)

struct Tikhonov <: DifferentiationMethod end


"""
	diffmatrix(x::AbstractVector)
"""
function diffmatrix(x::AbstractVector)
    D = zeros(eltype(x), length(x)-1, length(x))
    for i in 1:size(D,1)
        D[i,i]   = -1/(x[i+1]-x[i])
        D[i,i+1] =  1/(x[i+1]-x[i])
    end
    return D
end

"""
	spdiffmatrix(x::AbstractVector)
"""
function spdiffmatrix(x::AbstractVector)
    el = 1 ./diff(x)
    I = [1:length(x)-1;1:length(x)-1]
    J = [1:length(x)-1;2:length(x)]
    K = [-el;el]

    return sparse(I,J,K)
end

"""
	intgmatrix(x::AbstractVector)
"""
function intgmatrix(x::AbstractVector)
    K = zeros(eltype(x), length(x), length(x))
    for  j in 2:size(K,2), i in 1:size(K,1)
        if i>=j
            K[i,j-1] += .5*(x[j] - x[j-1])
            K[i,j]   += .5*(x[j] - x[j-1])
        end
        
    end
    return K
end

"""
	integrationoperator(x,y)
"""
function integrationoperator(x,y)
    retarr = zeros(eltype(x),size(x)) + zeros(eltype(y),size(y))
    for i in 2 : length(y)
        retarr[i] = retarr[i-1] + (x[i] - x[i-1]) * (y[i-1] + y[i])
    end
    return 0.5 * retarr
end

"""
	integrationadjoint(x,y)
"""
function integrationadjoint(x,y)
    z = zeros(eltype(x),size(x)) + zeros(eltype(y),size(y))
    N = length(z)
    
    z[N] = y[N]*(x[N]-x[N-1])

    sumy = y[N]
    
    for i in N-1:-1:2
        z[i] = (x[i]-x[i-1])*y[i] + (x[i+1]-x[i-1])*sumy
        sumy += y[i]
    end

    z[1] = (x[2]-x[1])*sumy
    
    return 0.5*z
end

"""
	Ψ(t, ϵ)

Regularised absolute value.
"""
function Ψ(t, ϵ)
    return 2*sqrt(t + ϵ^2)
end

"""
	Ψp(t, ϵ)

Derivative of regularised absolute value.
"""
function Ψp(t,ϵ)
    return 1/sqrt(t + ϵ^2)
end

"""
	diffoperator(du,Δx,D)
"""
function diffoperator(du,Δx,D,epsilon)
    psi = Diagonal(Ψp.(du.^2,epsilon))
    δx  = Diagonal(Δx)
    return (D')*(δx*psi)*D
end

"""
	derivative(x::AbstractVector, f::AbstractVector, ::TotalVariation, α::AbstractFloat, ::LaggedDiff)
"""
function differentiate(x::AbstractVector, f::AbstractVector, tv::TotalVariation, α, tol; maxit=2000, pbsize=:auto, plt=false, u0=[0;diff(f)])

    (length(x) == length(f)) || throw(DimensionMismatch("Inputs `x` and `f` should have same dimension"))

    c = 0
    u = u0
    Δx = diff(x)

    if plt
        figure()
        plot(x, u)
        draw()
    end

    if pbsize==:small || (pbsize==:auto && length(f)<1001)
        D = diffmatrix(x)
        A = intgmatrix(x)
        df = A'*(f .- f[1])
        
        keep = true

        while keep && c<maxit
            c += 1
            
            # make diff operator
            du = D*u
            L = diffoperator(du,Δx,D,tv.ϵ)
            # gradient
            g = A'*A*u - df + α*L*u
            # approx. hessian
            H = A'*A + α*L
            # QN step
            s = -H\g

            u += s

            keep = (norm(g)>tol)
            
            print("iteration $c | grad $(norm(g))\r")

            if plt
                cla()
                plot(x,u)
                draw()
            end

        end
    elseif pbsize==:large || (pbsize==:auto && length(f)>=1001)
        D = spdiffmatrix(x)
        df = integrationadjoint(x, f .- f[1])

        keep = true

        onevec = ones(length(x))
        
        while keep && c<maxit
            c += 1
            
            # make diff operator
            du = D*u
            L = diffoperator(du,Δx,D,tv.ϵ)
            # gradient
            g = integrationadjoint(x, integrationoperator(x,u)) - df + α*L*u
            # preconditioner
            #B = α*L + spdiagm(0=>integrationadjoint(x,integrationoperator(x,onevec)))
            #R = CholeskyPreconditioner(B,2,1e-2)
            # approx. hessian
            H = LinearMap(v -> integrationadjoint(x, integrationoperator(x,v)) + α*L*v, length(x))
            # QN step
            s = cg(H, -g)#, Pl=R)

            u += s

            keep = (norm(g)>tol)

            print("iteration $c | grad $(norm(g))\r")

            if plt
                cla()
                plot(x,u)
                draw()
            end
        end

    end
    
    if c==maxit
        println("max. number of iterations reached. result is shit.")
    end
    
    return u
end

function differentiate(x::AbstractVector, f::AbstractVector, ::Tikhonov, α; pbsize=:auto, u0=[0;diff(f)])
    if pbsize==:small || (pbsize==:auto && length(f)<1001)
        D = diffmatrix(x)
        A = intgmatrix(x)
        r = A'*(f .- f[1])
        δ = tr(D'D)/length(x)^3
        H = α*δ*D'D + A'A
        u = H\r        
    elseif pbsize==:large || (pbsize==:auto && length(f)>=1001)
        D = spdiffmatrix(x)
        δ = tr(D'D)/length(x)^3
        L = α*δ*D'D
        r = integrationadjoint(x, f .- f[1])
        H = LinearMap(v -> integrationadjoint(x, integrationoperator(x,v)) + L*v, length(x))
        u = cg(H, r)
    end

    return u
end
