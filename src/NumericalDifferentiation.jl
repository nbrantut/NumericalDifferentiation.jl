module NumericalDifferentiation

export differentiate, integrationoperator, TotalVariation, Tikhonov

using LinearAlgebra, SparseArrays
using LinearMaps, IterativeSolvers

abstract type DifferentiationMethod end

struct TotalVariation{T<:AbstractFloat} <: DifferentiationMethod
    ϵ::T
end
TotalVariation() = TotalVariation(1e-12)

struct Tikhonov <: DifferentiationMethod end

"""
	diffmatrix(x::AbstractVector)

Generate differentiation matrix from node positions `x`.
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

Generate sparse differentiation matrix from node positions `x`.
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

Generate integration matrix using Trapezoidal rule from node positions `x`.
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

Integration operator using trapezoidal rule. Return the integral ``\\int_0^x y(u)du`` for all `x`. This code is reproduced from NumericalIntegration.jl (copied to avoid unnecessary dependencies and compatibility issues).
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

Adjoint of integration operator. Constructed such that the corresponding matrix is truly the transpose of the matrix of the integration operator.
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

Construct differentiation operator for TV method.
"""
function diffoperator(du,Δx,D,epsilon)
    psi = Diagonal(Ψp.(du.^2,epsilon))
    δx  = Diagonal(Δx)
    return (D')*(δx*psi)*D
end

"""
	differentiate(x::AbstractVector, f::AbstractVector, tv::TotalVariation, α, tol; maxit=2000, pbsize=:auto, verbose=false, precond=false, u0=[0;diff(f)])

Compute the derivative of vector `f` defined at positions `x`, using Total Variation regularisation with parameter `α`. The method is described in Chartrand, R. (2011), "Numerical Differentiation of Noisy, Nonsmooth Data", ISRN Appl. Math., doi:10.5402/2011/164564.

# Arguments
- `x::AbstractVector`: x coordinates where vector f is defined.
- `f::AbstractVector`: input vector to take derivative of.
- `tv::TotalVariation`: instance of TotalVariation type to specify method.
- 'α': regularisation parameter.
- 'tol': tolerance on magnitude of gradient to stop iterations.

# Optional keyword arguments
- `maxit=2000`: maximum number of iterations.
- 'pbsize=:auto': if `:auto` (default), selects the matrix-based computation method only for problems with less than 1001 points. Enforce use of matrix-based method if set to `:small`, and enforce use of matrix-free method is set to `:large`.
- verbose=false. if true, shows convergence info.
- precond=false. if true, use simple preconditioner. Not used for small systems.
- 'u0=[0;diff(f)]': Can be used to set initial guess for derivative.

"""
function differentiate(x::AbstractVector, f::AbstractVector, tv::TotalVariation, α, tol; maxit=2000, pbsize=:auto, verbose=false, precond=false, u0=[0;diff(f)])

    (length(x) == length(f)) || throw(DimensionMismatch("Inputs `x` and `f` should have same dimension"))

    c = 0
    u = u0
    Δx = diff(x)

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

            keep = (norm(s)/norm(u)>tol)
            
            !verbose || print("iteration $c | change $(norm(s)/norm(u)) | grad $(norm(g))\r")

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

            # approx. hessian
            H = LinearMap(v -> integrationadjoint(x, integrationoperator(x,v)) + α*L*v, length(x))
            if precond
                P = Diagonal(α*diag(L) .+1)
            else
                P = I
            end
            # QN step
            s = cg(H, -g, Pl=P)

            u += s

            keep = (norm(s)/norm(u)>tol)

            !verbose || print("iteration $c | change $(norm(s)/norm(u)) | grad $(norm(g))\r")

        end

    end

    !verbose || println("\nMax. iterations $c")
    
    return u
end

"""
	differentiate(x::AbstractVector, f::AbstractVector, ::Tikhonov, α; pbsize=:auto)

Compute the derivative of vector `f` defined at positions `x`, using Tikhonov regularisation with parameter `α`. Optional keyword argument `pbsize` can be set to `:small` (uses matrix-based method), `:large` (uses matrix-free method), or `:auto` (default; chooses matrix-based method if `f` has less than 1001 points).

"""
function differentiate(x::AbstractVector, f::AbstractVector, ::Tikhonov, α; pbsize=:auto)
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

end
