module MDEforM

# required packages
using CairoMakie
using DifferentialEquations
using HCubature
using LaTeXStrings
using Optim
using Dates
using LinearAlgebra
using NaNMath
using QuadGK

# main code
export Fast_OU_ϵ, Fast_OU_∞, LDA, NLDAM, NSDP
export Langevin_ϵ, Langevin_∞, K, LDO, NLDO
export Langevin_ϵ_2D, Langevin_∞_2D
export Burger_ϵ, Burger_∞
export Fast_chaotic_ϵ, Fast_chaotic_∞
export produce_trajectory_1D, produce_trajectory_2D

include("multiscale_limit_pairs.jl")

export μ, ∂ϑ_μ, ∂Σ_μ

include("invariant_densities.jl")

export Δ, k
export Δ_Gaussian1D, Δ_Gaussian2D

include("MDE_functionals.jl")

export Δ_Gaussian1D_grad
export Δ_grad_ϑ, Δ_grad_Σ

include("MDE_gradients.jl")

export Σ_∞_QdP
export Σ_∞_QrP

include("MDE_asymptotic_variances.jl")

export MDE

include("MDE_optimizers.jl")

end
