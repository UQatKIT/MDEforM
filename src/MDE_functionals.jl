########################################################################################################
########################################################################################################
# A new approach for the drift estimation in multiscale settings based on a minimum distance estimation 
# method utilizing characteristic functions of the invariant distribution of the effective limit model.
# The weight function ϕ is chosen as a centered normal distribution with variance β², so that
# a lot of formulas simplify for the implementation. This script evaluates the MDE in the case of an
# invariant density with a non-quadratic potential in the exponent and in the case of a 
# Gaussian invariant density in one and two dimensions.
########################################################################################################
########################################################################################################
# Jaroslav Borodavka, 14.08.2024

########################################################################################################
## cost functionals for the MDE
########################################################################################################

## general case in 1D ##

# characteristic function of gaussian density for estimation, see calculations from main manuscript
@doc raw"""
    k(x, β)

Return `β` and function value of the characteristic function of a centered Gaussian density with standard deviation `β` at the point `x` as a tuple.

This characteristic function is given by
```math
\begin{aligned}
  k_\beta(x) = \exp\left( -\frac{\beta^2 x^2}{2} \right), \quad x \in \R.
\end{aligned}
```
It is used in the definition of the MDE and thoroughly outlined in the main manuscript in the numerics section.

---
# Arguments
- `x::Real`:         argument ``x`` at which to evaluate the function.
- `β::Real`:         positive number ``\beta``.
"""
function k(x, β=1)
  exp(-β^2*x^2/2), β
end

# convolution over which we have to integrate twice; once with respect to the data in a time integral,
# once over the whole domain of the invariant density, i.e. R, cf. numerics section of main manuscript
function inner_convol(x, ϑ, Σ, V)
  HCubature.hquadrature(y -> μ(x-t(y), ϑ, Σ, V)k(t(y))[1]dt(y), -1, 1)[1]
end

# space integral in cost functional, integration of inner_convol over R, see above
function convol(ϑ, Σ, V)
  f(y) = inner_convol(t(y), ϑ, Σ, V)μ(t(y), ϑ, Σ, V)dt(y)
  # functions are symmetric in the considered cases
  2HCubature.hquadrature(f, 0, 1)[1]
end

# time integral in cost functional, integration of inner_convol over data points, see above; serial version;
# no division by N here since we use this serial version for the parallel version below, division by N occurs there
function time_integral(data, ϑ, Σ, V)
  N = length(data)
  integral_val = 0.0

  for i in 1:N
    integral_val = integral_val + inner_convol(data[i], ϑ, Σ, V)  
  end

  -2integral_val
end

# time integral in cost functional; parallel version via multithreading; written with data-race freedom
function multi_time_integral(data, ϑ, Σ, V)
  N = length(data)
  # divison by 100 depending on number of threads
  data_batches = Iterators.partition(data, convert(Int, N/100))
  sum_atomic = Threads.Atomic{Float64}(0)

  @inbounds Threads.@threads for data_batch in collect(data_batches)
    res = time_integral(data_batch, ϑ, Σ, V)
    Threads.atomic_add!(sum_atomic, res)
  end 
  sum_atomic[]/N
end

# complete cost functional, parallel version via multi-threading
@doc raw"""
    Δ(data, ϑ, Σ, V)

Compute cost functional for given `data` and parameter values `ϑ` and `Σ`.

A properly discretized version of the cost functional, given by
```math
\begin{aligned}
  \Delta_T(X_\epsilon, \vartheta, \Sigma, V) = - \frac{2}{T} \int_0^T (\mu(\vartheta, \Sigma, V) \ast k_\beta)(X_\epsilon(t)) \, dt + \int_{\R} (\mu(\vartheta, \Sigma, V) \ast k_\beta)(x) \mu(\vartheta, \Sigma, x) \, dx,
\end{aligned}
```
is implemented and evaluated via [multithreading](https://docs.julialang.org/en/v1/manual/multi-threading/). Here, ``X_ϵ`` is a one-dimensional time series of length ``T``,
obtained from a multiscale SDE, ``\mu`` is the invariant density of the homogenized limit SDE, ``k_\beta`` refers to [`k`](@ref), and ``\ast`` is the convolution operator on ``\R``.
See the main manuscript for details on this functional. It is the core object of the MDE.

!!! warning 
    The computational cost of this function is quite high due to the integration of the convolutions, so if the data is finely discretized, then
    the running times for a single evalutation are relatively long. Remember that, in this case of a 
    non-quadratic potential, further simplifications of the above formula are not known thus far.

!!! note 
    When comparing the above formula with the formula from the main manuscript, then one notices that the double integral term is missing above. 
    This is on purpose because the double integral does not depend on any parameters with respect to which we will optimize.


---
# Arguments
- `data::Vector{Real}`:         one-dimensional time series ``X_ϵ``.
- `ϑ::Real`:                    positive drift coefficient ``\vartheta``.
- `Σ::Real`:                    positive diffusion coefficient ``\Sigma``.
- `V`:                          defining potential function ``V`` for the invariant density.

---
# Examples
```
$ julia --threads 10 --project=. # start julia with 10 threads and activate project
```
```julia-repl
julia> Threads.nthreads()
julia> using MDEforM
julia> data = Langevin_ϵ(1.0, func_config=NLDO(), α=2.0, σ=1.0, ϵ=0.1, T=100)[1]
julia> Δ(data, 1, 1, NLDO()[1])
```

---
See also [`Δ_Gaussian1D`](@ref), [`Δ_Gaussian2D`](@ref).
"""
function Δ(data, ϑ, Σ, V)
  time_stamp = Dates.format(now(), "HH:MM:SS")
  @info "⊙ $(time_stamp) - Functional call with parameter values ($(round(ϑ, digits=6)), $(round(Σ, digits=6)))."
  convol(ϑ, Σ, V) + multi_time_integral(data, ϑ, Σ, V)
end

## Gaussian case in 1D via exact distance formula ##

@doc raw"""
    Δ_Gaussian1D(data, ϑ, Σ)

Compute cost functional for given one-dimensonal `data` and parameter values `ϑ` and `Σ` in the case where the invariant density of the homogenized limit SDE is centered Gaussian.

A properly discretized version of the cost functional, given by
```math
\begin{aligned}
  \Delta_T(X_\epsilon, \vartheta, \Sigma, V) = -\frac{2}{T \sqrt{1 + \beta^2 \frac{\Sigma}{\vartheta}} } \int_0^T \exp\left( -\frac{\beta^2 X_\epsilon(t)^2}{2 (1 + \beta^2 \frac{\Sigma}{\vartheta})} \right) \, dt + \frac{1}{\sqrt{1 + 2 \beta^2 \frac{\Sigma}{\vartheta}}},
\end{aligned}
```
is implemented. Here, ``X_ϵ`` is a one-dimensional time series of length ``T``, obtained from a multiscale SDE, and ``\beta`` comes from [`k`](@ref). The potential is here ``V(x) = x^2/2``.
See the main manuscript for details on this functional. It is the core object of the MDE.

!!! note 
    The evaluation is, compared to [`Δ`](@ref), extremely fast, even for finely discretized data, which signifies the utility of choosing a centered Gaussian weight [`k`](@ref)
    in this Gaussian case.

---
# Arguments
- `data::Vector{Real}`:         one-dimensional time series ``X_ϵ``.
- `ϑ::Real`:                    positive drift coefficient ``\vartheta``.
- `Σ::Real`:                    positive diffusion coefficient ``\Sigma``.

---
# Examples
```julia-repl
julia> using MDEforM
julia> data = Langevin_ϵ(1.0, func_config=LDO(), α=2.0, σ=1.0, ϵ=0.1, T=100)[1]
julia> Δ_Gaussian1D(data, 1, 1)
```

---
See also [`Δ`](@ref), [`Δ_Gaussian2D`](@ref).
"""
function Δ_Gaussian1D(data, ϑ, Σ)
  β = k(0)[2]
  δ1 = 1/sqrt(1 + β^2*Σ/ϑ)
  δ2 = 1/sqrt(1 + 2β^2*Σ/ϑ)
  
  N = length(data)
  single_integral = 0.0

  for i in 1:N
    single_integral = single_integral + k(data[i]δ1)[1]
  end
  -2δ1/N*single_integral+δ2
end

## Gaussian case in 2D via exact distance formula ##

# transforming 2D data into 1D data for the exponential in the distance Δ and 
# calculating a determinant relevant for the distance formula;
# input arguments are the same as in Δ_Gaussian2D, see docs
# most appearing functions come from LinearAlgebra.jl
function transf_data_2D(data, ϑ, Σ)
  d = length(data[:,1])   # d=2 in our considered case
  N = length(data[1,:])
  I_d = I[1:d,1:d]  # identity matrix
  β = k(0)[2]
  ϑ_inv = inv([ϑ[1] ϑ[2]; ϑ[3] ϑ[4]])
  inverse_mat = inv(I_d + β^2*ϑ_inv*Σ)
  det_ϑ_Σ = det(I_d + β^2*ϑ_inv*Σ)

  #time_stamp = Dates.format(now(), "HH:MM:SS")
  #@info "⊙ $(time_stamp) - Covariance matrix equals $(ϑ_inv*Σ)."

  # NaNMath is used due to the way the optimizers in Optim.jl work; they relax optimization constraints which, however,
  # can yield DomainErrors; hence, the circumvention with NaNMath, cf. https://docs.sciml.ai/Optimization/stable/API/FAQ/
  transformed_data = [NaNMath.sqrt(data[:,i]' * inverse_mat * data[:,i]) for i ∈ 1:N]
  transformed_data, det_ϑ_Σ, ϑ_inv
end

@doc raw"""
    Δ_Gaussian2D(data, ϑ, Σ)

Compute cost functional for given two-dimensonal `data` and parameter values `ϑ` and `Σ` in the case where the invariant density of the homogenized limit SDE is centered Gaussian.

A properly discretized version of the cost functional, given by
```math
\begin{aligned}
  \Delta_T(X_\epsilon, \vartheta, \Sigma) &= - \frac{2}{T \sqrt{\det(I_d + \beta^2 M(\vartheta, \Sigma)) }} \int_0^T \exp\left( -\frac{\beta^2}{2} X_\epsilon(t)^\top \left( I_d + \beta^2 M(\vartheta, \Sigma) \right)^{-1} X_\epsilon(t) \right) \, dt \\[0.25cm]
    &+ \frac{1}{\sqrt{\det(I_d + 2 \beta^2 M(\vartheta, \Sigma))}},
\end{aligned}
```
is implemented. Here, ``X_ϵ`` is a two-dimensional time series of length ``T``, obtained from a multiscale SDE, ``M(\vartheta, \Sigma) \in \R^{2 \times 2}`` is a matrix depending on ``\vartheta`` and ``\Sigma``
and is given by the covariance matrix of the invariant Gaussian density, and ``\beta`` comes from [`k`](@ref). See the main manuscript for details on this functional and the appearing quantities. It is the core object of the MDE.

!!! note 
    The evaluation is, compared to [`Δ`](@ref), extremely fast, even for finely discretized two-dimensional data, which signifies the utility of choosing a two-dimensional centered Gaussian weight [`k`](@ref)
    in this Gaussian case.

---
# Arguments
- `data::Vector{Real}`:         two-dimensional time series ``X_ϵ``.
- `ϑ::Array{Real}`:             positive definite drift matrix ``\vartheta \in \mathbb{R}^{2 \times 2}``.
- `Σ::Array{Real}`:             positive definite diffusion matrix ``\Sigma \in \mathbb{R}^{2 \times 2}``.

---
# Examples
```julia-repl
julia> using MDEforM
julia> M=[4 2;2 3]
julia> σ = 5.0  
julia> data = Langevin_ϵ_2D([-5.0, -5.0], func_config=(x-> cos(x), x -> 1/2*cos(x)), M=M, σ=σ, ϵ=0.1, T=100.0)[1]
julia> CorrK = [K(x-> cos(x), σ) 0 ; 0 K(x -> 1/2*cos(x), σ)]
julia> ϑ = CorrK*M
julia> Σ = σ*CorrK
julia> Δ_Gaussian2D(data, ϑ, Σ)
```

---
See also [`Δ`](@ref), [`Δ_Gaussian1D`](@ref).
"""
function Δ_Gaussian2D(data, ϑ, Σ)
  #time_stamp = Dates.format(now(), "HH:MM:SS")
  #@info "⊙ $(time_stamp) - Function call with parameter value $(ϑ)."
  d = length(data[:,1])
  N = length(data[1,:])
  β = k(0)[2]
  I_d = I[1:d,1:d]
  transformed_data, det_ϑ_Σ, ϑ_inv = transf_data_2D(data, ϑ, Σ)

  δ1 = 1/NaNMath.sqrt(det_ϑ_Σ)
  δ2 = 1/NaNMath.sqrt(det(I_d + 2β^2*ϑ_inv*Σ))
  
  single_integral = 0.0

  for i in 1:N
    single_integral += k(transformed_data[i])[1]
  end
  
  -2δ1/N*single_integral+δ2
end