########################################################################################################
########################################################################################################
# Collecting a couple of gradients of the minimum distance approach for different types of effective 
# diffusion processes and their respective invariant density.
########################################################################################################
########################################################################################################
# Jaroslav Borodavka, 19.08.2024

########################################################################################################
## gradients for the MDE
########################################################################################################

## general case in 1D ##

## drift parameter estimation (e.g. Langevin) ##

# inner convolution expression in gradient of cost functional; derivative of inner_convol with respect to ϑ
function ∂ϑ_inner_convol(x, ϑ, Σ, V)
    HCubature.hquadrature(y -> ∂ϑ_μ(x-t(y), ϑ, Σ, V)k(t(y))[1]dt(y), -1, 1)[1]
end

# space integral in gradient of cost functional
function ∂ϑ_convol(ϑ, Σ, V)
    f(y) = 2inner_convol(t(y), ϑ, Σ, V)∂ϑ_μ(t(y), ϑ, Σ, V)dt(y)
    # functions are symmetric
    2HCubature.hquadrature(f, 0, 1)[1]
end

# time integral in gradient of cost functional, integration of ∂ϑ_inner_convol over data points, see above; serial version;
# no division by N here since we use this serial version for the parallel version below, division by N occurs there
function ∂ϑ_time_integral(data, ϑ, Σ, V)
    N = convert(Int, length(data))
    integral_val = 0.0
  
    for i in 1:N
      integral_val = integral_val + ∂ϑ_inner_convol(data[i], ϑ, Σ, V)  
    end
  
    -2integral_val
end

# time integral in gradient of cost functional; parallel version via multithreading; written with data-race freedom
function ∂ϑ_multi_time_integral(data, ϑ, Σ, V)
    N = length(data)
    # divison by 100 depending on number of threads
    data_batches = Iterators.partition(data, convert(Int, N/100))
    sum_atomic = Threads.Atomic{Float64}(0)

    @inbounds Threads.@threads for data_batch in collect(data_batches)
      res = ∂ϑ_time_integral(data_batch, ϑ, Σ, V)
      Threads.atomic_add!(sum_atomic, res)
    end 
    sum_atomic[]/N
end
  
# complete gradient of cost functional with respect to ϑ, parallel version via multi-threading
@doc raw"""
    Δ_grad_ϑ(data::Vector{<:Real}, ϑ::Real, Σ::Real, V::Function)

Compute gradient of cost functional [`Δ`](@ref) with respect to `ϑ` for given `data` and parameter values `ϑ` and `Σ` and a potential `V`.

A properly discretized version of the gradient of the cost functional, given by
```math
\begin{aligned}
  \partial_\vartheta \Delta_T(X_\epsilon, \vartheta, \Sigma, V) = -\frac{2}{T} \int_0^T (\partial_\vartheta \mu(\vartheta, \Sigma, V) \ast k_\beta)(X_\epsilon(t)) \, dt + 2 \int_{\R} (\mu(\vartheta, \Sigma, V) \ast k_\beta)(x) \partial_\vartheta \mu(\vartheta, \Sigma, x) \, dx,
\end{aligned}
```
is implemented and evaluated via [multithreading](https://docs.julialang.org/en/v1/manual/multi-threading/). Here, ``X_ϵ`` is a one-dimensional time series of length ``T``,
obtained from a multiscale SDE, ``\mu`` is the invariant density of the homogenized limit SDE corresponding to [`μ`](@ref), ``k_\beta`` refers to [`k`](@ref), and ``\ast`` is the convolution operator on ``\R``.

!!! warning 
    The computational cost of this function is quite high due to the integration of the convolutions, so if the data is finely discretized, then
    the running times for a single evalutation are relatively long. Remember that, in this case of a 
    non-quadratic potential, further simplifications of the above formula are not known thus far.

---
# Arguments
- `data::Vector{<:Real}`:       one-dimensional time series ``X_ϵ``.
- `ϑ::Real`:                    drift coefficient ``\vartheta``.
- `Σ::Real`:                    positive diffusion coefficient ``\Sigma``.
- `V::Function`:                defining potential function ``V`` for the invariant density.

---
# Examples
```
$ julia --threads 10 --project=. # start julia with 10 threads and activate project
```
```julia-repl
julia> Threads.nthreads()
julia> using MDEforM
julia> data = Langevin(1.0, 0.0, func_config=NLDO(), α=2.0, σ=1.0, ϵ=0.1, T=100)[1]
julia> Δ_grad_ϑ(data, 1, 1, NLDO()[1])
```

---
See also [`Δ`](@ref).
"""
function Δ_grad_ϑ(data::Vector{<:Real}, ϑ::Real, Σ::Real, V::Function)
    time_stamp = Dates.format(now(), "H:MM:SS")
    @info "∇ $(time_stamp) - Gradient call with parameter values ($(round(ϑ, digits=6)), $(round(Σ, digits=6)))."
    ∂ϑ_convol(ϑ, Σ, V) + ∂ϑ_multi_time_integral(data, ϑ, Σ, V)
end

## diffusion parameter estimation (e.g. Fast Chaotic Noise) ##

# inner convolution expression in gradient of cost functional; derivative of inner_convol with respect to Σ
function ∂Σ_inner_convol(x, ϑ, Σ, V)
    HCubature.hquadrature(y -> ∂Σ_μ(x-t(y), ϑ, Σ, V)k(t(y))[1]dt(y), -1, 1)[1]
end

# space integral in gradient of cost functional
function ∂Σ_convol(ϑ, Σ, V)
    f(y) = 2inner_convol(t(y), ϑ, Σ, V)∂Σ_μ(t(y), ϑ, Σ, V)dt(y)
    # functions are symmetric
    2HCubature.hquadrature(f, 0, 1)[1]
end

# time integral in gradient of cost functional, integration of ∂Σ_inner_convol over data points, see above; serial version;
# no division by N here since we use this serial version for the parallel version below, division by N occurs there
function ∂Σ_time_integral(data, ϑ, Σ, V)
    N = convert(Int, length(data))
    integral_val = 0.0
  
    for i in 1:N
      integral_val = integral_val + ∂Σ_inner_convol(data[i], ϑ, Σ, V)  
    end
  
    -2integral_val
end

# time integral in gradient of cost functional; parallel version via multithreading; written with data-race freedom
function ∂Σ_multi_time_integral(data, ϑ, Σ, V)
    N = length(data)
    # divison by 100 depending on number of threads
    data_batches = Iterators.partition(data, convert(Int, N/100))
    sum_atomic = Threads.Atomic{Float64}(0)

    @inbounds Threads.@threads for data_batch in collect(data_batches)
      res = ∂Σ_time_integral(data_batch, ϑ, Σ, V)
      Threads.atomic_add!(sum_atomic, res)
    end 
    sum_atomic[]/N
end
  
# complete gradient of cost functional with respect to Σ, parallel version via multi-threading
@doc raw"""
    Δ_grad_Σ(data::Vector{<:Real}, ϑ::Real, Σ::Real, V::Function)

Compute gradient of cost functional [`Δ`](@ref) with respect to `Σ` for given `data` and parameter values `ϑ` and `Σ` and a potential `V`.

A properly discretized version of the gradient of the cost functional, given by
```math
\begin{aligned}
  \partial_\Sigma \Delta_T(X_\epsilon, \vartheta, \Sigma, V) = -\frac{2}{T} \int_0^T (\partial_\Sigma \mu(\vartheta, \Sigma, V) \ast k_\beta)(X_\epsilon(t)) \, dt + 2 \int_{\R} (\mu(\vartheta, \Sigma, V) \ast k_\beta)(x) \partial_\Sigma \mu(\vartheta, \Sigma, x) \, dx,
\end{aligned}
```
is implemented and evaluated via [multithreading](https://docs.julialang.org/en/v1/manual/multi-threading/). Here, ``X_ϵ`` is a one-dimensional time series of length ``T``,
obtained from a multiscale SDE, ``\mu`` is the invariant density of the homogenized limit SDE corresponding to [`μ`](@ref), ``k_\beta`` refers to [`k`](@ref), and ``\ast`` is the convolution operator on ``\R``.

!!! warning 
    The computational cost of this function is quite high due to the integration of the convolutions, so if the data is finely discretized, then
    the running times for a single evalutation are relatively long. Remember that, in this case of a 
    non-quadratic potential, further simplifications of the above formula are not known thus far.

---
# Arguments
- `data::Vector{<:Real}`:       one-dimensional time series ``X_ϵ``.
- `ϑ::Real`:                    drift coefficient ``\vartheta``.
- `Σ::Real`:                    positive diffusion coefficient ``\Sigma``.
- `V::Function`:                defining potential function ``V`` for the invariant density.

---
# Examples
```
$ julia --threads 10 --project=. # start julia with 10 threads and activate project
```
```julia-repl
julia> Threads.nthreads()
julia> using MDEforM
julia> data = Langevin(1.0, 0.0, func_config=NLDO(), α=2.0, σ=1.0, ϵ=0.1, T=100)[1]
julia> Δ_grad_Σ(data, 1, 1, NLDO()[1])
```

---
See also [`Δ`](@ref).
"""
function Δ_grad_Σ(data::Vector{<:Real}, ϑ::Real, Σ::Real, V::Function)
    time_stamp = Dates.format(now(), "H:MM:SS")
    @info "∇ $(time_stamp) - Gradient call with parameter values ($(round(ϑ, digits=6)), $(round(Σ, digits=6)))."
    ∂Σ_convol(ϑ, Σ, V) + ∂Σ_multi_time_integral(data, ϑ, Σ, V)
end

## Gaussian case in 1D ##

@doc raw"""
    Δ_grad_ϑ(data::Vector{<:Real}, ϑ::Real, Σ::Real)

Compute gradient of cost functional [`Δ`](@ref) with respect to `ϑ` for given one-dimensonal `data` and parameter values `ϑ` and `Σ`.

A properly discretized version of the gradient of the cost functional, given by
```math
\begin{aligned}
    \partial_\vartheta \Delta_T(X_\epsilon, \vartheta, \Sigma, V) 
    = &\frac{\beta^2 \Sigma}{T \left( 1 + \beta^2 \frac{\Sigma}{\vartheta} \right)^{5/2} \vartheta^3} \int_0^T \left[ \left( X_\epsilon(t)^2 \beta^2 - 1 \right)\vartheta - \beta^2 \Sigma \right] \exp\left( -\frac{\beta^2 X_\epsilon(t)^2}{2 (1 + \beta^2 \frac{\Sigma}{\vartheta})} \right) \, dt \\[0.25cm]
    &+ \frac{\beta^2 \Sigma}{\left( 1 + 2 \beta^2 \frac{\Sigma}{\vartheta} \right)^{3/2} \vartheta^2}.
\end{aligned}
```
is implemented. Here, ``X_ϵ`` is a one-dimensional time series of length ``T``, obtained from a multiscale SDE, and ``\beta`` comes from [`k`](@ref). The potential is here ``V(x) = x^2/2``. 

---
# Arguments
- `data::Vector{Real}`:         one-dimensional time series ``X_ϵ``.
- `ϑ::Real`:                    drift coefficient ``\vartheta``.
- `Σ::Real`:                    positive diffusion coefficient ``\Sigma``.

---
# Examples
```julia-repl
julia> using MDEforM
julia> data = Langevin(1.0, 0.0, func_config=LDO(), α=2.0, σ=1.0, ϵ=0.1, T=100)[1]
julia> Δ_grad_ϑ(data, 1, 1)
```

---
See also [`Δ`](@ref).
"""
function Δ_grad_ϑ(data::Vector{<:Real}, ϑ::Real, Σ::Real)
    β = k(0)[2]
    δ1 = 1/sqrt(1 + β^2*Σ/ϑ)
    δ2 = β^2*Σ/(ϑ^3*(1+β^2*Σ/ϑ)^(5/2))
    δ3 = β^2*Σ/(ϑ^2*(1+2β^2*Σ/ϑ)^(3/2))
  
    N = length(data)
    single_integral = 0.0
  
    for i in 1:N
        single_integral += ((β^2*data[i]^2-1)ϑ-β^2*Σ)*k(data[i]δ1)[1]
    end
    δ2/N*single_integral+δ3
end