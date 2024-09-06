# Multiscale System and Homogenized Limit Pairs

The considered multiscale examples fit into the setting of homogenization for SDEs where there is a system of SDEs
```math
\begin{aligned}
    &d X_\epsilon = \left[ \frac{1}{\epsilon} a_0(X_\epsilon, Y_\epsilon) + a_1(X_\epsilon, Y_\epsilon) \right] dt + \alpha_0(X_\epsilon, Y_\epsilon) dU_t + \alpha_1(X_\epsilon, Y_\epsilon) dV_t, \qquad &\textit{\footnotesize "slow" dynamics} \\[0.25cm]
    &d Y_\epsilon = \left[ \frac{1}{\epsilon^2} b_0(X_\epsilon, Y_\epsilon) + \frac{1}{\epsilon} b_1(X_\epsilon, Y_\epsilon) + b_2(X_\epsilon, Y_\epsilon) \right] dt + \frac{1}{\epsilon} \beta(X_\epsilon, Y_\epsilon) dV_t, \qquad &\textit{\footnotesize "fast" dynamics}
\end{aligned}
```
depending on a small scale parameter ``\epsilon > 0`` and the slow process ``X_\epsilon`` converges weakly, in the sense of the induced probability measures, to the solution ``X`` of the homogenized limit SDE
```math
\begin{aligned}
    dX = f(X)dt + g(X)dW_t.
\end{aligned}
```
The majority of the following functions give realizations of the multiscale processes and their respective homogenized limit in form of a time series.

## Functions
```@docs
Fast_OU_ϵ
Fast_OU_∞
LDA
NLDAM
NSDP
Langevin_ϵ
K
Langevin_∞
LDO
NLDO
Langevin_ϵ_2D
Langevin_∞_2D
Burger_ϵ
Burger_∞
Fast_chaotic_ϵ
Fast_chaotic_∞
produce_trajectory
```

## Index

```@index
Pages = ["multiscale_limit_pairs.md"]
```
