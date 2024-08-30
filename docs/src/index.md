# Home

This is the documentation website for "MDE for Multiscale"! For detailed theoretical and numerical information on the minimum distance estimation (MDE) method please refer to the accompanying article XXX.

In order to use the module, clone the [GitHub repository](https://github.com/UQatKIT/MDEforM) to your local machine, navigate through the Terminal to the root directory containing the
Project.toml file, start Julia and activate the project in the Terminal via
```
$ julia --project=.
```
Then hit ] and instantiate the packages in the project
```julia-repl
(MDEforM) pkg> instantiate 
```
You may now use the module's functionality
```julia-repl
julia> using MDEforM
```
The functions listed under [Index](@ref Index_index) are exported by the module and thorough documentation of these functions 
can be found in the following list of [Contents](@ref Contents_index).

## [Contents](@id Contents_index)

```@contents
Pages = ["multiscale_limit_pairs.md", "invariant_densities.md", "MDE_functionals.md", "MDE_optimizers.md", "MDE_asymptotic_variances.md"]
Depth = 1
```

## [Index](@id Index_index)

```@index
```
