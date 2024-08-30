using Documenter, MDEforM

makedocs(
    sitename = "MDE for Multiscale Documentation",
    modules = [MDEforM],
    authors = "Jaroslav Borodavka",
    pages = [
        "Home" => "index.md",
        "Multiscale System and Homogenized Limit Pairs" => "multiscale_limit_pairs.md",
        "Invariant Densities" => "invariant_densities.md",
        "Cost Functionals for the MDE" => "MDE_functionals.md",
        "Optimization Task of the MDE" => "MDE_optimizers.md",
        "Asymptotic Variances of the MDE" => "MDE_asymptotic_variances.md",
    ],
)

deploydocs(
    repo = "github.com/UQatKIT/MDEforM.git",
)
