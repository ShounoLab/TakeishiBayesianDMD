using Distributions

include("$(@__DIR__)/ModelParams.jl")
include("$(@__DIR__)/TMCMCConfig.jl")

try
    include("./ComplexNormal.jl")
catch
end

mutable struct TBDMDParams
    # T :: the number of data
    # D :: the dimension of data
    # K :: the number of Koompan modes
    # n_iter :: the number of MCMC samples

    # λ <K>: eigenvalues of Koopman Operator
    # ϕ <K × T>: eigenfunctions of Koopman Operator
    # v <D × K>: coefficients of basis expansion of observations
    # w <D × K>: Koopman modes
    # σ <1>: noise variance

    λ :: Vector{Complex{Float64}}
    ϕ :: Matrix{Complex{Float64}}
    v :: Matrix{Float64}
    w :: Matrix{Complex{Float64}}
    σ² :: Float64
end

function init_tbdmd_params(model_params :: ModelParams,
                           mc_config :: TMCMCConfig;
                           init_with_prior :: Bool = false)
    n_modes = model_params.n_modes
    n_datadims = model_params.n_datadims
    n_data = model_params.n_data

    n_iter = mc_config.n_iter
    burnin = mc_config.burnin
    thinning = mc_config.thinning

    n_samples = n_iter

    # allocate memory for parameters and fill with 0 or 1
    dp_ary = [TBDMDParams(fill(0im, n_modes),
                          fill(0im, (n_modes, n_data)),
                          fill(1, (n_datadims, n_modes)),
                          fill(0im, (n_datadims, n_modes)),
                          1.0) for _ in 1:n_samples]

    if init_with_prior
        # generate initial values
        dp_ary[1].λ = rand(MvComplexNormal(fill(0, n_modes), diagm(fill(1, n_modes))))
        dp_ary[1].ϕ = rand(MvComplexNormal(fill(0, n_modes), diagm(fill(1, n_modes))), n_data)
        dp_ary[1].v = rand(InverseGamma(model_params.αᵥ, model_params.βᵥ), (n_datadims, n_modes))
        dp_ary[1].w = hcat([rand(MvComplexNormal(fill(0, n_datadims),
                                                 diagm(dp_ary[1].v[:, k])))
                            for k in 1:n_modes]...)
        dp_ary[1].σ² = rand(InverseGamma(model_params.αₛ, model_params.βₛ))
    end

    return dp_ary
end
