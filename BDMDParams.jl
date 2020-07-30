using Distributions

include("./ModelParams.jl")
include("./MCMCConfig.jl")

try
    include("./ComplexNormal.jl")
catch
end

mutable struct Param{T <: Union{Array{Int64}, Array{Float64},
                                Array{Complex{Int64}}, Array{Complex{Float64}}},
                     D <: Union{Sampleable, AbstractComplexDist}}
    samples :: T
    priors :: Union{Array{D}, D}

    function Param(samples :: T, priors :: Union{Array{D}, D};
                   check_dimensions :: Bool = true) where {T, D}

        if check_dimensions
        # create permutated sample array for checking dimension matching
        # e.g.) A <N × K × n_iter> -> A_perm <n_iter × N × K>

            perm = circshift(1:ndims(samples), 1)
            perm_samples = permutedims(samples, perm)

            if size(priors) == ()
                if !(size(perm_samples[1]) == size(priors) == ())
                    error("ERROR: dimensional mismatch")
                end
            elseif ndims(priors) == 1
                if length(perm_samples[1, :]) != length(priors)
                    error("ERROR: dimensional mismatch")
                end
            elseif ndims(priors) == 2
                if size(perm_samples[1, :, :]) != size(priors)
                    error("ERROR: dimensional mismatch")
                end
            else
                error("ERROR: the dimensions of parameters must be <= 2")
            end
        end

        return new{T, D}(samples, priors)
    end
end

mutable struct BDMDParams
    # T :: the number of data
    # D :: the dimension of data
    # K :: the number of Koompan modes
    # n_iter :: the number of MCMC samples

    # λ <K × n_iter>: eigenvalues of Koopman Operator
    # ϕ <K × T × n_iter>: eigenfunctions of Koopman Operator
    # v <D × K × n_iter>: coefficients of basis expansion of observations
    # w <D × K × n_iter>: Koopman modes
    # σ <n_iter × 1>: noise variance

    λ :: Param{Array{Complex{Float64}, 2}, D} where D <: AbstractComplexDist
    ϕ :: Param{Array{Complex{Float64}, 3}, D} where D <: AbstractComplexDist
    v :: Param{Array{Float64, 3}, D} where D <: Sampleable
    w :: Param{Array{Complex{Float64}, 3}, D} where D <: AbstractComplexDist
    σ² :: Param{Array{Float64, 1}, D} where D <: Sampleable
end

function init_dmd_params(model_params :: ModelParams, mc_config :: MCMCConfig;
                         init_with_prior :: Bool = false)
    n_modes = model_params.n_modes
    n_datadims = model_params.n_datadims
    n_data = model_params.n_data

    n_iter = mc_config.n_iter
    burnin = mc_config.burnin
    thinning = mc_config.thinning

    #n_samples = div((n_iter - burnin), thinning)
    n_samples = n_iter

    # grant memory regions of parameters and fill with 0 or 1
    λ = Param(Complex.(zeros(n_modes, n_samples)), fill(ComplexNormal(), n_modes))
    ϕ = Param(Complex.(zeros(n_modes, n_data - 1, n_samples)),
              fill(ComplexNormal(), (n_modes, n_data - 1)))
    v = Param(ones(n_datadims, n_modes, n_samples),
              fill(InverseGamma(model_params.αᵥ, model_params.βᵥ),
                   (n_datadims, n_modes)))
    σ² = Param(ones(n_samples), InverseGamma(model_params.αₛ, model_params.βₛ))

    if init_with_prior
        # generate initial values
        λ.samples[:, 1] = rand.(λ.priors)
        ϕ.samples[:, :, 1] = rand.(ϕ.priors)
        v.samples[:, :, 1] = rand.(v.priors)
        σ².samples[1] = rand(σ².priors)
    end

    # put conditional priors of Koopman modes w
    w_priors = ComplexNormal.(0.0 + 0.0im, v.samples)
    w = Param(Complex.(zeros(n_datadims, n_modes, n_samples)), w_priors, check_dimensions = false)

    if init_with_prior
        # generate initial values of Koopman modes w
        w.samples[:, :, 1] = rand.(w.priors[:, :, 1])
    end

    return (BDMDParams(λ, ϕ, v, w, σ²))
end
