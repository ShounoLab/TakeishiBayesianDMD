using ProgressMeter
using SparseArrays
include("$(@__DIR__)/TBDMDParams.jl")

function gibbs_for_w!(Y₀ :: Matrix{Union{R, C}},
                      Y₁ :: Matrix{Union{R, C}},
                      dp :: TBDMDParams,
                      dp_prev :: TBDMDParams,
                      mp :: ModelParams) where {R <: Real, C <: Complex}
    for k in 1:mp.n_modes
        indices_wo_k = deleteat!(collect(1:mp.n_modes), k)
        ϕₖ_L2² = 0.0
        ξ₋ₖ = Matrix{Complex{Float64}}(undef, (mp.n_datadims, mp.n_data - 1))
        η₋ₖ = Matrix{Complex{Float64}}(undef, (mp.n_datadims, mp.n_data - 1))
        sum_in_m = zeros(Complex{Float64}, mp.n_datadims)
        for j in 1:(mp.n_data - 1)
            ϕₖ_L2² += abs2(dp_prev.ϕ[k, j])
            ξ₋ₖ[:, j] = Y₀[:, j] - (dp_prev.w[:, indices_wo_k] *
                                    dp_prev.ϕ[indices_wo_k, j])
            η₋ₖ[:, j] = Y₁[:, j] - (dp_prev.w[:, indices_wo_k] *
                                    (dp_prev.λ[indices_wo_k] .*
                                     dp_prev.ϕ[indices_wo_k, j]))
            sum_in_m += conj(dp_prev.ϕ[k, j]) *
                        (ξ₋ₖ[:, j] .+ (conj(dp_prev.λ[k]) .* η₋ₖ[:, j]))
        end
        Pwₖ = Diagonal(dp_prev.v[:, k] .^ (-1)) +
              I * (1 + abs2(dp_prev.λ[k])) * ϕₖ_L2² / dp_prev.σ²
        P⁻¹ = Pwₖ ^ (-1)
        mwₖ = P⁻¹ * sum_in_m ./ dp_prev.σ²
        for d in 1:mp.n_datadims
            dp.w[d, k] = rand(ComplexNormal(mwₖ[d], √P⁻¹[d, d]))
        end
    end
end

function gibbs_for_v!(dp :: TBDMDParams,
                      mp :: ModelParams)
    for k in 1:mp.n_modes
        for d in 1:mp.n_datadims
            aᵥ = mp.αᵥ + 1
            bᵥ = mp.βᵥ + abs2(dp.w[d, k])
            dp.v[d, k] = rand(InverseGamma(aᵥ, bᵥ))
        end
    end
end

function gibbs_for_λ!(Y₁ :: Matrix{Union{R, C}},
                      dp :: TBDMDParams,
                      dp_prev :: TBDMDParams,
                      mp :: ModelParams) where {R <: Real, C <: Complex}
    for k in 1:mp.n_modes
        ϕₖ_L2² = sum(abs2.(dp_prev.ϕ[k, :]))

        indices_wo_k = deleteat!(collect(1:mp.n_modes), k)
        η₋ₖ = Matrix{Complex}(undef, (mp.n_datadims, mp.n_data - 1))
        sum_in_m = Complex{Float64}.(zeros(mp.n_datadims))
        for j in 1:(mp.n_data - 1)
            η₋ₖⱼ = Y₁[:, j] - (dp.w[:, indices_wo_k] *
                               (dp_prev.λ[indices_wo_k] .*
                                dp_prev.ϕ[indices_wo_k, j]))
            sum_in_m += conj(dp_prev.ϕ[k, j]) * η₋ₖⱼ
        end
        Pλ = 1 + dp.w[:, k]' * dp.w[:, k] * ϕₖ_L2² / dp_prev.σ²
        mλ = dp.w[:, k]' * sum_in_m / (dp_prev.σ² * Pλ)
        dp.λ[k] = rand(ComplexNormal(mλ, √real(Pλ ^ (-1))))
    end
end

function gibbs_for_ϕ!(Y₀ :: Matrix{Union{R, C}},
                      Y₁ :: Matrix{Union{R, C}},
                      dp :: TBDMDParams,
                      dp_prev :: TBDMDParams,
                      mp :: ModelParams) where {R <: Real, C <: Complex}
    W = dp.w
    Λ = Diagonal(dp.λ)
    Pϕ = I + (W' * W + conj(Λ) * W' * W * Λ) ./ dp_prev.σ²

    P⁻¹ = Pϕ ^ (-1)
    for j in 1:(mp.n_data - 1)
        mϕ = P⁻¹ ./ dp_prev.σ² * (W' * Y₀[:, j] + conj(Λ) * W' * Y₁[:, j])
        dp.ϕ[:, j] = rand(MvComplexNormal(mϕ, P⁻¹,
                                          check_posdef = false,
                                          check_hermitian = false))
    end
end

function gibbs_for_σ²!(Y₀ :: Matrix{Union{R, C}},
                       Y₁ :: Matrix{Union{R, C}},
                       dp :: TBDMDParams,
                       mp :: ModelParams) where {R <: Real, C <: Complex}
    W = dp.w
    Λ = Diagonal(dp.λ)

    sum_in_b₁ = 0.0
    sum_in_b₂ = 0.0
    for j in 1:(mp.n_data - 1)
        sum_in_b₁ += (Y₀[:, j] - W * dp.ϕ[:, j])' *
                     (Y₀[:, j] - W * dp.ϕ[:, j])
        sum_in_b₂ += (Y₁[:, j] - W * Λ * dp.ϕ[:, j])' *
                     (Y₁[:, j] - W * Λ * dp.ϕ[:, j])
    end
    a = mp.αₛ + 2 * (mp.n_data - 1) * mp.n_modes
    b = real(mp.βₛ + sum_in_b₁ + sum_in_b₂)
    dp.σ² = rand(InverseGamma(a, b))
end

function calc_log_lik(Y₀ :: Matrix{Union{R, C}},
                      Y₁ :: Matrix{Union{R, C}},
                      dp :: TBDMDParams,
                      mp :: ModelParams) where {R <: Real, C <: Complex}
    log_lik = 0.0
    for j in 1:(mp.n_data - 1)
        μ₀ = dp.w * dp.ϕ[:, j]
        μ₁ = dp.w * (Diagonal(dp.λ) * dp.ϕ[:, j])
        for d in 1:(mp.n_datadims)
            log_lik += loglikelihood(ComplexNormal(μ₀[d], √(dp.σ²)), Y₀[d, j]) +
                       loglikelihood(ComplexNormal(μ₁[d], √(dp.σ²)), Y₁[d, j])
        end
    end
    return log_lik
end

function drop_samples!(dp_ary :: Vector{TBDMDParams},
                       mp :: ModelParams,
                       mc :: TMCMCConfig)
    sample_indices = filter(j -> (j % mc.thinning == 0) && (j > mc.burnin), 1:mc.n_iter)
    n_samples = length(sample_indices)
    _mc = TMCMCConfig(n_samples, 0)
    dropped_dp_ary = init_dmd_params(mp, _mc)

    for (i, j) in enumerate(sample_indices)
        dropped_dp_ary[i].λ = copy(dp_ary[j].λ)
        dropped_dp_ary[i].ϕ = copy(dp_ary[j].ϕ)
        dropped_dp_ary[i].v = copy(dp_ary[j].v)
        dropped_dp_ary[i].w = copy(dp_ary[j].w)
        dropped_dp_ary[i].σ² = dp_ary[j].σ²
    end
    return dropped_dp_ary
end

function sort_samples!(dp :: TBDMDParams)
    perm = sortperm(dp.λ, by = abs, rev = true)
    dp.λ .= dp.λ[perm]
    dp.ϕ .= dp.ϕ[perm, :]
    dp.v .= dp.v[:, perm]
    dp.w .= dp.w[:, perm]

    return nothing
end

function bayesianDMD(Y :: Matrix{<:Union{Float64, ComplexF64}},
                     model_params :: ModelParams, mc_config :: TMCMCConfig)
    dmd_params = init_dmd_params(model_params, mc_config, init_with_prior = false)

    Y₀ = Y[:, 1:(end - 1)]
    Y₁ = Y[:, 2:end]

    log_liks = Vector(undef, mc_config.n_iter)
    log_liks[1] = calc_log_lik(Y₀, Y₁, dmd_params[1], model_params)

    progress = Progress(mc_config.n_iter)
    for iter in 2:mc_config.n_iter
        gibbs_for_w!(Y₀, Y₁, dmd_params[iter], dmd_params[iter - 1], model_params)
        gibbs_for_v!(dmd_params[iter], model_params)
        gibbs_for_λ!(Y₁, dmd_params[iter], dmd_params[iter - 1], model_params)
        gibbs_for_ϕ!(Y₀, Y₁, dmd_params[iter], dmd_params[iter - 1], model_params)
        gibbs_for_σ²!(Y₀, Y₁, dmd_params[iter], model_params)

        if mc_config.sortsamples
            sort_samples!(dmd_params[iter])
        end

        log_liks[iter] = calc_log_lik(Y₀, Y₁, dmd_params[iter], model_params)

        next!(progress)
    end

    # remove burnin and thinning intervals from samples
    dmd_params_dropped = drop_samples!(dmd_params, model_params, mc_config)

    return dmd_params_dropped, log_liks
end

function mean_bdmd(dp_ary :: Vector{TBDMDParams}, mp ::ModelParams)
    L = length(dp_ary)
    λ = mean([dp_ary[i].λ for i in 1:L])
    W = mean([dp_ary[i].w for i in 1:L])
    V = mean([dp_ary[i].v for i in 1:L])
    Φ = mean([dp_ary[i].ϕ for i in 1:L])
    σ² = mean([dp_ary[i].σ² for i in 1:L])
    return TBDMDParams(λ, Φ, V, W, σ²)
end

function reconstruct_pointest(dp_mean :: TBDMDParams, mp :: ModelParams)

    Y_pointest = Matrix{ComplexF64}(undef, (mp.n_datadims, mp.n_data))
    for j in 1:(mp.n_data - 1)
        Y_pointest[:, j] .= dp_mean.w * dp_mean.ϕ[:, j]
    end
    Y_pointest[:, mp.n_data] .= dp_mean.w * diagm(dp_mean.λ) * dp_mean.ϕ[:, mp.n_data - 1]
    return Y_pointest
end
