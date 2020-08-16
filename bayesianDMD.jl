using ProgressMeter
include("./BDMDParams.jl")

function gibbs_for_w(Y₀ :: Matrix{Union{R, C}}, Y₁ :: Matrix{Union{R, C}}, iter :: Int64,
                     dp :: BDMDParams, mp :: ModelParams) where {R <: Real, C <: Complex}
    for k in 1:mp.n_modes
        indices_wo_k = deleteat!(collect(1:mp.n_modes), k)
        ϕₖ_L2² = 0.0
        ξ₋ₖ = Matrix{Complex}(undef, (mp.n_datadims, mp.n_data - 1))
        η₋ₖ = Matrix{Complex}(undef, (mp.n_datadims, mp.n_data - 1))
        sum_in_m = Complex{Float64}.(zeros(mp.n_datadims))
        for j in 1:(mp.n_data - 1)
            ϕₖ_L2² += abs(dp.ϕ.samples[k, j, iter - 1]) .^ 2
            ξ₋ₖ[:, j] = Y₀[:, j] - (dp.w.samples[:, indices_wo_k, iter - 1] *
                               dp.ϕ.samples[indices_wo_k, j, iter - 1])
            η₋ₖ[:, j] = Y₁[:, j] - (dp.w.samples[:, indices_wo_k, iter - 1] *
                                    (dp.λ.samples[indices_wo_k, iter - 1] .*
                                     dp.ϕ.samples[indices_wo_k, j, iter - 1]))
            sum_in_m += conj(dp.ϕ.samples[k, j, iter - 1]) *
            (ξ₋ₖ[:, j] + conj(dp.λ.samples[k, iter - 1]) * η₋ₖ[:, j])
        end
        Pwₖ = Diagonal(dp.v.samples[:, k, iter - 1] .^ (-1)) +
              I * (1 + abs(dp.λ.samples[k, iter - 1]) ^ 2) * ϕₖ_L2² / dp.σ².samples[iter - 1]
        P⁻¹ = Pwₖ ^ (-1)
        mwₖ = P⁻¹ * sum_in_m ./ dp.σ².samples[iter - 1]
        dp.w.samples[:, k, iter] =
            rand(MvComplexNormal(mwₖ, Matrix(P⁻¹),
                                 check_posdef = false, check_hermitian = false))
    end
end

function gibbs_for_v(iter :: Int64, dp :: BDMDParams, mp :: ModelParams)
    for k in 1:mp.n_modes
        for d in 1:mp.n_datadims
            aᵥ = mp.αᵥ + 1
            bᵥ = mp.βᵥ + abs(dp.w.samples[d, k, iter]) ^ 2
            dp.v.samples[d, k, iter] = rand(InverseGamma(aᵥ, bᵥ))
        end
    end
end

function gibbs_for_λ(Y₁ :: Matrix{Union{R, C}}, iter :: Int64,
                     dp :: BDMDParams, mp :: ModelParams) where {R <: Real, C <: Complex}
    for k in 1:mp.n_modes
        ϕₖ_L2² = sum(abs.(dp.ϕ.samples[k, :, iter - 1]) .^ 2)

        indices_wo_k = deleteat!(collect(1:mp.n_modes), k)
        η₋ₖ = Matrix{Complex}(undef, (mp.n_datadims, mp.n_data - 1))
        sum_in_m = Complex{Float64}.(zeros(mp.n_datadims))
        for j in 1:(mp.n_data - 1)
            η₋ₖⱼ = Y₁[:, j] - (dp.w.samples[:, indices_wo_k, iter] *
                               (dp.λ.samples[indices_wo_k, iter - 1] .*
                                dp.ϕ.samples[indices_wo_k, j, iter - 1]))
            sum_in_m += conj(dp.ϕ.samples[k, j, iter - 1]) * η₋ₖⱼ
        end
        Pλ = 1 + dp.w.samples[:, k, iter]' * dp.w.samples[:, k, iter] *
             ϕₖ_L2² / dp.σ².samples[iter - 1]
        mλ = dp.w.samples[:, k, iter]' / (dp.σ².samples[iter - 1] * Pλ) * sum_in_m
        dp.λ.samples[k, iter] = rand(ComplexNormal(mλ, √real(Pλ ^ (-1))))
    end
end

function gibbs_for_ϕ(Y₀ :: Matrix{Union{R, C}}, Y₁ :: Matrix{Union{R, C}}, iter :: Int64,
                     dp :: BDMDParams, mp :: ModelParams) where {R <: Real, C <: Complex}
    W = dp.w.samples[:, :, iter]
    Λ = Diagonal(dp.λ.samples[:, iter])
    Pϕ = I + (W' * W + conj(Λ) * W' * W * Λ) ./ dp.σ².samples[iter - 1]

    P⁻¹ = Pϕ ^ (-1)
    for j in 1:(mp.n_data - 1)
        mϕ = P⁻¹ ./ dp.σ².samples[iter - 1] * (W' * Y₀[:, j] + conj(Λ) * W' * Y₁[:, j])
        dp.ϕ.samples[:, j, iter] =
            rand(MvComplexNormal(mϕ, P⁻¹,
                                 check_posdef = false, check_hermitian = false))
    end
end

function gibbs_for_σ²(Y₀ :: Matrix{Union{R, C}}, Y₁ :: Matrix{Union{R, C}}, iter :: Int64,
                      dp :: BDMDParams, mp :: ModelParams) where {R <: Real, C <: Complex}
    W = dp.w.samples[:, :, iter]
    Λ = Diagonal(dp.λ.samples[:, iter])

    sum_in_b₁ = 0.0
    sum_in_b₂ = 0.0
    for j in 1:(mp.n_data - 1)
        sum_in_b₁ += (Y₀[:, j] - W * dp.ϕ.samples[:, j, iter])' *
                     (Y₀[:, j] - W * dp.ϕ.samples[:, j, iter])
        sum_in_b₂ += (Y₁[:, j] - W * Λ * dp.ϕ.samples[:, j, iter])' *
                     (Y₁[:, j] - W * Λ * dp.ϕ.samples[:, j, iter])
    end
    a = mp.αₛ + 2 * (mp.n_data - 1) * mp.n_modes
    b = real(mp.βₛ + sum_in_b₁ + sum_in_b₂)
    dp.σ².samples[iter] = rand(InverseGamma(a, b))
end

function calc_log_lik(Y₀ :: Matrix{Union{R, C}}, Y₁ :: Matrix{Union{R, C}}, iter :: Int64,
                      dp :: BDMDParams, mp :: ModelParams) where {R <: Real, C <: Complex}
    log_lik = 0.0
    for j in 1:(mp.n_data - 1)
        μ₀ = dp.w.samples[:, :, iter] * dp.ϕ.samples[:, j, iter]
        μ₁ = dp.w.samples[:, :, iter] * (Diagonal(dp.λ.samples[:, iter]) * dp.ϕ.samples[:, j, iter])
        log_lik += loglikelihood(MvComplexNormal(μ₀, √(dp.σ².samples[iter])), Y₀[:, j]) +
                   loglikelihood(MvComplexNormal(μ₁, √(dp.σ².samples[iter])), Y₁[:, j])
    end
    return log_lik
end

function drop_samples(dp :: BDMDParams, mp :: ModelParams, mc :: MCMCConfig)
    sample_indices = filter(j -> (j % mc.thinning == 0) && (j > mc.burnin), 1:mc.n_iter)
    n_samples = length(sample_indices)
    _mc = MCMCConfig(n_samples, 0)
    dropped_dp = init_dmd_params(mp, _mc)

    for (i, j) in enumerate(sample_indices)
        dropped_dp.λ.samples[:, i] = copy(dp.λ.samples[:, j])
        dropped_dp.ϕ.samples[:, :, i] = copy(dp.ϕ.samples[:, :, j])
        dropped_dp.v.samples[:, :, i] = copy(dp.v.samples[:, :, j])
        dropped_dp.w.samples[:, :, i] = copy(dp.w.samples[:, :, j])
        dropped_dp.σ².samples[i] = dp.σ².samples[j]
    end
    return dropped_dp
end

function sort_samples(iter :: Int64, dp :: BDMDParams)
    perm = sortperm(dp.λ.samples[:, iter], by = abs, rev = true)
    dp.λ.samples[:, iter] .= dp.λ.samples[perm, iter]
    dp.ϕ.samples[:, :, iter] .= dp.ϕ.samples[perm, :, iter]
    dp.v.samples[:, :, iter] .= dp.v.samples[:, perm, iter]
    dp.w.samples[:, :, iter] .= dp.w.samples[:, perm, iter]

    return nothing
end

function bayesianDMD(Y :: Matrix{<:Union{Float64, ComplexF64}},
                     model_params :: ModelParams, mc_config :: MCMCConfig)
    dmd_params = init_dmd_params(model_params, mc_config, init_with_prior = false)

    Y₀ = Y[:, 1:(end - 1)]
    Y₁ = Y[:, 2:end]

    log_liks = Vector(undef, mc_config.n_iter)
    log_liks[1] = calc_log_lik(Y₀, Y₁, 1, dmd_params, model_params)

    progress = Progress(mc_config.n_iter)
    for iter in 2:mc_config.n_iter
        gibbs_for_w(Y₀, Y₁, iter, dmd_params, model_params)
        gibbs_for_v(iter, dmd_params, model_params)
        gibbs_for_λ(Y₁, iter, dmd_params, model_params)
        gibbs_for_ϕ(Y₀, Y₁, iter, dmd_params, model_params)
        gibbs_for_σ²(Y₀, Y₁, iter, dmd_params, model_params)

        if mc_config.sortsamples
            sort_samples(iter, dmd_params)
        end

        log_liks[iter] = calc_log_lik(Y₀, Y₁, iter, dmd_params, model_params)

        next!(progress)
    end

    # remove burnin and thinning intervals from samples
    dmd_params_dropped = drop_samples(dmd_params, model_params, mc_config)

    return dmd_params_dropped, log_liks
end

function calc_log_lik(Y₀ :: Matrix{Union{R, C}}, Y₁ :: Matrix{Union{R, C}}, iter :: Int64,
                      dp :: BDMDParams, mp :: ModelParams) where {R <: Real, C <: Complex}
    log_lik = 0.0
    for j in 1:(mp.n_data - 1)
        μ₀ = dp.w.samples[:, :, iter] * dp.ϕ.samples[:, j, iter]
        μ₁ = dp.w.samples[:, :, iter] * (Diagonal(dp.λ.samples[:, iter]) * dp.ϕ.samples[:, j, iter])
        log_lik += loglikelihood(MvComplexNormal(μ₀, √(dp.σ².samples[iter])), Y₀[:, j]) +
                   loglikelihood(MvComplexNormal(μ₁, √(dp.σ².samples[iter])), Y₁[:, j])
    end
    return log_lik
end

function mean_bdmd(dp :: BDMDParams, mp ::ModelParams)
    λ = reshape(mean(dp.λ.samples, dims = 2), :)
    W = reshape(mean(dp.w.samples, dims = 3), (mp.n_datadims, mp.n_modes))
    Φ = reshape(mean(dp.ϕ.samples, dims = 3), (mp.n_modes, mp.n_data - 1))
    σ² = mean(dp.σ².samples)
    return Dict("λ" => λ, "W" => W, "Φ" => Φ, "σ²" => σ²)
end

function reconstruct_pointest(dp_mean :: Dict, mp :: ModelParams)

    Y_pointest = Matrix{ComplexF64}(undef, (mp.n_datadims, mp.n_data))
    for j in 1:(mp.n_data - 1)
        Y_pointest[:, j] .= dp_mean["W"] * dp_mean["Φ"][:, j]
    end
    Y_pointest[:, mp.n_data] .= dp_mean["W"] * diagm(dp_mean["λ"]) * dp_mean["Φ"][:, mp.n_data - 1]
    return Y_pointest
end


