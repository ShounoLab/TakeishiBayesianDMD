using Distributions

struct LimitCycleConfig
    μ :: Real
    γ :: Real
    β :: Real
    Δt :: Real
    Nsamples :: Int64
end

struct LimitCycleStates
    r :: Vector{Float64}
    θ :: Vector{Float64}
    x :: Vector{Float64}
    y :: Vector{Float64}
end

function LimitCycleStates(r :: Vector{Float64},
                          θ :: Vector{Float64})
    x = r .* cos.(θ)
    y = r .* sin.(θ)
    return LimitCycleStates(r, θ, x, y)
end

function limit_cycle(r₀ :: Real, θ :: Real,
                     config :: LimitCycleConfig)
    r = zeros(Float64, config.Nsamples)
    θ = zeros(Float64, config.Nsamples)
    r[1], θ[1] = r₀, θ₀

    for t in 2:config.Nsamples
        r[t] = r[t - 1] + config.Δt * (config.μ * r[t - 1] - r[t - 1] ^ 3)
        θ[t] = θ[t - 1] + config.Δt * (config.γ - config.β * r[t - 1] ^ 2)
    end

    return LimitCycleStates(r, θ)
end

function limit_cycle_observations(
    states :: LimitCycleStates,
    dist :: Normal)

    Nsamples = length(states.r)
    Y = Matrix{Complex{Float64}}(undef, (5, Nsamples))
    for j in -2:2
        Y[j + 3, :] .= exp.(im * j .* states.θ)
    end
    return Y .+ rand(dist, size(Y))
end
