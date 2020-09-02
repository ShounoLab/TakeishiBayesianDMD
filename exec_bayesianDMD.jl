using CSV
using Random
using Plots
using JLD2
include("./bayesianDMD.jl")
include("./MLEDMD.jl")
include("./DMD.jl")
include("./Utils/NonlinearSimulations/NLSE/pseudospectral.jl")
include("./Utils/NonlinearSimulations/Burgers/cranknicolson.jl")

Random.seed!(123)

outdir = "output"

if !isdir(outdir)
    mkdir(outdir)
end


### oscillator
include("./Utils/toydata.jl")
D = 32
T = 128
K = 2
gen_oscillator("toydata_oscillator.csv", D, T, 5e-2, seed = 123)
X = CSV.read("data/toydata_oscillator.csv")
X = Matrix(transpose(Matrix(parse.(Complex{Float64}, X))))

mc_config = MCMCConfig(7500, 5000, thinning = 1)
mp = ModelParams(T, D, K, 1e-3, 1e-3, 1e-3, 1e-3)
dp, ll = bayesianDMD(X, mp, mc_config)

plot(ll)
dp_mean = mean_bdmd(dp, mp)
naive_dp = solve_dmd(X, K)
plot(real.(dp_mean.λ), imag.(dp_mean.λ), seriestype = :scatter, xlims = (-1, 1), ylims = (-1, 1), alpha = 0.5)
plot!(real.(naive_dp.λ), imag.(naive_dp.λ), seriestype = :scatter, alpha = 0.5)

X_pointest = reconstruct_pointest(dp_mean, mp)
plot(heatmap(real.(X), title = "original"),
     heatmap(real.(X_pointest), title = "reconst. (Takeishi)"),
     dpi = 200)
savefig("$outdir/oscillator_takeishi_reconst.png")

U, L, V = svd(X)
U_K, L_K, V_K = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]
W_mle, Λ_mle, σ²_mle, bic = solve_pdmd(X, K)
naive_dp = solve_dmd(X, K)

X1, X2, X3, X4 = real.(U_K), real.(naive_dp.W), real.(W_mle), real.(dp_mean["W"])
clims = (minimum(vcat(X1, X2, X3, X4)), maximum(vcat(X1, X2, X3, X4)))
p1 = plot(real.(U_K), title = "POD modes")
p2 = plot(real.(naive_dp.W), title = "DMD modes")
p3 = plot(real.(W_mle), title = "MLEDMD modes")
p4 = plot(real.(dp_mean["W"]), title = "Takeishi BDMD modes")
plot(p1, p2, p3, p4, dpi = 200, lw = 2, ylims = clims)
savefig("$outdir/oscillator_takeishi_modes.png")


### Nonlinear Schrodinger Equation
Random.seed!(123)
Ngrids = 256 # Number of Fourier modes
#Ngrids = 128 # Number of Fourier modes
L = 30.0 # Space period
#Δt = 2π / 21
Δt = 2π / 256
t_end = 2π
Nsteps = round(Int, t_end / Δt)

config = NLSESettings(Nsteps, Δt, t_end, Ngrids, L)

# initial state of the wave function
ψ₀ = 2.0 * sech.(config.gridpoints)

#result = SSFM(ψ₀, config)
result = PseudoSpectral(ψ₀, config)
X = result.ψ

D, T = size(result.ψ)
t_ary = collect(0:config.Δt:config.t_end)[1:end - 1]

K = 4
mc_config = MCMCConfig(7500, 5000, thinning = 1)
model_params = ModelParams(length(t_ary), config.Ngrids, K,
                           1e-3, 1e-3, 1e-3, 1e-3)

dmd_params, log_liks = bayesianDMD(X, model_params, mc_config)
plot(log_liks)

dp_mean = mean_bdmd(dmd_params, model_params)
X_pointest = reconstruct_pointest(dp_mean, model_params)
plot(heatmap(abs.(X), title = "original"),
     heatmap(abs.(X_pointest), title = "reconst. (Takeishi)"),
     dpi = 200)
savefig("$outdir/NLSE_takeishi.png")

U, L, V = svd(X)
U_K, L_K, V_K = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]
W_mle, Λ_mle, σ²_mle, bic = solve_pdmd(X, K)
naive_dp = solve_dmd(X, K)
X_reconst_dmd = reconstruct(t_ary, t_ary, naive_dp)
cmin = minimum(vcat(abs.(X), abs.(X_reconst_dmd), abs.(X_pointest)))
cmax = maximum(vcat(abs.(X), abs.(X_reconst_dmd), abs.(X_pointest)))
p1 = heatmap(abs.(X))
p2 = heatmap(abs.(X_reconst_dmd))
p3 = heatmap(abs.(X_pointest))
plot(p1, p2, p3, clims = (cmin, cmax))
@save "$outdir/mcmc_nlse_takeishi.jld2" result X_pointest mc_config model_params dmd_params
@save "$outdir/mcmc_nlse_takeishi_seed123.jld2" result X_pointest mc_config model_params dmd_params

X1, X2, X3, X4 = real.(U_K), real.(naive_dp.W), real.(W_mle), real.(dp_mean["W"])
clims = (minimum(vcat(X1, X2, X3, X4)), maximum(vcat(X1, X2, X3, X4)))
p1 = plot(config.gridpoints, real.(U_K), title = "POD modes")
p2 = plot(config.gridpoints, real.(naive_dp.W), title = "DMD modes")
p3 = plot(config.gridpoints, real.(W_mle), title = "MLEDMD modes")
p4 = plot(config.gridpoints, real.(dp_mean["W"]), title = "Takeishi BDMD modes")
#plot(p1, p2, p3, p4, dpi = 200, lw = 2, ylims = clims)
plot(p1, p2, p3, p4, dpi = 200, lw = 1)
savefig("$outdir/nlse_takeishi_modes.png")




### Burgers Equation
Random.seed!(123)
Ngrids = 256 # Number of Fourier modes
L = 30.0 # Space period
t_end = 30.0
Δt = 1.0
Nsteps = round(Int, t_end / Δt)
ν = 0.1

config = BurgersSettings(Nsteps, Δt, t_end, Ngrids, L, ν)


# initial state of the wave function
ψ₀ = exp.(- (config.gridpoints .+ 2) .^ 2)

result = BurgersCrankNicolson(ψ₀, config)
X = result.ψ

D, T = size(result.ψ)
t_ary = collect(0:config.Δt:config.t_end)

K = 7
mc_config = MCMCConfig(7500, 5000, thinning = 1, sortsamples = true)
model_params = ModelParams(length(t_ary), config.Ngrids, K,
                           1e-3, 1e-3, 1e-3, 1e-3)

dmd_params, log_liks = bayesianDMD(X, model_params, mc_config)
plot(log_liks)

dp_mean = mean_bdmd(dmd_params, model_params)
X_pointest = reconstruct_pointest(dp_mean, model_params)
@save "$outdir/mcmc_burgers_takeishi_seed123.jld2" X X_pointest dmd_params log_liks mc_config

U, L, V = svd(X)
U_K, L_K, V_K = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]
W_mle, Λ_mle, σ²_mle, bic = solve_pdmd(X, K)
naive_dp = solve_dmd(X, K)

X1, X2, X3, X4 = real.(U_K), real.(naive_dp.W), real.(W_mle), real.(dp_mean["W"])
clims = (minimum(vcat(X1, X2, X3, X4)), maximum(vcat(X1, X2, X3, X4)))
p1 = plot(config.gridpoints, real.(U_K), title = "POD modes")
p2 = plot(config.gridpoints, real.(naive_dp.W), title = "DMD modes")
p3 = plot(config.gridpoints, real.(W_mle), title = "MLEDMD modes")
p4 = plot(config.gridpoints, real.(dp_mean["W"]), title = "Takeishi BDMD modes")
#plot(p1, p2, p3, p4, dpi = 200, lw = 2, ylims = clims)
plot(p1, p2, p3, p4, dpi = 200, lw = 1)
savefig("$outdir/burgers_takeishi_modes.png")

X_reconst_dmd = reconstruct(t_ary, t_ary, naive_dp)
cmin = minimum(hcat(X, real.(X_reconst_dmd), real.(X_pointest)))
cmax = maximum(hcat(X, real.(X_reconst_dmd), real.(X_pointest)))
cmin = minimum(hcat(X, real.(X_reconst_dmd)))
cmax = maximum(hcat(X, real.(X_reconst_dmd)))
p1 = heatmap(X)
p2 = heatmap(real.(X_reconst_dmd))
p3 = heatmap(real.(X_pointest))
plot(p1, p2, p3, clims = (cmin, cmax))
plot(p1, p2, clims = (cmin, cmax))

X1, X2, X3 = real.(X), real.(X_reconst_dmd), real.(X_pointest)
clims = (minimum(vcat(X1, X2, X3)), maximum(vcat(X1, X2, X3)))
p1 = heatmap(X1, title = "original")
p2 = heatmap(X2, title = "DMD reconst.")
p3 = heatmap(X3, title = "Takeishi BDMD reconst.")
plot(p1, p2, p3, dpi = 200)
savefig("$outdir/burgers_takeishi_reconst.png")


### limit cycle
include("Utils/limitcycle.jl")
limconf = LimitCycleConfig(1, 1, 0, 0.01, 10000)
r₀ = √limconf.μ
θ₀ = 0

states = limit_cycle(r₀, θ₀, limconf)

dist = Normal(0, √(10 ^ (-4)))
Y = limit_cycle_observations(states, dist)

naive_dp = solve_dmd(Y, 5)

mc_config = MCMCConfig(3000, 1000, thinning = 1, sortsamples = true)
model_params = ModelParams(size(Y)[2], size(Y)[1], 5,
                           1e-3, 1e-3, 1e-3, 1e-3)

dmd_params, log_liks = bayesianDMD(Y, model_params, mc_config)
dp_mean = mean_bdmd(dmd_params, model_params)

