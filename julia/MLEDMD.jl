using LinearAlgebra

function solve_pdmd(data :: Matrix{<:Union{<:Real, ComplexF64}}, n_modes :: Int64)
    data = transpose(data)
    Y₀ = data[1:(end - 1), :]
    Y₁ = data[2:end, :]
    Y = hcat(Y₀, Y₁)

    n_data = size(data)[1] - 1
    n_datadims = size(data)[2]

    S = zeros(ComplexF64, (2 * n_datadims, 2 * n_datadims))
    for t in 1:n_data
        S .+= Y[t, :] * Y[t, :]'
    end
    S ./= n_data
    M, U = eigen(S)
    M = M[(2 * n_datadims):-1:1]
    U = U[:, (2 * n_datadims):-1:1]

    σ²_mle = sum(M[(n_modes + 1):end]) / (2 * n_datadims - n_modes)
    B_mle = U[:, 1:n_modes] * (diagm(M[1:n_modes]) - σ²_mle * I) ^ (1 / 2)
    W_mle = B_mle[1:n_datadims, :]
    Λ_mle = diag(pinv(W_mle) * B_mle[((n_datadims + 1):end), :])

    T = n_data
    D = n_datadims
    K = n_modes
    loglik = - 2 * T * D * log(π) - 2 * T * D -
            T * sum(log.(M[1:K])) -
            T * (2 * D - K) * log(σ²_mle)

    n_eff_params = 4 * D * K - K ^ 2 + 3 * K + 1

    bic = -2 * (loglik - n_eff_params / 2 * log(T))

    return W_mle, Λ_mle, σ²_mle, bic
end
