using DataFrames
using CSV

struct MCMCConfig
    n_iter :: Int64
    burnin :: Int64
    thinning :: Int64
    sortsamples :: Bool
end

function MCMCConfig(n_iter :: Int64, burnin :: Int64;
                    thinning :: Int64 = 1,
                    sortsamples :: Bool = false)
    if n_iter < burnin
        error("ERROR: the burnin number is must be less than iter number")
    end

    if n_iter - burnin < thinning
        error("ERROR: too long thinning")
    end

    return MCMCConfig(n_iter, burnin, thinning, sortsamples)
end

function write_config(filename :: String, mc_conf :: MCMCConfig)
    df = DataFrame()
    for config_sym in collect(fieldnames(MCMCConfig))
        df[config_sym] = getfield(mc_conf, config_sym)
    end
    CSV.write(filename, df)
    return nothing
end

function read_config(filename :: String)
    df = CSV.read(filename)
    if nrow(df) != 1
        error("nrow of ", filename, " must be 1")
    end

    return MCMCConfig(map(i -> df[1, i], 1:ncol(df))...)
end
