using DataFrames
using CSV

struct ModelParams
    n_data :: Int64
    n_datadims :: Int64
    n_modes :: Int64
    αᵥ :: Float64
    βᵥ :: Float64
    αₛ :: Float64
    βₛ :: Float64

    function ModelParams(n_data :: Int64, n_datadims :: Int64,
                         n_modes :: Int64, αᵥ :: Float64, βᵥ :: Float64,
                         αₛ :: Float64, βₛ :: Float64)
        if n_data <= 0
            error("ERROR: n_data must be > 0.")
        end
        if n_datadims <= 0
            error("ERROR: n_datadims must be > 0.")
        end
        if n_modes <= 0
            error("ERROR: n_modes must be > 0.")
        end
        if αᵥ <= 0.0
            error("ERROR: αᵥ must be > 0.0.")
        end
        if βᵥ <= 0.0
            error("ERROR: βᵥ must be > 0.0.")
        end
        if αₛ <= 0.0
            error("ERROR: αₛ must be > 0.0.")
        end
        if βₛ <= 0.0
            error("ERROR: βₛ must be > 0.0.")
        end
        return new(n_data :: Int64, n_datadims :: Int64,
                   n_modes :: Int64, αᵥ :: Float64, βᵥ :: Float64,
                   αₛ :: Float64, βₛ :: Float64)
    end
end

function write_config(filename :: String, mparams :: ModelParams)
    df = DataFrame()
    for config_sym in collect(fieldnames(ModelParams))
        df[config_sym] = getfield(mparams, config_sym)
    end
    CSV.write(filename, df)
    return nothing
end

function read_config(filename :: String)
    df = CSV.read(filename)
    if nrow(df) != 1
        error("nrow of ", filename, " must be 1")
    end

    return ModelParams(map(i -> df[1, i], 1:ncol(df))...)
end
