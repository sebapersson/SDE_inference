using Distributions 
using Random 
using LinearAlgebra 
using Plots
using StatsPlots

# For simulating data 
include(joinpath(@__DIR__, "Simulating_data.jl"))
# For inference
include(joinpath(@__DIR__, "..", "src", "SDE_inference.jl"))

# Define necessary functions for Ornstein model 
function alpha_ornstein_full(du, u, p, t)
    c = p.c
    du[1] = c[1] * (c[2] - u[1])
end
function beta_ornstein_full(du, u, p, t)
    c = p.c
    du[1, 1] = c[3]^2
end
function prob_ornstein_full(y_obs, y_mod, error_param, t, dim_obs)

    error_dist = Normal(0.0, error_param[1])
    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)

    # Log-scale for numerical accuracy
    prob = exp(prob)

    return prob
end
function ornstein_obs(y, u, p, t)
    y[1] = u[1]
end
function calc_x0_ornstein!(x0, ind_param)
    x0[1] = 0.0
end
function empty_func() # Can be useful to have 
end


# Setting up SDE-model and simulating data 
Random.seed!(123)
sde_mod = init_sde_model(alpha_ornstein_full, 
                         beta_ornstein_full, 
                         calc_x0_ornstein!, 
                         ornstein_obs, 
                         prob_ornstein_full,
                         1, 1, ones(Int64, 1, 1))

# Getting some dat 
data_obs = CSV.read(joinpath(@__DIR__, "Ornstein_model", "Simulated_data.csv"), DataFrame)
mod_param = DynModInput(exp.([0.1, 2.3, -0.9]), Float64[], Float64[])
dt, u0, tspan = 1e-2, [0.0], (0.0, 10.0)
# stand_step=false -> use same propagator as particle filter 
t_vec, u_vec = solve_sde_em(sde_mod, tspan, u0, mod_param, dt)
plot(t_vec, u_vec[1, :], label = "Model simulation")
plot!(data_obs[!, :time], data_obs[!, :obs], seriestype=:scatter, label="Observed data")

# Setup a Bootstrap filter 
dt=1e-2
filter_opt = BootstrapFilterEM(dt, 40, correlation=0.99)

# Needed to efficiently compute everything 
filter_cache = create_cache(filter_opt, Val(1), Val(sde_mod.dim_obs), Val(sde_mod.dim), sde_mod.P_mat)

path_data = joinpath(@__DIR__, "Ornstein_model", "Simulated_data.csv")
ind_data = init_ind_data(CSV.read(path_data, DataFrame), filter_opt)

mod_param = ModelParameters(DynModInput(exp.([0.1, 2.3, -0.9]), Float64[], Float64[]), Float64[], [0.3], Float64[])
# All random numbers used in the filter 
random_numbers = create_rand_num(ind_data, sde_mod, filter_opt)

bTime =  @elapsed log_lik = run_filter(filter_opt, mod_param, random_numbers, filter_cache, sde_mod, ind_data, Val(filter_opt.is_correlated))
log_lik
