"""
    McmcSamplerRandWalk

Essential parameters (Σ, λ) for Random-Walk sampler. 

Initalised by init_mcmc. When proposing, the step-length is multiplied to 
to the covariance matrix. 

See also: [`init_mcmc`](@ref)
"""
struct McmcSamplerRandWalk{T1<:Array{<:AbstractFloat, 2}, 
                           T2<:AbstractFloat, 
                           T3<:Signed, 
                           T4<:Array{<:AbstractFloat, 2}, 
                           T5<:Array{<:AbstractFloat, 1}}

    cov_mat::T1
    step_length::T2
    dim::T3
    name_sampler::String
    cov_mat_old::T4
    mean::T5
end
struct RandomWalk
end


"""
    McmcSamplerAM

Essential parameters (μ, Σ, γ0) for AM-sampler

Initalised by init_mcmc. When proposing, γ = γ0 / (iteration^alpha_power). 
μ, Σ, are updated after step_before_update steps, and are then updated 
every 30:th iteration. 

See also: [`init_mcmc`](@ref)
"""
struct McmcSamplerAM{T1<:Array{<:AbstractFloat, 2}, 
                     T2<:Array{<:AbstractFloat, 1}, 
                     T3<:AbstractFloat, 
                     T4<:Signed}
    
    cov_mat::T1
    # For tuning covariance matrix 
    mu::T2
    gamma0::T3
    alpha_power::T3
    lambda::T3
    steps_before_update::T4
    dim::T4
    name_sampler::String
    update_it::T4
end
struct AmSampler
end


"""
    McmcSamplerGenAM

Essential parameters (μ, Σ, log(λ), α∗, γ0) for General AM-sampler

Initalised by init_mcmc. When updating, γ = γ0 / (iteration^alpha_power). 
μ, Σ, log(λ) are updated after step_before_update steps. 

See also: [`init_mcmc`](@ref)
"""
struct McmcSamplerGenAM{T1<:Array{<:AbstractFloat, 2}, 
                        T2<:Array{<:AbstractFloat, 1},
                        T3<:Array{<:AbstractFloat, 1},
                        T4<:AbstractFloat, 
                        T5<:Signed}

    cov_mat::T1
    # Parameters for tuning covariance matrix 
    mu::T2
    log_lambda::T3
    alpha_target::T4
    gamma0::T4
    alpha_power::T4
    steps_before_update::T5
    dim::T5
    name_sampler::String
    update_it::T5
end
struct GenAmSampler
end


"""
    McmcSamplerRam

Essential parameters (Σ, α∗, γ0) for RAM-sampler

Initalised by init_mcmc. When updating, γ = γ0 / (iteration^alpha_power). 
Σ is updated after step_before_update steps. q_vec is the random normal 
numbers used in proposing new-parameters. 

See also: [`init_mcmc`](@ref)
"""
struct McmcSamplerRam{T1<:Array{<:AbstractFloat, 2}, 
                      T2<:Array{<:AbstractFloat, 1},
                      T3<:AbstractFloat, 
                      T4<:Signed}
    cov_mat::T1
    q_vec::T2
    alpha_target::T3
    gamma0::T3
    alpha_power::T3
    steps_before_update::T4
    dim::T4
    name_sampler::String
    update_it::T4
end
struct RamSampler
end


"""
    BootstrapFilterEm

Arrays (drift, diffusion and state-arrays + step-length) required for propegating particles using the Bootstrap-EM filter. 

Pre-allocated for computational efficiency. 
"""
struct BootstrapSolverObj{T1<:MArray,
                          T2<:MArray,
                          T3<:MArray,
                          T4<:AbstractFloat}

    alpha_vec::T1
    beta_mat::T2
    x_vec::T3
    delta_t::Vector{T4}
    sqrt_delta_t::Vector{T4}
end


struct BootstrapFilterArr{T1<:Array{UInt32, 1}, 
                          T2<:Array{Int64, 1},
                          T3<:Array{Float64, 2}, 
                          T4<:Array{Float64, 1}}
    i_resamp::T1
    i_sort_corr::T2
    x0_mat::T3
    x_curr::T3
    w_unormalised::T4
    w_normalised::T4
end


"""
    DiffBridgeSolverObj

Drift, diffusion, observation, and state-arrays + step-length when propegating via the modifed diffusion bridge. 

Pre-allocated for computational efficiency. 
"""
struct DiffBridgeSolverObj{T1<:MArray,
                           T2<:MArray,
                           T3<:MArray,
                           T4<:SArray,
                           T5<:SArray,
                           T6<:MArray,
                           T7<:Array{<:AbstractFloat, 1}}

    mean_vec::T1
    alpha_vec::T1
    cov_mat::T2
    beta_mat::T2
    sigma_mat::T3
    P_mat::T4
    P_mat_t::T5
    x_vec::T6
    delta_t::T7
    sqrt_delta_t::T7
end


"""
    SdeModel

Dimensions and drift, diffusion, and observation functions for a SDE-models.

Drift, diffusion, and observation functions all follow a specific arg-format
(see notebook). Calc_x0 calculates initial values using the individual parameters 
P_mat can be provided if the observation model is on the format y = P*X + ε, ε ~ N(0, σ^2) 
"""
struct SdeModel{F1<:Function,
                F2<:Function,   
                F3<:Function, 
                F4<:Function, 
                F5<:Function, 
                T1<:Signed, 
                T2<:SArray,
                T3<:SArray}
            
    calc_alpha::F1
    calc_beta::F2
    calc_x0!::F3
    dim::T1
    calc_obs::F4
    calc_prob_obs::F5
    dim_obs::T1
    P_mat::T2
    P_mat_trans::T3
end


"""
    DynModInput

Struct storing model-quantity values (c, ĸ, covariates) for an individual. 
"""
struct DynModInput{T1<:Array{<:AbstractFloat, 1}, 
                   T2<:Array{<:AbstractFloat, 1}, 
                   T3<:Array{<:AbstractFloat, 1}}

    c::T1
    kappa::T2
    covariates::T3
end


"""
    TimeStepInfo

Time-stepping data for SDE and Poison (tau-leaping) particle propagators. 
"""
struct TimeStepInfo{T1<:AbstractFloat, T2<:Signed}
    t_start::T1
    t_end::T1
    n_step::T2
end


"""
    IndData

Observed data (measurement y_mat and time t_vec) for individual i.

Initalised by init_ind_data. For a SDE-model n_step contains the number of 
time-steps to perform betwen t_vec[i-1] and t_vec[i]. For a SSA/Extrande-model it 
is empty. In the case of multiple observed time-series, each row of y-mat 
corresponds to one measured specie. Cov_val corresponds to potential covariate 
values (left empty if there aren't any covariates)

See also: [`init_ind_data`](@ref)
"""
struct IndData{T1<:AbstractFloat, 
               T2<:Signed}

    t_vec::Vector{T1}
    y_mat::Matrix{T1}
    n_step::Vector{T2}
    cov_val::Vector{T1}
end


"""
    RandomNumbers

Random-numbers for propegation and resampling SDE/Poisson-model particle-filters.

u_prop[i] contains random number to propegate between t_vec[i] to t_vec[i-1], 
and each entry can vary in size depending on spacing between observed 
time-points. 

See also: [`create_rand_num`](@ref)
"""
struct RandomNumbers{T1<:Array{<:Array{<:AbstractFloat, 2}, 1}, 
                     T2<:Array{<:AbstractFloat, 1}}

    u_prop::T1
    u_resamp::T2
end


"""
    ModelParameters

Rates, error-parameters, initial-values and covariates for individual i. 

If there are not any parameters, covariates is an empty vector. 

See also: [`init_model_parameters`](@ref)
"""
struct ModelParameters{T1<:Array{<:AbstractFloat, 1}, 
                       T2<:DynModInput, 
                       T3<:Array{<:Real, 1}}

    individual_parameters::T2
    x0::T3
    error_parameters::T1
    covariates::T1
end


"""
    BootstrapFilterEm

Options: time-step size (dt), number of particles, correlation level for Euler-Maruyama SDE bootstrap filter. 

If rho ∈ [0.0, 1.0) equals 0.0 the particles are uncorrelated. 
"""
struct BootstrapFilterEm{T1<:AbstractFloat, T2<:Signed}
    delta_t::T1
    n_particles::T2
    rho::T1    
end
struct BootstrapEm
end


"""
    ModDiffusionFilter

Options: time-step size (dt), number of particles, correlation level for modified diffusion bridge SDE bootstrap filter. 

If rho ∈ [0.0, 1.0) equals 0.0 the particles are uncorrelated. 
"""
struct ModDiffusionFilter{T1<:AbstractFloat, T2<:Signed}
    delta_t::T1
    n_particles::T2
    rho::T1    
end
struct ModDiffusion
end


"""
    TuneParticlesIndividual

Options when tuning particles for individual infernece. 

Intitialised by init_pilot_run. 

See also: [`init_pilot_run`](@ref)
"""
struct TuneParticlesIndividual{T1<:Signed,
                               T2<:Signed,
                               T3<:Signed,
                               T4<:Array{<:Signed, 1}, 
                               T5<:Array{Float64, 1}} 
                                
    n_particles_pilot::T1
    n_samples::T2
    n_particles_investigate::T4
    init_ind_param
    init_error
    n_times_run_filter::T3
    rho_list::T5
end


"""
    FileLocations

Directories of observed data, directory to save result, and name of model. 

Initalised by init_file_loc

See also: [`init_file_loc`](@ref)
"""
mutable struct FileLocations{T1<:Array{<:String, 1}, T2<:Array{<:AbstractFloat, 1}, T3<:Array{<:Integer, 1}}
    path_data::String
    model_name::String 
    dir_save::String 
    cov_name::T1
    cov_val::T2
    dist_id::T3
end


"""
    InitParameterInfo

Priors, initial-values, and parameter-characterics (log-scale and/or postive). 

For single-individual inference. 

Initialised by init_param. For both individual and error-parameters only 
a subset might be estimated on log-scale or be enforced as positive. The latter 
parameters are proposed via an exponential-transformation. 

See also: [`init_param`](@ref)
"""
struct InitParameterInfo{T1<:Array{<:Distribution, 1}, 
                         T2<:Array{<:Distribution, 1}, 
                         T3<:Array{<:AbstractFloat, 1}, 
                         T4<:Array{<:AbstractFloat, 1}, 
                         T5<:Signed}
                         
    prior_ind_param::T1
    prior_error_param::T2
    init_ind_param::T3
    init_error_param::T4
    n_param::T5
    ind_param_pos::Vector{Bool}
    error_param_pos::Vector{Bool}
    ind_param_log::Vector{Bool}
    error_param_log::Vector{Bool}
end



"""
    ParamInfoIndPre

Struct storing inference information for c_i the individual parameters. 

init_ind_param can be a matrix with values for each individual, array for value 
to use for all individuals, or (mean, median, random) to sample from the priors. 
"""
struct ParamInfoIndPre{T<:Any}
    init_ind_param::T
    pos_ind_param::Bool
    log_ind_param::Bool
    n_param::Int64
end


"""
    ParamInfoInd

Same as ParamInfoIndPre, except init_ind_param now has the stored inital values.  
"""
struct ParamInfoInd{T1<:Array{<:AbstractFloat, 1}, 
                    T2<:Signed}

    init_ind_param::T1
    pos_ind_param::Array{Bool, 1}
    log_ind_param::Array{Bool, 1}
    n_param::T2
end