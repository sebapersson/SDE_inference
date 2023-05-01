"""
    calc_norm_squared(x)

Calculate squared L2-norm of a vector x
"""
function calc_norm_squared(x)
    return sum(x.^2)
end


"""
    init_sol_object_bootstrap(::Val{dim_model}, ::Val{dim_model_obs}, sde_mod::SdeModel)::BootstrapSolverObj where {dim_model, dim_model_obs}

Initialise solution-struct (BootstrapSolverObj) to pre-allocate matrices and vectors for propgating particles. 

Pre-allocates drift-vector, diffusion-matrix, current particle-values at time, and step-length to propegate the 
particles in a memory efficient manner. As StaticArrays-are employed Val-input required to help compiler. 
"""
function init_sol_object_bootstrap(::Val{dim_model}, ::Val{dim_model_obs}, sde_mod::SdeModel)::BootstrapSolverObj where {dim_model, dim_model_obs}
    
    alpha_vec = zeros(MVector{dim_model, Float64})
    
    beta_mat = zeros(MMatrix{dim_model, dim_model, Float64})
    
    x_curr = zeros(MVector{dim_model, Float64})

    Δt = Array{Float64, 1}(undef, 1)
    sqrt_Δt = Array{Float64, 1}(undef, 1)

    solver_obj = BootstrapSolverObj(alpha_vec,
                                    beta_mat,
                                    x_curr,
                                    Δt,
                                    sqrt_Δt)

    return solver_obj
end


"""
    step_em_bootstrap!(p::DynModInput, 
                       so::BootstrapSolverObj, 
                       sde_mod::SdeModel,
                       u_vec, 
                       t::Float64) where T1<:Signed 

Propegate the particles one time-step for the Euler-Maruyama bootstrap filter. 

See also: [`propegate_em_bootstrap!`]
"""
function step_em_bootstrap!(p::DynModInput, 
                            so::BootstrapSolverObj, 
                            sde_mod::S,
                            u_vec::V, 
                            t::Float64) where {S<:SdeModel, V<:AbstractVector}

    Δt::Float64 = so.Δt[1]
    sqrt_Δt::Float64 = so.sqrt_Δt[1]

    # Calculate beta and alpha arrays 
    sde_mod.calc_alpha(so.alpha_vec, so.x_vec, p, t)
    sde_mod.calc_beta(so.beta_mat, so.x_vec, p, t)

    # Cholesky, overwrite beta-matrix for propegation 
    calc_cholesky!(so.beta_mat, sde_mod.dim)

    so.x_vec .+= so.alpha_vec*Δt .+ so.beta_mat*u_vec .* sqrt_Δt

end


"""
    propegate_em_bootstrap!(x::Array{Float64, 2}, 
                            p::DynModInput, 
                            solver_obj::BootstrapSolverObj,
                            t_step_info::TimeStepInfo, 
                            sde_mod::SdeModel, 
                            n_particles::T1, 
                            u::Array{Float64, 2}) where {T1<:Signed}

Propegate n-particles (x) in the bootstrap filter for a SDE-model using Euler-Maruyama stepper. 

Propegates n-particles for an individual with parameters p between time-points t_step_info.t_start 
and t_step_info.t_end using t_step_info.n_step. Old particle values x are overwritten for memory 
efficiency. Negative values are set to 0 to avoid negative square-roots. The auxillerary variables 
contain random normal numbers used to propegate, and the solver_obj contains pre-allocated 
matrices and vectors. 
"""
function propegate_em_bootstrap!(x::Array{Float64, 2}, 
                                 p::DynModInput, 
                                 solver_obj::BootstrapSolverObj,
                                 t_step_info::TimeStepInfo, 
                                 sde_mod::SdeModel, 
                                 n_particles::T1, 
                                 u::Array{Float64, 2}) where {T1<:Signed}
    
    # Stepping options for the EM-stepper
    Δt::Float64 = (t_step_info.t_end - t_step_info.t_start) / t_step_info.n_step
    solver_obj.Δt[1] = Δt
    solver_obj.sqrt_Δt[1] = sqrt(Δt)
    t_vec = t_step_info.t_start:Δt:t_step_info.t_end
    
    # Update each particle (note x is overwritten)
    @inbounds for i in 1:n_particles
        i_acc = 1:sde_mod.dim
        solver_obj.x_vec .= x[:, i]
        
        @inbounds for j in 1:t_step_info.n_step
            
            u_vec = @view u[i_acc, i] 
            step_em_bootstrap!(p, solver_obj, sde_mod, u_vec, t_vec[j])

            map_to_zero!(solver_obj.x_vec, sde_mod.dim)

            i_acc = i_acc .+ sde_mod.dim
        end

        x[:, i] .= solver_obj.x_vec
    end

end


"""
    run_filter(filt_opt::BootstrapFilterEM,
               model_parameters::ModelParameters, 
               random_numbers::RandomNumbers, 
               sde_mod::SdeModel, 
               individual_data::IndData)::Float64

Run bootstrap filter for Euler-Maruyama SDE-stepper to obtain unbiased likelihood estimate. 

Each filter takes the input filt_opt, model-parameter, random-numbers, model-struct and 
individual_data. The filter is optmised to be fast and memory efficient on a single-core. 

# Args
- `filt_opt`: filter options (BootstrapFilterEM-struct)
- `model_parameters`: none-transfmored unknown model-parameters (ModelParameters)
- `random_numbers`: auxillerary variables, random-numbers, used to estimate the likelihood (RandomNumbers-struct)
- `sde_mod`: underlaying SDE-model for calculating likelihood (SdeModel struct)
- `individual_data`: observed data, and number of time-steps to perform between data-points (IndData-struct)

See also: [`BootstrapFilterEM`, `ModelParameters`, `RandomNumbers`, `SdeModel`, `IndData`]
"""
function run_filter(filt_opt::BootstrapFilterEM,
                    model_parameters::ModelParameters, 
                    random_numbers::RandomNumbers, 
                    sde_mod::SdeModel, 
                    individual_data::IndData)::Float64

    # Nested function that updates the weights (normalised and non-normalised)
    # for the bootstrap filter. (Nested function typically do
    # not decrease performance in Julia)
    @inline function calc_weights!(w_unormalised, w_normalised, i_t_vec)::Float64

        y_obs_sub = SubArray(y_mat, (i_dim_obs, i_t_vec))
        @inbounds for i in 1:n_particles
            x_curr_sub = @view x_curr[:, i] 
            sde_mod.calc_obs(y_mod_vec, x_curr_sub, c, t_vec[i_t_vec])

            w_unormalised[i] = sde_mod.calc_prob_obs(y_obs_sub, y_mod_vec, error_param, t_vec[i_t_vec], sde_mod.dim_obs)
        end
        sum_w_unormalised_ret::Float64 = sum(w_unormalised)
        w_normalised .= w_unormalised ./ sum_w_unormalised_ret

        return sum_w_unormalised_ret
    end

    # Extract individual parameters for propegation 
    n_particles::Int64 = filt_opt.n_particles
    c::DynModInput = model_parameters.individual_parameters
    error_param::Array{Float64, 1} = model_parameters.error_parameters

    # Extract individual data and discretization level between time-points 
    t_vec::Array{Float64, 1} = individual_data.t_vec
    y_mat::Array{Float64, 2} = individual_data.y_mat
    n_step_vec::Array{Int16, 1} = individual_data.n_step
    len_t_vec::Int64 = length(t_vec)
    
    # Pre-allocated variables required for looping in the filter 
    x0_mat::Array{Float64, 2} = reshape(repeat(model_parameters.x0, n_particles), (sde_mod.dim, n_particles))
    # Calculate initial values for particles (states)
    for i in 1:n_particles
        sde_mod.calc_x0!((@view x0_mat[:, i]), model_parameters)
    end
    x_curr::Array{Float64, 2} = deepcopy(x0_mat)
    w_unormalised::Array{Float64, 1} = Array{Float64, 1}(undef, n_particles)
    w_normalised::Array{Float64, 1} = Array{Float64, 1}(undef, n_particles)
    y_mod_vec::Array{Float64, 1} = Array{Float64, 1}(undef, sde_mod.dim_obs)
    i_dim_obs = 1:sde_mod.dim_obs
    i_dim_mod = 1:sde_mod.dim
    n_particles_inv::Float64 = convert(Float64, 1 / n_particles)

    log_lik::Float64 = 0.0

    # If correlated-filter, convert standard-normal resampling numbers to 
    # standard uniform 
    if filt_opt.ρ != 0
        u_resamp_vec_tmp = deepcopy(random_numbers.u_resamp)
        u_resamp_vec_tmp = cdf(Normal(), u_resamp_vec_tmp)
    else
        u_resamp_vec_tmp = deepcopy(random_numbers.u_resamp)
    end
    u_resamp_vec::Array{Float64, 1} = u_resamp_vec_tmp

    # Propegate particles for t1 
    i_u_prop::Int64 = 1  # Which discretization level to access 
    i_col_u_mat = 1:n_particles  # Which random numbers to use for propegation 

    # Struct used when propegating 
    solver_obj::BootstrapSolverObj = init_sol_object_bootstrap(Val(sde_mod.dim), Val(sde_mod.dim_obs), sde_mod)
    
    # Special case where t = 0 is not observed 
    if t_vec[1] > 0.0
        t_step_info = TimeStepInfo(0.0, t_vec[1], n_step_vec[i_u_prop])
        try 
            propegate_em_bootstrap!(x_curr, c, solver_obj, t_step_info, sde_mod, n_particles, random_numbers.u_prop[i_u_prop])
        catch 
            return -Inf 
        end

        i_u_prop += 1
    end

    # Update likelihood first time
    sum_w_unormalised::Float64 = calc_weights!(w_unormalised, w_normalised, 1)
    log_lik += log(sum_w_unormalised * n_particles_inv)

    # Indices for resampling 
    i_resamp::Array{UInt32, 1} = Array{UInt32, 1}(undef, n_particles)

    # Propegate over remaning time-steps 
    for i_t_vec in 2:1:len_t_vec    
        
        # If correlated, sort x_curr
        if filt_opt.ρ != 0
            data_sort = sum(x_curr.^2, dims=1)[1, :]
            i_sort = sortperm(data_sort)
            x_curr = x_curr[:, i_sort]
            w_normalised = w_normalised[i_sort]
        end

        u_resample = u_resamp_vec[i_t_vec-1]
        systematic_resampling!(i_resamp, w_normalised, n_particles, u_resample)
        x_curr = x_curr[:, i_resamp]
        
        # Variables for propeating correct particles  
        t_step_info = TimeStepInfo(t_vec[i_t_vec-1], t_vec[i_t_vec], n_step_vec[i_u_prop])
           
        try 
            propegate_em_bootstrap!(x_curr, c, solver_obj, t_step_info, sde_mod, n_particles, random_numbers.u_prop[i_u_prop])         
        catch 
            return -Inf 
        end
        i_u_prop += 1
        
        # Update weights and calculate likelihood
        sum_w_unormalised = calc_weights!(w_unormalised, w_normalised, i_t_vec)
        log_lik += log(sum_w_unormalised * n_particles_inv)
    end

    return log_lik
end
