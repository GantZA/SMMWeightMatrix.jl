mutable struct WeightMatrix
    seed::Int64
    log_prices::Array{Float64,1}
    block_size::Int64
    num_bootstap::Int64
end

function block_bootstrap_index(block_ind, n, b)
    rand_blocks = sample(block_ind, floor(Int,n/b))
    sample_ind = transpose(repeat(rand_blocks, 1, b))
    sample_ind = sample_ind[:]
    addition_vec = repeat(0:b-1,floor(Int,n/b))
    sample_ind = sample_ind + addition_vec
    return sample_ind
end

function block_bootstrap_estimator(wm::WeightMatrix)
    # Step 1: Apply a Moving Block Bootstrap to the Measured Series
    Random.seed!(wm.seed)

    n = size(wm.log_prices, 1)
    block_ind = 1:n-wm.block_size+1
    # b_samples = [log_prices[block_bootstrap_index(block_ind, n, b)] for i in 1:num_bootstap]
    b_samples = Array{Float64,2}(undef, n, wm.num_bootstap)
    for i in 1:wm.num_bootstap
        b_samples[:,i] = wm.log_prices[block_bootstrap_index(block_ind, n, wm.block_size)]
    end

    # Step 2: Calculate Distributions for Each Moment and Test Statistic
    dist = get_summary_stats(b_samples, wm.log_prices)
    W = inv(cov(dist))
    return W
end

function(wm::WeightMatrix)(algo=block_bootstrap_estimator)
    return block_bootstrap_estimator(wm)
end

function select_moments(log_prices)
    mean_log = mean(log_prices)
    std_log = std(log_prices)
    kurt_log = normal_kurtosis(log_prices)
    ks_stat = 0.0
    hurst_log = generalized_hurts_exp(log_prices)

    return [mean_log, std_log, kurt_log, ks_stat, hurst_log]
end


# function identity_estimator(moments, simulated_moments)
#     return Matrix{Float64}(I, size(moments,1), size(moments,1))
# end
#
# function two_step_cov_estimator(moments, simulated_moments)
#     weight_matrix_1 = identity_estimator(moments, simulated_moments)
#     smm_estimated_parameters =
#     moment_error =
#     Ω = 1/N * (moment_error * moment_error')
#     weight_matrix_2 = inv(Ω)
#     return weight_matrix_2
# end
#
# function iterated_var_cov_estimator(moments, simulated_moments)
#
# end
#
# function newey_west_consistent_estimator(moments, simulated_moments)
#
# end
