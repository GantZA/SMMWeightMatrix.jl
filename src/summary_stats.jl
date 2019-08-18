normal_kurtosis(x) = kurtosis(x, false)

function ks_test(a,b)
    x = ApproximateTwoSampleKSTest(a[2:end],b[2:end])
    return sqrt(x.n_x*x.n_y/(x.n_x+x.n_y))*x.δ
end


function generalized_hurts_exp(log_prices)
    q = 1
    max_τ = 19

    k = 0
    H = zeros(15)
    L = size(log_prices, 1)

    for iter_τ = 5:max_τ
        k += 1

        x = collect(1:iter_τ)
        k_q_t = zeros(iter_τ)
        for τ = 1:iter_τ
            numer = log_prices[(τ+1):τ:L] - log_prices[1:τ:(L-τ)]
            denom = log_prices[1:τ:L]

            # Determine Drift
            N = size(numer, 1) + 1
            X = collect(1:N)
            Y = denom
            mx = sum(X)/N
            SSxx = sum(X.^2) - N * mx^2
            my = sum(Y) / N
            SSxy = sum(X .* Y) - N * mx * my
            cc = [SSxy/SSxx, my - SSxy/SSxx*mx]

            # Subtract Drift
            numer = numer .- cc[1]
            denom = denom .- cc[1] * collect(1:N) .- cc[2]

            k_q_t[τ] = mean(abs.(numer).^q) / mean(abs.(denom).^q)
        end

        # Calculate Hurst Exponent for current iteration
        log_10_x = log10.(x)
        mx = mean(log_10_x)
        SSxx = sum(log_10_x.^2) - iter_τ * mx^2
        log_10_k_q_t = log10.(k_q_t)
        my = mean(log_10_k_q_t)
        SSxy = sum(log_10_x .* log_10_k_q_t) - iter_τ * mx * my
        H[k] = SSxy/SSxx
    end

    H = mean(H) / q
    return H
end

function get_summary_stats(simualted_log_prices, log_prices)

    num_replications = size(simualted_log_prices,2)
    stats_mat = zeros(num_replications,5)

    stats_mat[:,1] = mean(simualted_log_prices, dims=1)
    stats_mat[:,2] = std(simualted_log_prices, dims=1)
    stats_mat[:,3] = mapslices(normal_kurtosis, simualted_log_prices, dims=1)

    ks_test_stat(x) = ks_test(log_prices, x)
    stats_mat[:,4] = mapslices(ks_test_stat, simualted_log_prices, dims=1)
    stats_mat[:,5] = mapslices(generalized_hurts_exp, simualted_log_prices,
        dims=1)
    return stats_mat
end
