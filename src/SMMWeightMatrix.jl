module SMMWeightMatrix

using Statistics
using Distributions
using HypothesisTests
using Random
using LinearAlgebra

include("summary_stats.jl")
include("weight_matrix.jl")

export WeightMatrix, block_bootstrap_estimator, select_moments

end # module
