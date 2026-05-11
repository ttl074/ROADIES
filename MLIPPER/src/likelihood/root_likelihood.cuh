#pragma once
#ifndef TREE_GENERATION_ROOT_LIKELIHOOD_CUH
#define TREE_GENERATION_ROOT_LIKELIHOOD_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include "tree/tree.hpp"
#include "partial_likelihood.cuh"

namespace root_likelihood {

__device__ double compute_root_loglikelihood_at_site(
    const DeviceTree& D,
    const NodeOpInfo& op,
    const fp_t* freqs,
    const fp_t* rate_weights,
    const unsigned* pattern_w,
    const int* invar_indices,
    double invar_proportion,
    unsigned int site_idx);

double compute_root_loglikelihood_total(
    const DeviceTree& D,
    int root_id,
    const unsigned* d_pattern_w,
    const int* d_invar_indices,
    double invar_proportion,
    cudaStream_t stream = 0);

// Compute a combined placement log-likelihood using query CLV (pendant),
// parent_down*sibling_up (distal), and target_up (proximal) with provided PMATs.
double compute_combined_placement_loglikelihood(
    const DeviceTree& D,
    int target_id,
    const fp_t* d_pendant_pmat, // [rate_cats * states * states]
    const fp_t* d_distal_pmat,  // [rate_cats * states * states]
    const fp_t* d_proximal_pmat,// [rate_cats * states * states]
    cudaStream_t stream = 0);

// Compute combined log-likelihood per placement op (parallel across ops).
void compute_combined_loglik_per_op(
    const DeviceTree& D,
    const NodeOpInfo* d_ops,
    const int* d_op_indices,
    int num_ops,
    const fp_t* d_pendant_pmats, // [num_ops * rate_cats * states * states]
    const fp_t* d_distal_pmats,  // [N * rate_cats * states * states]
    const fp_t* d_proximal_pmats,// [N * rate_cats * states * states]
    fp_t* d_likelihoods,
    cudaStream_t stream = 0);

// Compute combined log-likelihood per placement op into device buffer.
void compute_combined_loglik_per_op_device(
    const DeviceTree& D,
    const NodeOpInfo* d_ops,
    const int* d_op_indices,
    int num_ops,
    const fp_t* d_pendant_pmats, // [num_ops * rate_cats * states * states]
    const fp_t* d_distal_pmats,  // [N * rate_cats * states * states]
    const fp_t* d_proximal_pmats,// [N * rate_cats * states * states]
    fp_t* d_out,                  // [num_ops]
    cudaStream_t stream = 0);

} // namespace root_likelihood

#endif // TREE_GENERATION_ROOT_LIKELIHOOD_CUH
