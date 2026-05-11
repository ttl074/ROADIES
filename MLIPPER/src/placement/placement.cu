#include <vector>
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "placement.cuh"
#include "util/mlipper_util.h"
#include "pmatrix/pmat.h"
#include "pmatrix/pmat_gpu.cuh"
#include "tree/tree.hpp"
#include "likelihood/partial_likelihood.cuh"
#include "likelihood/root_likelihood.cuh"
#include "derivative.cuh"


namespace {
constexpr int kDefaultFullOptPasses = 4;
constexpr int kDefaultRefineGlobalPasses = 0;
constexpr int kDefaultRefineExtraPasses = 0;
constexpr int kDefaultDetectTopK = 0;
constexpr int kDefaultRefineTopK = 0;
constexpr double kRefineGapTop2 = 0.25;
constexpr double kRefineGapTop5 = 1.0;
constexpr double kRefineConvergedLoglkEps = 1e-2;
constexpr double kRefineConvergedLengthEps = 1e-4;
constexpr int kExportPlacementTopK = 5;

struct RefineConfig {
    int full_opt_passes = kDefaultFullOptPasses;
    int global_opt_passes = kDefaultRefineGlobalPasses;
    int refine_extra_passes = kDefaultRefineExtraPasses;
    int detect_topk_limit = kDefaultDetectTopK;
    int refine_topk_limit = kDefaultRefineTopK;
    double gap_top2 = kRefineGapTop2;
    double gap_top5 = kRefineGapTop5;
    double converged_loglk_eps = kRefineConvergedLoglkEps;
    double converged_length_eps = kRefineConvergedLengthEps;
    bool enable_convergence_check = false;
};

static int getenv_int_or_default(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    return std::max(0, std::atoi(value));
}

static int getenv_signed_int_or_default(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    return std::atoi(value);
}

static double getenv_double_or_default(const char* name, double default_value) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    return std::atof(value);
}

static RefineConfig load_refine_config() {
    RefineConfig cfg;
    cfg.full_opt_passes =
        getenv_int_or_default("MLIPPER_FULL_OPT_PASSES", cfg.full_opt_passes);
    cfg.global_opt_passes =
        getenv_int_or_default("MLIPPER_REFINE_GLOBAL_PASSES", cfg.global_opt_passes);
    cfg.refine_extra_passes =
        getenv_int_or_default("MLIPPER_REFINE_EXTRA_PASSES", cfg.refine_extra_passes);
    cfg.detect_topk_limit =
        getenv_int_or_default("MLIPPER_REFINE_DETECT_TOPK", cfg.detect_topk_limit);
    cfg.refine_topk_limit =
        getenv_int_or_default("MLIPPER_REFINE_TOPK", cfg.refine_topk_limit);
    cfg.gap_top2 =
        getenv_double_or_default("MLIPPER_REFINE_GAP_TOP2", cfg.gap_top2);
    cfg.gap_top5 =
        getenv_double_or_default("MLIPPER_REFINE_GAP_TOP5", cfg.gap_top5);
    cfg.converged_loglk_eps =
        getenv_double_or_default("MLIPPER_REFINE_CONVERGED_LOGLK_EPS", cfg.converged_loglk_eps);
    cfg.converged_length_eps =
        getenv_double_or_default("MLIPPER_REFINE_CONVERGED_LENGTH_EPS", cfg.converged_length_eps);
    cfg.enable_convergence_check =
        getenv_int_or_default("MLIPPER_ENABLE_CONVERGENCE_CHECK", 0) != 0;
    return cfg;
}

static int target_id_from_op(const NodeOpInfo& op) {
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const bool target_is_right = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
    return target_is_left ? op.left_id : (target_is_right ? op.right_id : op.parent_id);
}

static std::vector<double> compute_like_weight_ratios(
    const std::vector<fp_t>& top_values)
{
    std::vector<double> ratios(top_values.size(), 0.0);
    if (top_values.empty()) {
        return ratios;
    }

    const double max_ll = static_cast<double>(top_values.front());
    double sum_weights = 0.0;
    for (size_t i = 0; i < top_values.size(); ++i) {
        const double weight = std::exp(static_cast<double>(top_values[i]) - max_ll);
        ratios[i] = weight;
        sum_weights += weight;
    }
    if (sum_weights <= 0.0 || !std::isfinite(sum_weights)) {
        ratios.assign(top_values.size(), 0.0);
        ratios.front() = 1.0;
        return ratios;
    }
    for (double& value : ratios) {
        value /= sum_weights;
    }
    return ratios;
}

static int export_placement_topk();
static bool local_child_refine_enabled();

struct LocalChildRefineFamilyOps {
    int selected_op = -1;
    int child_left_op = -1;
    int child_right_op = -1;
};

static std::vector<NodeOpInfo> load_host_ops_for_local_child_refine(
    const NodeOpInfo* d_ops,
    int num_ops,
    cudaStream_t stream)
{
    std::vector<NodeOpInfo> host_ops;
    if (!d_ops || num_ops <= 0) {
        return host_ops;
    }
    host_ops.resize(static_cast<size_t>(num_ops));
    CUDA_CHECK(cudaMemcpyAsync(
        host_ops.data(),
        d_ops,
        sizeof(NodeOpInfo) * static_cast<size_t>(num_ops),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host_ops;
}

static LocalChildRefineFamilyOps find_local_child_refine_family_ops(
    const std::vector<NodeOpInfo>& host_ops,
    int selected_target_id)
{
    LocalChildRefineFamilyOps family;
    if (selected_target_id < 0) {
        return family;
    }
    for (size_t op_idx = 0; op_idx < host_ops.size(); ++op_idx) {
        const NodeOpInfo& op = host_ops[op_idx];
        const int target_id = target_id_from_op(op);
        if (target_id == selected_target_id && family.selected_op < 0) {
            family.selected_op = static_cast<int>(op_idx);
        }
        if (op.parent_id != selected_target_id) continue;
        if (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT)) {
            family.child_left_op = static_cast<int>(op_idx);
        } else if (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT)) {
            family.child_right_op = static_cast<int>(op_idx);
        }
    }
    return family;
}

static unsigned int host_combined_scaler_shift_at(
    const std::vector<unsigned>& scaler_slice,
    const DeviceTree& D,
    unsigned int site_idx,
    int rate_idx)
{
    if (scaler_slice.empty()) return 0u;
    if (D.per_rate_scaling) {
        return scaler_slice[(size_t)site_idx * (size_t)D.rate_cats + (size_t)rate_idx];
    }
    return scaler_slice[(size_t)site_idx];
}

static unsigned host_pattern_weight_at(
    const std::vector<unsigned>& pattern_weights,
    unsigned int site_idx)
{
    return pattern_weights.empty() ? 1u : pattern_weights[(size_t)site_idx];
}

struct DoubleRerankCandidateBuffers {
    std::vector<fp_t> pendant_pmat;
    std::vector<fp_t> distal_pmat;
    std::vector<fp_t> proximal_pmat;
    std::vector<fp_t> distal_clv;
    std::vector<fp_t> proximal_clv;
    std::vector<unsigned> distal_scalers;
    std::vector<unsigned> proximal_scalers;
};

static DoubleRerankCandidateBuffers load_double_rerank_candidate_buffers(
    const DeviceTree& D,
    int op_index,
    int target_id)
{
    DoubleRerankCandidateBuffers out;
    const size_t per_query = D.pmat_per_node_elems();
    const size_t per_node_pmat = D.pmat_per_node_elems();
    const size_t per_site = (size_t)D.rate_cats * (size_t)D.states;
    const size_t per_node_clv = D.sites * per_site;
    const size_t scaler_span = D.per_rate_scaling
        ? (D.sites * (size_t)D.rate_cats)
        : D.sites;

    out.pendant_pmat.resize(per_query);
    out.distal_pmat.resize(per_node_pmat);
    out.proximal_pmat.resize(per_node_pmat);
    out.distal_clv.resize(per_node_clv);
    out.proximal_clv.resize(per_node_clv);
    if (scaler_span > 0) {
        out.distal_scalers.resize(scaler_span);
        out.proximal_scalers.resize(scaler_span);
    }

    CUDA_CHECK(cudaMemcpy(
        out.pendant_pmat.data(),
        D.d_query_pmat + (size_t)op_index * per_query,
        sizeof(fp_t) * per_query,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        out.distal_pmat.data(),
        D.d_pmat_mid_dist + (size_t)target_id * per_node_pmat,
        sizeof(fp_t) * per_node_pmat,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        out.proximal_pmat.data(),
        D.d_pmat_mid_prox + (size_t)target_id * per_node_pmat,
        sizeof(fp_t) * per_node_pmat,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        out.distal_clv.data(),
        D.d_clv_mid_base + (size_t)target_id * per_node_clv,
        sizeof(fp_t) * per_node_clv,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        out.proximal_clv.data(),
        D.d_clv_up + (size_t)target_id * per_node_clv,
        sizeof(fp_t) * per_node_clv,
        cudaMemcpyDeviceToHost));
    if (scaler_span > 0) {
        if (D.d_site_scaler_mid_base) {
            CUDA_CHECK(cudaMemcpy(
                out.distal_scalers.data(),
                D.d_site_scaler_mid_base + (size_t)target_id * scaler_span,
                sizeof(unsigned) * scaler_span,
                cudaMemcpyDeviceToHost));
        }
        if (D.d_site_scaler_up) {
            CUDA_CHECK(cudaMemcpy(
                out.proximal_scalers.data(),
                D.d_site_scaler_up + (size_t)target_id * scaler_span,
                sizeof(unsigned) * scaler_span,
                cudaMemcpyDeviceToHost));
        }
    }
    return out;
}

static double recompute_candidate_loglikelihood_double(
    const DeviceTree& D,
    const std::vector<fp_t>& query_clv,
    const std::vector<fp_t>& rate_weights,
    const std::vector<fp_t>& frequencies,
    const std::vector<unsigned>& pattern_weights,
    const DoubleRerankCandidateBuffers& candidate)
{
    const size_t states = (size_t)D.states;
    const size_t rate_cats = (size_t)D.rate_cats;
    const size_t per_site = rate_cats * states;
    const double eps = 1e-300;
    constexpr double kLn2Host = 0.69314718055994530942;

    double total = 0.0;
    for (unsigned int site = 0; site < D.sites; ++site) {
        const size_t site_off = (size_t)site * per_site;
        std::vector<double> rate_vals(rate_cats, 0.0);
        std::vector<unsigned int> rate_shifts(rate_cats, 0u);
        unsigned int site_min_shift = 0u;
        bool have_positive = false;

        for (int rc = 0; rc < D.rate_cats; ++rc) {
            const size_t rc_off = (size_t)rc * states;
            const size_t matrix_off = (size_t)rc * states * states;
            double rate_sum = 0.0;
            for (int s = 0; s < D.states; ++s) {
                double acc_pend = 0.0;
                double acc_dist = 0.0;
                double acc_prox = 0.0;
                const size_t row_off = matrix_off + (size_t)s * states;
                for (int k = 0; k < D.states; ++k) {
                    const size_t idx = row_off + (size_t)k;
                    acc_pend += static_cast<double>(candidate.pendant_pmat[idx]) *
                                static_cast<double>(query_clv[site_off + rc_off + (size_t)k]);
                    acc_dist += static_cast<double>(candidate.distal_pmat[idx]) *
                                static_cast<double>(candidate.distal_clv[site_off + rc_off + (size_t)k]);
                    acc_prox += static_cast<double>(candidate.proximal_pmat[idx]) *
                                static_cast<double>(candidate.proximal_clv[site_off + rc_off + (size_t)k]);
                }
                rate_sum += acc_pend * acc_dist * acc_prox * static_cast<double>(frequencies[(size_t)s]);
            }

            const unsigned int distal_shift =
                host_combined_scaler_shift_at(candidate.distal_scalers, D, site, rc);
            const unsigned int prox_shift =
                host_combined_scaler_shift_at(candidate.proximal_scalers, D, site, rc);
            const unsigned int total_shift = distal_shift + prox_shift;
            rate_vals[(size_t)rc] = rate_sum;
            rate_shifts[(size_t)rc] = total_shift;
            if (rate_sum > 0.0) {
                if (!have_positive || total_shift < site_min_shift) {
                    site_min_shift = total_shift;
                }
                have_positive = true;
            }
        }

        double site_lk = 0.0;
        for (int rc = 0; rc < D.rate_cats; ++rc) {
            double val = rate_vals[(size_t)rc];
            if (val > 0.0) {
                const int diff = static_cast<int>(rate_shifts[(size_t)rc]) - static_cast<int>(site_min_shift);
                if (diff > 0) val = std::ldexp(val, -diff);
                site_lk += static_cast<double>(rate_weights[(size_t)rc]) * val;
            }
        }

        total += static_cast<double>(host_pattern_weight_at(pattern_weights, site)) *
            (std::log(site_lk > eps ? site_lk : eps) - static_cast<double>(site_min_shift) * kLn2Host);
    }
    return total;
}

#if !defined(MLIPPER_USE_DOUBLE)
constexpr int kDefaultDoubleRerankUlpFactor = 4;

static bool double_rerank_enabled() {
    const char* value = std::getenv("MLIPPER_DOUBLE_RERANK");
    if (!value || !value[0]) return true;
    return std::atoi(value) != 0;
}

static int double_rerank_ulp_factor() {
    return getenv_int_or_default("MLIPPER_DOUBLE_RERANK_ULP_FACTOR", kDefaultDoubleRerankUlpFactor);
}

static double double_rerank_gap_floor() {
    return getenv_double_or_default("MLIPPER_DOUBLE_RERANK_GAP_TOP2", 0.0);
}

static double float_loglik_ulp(double value) {
    const float here = static_cast<float>(value);
    const float next = std::nextafter(here, std::numeric_limits<float>::infinity());
    return std::fabs(static_cast<double>(next) - static_cast<double>(here));
}

static void maybe_apply_double_rerank(
    const DeviceTree& D,
    const NodeOpInfo* d_ops,
    const std::vector<int>& top_indices,
    PlacementResult& result,
    cudaStream_t stream)
{
    if (!double_rerank_enabled()) return;
    if (!d_ops) return;
    if (top_indices.size() < 2 || result.top_placements.size() < 2) return;

    const double best_ll = result.top_placements.front().loglikelihood;
    const double gap_top2 = result.top_placements[0].loglikelihood - result.top_placements[1].loglikelihood;
    const double ulp_gap = float_loglik_ulp(best_ll) * static_cast<double>(double_rerank_ulp_factor());
    const double trigger_gap = std::max(double_rerank_gap_floor(), ulp_gap);
    if (!(gap_top2 <= trigger_gap)) return;

    size_t rerank_count = 1;
    while (rerank_count < result.top_placements.size()) {
        const double gap = result.top_placements[0].loglikelihood - result.top_placements[rerank_count].loglikelihood;
        if (gap > trigger_gap) break;
        ++rerank_count;
    }
    if (rerank_count < 2) return;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const size_t per_site = (size_t)D.rate_cats * (size_t)D.states;
    std::vector<fp_t> query_clv(D.sites * per_site, fp_t(0));
    std::vector<fp_t> rate_weights((size_t)D.rate_cats, fp_t(0));
    std::vector<fp_t> frequencies((size_t)D.states, fp_t(0));
    std::vector<unsigned> pattern_weights(D.sites, 1u);

    if (!query_clv.empty()) {
        CUDA_CHECK(cudaMemcpy(
            query_clv.data(),
            D.d_query_clv,
            sizeof(fp_t) * query_clv.size(),
            cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(
        rate_weights.data(),
        D.d_rate_weights,
        sizeof(fp_t) * rate_weights.size(),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        frequencies.data(),
        D.d_frequencies,
        sizeof(fp_t) * frequencies.size(),
        cudaMemcpyDeviceToHost));
    if (D.d_pattern_weights_u && D.sites > 0) {
        CUDA_CHECK(cudaMemcpy(
            pattern_weights.data(),
            D.d_pattern_weights_u,
            sizeof(unsigned) * D.sites,
            cudaMemcpyDeviceToHost));
    }

    struct RankedWithOriginal {
        PlacementResult::RankedPlacement placement;
        size_t original_rank = 0;
    };

    std::vector<RankedWithOriginal> reranked;
    reranked.reserve(result.top_placements.size());
    for (size_t i = 0; i < result.top_placements.size(); ++i) {
        reranked.push_back(RankedWithOriginal{result.top_placements[i], i});
    }

    for (size_t i = 0; i < rerank_count; ++i) {
        const int op_index = top_indices[i];
        if (op_index < 0) continue;

        NodeOpInfo host_op{};
        CUDA_CHECK(cudaMemcpy(
            &host_op,
            d_ops + op_index,
            sizeof(NodeOpInfo),
            cudaMemcpyDeviceToHost));

        const int target_id = target_id_from_op(host_op);
        if (target_id < 0 || target_id >= D.N) continue;

        const DoubleRerankCandidateBuffers candidate =
            load_double_rerank_candidate_buffers(D, op_index, target_id);
        reranked[i].placement.loglikelihood = recompute_candidate_loglikelihood_double(
            D,
            query_clv,
            rate_weights,
            frequencies,
            pattern_weights,
            candidate);
    }

    std::stable_sort(
        reranked.begin(),
        reranked.end(),
        [](const RankedWithOriginal& lhs, const RankedWithOriginal& rhs) {
            if (lhs.placement.loglikelihood != rhs.placement.loglikelihood) {
                return lhs.placement.loglikelihood > rhs.placement.loglikelihood;
            }
            return lhs.original_rank < rhs.original_rank;
        });

    result.top_placements.clear();
    result.top_placements.reserve(reranked.size());
    std::vector<fp_t> reranked_logliks;
    reranked_logliks.reserve(reranked.size());
    for (const RankedWithOriginal& entry : reranked) {
        result.top_placements.push_back(entry.placement);
        reranked_logliks.push_back(static_cast<fp_t>(entry.placement.loglikelihood));
    }

    const std::vector<double> like_weight_ratios = compute_like_weight_ratios(reranked_logliks);
    for (size_t i = 0; i < result.top_placements.size() && i < like_weight_ratios.size(); ++i) {
        result.top_placements[i].like_weight_ratio = like_weight_ratios[i];
    }

    result.target_id = result.top_placements.front().target_id;
    result.loglikelihood = result.top_placements.front().loglikelihood;
    result.proximal_length = result.top_placements.front().proximal_length;
    result.pendant_length = result.top_placements.front().pendant_length;
    result.gap_top2 = std::numeric_limits<double>::infinity();
    result.gap_top5 = std::numeric_limits<double>::infinity();
    if (result.top_placements.size() > 1) {
        result.gap_top2 = result.top_placements[0].loglikelihood - result.top_placements[1].loglikelihood;
        const size_t top5_idx = std::min<size_t>(4, result.top_placements.size() - 1);
        result.gap_top5 = result.top_placements[0].loglikelihood - result.top_placements[top5_idx].loglikelihood;
    }
}

#else
static void maybe_apply_double_rerank(
    const DeviceTree&,
    const NodeOpInfo*,
    const std::vector<int>&,
    PlacementResult&,
    cudaStream_t)
{}

#endif

static void rerank_selected_target_and_children(
    const DeviceTree& D,
    const NodeOpInfo* d_ops,
    int num_ops,
    PlacementResult& result,
    cudaStream_t stream)
{
    if (!local_child_refine_enabled()) return;
    if (!d_ops || num_ops <= 0) return;
    if (result.target_id < 0 || result.target_id >= D.N) return;

    const std::vector<NodeOpInfo> host_ops =
        load_host_ops_for_local_child_refine(d_ops, num_ops, stream);
    if (host_ops.empty()) return;
    const LocalChildRefineFamilyOps family =
        find_local_child_refine_family_ops(host_ops, result.target_id);
    if (family.selected_op < 0) return;
    if (family.child_left_op < 0 && family.child_right_op < 0) return;

    const size_t per_site = (size_t)D.rate_cats * (size_t)D.states;
    std::vector<fp_t> query_clv(D.sites * per_site, fp_t(0));
    std::vector<fp_t> rate_weights((size_t)D.rate_cats, fp_t(0));
    std::vector<fp_t> frequencies((size_t)D.states, fp_t(0));
    std::vector<unsigned> pattern_weights(D.sites, 1u);

    if (!query_clv.empty()) {
        CUDA_CHECK(cudaMemcpy(
            query_clv.data(),
            D.d_query_clv,
            sizeof(fp_t) * query_clv.size(),
            cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(
        rate_weights.data(),
        D.d_rate_weights,
        sizeof(fp_t) * rate_weights.size(),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        frequencies.data(),
        D.d_frequencies,
        sizeof(fp_t) * frequencies.size(),
        cudaMemcpyDeviceToHost));
    if (D.d_pattern_weights_u && D.sites > 0) {
        CUDA_CHECK(cudaMemcpy(
            pattern_weights.data(),
            D.d_pattern_weights_u,
            sizeof(unsigned) * D.sites,
            cudaMemcpyDeviceToHost));
    }

    struct LocalCandidate {
        PlacementResult::RankedPlacement placement;
        int op_index = -1;
    };

    std::vector<LocalCandidate> local_candidates;
    local_candidates.reserve(3);
    auto append_local_candidate = [&](int op_index) {
        if (op_index < 0) return;
        const NodeOpInfo& op = host_ops[(size_t)op_index];
        const int target_id = target_id_from_op(op);
        if (target_id < 0 || target_id >= D.N) return;

        LocalCandidate candidate;
        candidate.op_index = op_index;
        candidate.placement.target_id = target_id;

        fp_t pendant_length = fp_t(0);
        fp_t proximal_length = fp_t(0);
        CUDA_CHECK(cudaMemcpy(
            &pendant_length,
            D.d_prev_pendant_length + target_id,
            sizeof(fp_t),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            &proximal_length,
            D.d_prev_proximal_length + target_id,
            sizeof(fp_t),
            cudaMemcpyDeviceToHost));
        candidate.placement.pendant_length = static_cast<double>(pendant_length);
        candidate.placement.proximal_length = static_cast<double>(proximal_length);

        const DoubleRerankCandidateBuffers buffers =
            load_double_rerank_candidate_buffers(D, op_index, target_id);
        candidate.placement.loglikelihood = recompute_candidate_loglikelihood_double(
            D,
            query_clv,
            rate_weights,
            frequencies,
            pattern_weights,
            buffers);
        local_candidates.push_back(candidate);
    };

    append_local_candidate(family.selected_op);
    append_local_candidate(family.child_left_op);
    append_local_candidate(family.child_right_op);
    if (local_candidates.size() < 2) return;

    std::stable_sort(
        local_candidates.begin(),
        local_candidates.end(),
        [](const LocalCandidate& lhs, const LocalCandidate& rhs) {
            if (lhs.placement.loglikelihood != rhs.placement.loglikelihood) {
                return lhs.placement.loglikelihood > rhs.placement.loglikelihood;
            }
            return lhs.op_index < rhs.op_index;
        });

    std::vector<PlacementResult::RankedPlacement> merged = result.top_placements;
    for (const LocalCandidate& local : local_candidates) {
        bool replaced = false;
        for (PlacementResult::RankedPlacement& existing : merged) {
            if (existing.target_id == local.placement.target_id) {
                existing = local.placement;
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            merged.push_back(local.placement);
        }
    }

    auto existing_rank = [&](int target_id) -> size_t {
        for (size_t rank = 0; rank < result.top_placements.size(); ++rank) {
            if (result.top_placements[rank].target_id == target_id) return rank;
        }
        return result.top_placements.size();
    };

    std::stable_sort(
        merged.begin(),
        merged.end(),
        [&](const PlacementResult::RankedPlacement& lhs, const PlacementResult::RankedPlacement& rhs) {
            if (lhs.loglikelihood != rhs.loglikelihood) {
                return lhs.loglikelihood > rhs.loglikelihood;
            }
            return existing_rank(lhs.target_id) < existing_rank(rhs.target_id);
        });

    const size_t keep = std::max<size_t>(export_placement_topk(), 3);
    if (merged.size() > keep) {
        merged.resize(keep);
    }
    std::vector<fp_t> merged_logliks;
    merged_logliks.reserve(merged.size());
    for (const PlacementResult::RankedPlacement& placement : merged) {
        merged_logliks.push_back(static_cast<fp_t>(placement.loglikelihood));
    }
    const std::vector<double> like_weight_ratios = compute_like_weight_ratios(merged_logliks);
    for (size_t i = 0; i < merged.size() && i < like_weight_ratios.size(); ++i) {
        merged[i].like_weight_ratio = like_weight_ratios[i];
    }

    result.top_placements.swap(merged);
    result.target_id = result.top_placements.front().target_id;
    result.loglikelihood = result.top_placements.front().loglikelihood;
    result.proximal_length = result.top_placements.front().proximal_length;
    result.pendant_length = result.top_placements.front().pendant_length;
    result.gap_top2 = std::numeric_limits<double>::infinity();
    result.gap_top5 = std::numeric_limits<double>::infinity();
    if (result.top_placements.size() > 1) {
        result.gap_top2 = result.top_placements[0].loglikelihood - result.top_placements[1].loglikelihood;
        const size_t top5_idx = std::min<size_t>(4, result.top_placements.size() - 1);
        result.gap_top5 = result.top_placements[0].loglikelihood - result.top_placements[top5_idx].loglikelihood;
    }
}

static std::vector<PlacementResult::RankedPlacement> build_top_ranked_placements(
    const DeviceTree& D,
    const NodeOpInfo* d_ops,
    const std::vector<int>& top_indices,
    const std::vector<fp_t>& top_values)
{
    std::vector<PlacementResult::RankedPlacement> ranked;
    const size_t keep = std::min(top_indices.size(), top_values.size());
    ranked.reserve(keep);

    const std::vector<double> like_weight_ratios = compute_like_weight_ratios(top_values);
    for (size_t i = 0; i < keep; ++i) {
        const int op_index = top_indices[i];
        if (op_index < 0) {
            continue;
        }

        NodeOpInfo host_op{};
        CUDA_CHECK(cudaMemcpy(
            &host_op,
            d_ops + op_index,
            sizeof(NodeOpInfo),
            cudaMemcpyDeviceToHost));

        const int target_id = target_id_from_op(host_op);
        if (target_id < 0 || target_id >= D.N) {
            continue;
        }

        fp_t pendant_length = fp_t(0);
        fp_t proximal_length = fp_t(0);
        CUDA_CHECK(cudaMemcpy(
            &pendant_length,
            D.d_prev_pendant_length + target_id,
            sizeof(fp_t),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            &proximal_length,
            D.d_prev_proximal_length + target_id,
            sizeof(fp_t),
            cudaMemcpyDeviceToHost));

        PlacementResult::RankedPlacement candidate;
        candidate.target_id = target_id;
        candidate.loglikelihood = static_cast<double>(top_values[i]);
        candidate.proximal_length = static_cast<double>(proximal_length);
        candidate.pendant_length = static_cast<double>(pendant_length);
        candidate.like_weight_ratio = like_weight_ratios[i];
        ranked.push_back(candidate);
    }
    return ranked;
}

static int export_placement_topk() {
    const char* value = std::getenv("MLIPPER_EXPORT_PLACEMENT_TOPK");
    if (!value || !value[0]) return kExportPlacementTopK;
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : kExportPlacementTopK;
}

static bool local_child_refine_enabled() {
    return getenv_int_or_default("MLIPPER_LOCAL_CHILD_REFINE", 1) != 0;
}

static bool newton_debug_enabled() {
    return getenv_int_or_default("MLIPPER_DEBUG_NEWTON", 0) != 0;
}

static int newton_debug_limit() {
    return getenv_int_or_default("MLIPPER_DEBUG_NEWTON_LIMIT", 8);
}

static int newton_debug_all_iters() {
    return getenv_int_or_default("MLIPPER_DEBUG_NEWTON_ALL_ITERS", 0);
}

static int newton_debug_target_id() {
    return getenv_signed_int_or_default("MLIPPER_DEBUG_TARGET_ID", -1);
}

static bool sumtable_debug_enabled() {
    return getenv_int_or_default("MLIPPER_DEBUG_SUMTABLE", 0) != 0;
}

static int sumtable_debug_site() {
    return getenv_signed_int_or_default("MLIPPER_DEBUG_SUMTABLE_SITE", -1);
}

static int sumtable_debug_rate() {
    return getenv_signed_int_or_default("MLIPPER_DEBUG_SUMTABLE_RATE", 0);
}

static bool midbase_snapshot_enabled() {
    return getenv_int_or_default("MLIPPER_DEBUG_MIDBASE_SNAPSHOT", 0) != 0;
}

static int midbase_snapshot_target() {
    return getenv_signed_int_or_default("MLIPPER_DEBUG_MIDBASE_TARGET", -1);
}

static int midbase_snapshot_site() {
    return getenv_signed_int_or_default("MLIPPER_DEBUG_MIDBASE_SITE", -1);
}

static int midbase_snapshot_rate() {
    return getenv_signed_int_or_default("MLIPPER_DEBUG_MIDBASE_RATE", 0);
}

static bool scaler_scan_enabled() {
    return getenv_int_or_default("MLIPPER_DEBUG_SCALER_SCAN", 0) != 0;
}

static int scaler_scan_site() {
    return getenv_signed_int_or_default("MLIPPER_DEBUG_SCALER_SCAN_SITE", -1);
}

static int scaler_scan_rate() {
    return getenv_signed_int_or_default("MLIPPER_DEBUG_SCALER_SCAN_RATE", 0);
}

template <typename T>
static void cuda_free_noexcept(T*& ptr) noexcept {
    if (!ptr) return;
    cudaFree(ptr);
    ptr = nullptr;
}

struct PlacementKernelDebugConfig {
    int newton = 0;
    int newton_limit = 0;
    int newton_all_iters = 0;
    int newton_target = -1;
    int sumtable = 0;
    int sumtable_site = -1;
    int sumtable_rate = 0;
};

static PlacementKernelDebugConfig load_placement_kernel_debug_config() {
    PlacementKernelDebugConfig cfg;
    cfg.newton = newton_debug_enabled() ? 1 : 0;
    cfg.newton_limit = newton_debug_limit();
    cfg.newton_all_iters = newton_debug_all_iters();
    cfg.newton_target = newton_debug_target_id();
    cfg.sumtable = sumtable_debug_enabled() ? 1 : 0;
    cfg.sumtable_site = sumtable_debug_site();
    cfg.sumtable_rate = sumtable_debug_rate();
    return cfg;
}

struct PlacementKernelScratchBuffers {
    fp_t* d_prev_loglk = nullptr;
    fp_t* d_last_loglk = nullptr;
    int* d_active_ops = nullptr;
    int* d_refine_op_indices = nullptr;
    int* d_refine_active_ops = nullptr;
    int* d_any_active_flag = nullptr;
    void* d_any_active_temp = nullptr;
    fp_t* d_last_pendant_length = nullptr;
    fp_t* d_last_proximal_length = nullptr;
    size_t any_active_temp_bytes = 0;
    int refine_capacity = 0;
    std::vector<fp_t> host_loglk_cache;
    std::vector<int> host_order_cache;

    ~PlacementKernelScratchBuffers() {
        release();
    }

    void release() noexcept {
        cuda_free_noexcept(d_last_proximal_length);
        cuda_free_noexcept(d_last_pendant_length);
        cuda_free_noexcept(d_last_loglk);
        cuda_free_noexcept(d_prev_loglk);
        cuda_free_noexcept(d_any_active_temp);
        cuda_free_noexcept(d_any_active_flag);
        cuda_free_noexcept(d_refine_active_ops);
        cuda_free_noexcept(d_refine_op_indices);
        cuda_free_noexcept(d_active_ops);
        any_active_temp_bytes = 0;
        refine_capacity = 0;
    }

    template <typename CheckCudaFn>
    void ensure_refine_capacity(int required_capacity, CheckCudaFn&& check_cuda) {
        if (required_capacity <= refine_capacity) return;
        cuda_free_noexcept(d_refine_op_indices);
        cuda_free_noexcept(d_refine_active_ops);
        check_cuda(
            "cudaMalloc d_refine_op_indices",
            cudaMalloc(&d_refine_op_indices, sizeof(int) * (size_t)required_capacity));
        check_cuda(
            "cudaMalloc d_refine_active_ops",
            cudaMalloc(&d_refine_active_ops, sizeof(int) * (size_t)required_capacity));
        refine_capacity = required_capacity;
    }
};

static double restored_abs_max(const fp_t* values, int count, unsigned shift) {
    double best = 0.0;
    for (int idx = 0; idx < count; ++idx) {
        const double restored =
            std::ldexp(static_cast<double>(values[idx]), -static_cast<int>(shift));
        best = std::max(best, std::abs(restored));
    }
    return best;
}

static double raw_abs_max(const fp_t* values, int count) {
    double best = 0.0;
    for (int idx = 0; idx < count; ++idx) {
        best = std::max(best, std::abs(static_cast<double>(values[idx])));
    }
    return best;
}

static void maybe_dump_scaler_scan(
    const char* stage,
    const DeviceTree& D,
    const NodeOpInfo* d_ops,
    int num_ops,
    cudaStream_t stream)
{
    if (!scaler_scan_enabled()) return;
    const int site = scaler_scan_site();
    const int rate = scaler_scan_rate();
    if (site < 0 || static_cast<size_t>(site) >= D.sites) return;
    if (rate < 0 || rate >= D.rate_cats) return;
    if (!D.d_clv_down || !D.d_clv_up || !D.d_clv_mid_base) return;

    const size_t per_node = static_cast<size_t>(D.sites) * static_cast<size_t>(D.rate_cats) * static_cast<size_t>(D.states);
    const size_t site_span = static_cast<size_t>(D.rate_cats) * static_cast<size_t>(D.states);
    const size_t scaler_node_span = D.per_rate_scaling
        ? (D.sites * static_cast<size_t>(D.rate_cats))
        : D.sites;
    const size_t scaler_site_off = D.per_rate_scaling
        ? (static_cast<size_t>(site) * static_cast<size_t>(D.rate_cats) + static_cast<size_t>(rate))
        : static_cast<size_t>(site);

    std::vector<fp_t> host_down(static_cast<size_t>(D.states));
    std::vector<fp_t> host_up(static_cast<size_t>(D.states));
    std::vector<fp_t> host_mid_base(static_cast<size_t>(D.states));
    std::vector<fp_t> host_mid(static_cast<size_t>(D.states));
    std::vector<NodeOpInfo> host_ops;
    std::vector<int> parent_of_target(static_cast<size_t>(D.N), -1);
    std::vector<int> sibling_of_target(static_cast<size_t>(D.N), -1);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (d_ops && num_ops > 0) {
        host_ops.resize(static_cast<size_t>(num_ops));
        CUDA_CHECK(cudaMemcpy(
            host_ops.data(),
            d_ops,
            sizeof(NodeOpInfo) * static_cast<size_t>(num_ops),
            cudaMemcpyDeviceToHost));
        for (const NodeOpInfo& op : host_ops) {
            const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
            const bool target_is_right = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
            if (!target_is_left && !target_is_right) continue;
            const int target_id = target_is_left ? op.left_id : op.right_id;
            const int sibling_id = target_is_left ? op.right_id : op.left_id;
            if (target_id < 0 || target_id >= D.N) continue;
            parent_of_target[static_cast<size_t>(target_id)] = op.parent_id;
            sibling_of_target[static_cast<size_t>(target_id)] = sibling_id;
        }
    }
    std::fprintf(stderr,
        "[scaler-scan] stage=%s site=%d rate=%d nodes=%d states=%d\n",
        stage ? stage : "<null>",
        site,
        rate,
        D.N,
        D.states);

    for (int node = 0; node < D.N; ++node) {
        const size_t base =
            static_cast<size_t>(node) * per_node +
            static_cast<size_t>(site) * site_span +
            static_cast<size_t>(rate) * static_cast<size_t>(D.states);
        const size_t scaler_base =
            static_cast<size_t>(node) * scaler_node_span + scaler_site_off;

        unsigned down_shift = 0u;
        unsigned up_shift = 0u;
        unsigned mid_base_shift = 0u;
        unsigned mid_shift = 0u;
        const int parent_id = parent_of_target[static_cast<size_t>(node)];
        const int sibling_id = sibling_of_target[static_cast<size_t>(node)];
        unsigned parent_down_shift = 0u;
        unsigned sibling_up_shift = 0u;

        CUDA_CHECK(cudaMemcpy(
            host_down.data(),
            D.d_clv_down + base,
            sizeof(fp_t) * static_cast<size_t>(D.states),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            host_up.data(),
            D.d_clv_up + base,
            sizeof(fp_t) * static_cast<size_t>(D.states),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            host_mid_base.data(),
            D.d_clv_mid_base + base,
            sizeof(fp_t) * static_cast<size_t>(D.states),
            cudaMemcpyDeviceToHost));
        if (D.d_clv_mid) {
            CUDA_CHECK(cudaMemcpy(
                host_mid.data(),
                D.d_clv_mid + base,
                sizeof(fp_t) * static_cast<size_t>(D.states),
                cudaMemcpyDeviceToHost));
        } else {
            std::fill(host_mid.begin(), host_mid.end(), fp_t(0));
        }

        if (D.d_site_scaler_down) {
            CUDA_CHECK(cudaMemcpy(&down_shift, D.d_site_scaler_down + scaler_base,
                                  sizeof(unsigned), cudaMemcpyDeviceToHost));
        }
        if (D.d_site_scaler_up) {
            CUDA_CHECK(cudaMemcpy(&up_shift, D.d_site_scaler_up + scaler_base,
                                  sizeof(unsigned), cudaMemcpyDeviceToHost));
        }
        if (D.d_site_scaler_mid_base) {
            CUDA_CHECK(cudaMemcpy(&mid_base_shift, D.d_site_scaler_mid_base + scaler_base,
                                  sizeof(unsigned), cudaMemcpyDeviceToHost));
        }
        if (D.d_site_scaler_mid) {
            CUDA_CHECK(cudaMemcpy(&mid_shift, D.d_site_scaler_mid + scaler_base,
                                  sizeof(unsigned), cudaMemcpyDeviceToHost));
        }
        if (parent_id >= 0 && D.d_site_scaler_down) {
            const size_t parent_scaler_base =
                static_cast<size_t>(parent_id) * scaler_node_span + scaler_site_off;
            CUDA_CHECK(cudaMemcpy(&parent_down_shift, D.d_site_scaler_down + parent_scaler_base,
                                  sizeof(unsigned), cudaMemcpyDeviceToHost));
        }
        if (sibling_id >= 0 && D.d_site_scaler_up) {
            const size_t sibling_scaler_base =
                static_cast<size_t>(sibling_id) * scaler_node_span + scaler_site_off;
            CUDA_CHECK(cudaMemcpy(&sibling_up_shift, D.d_site_scaler_up + sibling_scaler_base,
                                  sizeof(unsigned), cudaMemcpyDeviceToHost));
        }

        const int inherited_down_shift =
            (parent_id >= 0 && sibling_id >= 0)
                ? static_cast<int>(parent_down_shift + sibling_up_shift)
                : -1;
        const int down_local_shift =
            (inherited_down_shift >= 0)
                ? static_cast<int>(down_shift) - inherited_down_shift
                : -1;

        std::fprintf(stderr,
            "[scaler-scan] stage=%s node=%d "
            "parent=%d sibling=%d parent_down_shift=%u sibling_up_shift=%u inherited_down_shift=%d down_local_shift=%d "
            "down_shift=%u up_shift=%u mid_base_shift=%u mid_shift=%u "
            "down_raw_max=%.12e down_restored_max=%.12e "
            "up_raw_max=%.12e up_restored_max=%.12e "
            "mid_base_raw_max=%.12e mid_base_restored_max=%.12e "
            "mid_raw_max=%.12e mid_restored_max=%.12e\n",
            stage ? stage : "<null>",
            node,
            parent_id,
            sibling_id,
            parent_down_shift,
            sibling_up_shift,
            inherited_down_shift,
            down_local_shift,
            down_shift,
            up_shift,
            mid_base_shift,
            mid_shift,
            raw_abs_max(host_down.data(), D.states),
            restored_abs_max(host_down.data(), D.states, down_shift),
            raw_abs_max(host_up.data(), D.states),
            restored_abs_max(host_up.data(), D.states, up_shift),
            raw_abs_max(host_mid_base.data(), D.states),
            restored_abs_max(host_mid_base.data(), D.states, mid_base_shift),
            raw_abs_max(host_mid.data(), D.states),
            restored_abs_max(host_mid.data(), D.states, mid_shift));
    }
    std::fflush(stderr);
}

static void maybe_dump_midbase_snapshot(
    const char* stage,
    const DeviceTree& D,
    cudaStream_t stream)
{
    if (!midbase_snapshot_enabled()) return;
    const int target_id = midbase_snapshot_target();
    const int site = midbase_snapshot_site();
    const int rate = midbase_snapshot_rate();
    if (target_id < 0 || target_id >= D.N) return;
    if (site < 0 || static_cast<size_t>(site) >= D.sites) return;
    if (rate < 0 || rate >= D.rate_cats) return;
    if (!D.d_clv_mid_base || !D.d_clv_up) return;

    const size_t per_node = static_cast<size_t>(D.sites) * static_cast<size_t>(D.rate_cats) * static_cast<size_t>(D.states);
    const size_t site_span = static_cast<size_t>(D.rate_cats) * static_cast<size_t>(D.states);
    const size_t base =
        static_cast<size_t>(target_id) * per_node +
        static_cast<size_t>(site) * site_span +
        static_cast<size_t>(rate) * static_cast<size_t>(D.states);

    fp_t host_mid_base[4] = {fp_t(0), fp_t(0), fp_t(0), fp_t(0)};
    fp_t host_up[4] = {fp_t(0), fp_t(0), fp_t(0), fp_t(0)};
    fp_t host_mid[4] = {fp_t(0), fp_t(0), fp_t(0), fp_t(0)};
    unsigned host_down_shift = 0u;
    unsigned host_up_shift = 0u;
    unsigned host_mid_shift = 0u;

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(
        host_mid_base,
        D.d_clv_mid_base + base,
        sizeof(fp_t) * static_cast<size_t>(D.states),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        host_up,
        D.d_clv_up + base,
        sizeof(fp_t) * static_cast<size_t>(D.states),
        cudaMemcpyDeviceToHost));
    if (D.d_clv_mid) {
        CUDA_CHECK(cudaMemcpy(
            host_mid,
            D.d_clv_mid + base,
            sizeof(fp_t) * static_cast<size_t>(D.states),
            cudaMemcpyDeviceToHost));
    }
    if (D.d_site_scaler_down) {
        const size_t scaler_base = D.per_rate_scaling
            ? (static_cast<size_t>(target_id) * static_cast<size_t>(D.sites) + static_cast<size_t>(site)) * static_cast<size_t>(D.rate_cats) + static_cast<size_t>(rate)
            : static_cast<size_t>(target_id) * static_cast<size_t>(D.sites) + static_cast<size_t>(site);
        CUDA_CHECK(cudaMemcpy(
            &host_down_shift,
            D.d_site_scaler_down + scaler_base,
            sizeof(unsigned),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            &host_up_shift,
            D.d_site_scaler_up + scaler_base,
            sizeof(unsigned),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            &host_mid_shift,
            D.d_site_scaler_mid_base + scaler_base,
            sizeof(unsigned),
            cudaMemcpyDeviceToHost));
    }

    std::fprintf(
        stderr,
        "[midbase-snapshot] stage=%s target=%d site=%d rate=%d "
        "down_shift=%u up_shift=%u mid_shift=%u "
        "mid_base=(%.12e,%.12e,%.12e,%.12e) "
        "up=(%.12e,%.12e,%.12e,%.12e) "
        "mid=(%.12e,%.12e,%.12e,%.12e)\n",
        stage ? stage : "<null>",
        target_id,
        site,
        rate,
        host_down_shift,
        host_up_shift,
        host_mid_shift,
        static_cast<double>(host_mid_base[0]),
        static_cast<double>(host_mid_base[1]),
        static_cast<double>(host_mid_base[2]),
        static_cast<double>(host_mid_base[3]),
        static_cast<double>(host_up[0]),
        static_cast<double>(host_up[1]),
        static_cast<double>(host_up[2]),
        static_cast<double>(host_up[3]),
        static_cast<double>(host_mid[0]),
        static_cast<double>(host_mid[1]),
        static_cast<double>(host_mid[2]),
        static_cast<double>(host_mid[3]));
    std::fflush(stderr);
}
}

__global__ void BuildOpPendantLengthsKernel(
    const NodeOpInfo* ops,
    const fp_t* node_lengths,
    fp_t* op_lengths,
    int num_ops,
    int total_nodes,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int op_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_local >= num_ops) return;
    if (!ops || !op_lengths) return;

    fp_t branch_length = default_len;
    if (node_lengths) {
        const NodeOpInfo op = ops[op_local];
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const int target_id = target_is_left ? op.left_id : op.right_id;
        if (target_id >= 0 && target_id < total_nodes) {
            branch_length = node_lengths[target_id];
        }
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;
    op_lengths[op_local] = branch_length;
}

__global__ void BuildOpDistalLengthsKernel(
    const NodeOpInfo* ops,
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    fp_t* op_lengths,
    int num_ops,
    int total_nodes,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int op_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_local >= num_ops) return;
    if (!ops || !op_lengths) return;

    fp_t branch_length = default_len;
    if (total_lengths && proximal_lengths) {
        const NodeOpInfo op = ops[op_local];
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const int target_id = target_is_left ? op.left_id : op.right_id;
        if (target_id >= 0 && target_id < total_nodes) {
            branch_length = total_lengths[target_id] - proximal_lengths[target_id];
        }
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;
    op_lengths[op_local] = branch_length;
}

__global__ void BuildNodePendantLengthsKernel(
    const fp_t* node_lengths,
    fp_t* out_lengths,
    int total_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= total_nodes) return;
    if (!out_lengths) return;

    fp_t branch_length = default_len;
    if (node_lengths) {
        branch_length = node_lengths[node_id];
    }
    if (node_id == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;
    out_lengths[node_id] = branch_length;
}

__global__ void BuildInitialProximalLengthsKernel(
    const fp_t* node_lengths,
    fp_t* out_lengths,
    int total_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= total_nodes) return;
    if (!out_lengths) return;

    fp_t branch_length = default_len;
    if (node_lengths) {
        branch_length = static_cast<fp_t>(0.5) * node_lengths[node_id];
    }
    if (node_id == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;
    out_lengths[node_id] = branch_length;
}

__global__ void FillSequentialIndicesKernel(
    int* out_indices,
    int count)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_index >= count) return;
    if (!out_indices) return;
    out_indices[flat_index] = flat_index;
}

__global__ void BuildNodeDistalLengthsKernel(
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    fp_t* out_lengths,
    int total_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= total_nodes) return;
    if (!out_lengths) return;

    fp_t branch_length = default_len;
    if (total_lengths && proximal_lengths) {
        branch_length = total_lengths[node_id] - proximal_lengths[node_id];
    }
    if (node_id == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;
    out_lengths[node_id] = branch_length;
}

// Keep per-op best log-likelihood; rollback branch lengths if current pass is worse.
__global__ void KeepBestBranchLengthsKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    fp_t* curr_loglk,
    fp_t* prev_loglk,
    fp_t* curr_pendant,
    fp_t* curr_proximal,
    fp_t* prev_pendant,
    fp_t* prev_proximal,
    int* active_ops,
    int num_ops,
    int total_nodes,
    int debug_target)
{
    const int op_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_local >= num_ops) return;
    if (!ops || !curr_loglk || !prev_loglk ||
        !curr_pendant || !curr_proximal || !prev_pendant || !prev_proximal) return;
    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= total_nodes) return;
    const fp_t curr = curr_loglk[op_idx];
    const fp_t prev = prev_loglk[op_idx];
    if (target_id == debug_target) {
        printf(
            "[keepbest-debug] op=%d target=%d curr_ll=%.12f prev_ll=%.12f curr_pend=%.12f prev_pend=%.12f curr_prox=%.12f prev_prox=%.12f\n",
            op_idx,
            target_id,
            static_cast<double>(curr),
            static_cast<double>(prev),
            static_cast<double>(curr_pendant[target_id]),
            static_cast<double>(prev_pendant[target_id]),
            static_cast<double>(curr_proximal[target_id]),
            static_cast<double>(prev_proximal[target_id]));
    }
    if (curr < prev) {
        curr_loglk[op_idx] = prev;
        curr_pendant[target_id] = prev_pendant[target_id];
        curr_proximal[target_id] = prev_proximal[target_id];
        if (active_ops) active_ops[op_local] = 0;
    } else {
        prev_loglk[op_idx] = curr;
        prev_pendant[target_id] = curr_pendant[target_id];
        prev_proximal[target_id] = curr_proximal[target_id];
    }
}

__global__ void UpdateActiveOpsByConvergenceKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    const fp_t* curr_loglk,
    const fp_t* prev_loglk,
    const fp_t* curr_pendant,
    const fp_t* prev_pendant,
    const fp_t* curr_proximal,
    const fp_t* prev_proximal,
    int* active_ops,
    int num_ops,
    int total_nodes,
    double loglk_eps,
    double length_eps)
{
    const int op_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_local >= num_ops) return;
    if (!ops || !curr_loglk || !prev_loglk ||
        !curr_pendant || !prev_pendant || !curr_proximal || !prev_proximal || !active_ops) {
        return;
    }
    if (active_ops[op_local] == 0) return;

    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;
    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= total_nodes) return;

    const double d_ll = fabs(static_cast<double>(curr_loglk[op_idx]) - static_cast<double>(prev_loglk[op_idx]));
    const double d_pendant = fabs(static_cast<double>(curr_pendant[target_id]) - static_cast<double>(prev_pendant[target_id]));
    const double d_proximal = fabs(static_cast<double>(curr_proximal[target_id]) - static_cast<double>(prev_proximal[target_id]));

    if (d_ll < loglk_eps && d_pendant < length_eps && d_proximal < length_eps) {
        active_ops[op_local] = 0;
    }
}

// Build per-placement pendant PMATs directly from the target branch lengths.
__global__ void BuildPendantPMATPerOpKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    const fp_t* node_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    fp_t p,
    fp_t* P,
    int states,
    int rate_cats,
    int num_ops,
    int total_nodes,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = num_ops * rate_cats;
    if (flat_index >= total_entries) return;

    const int op_local = flat_index / rate_cats;
    const int rate_idx = flat_index - op_local * rate_cats;
    if (!ops || op_local >= num_ops || rate_idx >= rate_cats) return;
    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;

    fp_t branch_length = default_len;
    if (node_lengths && target_id >= 0 && target_id < total_nodes) {
        branch_length = node_lengths[target_id];
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;

    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
    const size_t matrix_span = state_count * state_count;
    const fp_t* rate_lambdas = lambdas + rate_offset;
    fp_t* out_pmat = P + static_cast<size_t>(flat_index) * matrix_span;
    pmatrix_from_triple_device(Vinv, V, rate_lambdas, fp_t(1.0), branch_length, p, out_pmat, states);
}

// Build proximal PMATs for every node from the current proximal branch lengths.
__global__ void BuildNodeProximalPMATKernel(
    const fp_t* node_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    fp_t p,
    fp_t* P,
    int states,
    int rate_cats,
    int num_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = num_nodes * rate_cats;
    if (flat_index >= total_entries) return;

    const int node_idx = flat_index / rate_cats;
    const int rate_idx = flat_index - node_idx * rate_cats;
    if (node_idx >= num_nodes || rate_idx >= rate_cats) return;

    fp_t branch_length = default_len;
    if (node_lengths) {
        branch_length = node_lengths[node_idx];
    }
    if (node_idx == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;

    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_count = static_cast<size_t>(rate_cats);
    const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
    const size_t matrix_span = state_count * state_count;
    const size_t output_base =
        (static_cast<size_t>(node_idx) * rate_count + static_cast<size_t>(rate_idx)) * matrix_span;
    const fp_t* rate_lambdas = lambdas + rate_offset;
    fp_t* out_pmat = P + output_base;
    pmatrix_from_triple_device(Vinv, V, rate_lambdas, fp_t(1.0), branch_length, p, out_pmat, states);
}

// Build distal PMATs for every node from total branch length minus proximal length.
__global__ void BuildNodeDistalPMATKernel(
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    fp_t p,
    fp_t* P,
    int states,
    int rate_cats,
    int num_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = num_nodes * rate_cats;
    if (flat_index >= total_entries) return;

    const int node_idx = flat_index / rate_cats;
    const int rate_idx = flat_index - node_idx * rate_cats;
    if (node_idx >= num_nodes || rate_idx >= rate_cats) return;
    if (!total_lengths || !proximal_lengths) return;

    fp_t branch_length = total_lengths[node_idx] - proximal_lengths[node_idx];
    if (node_idx == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;

    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_count = static_cast<size_t>(rate_cats);
    const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
    const size_t matrix_span = state_count * state_count;
    const size_t output_base =
        (static_cast<size_t>(node_idx) * rate_count + static_cast<size_t>(rate_idx)) * matrix_span;
    const fp_t* rate_lambdas = lambdas + rate_offset;
    fp_t* out_pmat = P + output_base;
    pmatrix_from_triple_device(Vinv, V, rate_lambdas, fp_t(1.0), branch_length, p, out_pmat, states);
}

// Refresh proximal PMATs only for the currently selected placement targets.
__global__ void BuildSelectedNodeProximalPMATKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    const fp_t* node_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    fp_t p,
    fp_t* P,
    int states,
    int rate_cats,
    int num_ops,
    int total_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = num_ops * rate_cats;
    if (flat_index >= total_entries) return;

    const int op_local = flat_index / rate_cats;
    const int rate_idx = flat_index - op_local * rate_cats;
    if (!ops || op_local >= num_ops || rate_idx >= rate_cats) return;
    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= total_nodes) return;

    fp_t branch_length = default_len;
    if (node_lengths) {
        branch_length = node_lengths[target_id];
    }
    if (target_id == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;

    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_count = static_cast<size_t>(rate_cats);
    const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
    const size_t matrix_span = state_count * state_count;
    const size_t output_base =
        (static_cast<size_t>(target_id) * rate_count + static_cast<size_t>(rate_idx)) * matrix_span;
    const fp_t* rate_lambdas = lambdas + rate_offset;
    fp_t* out_pmat = P + output_base;
    pmatrix_from_triple_device(Vinv, V, rate_lambdas, fp_t(1.0), branch_length, p, out_pmat, states);
}

// Refresh distal PMATs only for the currently selected placement targets.
__global__ void BuildSelectedNodeDistalPMATKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    fp_t p,
    fp_t* P,
    int states,
    int rate_cats,
    int num_ops,
    int total_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = num_ops * rate_cats;
    if (flat_index >= total_entries) return;

    const int op_local = flat_index / rate_cats;
    const int rate_idx = flat_index - op_local * rate_cats;
    if (!ops || op_local >= num_ops || rate_idx >= rate_cats) return;
    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= total_nodes) return;
    if (!total_lengths || !proximal_lengths) return;

    fp_t branch_length = total_lengths[target_id] - proximal_lengths[target_id];
    if (target_id == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;

    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_count = static_cast<size_t>(rate_cats);
    const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
    const size_t matrix_span = state_count * state_count;
    const size_t output_base =
        (static_cast<size_t>(target_id) * rate_count + static_cast<size_t>(rate_idx)) * matrix_span;
    const fp_t* rate_lambdas = lambdas + rate_offset;
    fp_t* out_pmat = P + output_base;
    pmatrix_from_triple_device(Vinv, V, rate_lambdas, fp_t(1.0), branch_length, p, out_pmat, states);
}

// Per-site placement kernel: build midpoint CLV for placement.
__global__ void BuildMidpointForPlacementKernel(
    DeviceTree D,
    const NodeOpInfo* d_ops,
    const int* d_op_indices,
    int op_offset,
    int num_ops,
    bool proximal_mode)
{
    const int op_local = op_offset + (int)blockIdx.y;
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (!d_ops || op_local >= num_ops) return;
    const int op_idx = d_op_indices ? d_op_indices[op_local] : op_local;
    if (op_idx < 0) return;
    const NodeOpInfo op = d_ops[op_idx];
    const bool active_thread = (tid < D.sites);
    __shared__ fp_t shared_target_mat[8 * 16];
    __shared__ fp_t shared_parent_mat[8 * 16];
    if (D.states == 4) {
        switch (D.rate_cats) {
            case 1:
                compute_midpoint_inner_inner_ratecat<1>(
                    D,
                    op,
                    tid,
                    proximal_mode,
                    op_idx,
                    active_thread,
                    shared_target_mat,
                    shared_parent_mat);
                break;
            case 4:
                compute_midpoint_inner_inner_ratecat<4>(
                    D,
                    op,
                    tid,
                    proximal_mode,
                    op_idx,
                    active_thread,
                    shared_target_mat,
                    shared_parent_mat);
                break;
            case 8:
                compute_midpoint_inner_inner_ratecat<8>(
                    D,
                    op,
                    tid,
                    proximal_mode,
                    op_idx,
                    active_thread,
                    shared_target_mat,
                    shared_parent_mat);
                break;
            default:
                // Generic version not implemented for midpoint helper.
                break;
        }
    }
}

// Per-site root likelihood kernel: assumes midpoint CLV already computed.
PlacementResult PlacementEvaluationKernel (
    const DeviceTree& D,
    const NodeOpInfo* d_ops,
    int num_ops,
    int smoothing,
    cudaStream_t stream
){
    PlacementResult result;
    assert(num_ops > 0 && "num_ops must be positive");
    if (num_ops <= 0) return result;
    assert(smoothing > 0 && "smoothing must be positive");

    // Stage 1: validate inputs and plan the runtime/launch configuration.
    const size_t sumtable_stride = (size_t)D.sites * (size_t)D.rate_cats * (size_t)D.states;
    if ((size_t)num_ops > D.sumtable_capacity_ops || (size_t)num_ops > D.likelihood_capacity_ops) {
        throw std::runtime_error("DeviceTree buffers too small for num_ops.");
    }
    auto check_launch = [&](const char* stage) {
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(err));
        }
    };
    auto check_cuda = [&](const char* stage, cudaError_t err) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(err));
        }
    };
    PlacementKernelScratchBuffers scratch;
    fp_t*& d_prev_loglk = scratch.d_prev_loglk;
    fp_t*& d_last_loglk = scratch.d_last_loglk;
    int*& d_active_ops = scratch.d_active_ops;
    int*& d_refine_op_indices = scratch.d_refine_op_indices;
    int*& d_refine_active_ops = scratch.d_refine_active_ops;
    int*& d_any_active_flag = scratch.d_any_active_flag;
    void*& d_any_active_temp = scratch.d_any_active_temp;
    fp_t*& d_last_pendant_length = scratch.d_last_pendant_length;
    fp_t*& d_last_proximal_length = scratch.d_last_proximal_length;
    size_t& any_active_temp_bytes = scratch.any_active_temp_bytes;
    std::vector<fp_t>& host_loglk_cache = scratch.host_loglk_cache;
    std::vector<int>& host_order_cache = scratch.host_order_cache;
    fp_t* d_likelihoods = D.d_likelihoods;
    fp_t* d_sumtable = D.d_sumtable;

    const size_t diag_shared = (size_t)D.rate_cats * (size_t)D.states * 4;
    const RefineConfig refine_cfg = load_refine_config();
    const PlacementKernelDebugConfig debug_cfg =
        load_placement_kernel_debug_config();
    const bool use_selective_refine =
        refine_cfg.refine_extra_passes > 0 &&
        refine_cfg.detect_topk_limit > 0 &&
        refine_cfg.refine_topk_limit > 0 &&
        refine_cfg.global_opt_passes > 0;
    const size_t midpoint_pmat_shared = (size_t)D.rate_cats * 16 * 2;
    size_t shmem_bytes = sizeof(fp_t) * diag_shared;
    shmem_bytes += sizeof(fp_t) * midpoint_pmat_shared;

    // Pendant and proximal derivative kernels have different register pressure.
    // Size them independently so one kernel does not inherit an invalid launch shape.
    int pendant_block_threads = 512;
    int max_blocks_per_sm = 0;
    cudaFuncAttributes attr{};
    check_cuda("cudaFuncGetAttributes pendant", cudaFuncGetAttributes(&attr, LikelihoodDerivativePendantKernel));
    while (pendant_block_threads >= 32) {
        check_cuda("cudaOccupancyMaxActiveBlocksPerMultiprocessor pendant", cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            LikelihoodDerivativePendantKernel,
            pendant_block_threads,
            shmem_bytes));
        if (max_blocks_per_sm > 0) break;
        pendant_block_threads /= 2;
    }
    if (max_blocks_per_sm == 0) {
        throw std::runtime_error("No valid block size for LikelihoodDerivativePendantKernel on this GPU.");
    }

    int proximal_block_threads = 512;
    max_blocks_per_sm = 0;
    check_cuda("cudaFuncGetAttributes proximal", cudaFuncGetAttributes(&attr, LikelihoodDerivativeProximalKernel));
    while (proximal_block_threads >= 32) {
        check_cuda("cudaOccupancyMaxActiveBlocksPerMultiprocessor proximal", cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            LikelihoodDerivativeProximalKernel,
            proximal_block_threads,
            shmem_bytes));
        if (max_blocks_per_sm > 0) break;
        proximal_block_threads /= 2;
    }
    if (max_blocks_per_sm == 0) {
        throw std::runtime_error("No valid block size for LikelihoodDerivativeProximalKernel on this GPU.");
    }

    const int midpoint_block_threads = 256;
    const int pmat_block_threads = 128;
    dim3 pendant_block(pendant_block_threads);
    dim3 proximal_block(proximal_block_threads);
    dim3 midpoint_block(midpoint_block_threads);
    dim3 pmat_block(pmat_block_threads);
    dim3 midpoint_grid((D.sites + midpoint_block.x - 1) / midpoint_block.x, (unsigned)num_ops);

    // Stage 2: allocate scratch buffers and initialize branch-length state.
    check_cuda("cudaMalloc d_prev_loglk", cudaMalloc(&d_prev_loglk, sizeof(fp_t) * (size_t)num_ops));
    check_cuda("cudaMalloc d_last_loglk", cudaMalloc(&d_last_loglk, sizeof(fp_t) * (size_t)num_ops));
    check_cuda("cudaMalloc d_active_ops", cudaMalloc(&d_active_ops, sizeof(int) * (size_t)num_ops));
    check_cuda("cudaMemset d_active_ops", cudaMemset(d_active_ops, 1, sizeof(int) * (size_t)num_ops));
    maybe_dump_midbase_snapshot("placement-entry", D, stream);
    maybe_dump_scaler_scan("placement-entry", D, d_ops, num_ops, stream);
    {
        const size_t total_nodes = (size_t)D.N;
        dim3 init_block(256);
        dim3 init_grid((unsigned)((total_nodes + init_block.x - 1) / init_block.x));
        BuildNodePendantLengthsKernel<<<init_grid, init_block, 0, stream>>>(
            nullptr,
            D.d_prev_pendant_length,
            D.N,
            D.root_id,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        check_launch("BuildNodePendantLengthsKernel");
        BuildInitialProximalLengthsKernel<<<init_grid, init_block, 0, stream>>>(
            D.d_blen,
            D.d_prev_proximal_length,
            D.N,
            D.root_id,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        check_launch("BuildInitialProximalLengthsKernel");
    }

    check_cuda("cudaMalloc d_last_pendant_length", cudaMalloc(&d_last_pendant_length, sizeof(fp_t) * (size_t)D.N));
    check_cuda("cudaMalloc d_last_proximal_length", cudaMalloc(&d_last_proximal_length, sizeof(fp_t) * (size_t)D.N));
    if (use_selective_refine || refine_cfg.enable_convergence_check) {
        check_cuda("cudaMalloc d_any_active_flag", cudaMalloc(&d_any_active_flag, sizeof(int)));
        check_cuda("cub::DeviceReduce::Max size query", cub::DeviceReduce::Max(
            d_any_active_temp,
            any_active_temp_bytes,
            d_active_ops,
            d_any_active_flag,
            num_ops,
            stream));
        check_cuda("cudaMalloc d_any_active_temp", cudaMalloc(&d_any_active_temp, any_active_temp_bytes));
    }

    // Stage 3: define host-side helpers used by the optimization loop.
    auto fetch_topk_loglk =
        [&](const fp_t* d_source, int topk, std::vector<int>& top_indices, std::vector<fp_t>& top_values) {
            if (topk <= 0) return;
            host_loglk_cache.resize((size_t)num_ops);
            check_cuda("cudaMemcpyAsync host_loglk_cache", cudaMemcpyAsync(
                host_loglk_cache.data(),
                d_source,
                sizeof(fp_t) * (size_t)num_ops,
                cudaMemcpyDeviceToHost,
                stream));
            check_cuda("cudaStreamSynchronize host_loglk_cache", cudaStreamSynchronize(stream));
            const int actual_topk = std::min(num_ops, topk);
            host_order_cache.resize((size_t)num_ops);
            std::iota(host_order_cache.begin(), host_order_cache.end(), 0);
            std::partial_sort(
                host_order_cache.begin(),
                host_order_cache.begin() + actual_topk,
                host_order_cache.end(),
                [&](int lhs, int rhs) {
                    return host_loglk_cache[(size_t)lhs] > host_loglk_cache[(size_t)rhs];
                });
            top_indices.resize((size_t)actual_topk);
            top_values.resize((size_t)actual_topk);
            for (int i = 0; i < actual_topk; ++i) {
                const int op_idx = host_order_cache[(size_t)i];
                top_indices[(size_t)i] = op_idx;
                top_values[(size_t)i] = host_loglk_cache[(size_t)op_idx];
            }
        };
    auto any_active_on_device =
        [&](const int* d_active_mask, int count) {
            if (!d_active_mask || count <= 0) return false;
            int h_any_active = 0;
            check_cuda("cub::DeviceReduce::Max active mask", cub::DeviceReduce::Max(
                d_any_active_temp,
                any_active_temp_bytes,
                d_active_mask,
                d_any_active_flag,
                count,
                stream));
            check_cuda("cudaMemcpyAsync h_any_active", cudaMemcpyAsync(
                &h_any_active,
                d_any_active_flag,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream));
            check_cuda("cudaStreamSynchronize h_any_active", cudaStreamSynchronize(stream));
            return h_any_active != 0;
        };

    // Stage 4: build the baseline placement state and score every op once.
    {
        dim3 pmat_grid((unsigned)((num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
        BuildPendantPMATPerOpKernel<<<pmat_grid, pmat_block, 0, stream>>>(
            d_ops,
            nullptr,
            D.d_prev_pendant_length,
            D.d_Vinv,
            D.d_V,
            D.d_lambdas,
            0.0,
            D.d_query_pmat,
            D.states,
            D.rate_cats,
            num_ops,
            D.N,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        check_launch("BuildPendantPMATPerOpKernel baseline");

        dim3 node_grid((unsigned)((D.N * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
        BuildNodeProximalPMATKernel<<<node_grid, pmat_block, 0, stream>>>(
            D.d_prev_proximal_length,
            D.d_Vinv,
            D.d_V,
            D.d_lambdas,
            0.0,
            D.d_pmat_mid_prox,
            D.states,
            D.rate_cats,
            D.N,
            D.root_id,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        check_launch("BuildNodeProximalPMATKernel baseline");

        BuildNodeDistalPMATKernel<<<node_grid, pmat_block, 0, stream>>>(
            D.d_blen,
            D.d_prev_proximal_length,
            D.d_Vinv,
            D.d_V,
            D.d_lambdas,
            0.0,
            D.d_pmat_mid_dist,
            D.states,
            D.rate_cats,
            D.N,
            D.root_id,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        check_launch("BuildNodeDistalPMATKernel baseline");

        BuildMidpointForPlacementKernel<<<midpoint_grid, midpoint_block, 0, stream>>>(
            D,
            d_ops,
            nullptr,
            0,
            num_ops,
            false);
        check_launch("BuildMidpointForPlacementKernel baseline");
        maybe_dump_midbase_snapshot("after-build-midpoint-baseline", D, stream);

        root_likelihood::compute_combined_loglik_per_op_device(
            D,
            d_ops,
            nullptr,
            num_ops,
            D.d_query_pmat,
            D.d_pmat_mid_dist,
            D.d_pmat_mid_prox,
            d_prev_loglk,
            stream);
        maybe_dump_midbase_snapshot("after-rootlik-baseline", D, stream);
    }

    check_cuda("cudaMemcpyAsync d_last_loglk", cudaMemcpyAsync(
        d_last_loglk,
        d_prev_loglk,
        sizeof(fp_t) * (size_t)num_ops,
        cudaMemcpyDeviceToDevice,
        stream));
    check_cuda("cudaMemcpyAsync d_last_pendant_length", cudaMemcpyAsync(
        d_last_pendant_length,
        D.d_prev_pendant_length,
        sizeof(fp_t) * (size_t)D.N,
        cudaMemcpyDeviceToDevice,
        stream));
    check_cuda("cudaMemcpyAsync d_last_proximal_length", cudaMemcpyAsync(
        d_last_proximal_length,
        D.d_prev_proximal_length,
        sizeof(fp_t) * (size_t)D.N,
        cudaMemcpyDeviceToDevice,
        stream));

    // Stage 5: iteratively optimize pendant/proximal branch lengths per op.
    const int opt_passes = use_selective_refine
        ? std::max(refine_cfg.global_opt_passes + refine_cfg.refine_extra_passes, smoothing)
        : std::max(refine_cfg.full_opt_passes, smoothing);
    bool restrict_to_refine_topk = false;
    int best_op_after_global_pass1 = -1;
    int refine_op_count = 0;
    for (int pass = 0; pass < opt_passes; ++pass) {
        if (use_selective_refine && pass >= refine_cfg.global_opt_passes && !restrict_to_refine_topk) {
            break;
        }
        const bool use_compact_refine =
            use_selective_refine && restrict_to_refine_topk && refine_op_count > 0;
        const int current_num_ops = use_compact_refine ? refine_op_count : num_ops;
        const int* current_op_indices = use_compact_refine ? d_refine_op_indices : nullptr;
        int* current_active_ops = use_compact_refine ? d_refine_active_ops : d_active_ops;
        dim3 current_deriv_grid((unsigned)current_num_ops);
        dim3 current_pmat_grid((unsigned)((current_num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
        if (use_selective_refine &&
            restrict_to_refine_topk &&
            pass >= refine_cfg.global_opt_passes) {
            check_cuda("cudaMemcpyAsync d_last_loglk", cudaMemcpyAsync(
                d_last_loglk,
                d_prev_loglk,
                sizeof(fp_t) * (size_t)num_ops,
                cudaMemcpyDeviceToDevice,
                stream));
            check_cuda("cudaMemcpyAsync d_last_pendant_length", cudaMemcpyAsync(
                d_last_pendant_length,
                D.d_prev_pendant_length,
                sizeof(fp_t) * (size_t)D.N,
                cudaMemcpyDeviceToDevice,
                stream));
            check_cuda("cudaMemcpyAsync d_last_proximal_length", cudaMemcpyAsync(
                d_last_proximal_length,
                D.d_prev_proximal_length,
                sizeof(fp_t) * (size_t)D.N,
                cudaMemcpyDeviceToDevice,
                stream));
        }
        LikelihoodDerivativePendantKernel<<<current_deriv_grid, pendant_block, shmem_bytes, stream>>>(
            D,
            d_ops,
            0,
            current_op_indices,
            nullptr,
            nullptr,
            0.0,
            d_sumtable,
            D.d_pattern_weights_u,
            30,
            D.d_new_pendant_length,
            sumtable_stride,
            D.d_prev_pendant_length,
            current_active_ops,
            debug_cfg.newton,
            debug_cfg.newton_all_iters,
            debug_cfg.newton_limit,
            debug_cfg.newton_target,
            debug_cfg.sumtable,
            debug_cfg.sumtable_site,
            debug_cfg.sumtable_rate);
        check_launch("LikelihoodDerivativePendantKernel");
        if (pass == 0) {
            maybe_dump_midbase_snapshot("after-pendant-pass0", D, stream);
        }

        // Rebuild query-side PMATs from the updated pendant lengths.
        BuildPendantPMATPerOpKernel<<<current_pmat_grid, pmat_block, 0, stream>>>(
            d_ops,
            current_op_indices,
            D.d_new_pendant_length,
            D.d_Vinv,
            D.d_V,
            D.d_lambdas,
            0.0,
            D.d_query_pmat,
            D.states,
            D.rate_cats,
            current_num_ops,
            D.N,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        check_launch("BuildPendantPMATPerOpKernel refine");
        LikelihoodDerivativeProximalKernel<<<current_deriv_grid, proximal_block, shmem_bytes, stream>>>(
            D,
            d_ops,
            0,
            current_op_indices,
            nullptr,
            nullptr,
            0.0,
            d_sumtable,
            D.d_pattern_weights_u,
            30,
            D.d_new_proximal_length,
            sumtable_stride,
            D.d_prev_proximal_length,
            current_active_ops,
            debug_cfg.newton,
            debug_cfg.newton_all_iters,
            debug_cfg.newton_limit,
            debug_cfg.newton_target,
            debug_cfg.sumtable,
            debug_cfg.sumtable_site,
            debug_cfg.sumtable_rate);
        check_launch("LikelihoodDerivativeProximalKernel");

        // Rebuild midpoint PMATs from the updated proximal lengths.
        {
            if (use_compact_refine) {
                dim3 pmat_grid((unsigned)((current_num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
                BuildSelectedNodeProximalPMATKernel<<<pmat_grid, pmat_block, 0, stream>>>(
                    d_ops,
                    current_op_indices,
                    D.d_new_proximal_length,
                    D.d_Vinv,
                    D.d_V,
                    D.d_lambdas,
                    0.0,
                    D.d_pmat_mid_prox,
                    D.states,
                    D.rate_cats,
                    current_num_ops,
                    D.N,
                    D.root_id,
                    OPT_BRANCH_LEN_MIN,
                    OPT_BRANCH_LEN_MAX,
                    DEFAULT_BRANCH_LENGTH);
                check_launch("BuildSelectedNodeProximalPMATKernel refine");
            } else {
                dim3 pmat_grid((unsigned)((D.N * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
                BuildNodeProximalPMATKernel<<<pmat_grid, pmat_block, 0, stream>>>(
                    D.d_new_proximal_length,
                    D.d_Vinv,
                    D.d_V,
                    D.d_lambdas,
                    0.0,
                    D.d_pmat_mid_prox,
                    D.states,
                    D.rate_cats,
                    D.N,
                    D.root_id,
                    OPT_BRANCH_LEN_MIN,
                    OPT_BRANCH_LEN_MAX,
                    DEFAULT_BRANCH_LENGTH);
                check_launch("BuildNodeProximalPMATKernel refine");
            }
        }

        // Rebuild distal PMATs from total branch length minus proximal length.
        {
            if (use_compact_refine) {
                dim3 pmat_grid((unsigned)((current_num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
                BuildSelectedNodeDistalPMATKernel<<<pmat_grid, pmat_block, 0, stream>>>(
                    d_ops,
                    current_op_indices,
                    D.d_blen,
                    D.d_new_proximal_length,
                    D.d_Vinv,
                    D.d_V,
                    D.d_lambdas,
                    0.0,
                    D.d_pmat_mid_dist,
                    D.states,
                    D.rate_cats,
                    current_num_ops,
                    D.N,
                    D.root_id,
                    OPT_BRANCH_LEN_MIN,
                    OPT_BRANCH_LEN_MAX,
                    DEFAULT_BRANCH_LENGTH);
                check_launch("BuildSelectedNodeDistalPMATKernel refine");
            } else {
                dim3 pmat_grid((unsigned)((D.N * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
                BuildNodeDistalPMATKernel<<<pmat_grid, pmat_block, 0, stream>>>(
                    D.d_blen,
                    D.d_new_proximal_length,
                    D.d_Vinv,
                    D.d_V,
                    D.d_lambdas,
                    0.0,
                    D.d_pmat_mid_dist,
                    D.states,
                    D.rate_cats,
                    D.N,
                    D.root_id,
                    OPT_BRANCH_LEN_MIN,
                    OPT_BRANCH_LEN_MAX,
                    DEFAULT_BRANCH_LENGTH);
                check_launch("BuildNodeDistalPMATKernel refine");
            }
        }

        // Score each placement op after the pendant/proximal updates.
        root_likelihood::compute_combined_loglik_per_op_device(
            D,
            d_ops,
            current_op_indices,
            current_num_ops,
            D.d_query_pmat,
            D.d_pmat_mid_dist,
            D.d_pmat_mid_prox,
            d_likelihoods,
            stream);

        dim3 keep_block(256);
        dim3 keep_grid((unsigned)((current_num_ops + keep_block.x - 1) / keep_block.x));
        KeepBestBranchLengthsKernel<<<keep_grid, keep_block, 0, stream>>>(
            d_ops,
            current_op_indices,
            d_likelihoods,
            d_prev_loglk,
            D.d_new_pendant_length,
            D.d_new_proximal_length,
            D.d_prev_pendant_length,
            D.d_prev_proximal_length,
            current_active_ops,
            current_num_ops,
            D.N,
            -1);
        check_launch("KeepBestBranchLengthsKernel");

        if (use_selective_refine && pass + 1 == 1 && refine_cfg.global_opt_passes > 1) {
            std::vector<int> top_idx;
            std::vector<fp_t> top_vals;
            fetch_topk_loglk(d_prev_loglk, 1, top_idx, top_vals);
            best_op_after_global_pass1 = top_idx.empty() ? -1 : top_idx[0];
        }

        if (use_selective_refine && pass + 1 == refine_cfg.global_opt_passes) {
            if (refine_cfg.detect_topk_limit <= 0 || refine_cfg.refine_topk_limit <= 0) {
                break;
            }
            const int topk = std::min(num_ops, refine_cfg.detect_topk_limit);
            std::vector<int> order;
            std::vector<fp_t> host_topk_ll;
            fetch_topk_loglk(d_prev_loglk, topk, order, host_topk_ll);
            if (best_op_after_global_pass1 < 0 && !order.empty()) {
                best_op_after_global_pass1 = order[0];
            }

            const double best_ll = (topk > 0)
                ? static_cast<double>(host_topk_ll[0])
                : -std::numeric_limits<double>::infinity();
            const int best_op_after_global_pass2 = (topk > 0) ? order[0] : -1;
            const double gap12 = (topk > 1)
                ? (best_ll - static_cast<double>(host_topk_ll[1]))
                : std::numeric_limits<double>::infinity();
            const int top5 = std::min(topk, 5);
            const double gap15 = (top5 > 1)
                ? (best_ll - static_cast<double>(host_topk_ll[top5 - 1]))
                : std::numeric_limits<double>::infinity();

            const bool ambiguous =
                (gap12 < refine_cfg.gap_top2) ||
                (gap15 < refine_cfg.gap_top5) ||
                (best_op_after_global_pass1 != best_op_after_global_pass2);

            if (!ambiguous) {
                break;
            }

            const int refine_topk = std::min(num_ops, refine_cfg.refine_topk_limit);
            std::vector<int> refine_indices((size_t)refine_topk, 0);
            std::vector<int> refine_active((size_t)refine_topk, 1);
            for (int rank = 0; rank < refine_topk; ++rank) {
                refine_indices[(size_t)rank] = order[(size_t)rank];
            }
            scratch.ensure_refine_capacity(refine_topk, check_cuda);
            CUDA_CHECK(cudaMemcpy(
                d_refine_op_indices,
                refine_indices.data(),
                sizeof(int) * (size_t)refine_topk,
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_refine_active_ops,
                refine_active.data(),
                sizeof(int) * (size_t)refine_topk,
                cudaMemcpyHostToDevice));
            refine_op_count = refine_topk;
            restrict_to_refine_topk = true;
            continue;
        }

        if (refine_cfg.enable_convergence_check && (!use_selective_refine || restrict_to_refine_topk)) {
            dim3 conv_block(256);
            const int active_count = use_selective_refine ? refine_op_count : current_num_ops;
            dim3 conv_grid((unsigned)((active_count + conv_block.x - 1) / conv_block.x));
            current_active_ops = use_selective_refine ? d_refine_active_ops : d_active_ops;
            UpdateActiveOpsByConvergenceKernel<<<conv_grid, conv_block, 0, stream>>>(
                d_ops,
                current_op_indices,
                d_prev_loglk,
                d_last_loglk,
                D.d_prev_pendant_length,
                d_last_pendant_length,
                D.d_prev_proximal_length,
                d_last_proximal_length,
                current_active_ops,
                active_count,
                D.N,
                refine_cfg.converged_loglk_eps,
                refine_cfg.converged_length_eps);
            CUDA_CHECK(cudaGetLastError());

            if (!any_active_on_device(current_active_ops, active_count)) {
                break;
            }
        }

        if (pass + 1 < opt_passes) {
            check_cuda("cudaMemcpyAsync d_last_loglk", cudaMemcpyAsync(
                d_last_loglk,
                d_prev_loglk,
                sizeof(fp_t) * (size_t)num_ops,
                cudaMemcpyDeviceToDevice,
                stream));
            check_cuda("cudaMemcpyAsync d_last_pendant_length", cudaMemcpyAsync(
                d_last_pendant_length,
                D.d_prev_pendant_length,
                sizeof(fp_t) * (size_t)D.N,
                cudaMemcpyDeviceToDevice,
                stream));
            check_cuda("cudaMemcpyAsync d_last_proximal_length", cudaMemcpyAsync(
                d_last_proximal_length,
                D.d_prev_proximal_length,
                sizeof(fp_t) * (size_t)D.N,
                cudaMemcpyDeviceToDevice,
                stream));
        }
    }

    // Stage 6: collect the final top-k ranking and assemble the result.
    std::vector<int> final_top_indices;
    std::vector<fp_t> final_top_values;
    const int export_topk = export_placement_topk();
    fetch_topk_loglk(d_prev_loglk, export_topk, final_top_indices, final_top_values);
    if (local_child_refine_enabled() &&
        d_ops &&
        num_ops > 0 &&
        !final_top_indices.empty() &&
        !final_top_values.empty() &&
        host_loglk_cache.size() == static_cast<size_t>(num_ops)) {
        const std::vector<NodeOpInfo> host_ops =
            load_host_ops_for_local_child_refine(d_ops, num_ops, stream);
        if (!host_ops.empty()) {
            const int best_op_index = final_top_indices.front();
            if (best_op_index >= 0 && best_op_index < num_ops) {
                const int best_target_id =
                    target_id_from_op(host_ops[static_cast<size_t>(best_op_index)]);
                if (best_target_id >= 0 && best_target_id < D.N) {
                    const LocalChildRefineFamilyOps family =
                        find_local_child_refine_family_ops(host_ops, best_target_id);
                    if (family.child_left_op >= 0 || family.child_right_op >= 0) {
                        std::vector<int> augmented_indices = final_top_indices;
                        if (family.child_left_op >= 0) {
                            augmented_indices.push_back(family.child_left_op);
                        }
                        if (family.child_right_op >= 0) {
                            augmented_indices.push_back(family.child_right_op);
                        }
                        std::sort(
                            augmented_indices.begin(),
                            augmented_indices.end(),
                            [&](int lhs, int rhs) {
                                const fp_t lhs_ll = host_loglk_cache[static_cast<size_t>(lhs)];
                                const fp_t rhs_ll = host_loglk_cache[static_cast<size_t>(rhs)];
                                if (lhs_ll == rhs_ll) return lhs < rhs;
                                return lhs_ll > rhs_ll;
                            });
                        augmented_indices.erase(
                            std::unique(augmented_indices.begin(), augmented_indices.end()),
                            augmented_indices.end());

                        const int keep = std::min<int>(
                            std::max<int>(1, export_placement_topk()),
                            augmented_indices.size());
                        augmented_indices.resize(static_cast<size_t>(keep));

                        std::vector<fp_t> augmented_values(static_cast<size_t>(keep), fp_t(0));
                        for (int i = 0; i < keep; ++i) {
                            augmented_values[static_cast<size_t>(i)] =
                                host_loglk_cache[static_cast<size_t>(
                                    augmented_indices[static_cast<size_t>(i)])];
                        }

                        final_top_indices.swap(augmented_indices);
                        final_top_values.swap(augmented_values);
                    }
                }
            }
        }
    }
    if (final_top_indices.empty() || final_top_values.empty()) {
        throw std::runtime_error("PlacementEvaluationKernel: no placement candidates produced");
    }

    result.top_placements = build_top_ranked_placements(
        D,
        d_ops,
        final_top_indices,
        final_top_values);
    if (result.top_placements.empty()) {
        throw std::runtime_error("PlacementEvaluationKernel: failed to materialize ranked placements");
    }

    scratch.release();
    result.target_id = result.top_placements.front().target_id;
    result.loglikelihood = result.top_placements.front().loglikelihood;
    result.proximal_length = result.top_placements.front().proximal_length;
    result.pendant_length = result.top_placements.front().pendant_length;
    if (final_top_values.size() > 1) {
        result.gap_top2 =
            static_cast<double>(final_top_values[0]) - static_cast<double>(final_top_values[1]);
        const size_t top5_idx = std::min<size_t>(4, final_top_values.size() - 1);
        result.gap_top5 =
            static_cast<double>(final_top_values[0]) - static_cast<double>(final_top_values[top5_idx]);
    }

    // Stage 7: apply host-side postprocessing to the assembled ranking.
    maybe_apply_double_rerank(
        D,
        d_ops,
        final_top_indices,
        result,
        stream);
    rerank_selected_target_and_children(
        D,
        d_ops,
        num_ops,
        result,
        stream);
    return result;
}

PlacementResult PlacementEvaluationKernelPreorderPruned(
    const DeviceTree& D,
    const TreeBuildResult& T,
    const NodeOpInfo* d_ops,
    int num_ops,
    int smoothing,
    const PlacementPruneConfig& prune_cfg,
    cudaStream_t stream,
    int pseudo_root_id)
{
    if (!prune_cfg.enable_pruning || T.root_id < 0 || T.preorder.empty()) {
        return PlacementEvaluationKernel(
            D,
            d_ops,
            num_ops,
            smoothing,
            stream);
    }
    if (!d_ops || num_ops <= 0) {
        return PlacementResult{};
    }

    std::vector<NodeOpInfo> host_ops((size_t)num_ops);
    CUDA_CHECK(cudaMemcpyAsync(
        host_ops.data(),
        d_ops,
        sizeof(NodeOpInfo) * (size_t)num_ops,
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int num_parents = std::max(0, D.N);
    std::vector<int> parent_counts((size_t)num_parents, 0);
    for (int i = 0; i < num_ops; ++i) {
        const int pid = host_ops[(size_t)i].parent_id;
        if (pid >= 0 && pid < num_parents) {
            ++parent_counts[(size_t)pid];
        }
    }
    std::vector<int> parent_offsets((size_t)num_parents + 1, 0);
    for (int p = 0; p < num_parents; ++p) {
        parent_offsets[(size_t)p + 1] = parent_offsets[(size_t)p] + parent_counts[(size_t)p];
    }
    std::vector<int> parent_cursor = parent_offsets;
    std::vector<int> child_op_indices((size_t)num_ops, -1);
    for (int i = 0; i < num_ops; ++i) {
        const int pid = host_ops[(size_t)i].parent_id;
        if (pid >= 0 && pid < num_parents) {
            const int pos = parent_cursor[(size_t)pid]++;
            child_op_indices[(size_t)pos] = i;
        }
    }

    NodeOpInfo* d_batch_ops = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_ops, sizeof(NodeOpInfo) * (size_t)num_ops));

    auto target_id_of = [](const NodeOpInfo& op) -> int {
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const bool target_is_right = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
        return target_is_left ? op.left_id : (target_is_right ? op.right_id : op.parent_id);
    };

    struct SearchState {
        int op_idx = -1;
        double parent_ll = -std::numeric_limits<double>::infinity();
        int drop_streak = 0;
    };

    std::vector<SearchState> frontier;
    int start_node_id = T.root_id;
    if (pseudo_root_id >= 0 &&
        pseudo_root_id < (int)T.nodes.size() &&
        pseudo_root_id < D.N &&
        !T.nodes[(size_t)pseudo_root_id].is_tip) {
        start_node_id = pseudo_root_id;
    }
    if (start_node_id >= 0 && start_node_id < D.N) {
        const int start_begin = parent_offsets[(size_t)start_node_id];
        const int start_end = parent_offsets[(size_t)start_node_id + 1];
        frontier.reserve((size_t)std::max(0, start_end - start_begin));
        for (int pos = start_begin; pos < start_end; ++pos) {
            frontier.push_back(SearchState{
                child_op_indices[(size_t)pos],
                -std::numeric_limits<double>::infinity(),
                0});
        }
    }

    PlacementResult best{};
    best.loglikelihood = -std::numeric_limits<double>::infinity();
    const int max_consecutive_drops = std::max(1, prune_cfg.max_consecutive_drops);
    std::vector<SearchState> next_frontier;
    std::vector<SearchState> eval_states;
    std::vector<NodeOpInfo> batch_ops;
    std::vector<fp_t> batch_ll;
    eval_states.reserve((size_t)num_ops);
    batch_ops.reserve((size_t)num_ops);
    batch_ll.reserve((size_t)num_ops);
    next_frontier.reserve((size_t)num_ops);

    while (!frontier.empty()) {        
        eval_states.clear();
        batch_ops.clear();
        for (size_t i = 0; i < frontier.size(); ++i) {
            const SearchState s = frontier[i];
            const int op_idx = s.op_idx;
            if (op_idx < 0 || op_idx >= num_ops) continue;
            eval_states.push_back(s);
            batch_ops.push_back(host_ops[(size_t)op_idx]);
        }
        const int batch_n = (int)eval_states.size();
        if (batch_n <= 0) break;
        if (prune_cfg.enable_small_frontier_fallback &&
            prune_cfg.small_frontier_threshold > 0 &&
            batch_n < prune_cfg.small_frontier_threshold) {
            CUDA_CHECK(cudaFree(d_batch_ops));
            return PlacementEvaluationKernel(
                D,
                d_ops,
                num_ops,
                smoothing,
                stream);
        }

        CUDA_CHECK(cudaMemcpyAsync(
            d_batch_ops,
            batch_ops.data(),
            sizeof(NodeOpInfo) * (size_t)batch_n,
            cudaMemcpyHostToDevice,
            stream));

        PlacementEvaluationKernel(
            D,
            d_batch_ops,
            batch_n,
            smoothing,
            stream);

        batch_ll.resize((size_t)batch_n);
        CUDA_CHECK(cudaMemcpyAsync(
            batch_ll.data(),
            D.d_likelihoods,
            sizeof(fp_t) * (size_t)batch_n,
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        next_frontier.clear();
        next_frontier.reserve((size_t)batch_n * 2);

        for (int batch_i = 0; batch_i < batch_n; ++batch_i) {
            const SearchState cur_state = eval_states[(size_t)batch_i];
            const NodeOpInfo& op = host_ops[(size_t)cur_state.op_idx];
            const int target_id = target_id_of(op);
            const double cur_ll = static_cast<double>(batch_ll[(size_t)batch_i]);

            if (target_id >= 0 && target_id < D.N && cur_ll > best.loglikelihood) {
                best.target_id = target_id;
                best.loglikelihood = cur_ll;
            }

            int streak = 0;
            if (cur_state.parent_ll > -std::numeric_limits<double>::infinity() && cur_ll < cur_state.parent_ll) {
                streak = cur_state.drop_streak + 1;
            }
            const bool below_best_threshold = (cur_ll < (best.loglikelihood - prune_cfg.drop_threshold));
            const bool should_prune = (streak >= max_consecutive_drops) && below_best_threshold;
            if (should_prune) continue;

            if (target_id < 0 || target_id >= (int)T.nodes.size()) continue;
            if (T.nodes[(size_t)target_id].is_tip) continue;
            if (target_id < 0 || target_id >= D.N) continue;

            const int child_begin = parent_offsets[(size_t)target_id];
            const int child_end = parent_offsets[(size_t)target_id + 1];
            for (int pos = child_begin; pos < child_end; ++pos) {
                next_frontier.push_back(SearchState{child_op_indices[(size_t)pos], cur_ll, streak});
            }
        }
        frontier.swap(next_frontier);
    }

    CUDA_CHECK(cudaFree(d_batch_ops));
    if (best.loglikelihood == -std::numeric_limits<double>::infinity()) {
        return PlacementEvaluationKernel(
            D,
            d_ops,
            num_ops,
            smoothing,
            stream);
    }
    if (best.target_id >= 0 && best.target_id < D.N) {
        fp_t pendant_length = fp_t(0);
        fp_t proximal_length = fp_t(0);
        CUDA_CHECK(cudaMemcpyAsync(
            &pendant_length,
            D.d_prev_pendant_length + best.target_id,
            sizeof(fp_t),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            &proximal_length,
            D.d_prev_proximal_length + best.target_id,
            sizeof(fp_t),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        best.pendant_length = static_cast<double>(pendant_length);
        best.proximal_length = static_cast<double>(proximal_length);
    }
    return best;
}
