#include <vector>
#include <stdexcept>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <cub/cub.cuh>
#include "placement.cuh"
#include "util/mlipper_util.h"
#include "tree/tree.hpp"
#include "likelihood/partial_likelihood.cuh"
#include "pmatrix/pmat_gpu.cuh"
#include "likelihood/root_likelihood.cuh"
#include "derivative.cuh"

static __device__ __forceinline__ unsigned int threshold_scale_shift(fp_t max_val)
{
    const fp_t scale_threshold = fp_ldexp(fp_t(1), SCALE_THRESHOLD_EXPONENT);
    return (max_val < scale_threshold)
        ? static_cast<unsigned int>(-SCALE_THRESHOLD_EXPONENT)
        : 0u;
}

// Branch-diagonal helpers.
// Build exp(lambda*t) and its first/second derivatives for a branch into caller-provided buffer.
static __device__ __forceinline__ void build_diagtable_for_branch(
    const DeviceTree& D,
    fp_t branch_length,
    fp_t* diag_out)
{
    if (!diag_out || !D.d_lambdas) return;
    const unsigned int total = (unsigned int)(D.rate_cats * D.states);
    const unsigned int idx = threadIdx.x;
    if (idx >= total) return;
    const int rc = idx / D.states;
    const int st = idx - rc * D.states;
    const fp_t lambda = D.d_lambdas[(size_t)rc * D.states + st];
    const fp_t e = fp_exp(lambda * branch_length);
    const size_t base = (size_t)idx * 4;
    diag_out[base + 0] = e;
    diag_out[base + 1] = lambda * e;
    diag_out[base + 2] = lambda * lambda * e;
    diag_out[base + 3] = fp_t(0);
}

template<int RATE_CATS>
static __device__ __forceinline__ void build_diagtable_states4(
    const DeviceTree& D,
    fp_t branch_length,
    fp_t* diag_out)
{
    if (!diag_out || !D.d_lambdas) return;
    const unsigned int idx = threadIdx.x;
    if (idx >= (unsigned int)(RATE_CATS * 4)) return;
    const fp_t lambda = D.d_lambdas[idx];
    const fp_t e = fp_exp(lambda * branch_length);
    const size_t base = (size_t)idx * 4;
    diag_out[base + 0] = e;
    diag_out[base + 1] = lambda * e;
    diag_out[base + 2] = lambda * lambda * e;
    diag_out[base + 3] = fp_t(0);
}

static __device__ __forceinline__ bool build_diagtable_states4_dispatch(
    const DeviceTree& D,
    fp_t branch_length,
    fp_t* diag_out)
{
    switch (D.rate_cats) {
        case 1:
            build_diagtable_states4<1>(D, branch_length, diag_out);
            return true;
        case 4:
            build_diagtable_states4<4>(D, branch_length, diag_out);
            return true;
        case 8:
            build_diagtable_states4<8>(D, branch_length, diag_out);
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__
unsigned int scaler_shift_at_site(
    const DeviceTree& D,
    const unsigned* __restrict__ scaler_base,
    unsigned int site_idx,
    int rate_idx);

static __device__ __forceinline__
unsigned int scaler_shift_from_site_ptr(
    const DeviceTree& D,
    const unsigned* __restrict__ scaler_site_ptr,
    int rate_idx);

static __device__ __forceinline__ void maybe_print_newton_debug(
    int debug_enabled,
    int debug_all_iters,
    int debug_limit,
    int debug_target_id,
    const char* kernel_name,
    int op_global,
    int target_id,
    int iter,
    double branch_before,
    double branch_after,
    double block_df,
    double block_ddf,
    double branch_lower_bound,
    double branch_upper_bound,
    double max_step)
{
    if (!debug_enabled) return;
    if (debug_target_id >= 0 && target_id != debug_target_id) return;
    if (!debug_all_iters && iter != 0) return;
    if (debug_limit > 0 && iter >= debug_limit) return;
    printf(
        "[newton-debug] kernel=%s op=%d target=%d iter=%d branch_before=%.12f "
        "branch_after=%.12f df=%.12e ddf=%.12e lb=%.12f ub=%.12f max_step=%.12f\n",
        kernel_name ? kernel_name : "<null>",
        op_global,
        target_id,
        iter,
        branch_before,
        branch_after,
        block_df,
        block_ddf,
        branch_lower_bound,
        branch_upper_bound,
        max_step);
}

static __device__ __forceinline__ void maybe_print_sumtable_debug(
    int enabled,
    int debug_target_id,
    int debug_site,
    int debug_rate,
    const char* kernel_name,
    const char* left_source,
    const char* right_source,
    int target_id,
    unsigned int site_idx,
    int rate_idx,
    const fp4_t& qclv,
    const fp4_t& pbase,
    const fp4_t& pup,
    const fp4_t& pclv,
    const fp4_t& left_proj,
    const fp4_t& right_proj,
    const fp_t* target_mat,
    const fp_t* parent_mat,
    const fp_t* sumtable_row,
    unsigned int down_shift,
    unsigned int up_shift,
    unsigned int inherited_shift,
    unsigned int midpoint_shift)
{
    if (!enabled) return;
    if (debug_target_id >= 0 && target_id != debug_target_id) return;
    if (debug_site < 0 || static_cast<int>(site_idx) != debug_site) return;
    if (debug_rate >= 0 && rate_idx != debug_rate) return;
    printf(
        "[sumtable-proj-debug] kernel=%s left_source=%s right_source=%s target=%d site=%u rate=%d "
        "down_shift=%u up_shift=%u inherited_shift=%u midpoint_shift=%u "
        "Qclv=(%.12e,%.12e,%.12e,%.12e) "
        "Pbase=(%.12e,%.12e,%.12e,%.12e) "
        "Pup=(%.12e,%.12e,%.12e,%.12e) "
        "Pclv=(%.12e,%.12e,%.12e,%.12e) "
        "left_proj=(%.12e,%.12e,%.12e,%.12e) "
        "right_proj=(%.12e,%.12e,%.12e,%.12e) "
        "sum=(%.12e,%.12e,%.12e,%.12e)\n",
        kernel_name ? kernel_name : "<null>",
        left_source ? left_source : "<null>",
        right_source ? right_source : "<null>",
        target_id,
        site_idx,
        rate_idx,
        down_shift,
        up_shift,
        inherited_shift,
        midpoint_shift,
        static_cast<double>(qclv.x),
        static_cast<double>(qclv.y),
        static_cast<double>(qclv.z),
        static_cast<double>(qclv.w),
        static_cast<double>(pbase.x),
        static_cast<double>(pbase.y),
        static_cast<double>(pbase.z),
        static_cast<double>(pbase.w),
        static_cast<double>(pup.x),
        static_cast<double>(pup.y),
        static_cast<double>(pup.z),
        static_cast<double>(pup.w),
        static_cast<double>(pclv.x),
        static_cast<double>(pclv.y),
        static_cast<double>(pclv.z),
        static_cast<double>(pclv.w),
        static_cast<double>(left_proj.x),
        static_cast<double>(left_proj.y),
        static_cast<double>(left_proj.z),
        static_cast<double>(left_proj.w),
        static_cast<double>(right_proj.x),
        static_cast<double>(right_proj.y),
        static_cast<double>(right_proj.z),
        static_cast<double>(right_proj.w),
        static_cast<double>(sumtable_row[0]),
        static_cast<double>(sumtable_row[1]),
        static_cast<double>(sumtable_row[2]),
        static_cast<double>(sumtable_row[3]));
}

// Midpoint PMAT staging helpers.
template<int RATE_CATS>
__device__ __forceinline__ void load_midpoint_pmat_pair(
    fp_t* shared_target_mat,
    fp_t* shared_parent_mat,
    const fp_t* target_mat,
    const fp_t* parent_mat)
{
    const int total_mat_elems = RATE_CATS * 16;
    for (int idx = threadIdx.x; idx < total_mat_elems; idx += blockDim.x) {
        shared_target_mat[idx] = target_mat[idx];
        shared_parent_mat[idx] = parent_mat[idx];
    }
    __syncthreads();
}

static __device__ __forceinline__
bool load_midpoint_pmat_pair_dispatch(
    const DeviceTree& D,
    fp_t* shared_target_mat,
    fp_t* shared_parent_mat,
    const fp_t* target_mat,
    const fp_t* parent_mat)
{
    switch (D.rate_cats) {
        case 1:
            load_midpoint_pmat_pair<1>(
                shared_target_mat, shared_parent_mat, target_mat, parent_mat);
            return true;
        case 4:
            load_midpoint_pmat_pair<4>(
                shared_target_mat, shared_parent_mat, target_mat, parent_mat);
            return true;
        case 8:
            load_midpoint_pmat_pair<8>(
                shared_target_mat, shared_parent_mat, target_mat, parent_mat);
            return true;
        default:
            return false;
    }
}

// Small matrix helpers.
static __device__ __forceinline__
fp4_t matvec4_rows(const fp_t* mat_rows, const fp4_t& vec)
{
    const fp4_t* rows = reinterpret_cast<const fp4_t*>(mat_rows);
    return make_fp4(
        fp_dot4(rows[0], vec),
        fp_dot4(rows[1], vec),
        fp_dot4(rows[2], vec),
        fp_dot4(rows[3], vec));
}

static __device__ __forceinline__
fp4_t matvec4_cols(const fp_t* mat_cols, const fp4_t& vec)
{
    return make_fp4(
        fp_fma(mat_cols[0], vec.x, fp_fma(mat_cols[4], vec.y, fp_fma(mat_cols[8], vec.z, mat_cols[12] * vec.w))),
        fp_fma(mat_cols[1], vec.x, fp_fma(mat_cols[5], vec.y, fp_fma(mat_cols[9], vec.z, mat_cols[13] * vec.w))),
        fp_fma(mat_cols[2], vec.x, fp_fma(mat_cols[6], vec.y, fp_fma(mat_cols[10], vec.z, mat_cols[14] * vec.w))),
        fp_fma(mat_cols[3], vec.x, fp_fma(mat_cols[7], vec.y, fp_fma(mat_cols[11], vec.z, mat_cols[15] * vec.w))));
}

// Build one pendant-side midpoint vector for a single site and rate category.
template<int RATE_CATS>
__device__ __forceinline__ fp4_t build_pendant_midpoint_site_rate(
    const DeviceTree& D,
    int target_id,
    unsigned int site_idx,
    int rate_idx,
    fp_t* shared_target_mat,
    fp_t* shared_parent_mat,
    unsigned int* total_shift_out,
    unsigned int* down_shift_out,
    unsigned int* up_shift_out,
    fp4_t* pbase_out,
    fp4_t* pup_out)
{
    const size_t per_node = per_node_span(D);
    const size_t rate_count = static_cast<size_t>(RATE_CATS);
    const size_t site_span = rate_count * 4;
    const size_t node_base = static_cast<size_t>(target_id) * per_node;
    const size_t site_base = static_cast<size_t>(site_idx) * site_span;
    const size_t rate_offset = static_cast<size_t>(rate_idx);
    const size_t rate_base = rate_offset * 4;

    const fp_t* mid_base = D.d_clv_mid_base + node_base + site_base;
    const fp_t* target_up = D.d_clv_up + node_base + site_base;
    const unsigned* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site_idx);
    const unsigned* up_scaler = up_scaler_ptr(D, target_id, site_idx);

    const unsigned int down_shift = scaler_shift_from_site_ptr(D, mid_base_scaler, rate_idx);
    const unsigned int up_shift = scaler_shift_from_site_ptr(D, up_scaler, rate_idx);
    unsigned int inherited_shift = down_shift + up_shift;

    const size_t mat_base = rate_offset * 16;
    const fp_t* Mtarget = shared_target_mat + mat_base;
    const fp_t* Mparent = shared_parent_mat + mat_base;
    const fp4_t Pup = reinterpret_cast<const fp4_t*>(target_up + rate_base)[0];
    const fp4_t Pbase = reinterpret_cast<const fp4_t*>(mid_base + rate_base)[0];
    if (pbase_out) *pbase_out = Pbase;
    if (pup_out) *pup_out = Pup;
    if (down_shift_out) *down_shift_out = down_shift;
    if (up_shift_out) *up_shift_out = up_shift;

    fp4_t midpoint = matvec4_rows(Mparent, Pbase);
    const fp4_t target_proj = matvec4_rows(Mtarget, Pup);
    midpoint.x *= target_proj.x;
    midpoint.y *= target_proj.y;
    midpoint.z *= target_proj.z;
    midpoint.w *= target_proj.w;

    fp_t row_max = fp_hmax4(midpoint.x, midpoint.y, midpoint.z, midpoint.w);
    unsigned int total_shift = inherited_shift;
    {
        const unsigned int shift = threshold_scale_shift(row_max);
        if (shift) {
        total_shift += shift;
        midpoint.x = fp_ldexp(midpoint.x, shift);
        midpoint.y = fp_ldexp(midpoint.y, shift);
        midpoint.z = fp_ldexp(midpoint.z, shift);
        midpoint.w = fp_ldexp(midpoint.w, shift);
        }
    }

    if (total_shift_out) *total_shift_out = total_shift;
    return midpoint;
}

// Build proximal-side midpoint vectors for a single site across all rate categories.
template<int RATE_CATS>
__device__ __forceinline__ void build_proximal_midpoint_site(
    const DeviceTree& D,
    int target_id,
    unsigned int site_idx,
    fp_t* shared_target_mat,
    fp_t* shared_parent_mat,
    fp4_t* midpoint_rows,
    unsigned int* midpoint_shifts,
    fp4_t* mid_base_rows,
    fp4_t* query_rows,
    fp4_t* parent_proj_rows,
    fp4_t* target_proj_rows)
{
    const size_t per_node = per_node_span(D);
    const size_t rate_count = static_cast<size_t>(RATE_CATS);
    const size_t site_span = rate_count * 4;
    const size_t node_base = static_cast<size_t>(target_id) * per_node;
    const size_t site_base = static_cast<size_t>(site_idx) * site_span;
    const fp_t* mid_base = D.d_clv_mid_base + node_base + site_base;
    const fp_t* query_clv = D.d_query_clv + site_base;
    const unsigned* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site_idx);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        unsigned int inherited_shift = scaler_shift_from_site_ptr(D, mid_base_scaler, r);

        const size_t rate_offset = static_cast<size_t>(r);
        const size_t rate_base = rate_offset * 4;
        const size_t mat_base = rate_offset * 16;
        const fp_t* Mtarget = shared_target_mat + mat_base;
        const fp_t* Mparent = shared_parent_mat + mat_base;
        const fp4_t Pup = reinterpret_cast<const fp4_t*>(query_clv + rate_base)[0];
        const fp4_t Pbase = reinterpret_cast<const fp4_t*>(mid_base + rate_base)[0];
        const fp4_t parent_proj = matvec4_rows(Mparent, Pbase);
        const fp4_t target_proj = matvec4_rows(Mtarget, Pup);

        if (mid_base_rows) mid_base_rows[r] = Pbase;
        if (query_rows) query_rows[r] = Pup;
        if (parent_proj_rows) parent_proj_rows[r] = parent_proj;
        if (target_proj_rows) target_proj_rows[r] = target_proj;

        fp_t p0 = parent_proj.x * target_proj.x;
        fp_t p1 = parent_proj.y * target_proj.y;
        fp_t p2 = parent_proj.z * target_proj.z;
        fp_t p3 = parent_proj.w * target_proj.w;

        fp_t row_max = fp_hmax4(p0, p1, p2, p3);
        unsigned int total_shift = inherited_shift;
        {
            const unsigned int shift = threshold_scale_shift(row_max);
            if (shift) {
            total_shift += shift;
            p0 = fp_ldexp(p0, shift);
            p1 = fp_ldexp(p1, shift);
            p2 = fp_ldexp(p2, shift);
            p3 = fp_ldexp(p3, shift);
            }
        }

        midpoint_rows[r] = make_fp4(p0, p1, p2, p3);
        midpoint_shifts[r] = total_shift;
    }
}

// Build one site's derivative sumtable rows for the pendant-side branch update.
template<int RATE_CATS>
__device__ __forceinline__ void update_pendant_sumtable_site(
    DeviceTree D,
    int target_id,
    const fp_t* __restrict__ left_clv_base,
    unsigned int site_idx,
    fp_t* sumtable,
    fp_t* shared_target_mat,
    fp_t* shared_parent_mat,
    int debug_sumtable,
    int debug_target_id,
    int debug_sumtable_site,
    int debug_sumtable_rate)
{
    const size_t rate_count = static_cast<size_t>(D.rate_cats);
    const size_t state_count = static_cast<size_t>(D.states);
    const size_t site_span = rate_count * state_count;
    const size_t site_base = static_cast<size_t>(site_idx) * site_span;
    const fp_t* left_clv = left_clv_base + site_base;
    fp_t* sumtable_ptr = sumtable + site_base;

    unsigned int midpoint_shifts[RATE_CATS];
    unsigned int active_rate_mask = 0u;
    unsigned int site_min_shift = 0u;
    bool have_signal = false;

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const size_t rate_offset = static_cast<size_t>(r);
        const size_t rate_base = rate_offset * 4;
        const fp4_t Qclv = reinterpret_cast<const fp4_t*>(left_clv + rate_base)[0];
        fp4_t Pbase = make_fp4(fp_t(0), fp_t(0), fp_t(0), fp_t(0));
        fp4_t Pup = make_fp4(fp_t(0), fp_t(0), fp_t(0), fp_t(0));
        unsigned int down_shift = 0u;
        unsigned int up_shift = 0u;
        const fp4_t Pclv = build_pendant_midpoint_site_rate<RATE_CATS>(
            D,
            target_id,
            site_idx,
            r,
            shared_target_mat,
            shared_parent_mat,
            &midpoint_shifts[r],
            &down_shift,
            &up_shift,
            &Pbase,
            &Pup);
        fp_t* sumtable_row = sumtable_ptr + rate_base;
        const fp_t* Mtarget = shared_target_mat + static_cast<size_t>(r) * 16;
        const fp_t* Mparent = shared_parent_mat + static_cast<size_t>(r) * 16;

        const fp4_t piq = make_fp4(
            D.d_frequencies[0] * Qclv.x,
            D.d_frequencies[1] * Qclv.y,
            D.d_frequencies[2] * Qclv.z,
            D.d_frequencies[3] * Qclv.w);
        // MLIPPER stores PMAT factors in the standard order:
        //   P = V * diag(exp(lambda * t)) * Vinv
        // The sumtable projection must use the same basis convention:
        //   left[j]  = sum_k piq[k] * V[k,j]
        //   right[j] = sum_k Vinv[j,k] * clv[k]
        const fp4_t left_proj = matvec4_cols(D.d_V, piq);
        const fp4_t right_proj = matvec4_rows(D.d_Vinv, Pclv);

        sumtable_row[0] = left_proj.x * right_proj.x;
        sumtable_row[1] = left_proj.y * right_proj.y;
        sumtable_row[2] = left_proj.z * right_proj.z;
        sumtable_row[3] = left_proj.w * right_proj.w;
        maybe_print_sumtable_debug(
            debug_sumtable,
            debug_target_id,
            debug_sumtable_site,
            debug_sumtable_rate,
            "pendant",
            "query_clv",
            "midpoint(mid_base,target_up)",
            target_id,
            site_idx,
            r,
            Qclv,
            Pbase,
            Pup,
            Pclv,
            left_proj,
            right_proj,
            Mtarget,
            Mparent,
            sumtable_row,
            down_shift,
            up_shift,
            down_shift + up_shift,
            midpoint_shifts[r]);

        const fp_t sum_row_max = fp_fmax(
            fp_fmax(sumtable_row[0], sumtable_row[1]),
            fp_fmax(sumtable_row[2], sumtable_row[3]));
        if (sum_row_max > fp_t(0)) {
            if (!have_signal || midpoint_shifts[r] < site_min_shift) {
                site_min_shift = midpoint_shifts[r];
            }
            active_rate_mask |= (1u << r);
            have_signal = true;
        }
    }

    if (!have_signal) return;

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        if ((active_rate_mask & (1u << r)) == 0u) continue;
        const int diff = (int)midpoint_shifts[r] - (int)site_min_shift;
        if (diff <= 0) continue;
        const size_t rate_base = static_cast<size_t>(r) * 4;
        fp_t* sumtable_row = sumtable_ptr + rate_base;
        sumtable_row[0] = fp_ldexp(sumtable_row[0], -diff);
        sumtable_row[1] = fp_ldexp(sumtable_row[1], -diff);
        sumtable_row[2] = fp_ldexp(sumtable_row[2], -diff);
        sumtable_row[3] = fp_ldexp(sumtable_row[3], -diff);
    }

    if (debug_sumtable &&
        (debug_target_id < 0 || target_id == debug_target_id) &&
        static_cast<int>(site_idx) == debug_sumtable_site) {
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            if ((active_rate_mask & (1u << r)) == 0u) continue;
            if (debug_sumtable_rate >= 0 && r != debug_sumtable_rate) continue;
            const int diff = static_cast<int>(midpoint_shifts[r]) - static_cast<int>(site_min_shift);
            const size_t rate_base = static_cast<size_t>(r) * 4;
            const fp_t* sumtable_row = sumtable_ptr + rate_base;
            printf("[sumtable-common-debug] kernel=pendant left_source=query_clv right_source=midpoint(mid_base,target_up) target=%d site=%u rate=%d site_min_shift=%u rate_shift=%u diff=%d "
                   "sum_common=(%.12e,%.12e,%.12e,%.12e) "
                   "sum_restored=(%.12e,%.12e,%.12e,%.12e)\n",
                   target_id,
                   site_idx,
                   r,
                   site_min_shift,
                   midpoint_shifts[r],
                   diff,
                   static_cast<double>(sumtable_row[0]),
                   static_cast<double>(sumtable_row[1]),
                   static_cast<double>(sumtable_row[2]),
                   static_cast<double>(sumtable_row[3]),
                   static_cast<double>(fp_ldexp(sumtable_row[0], -static_cast<int>(site_min_shift))),
                   static_cast<double>(fp_ldexp(sumtable_row[1], -static_cast<int>(site_min_shift))),
                   static_cast<double>(fp_ldexp(sumtable_row[2], -static_cast<int>(site_min_shift))),
                   static_cast<double>(fp_ldexp(sumtable_row[3], -static_cast<int>(site_min_shift))));
        }
    }
}

// Build one site's derivative sumtable rows for the proximal-side branch update.
template<int RATE_CATS>
__device__ __forceinline__ void update_proximal_sumtable_site(
    DeviceTree D,
    int target_id,
    const fp_t* __restrict__ left_clv_base,
    const unsigned* __restrict__ left_scaler_base,
    unsigned int site_idx,
    fp_t* sumtable,
    fp_t* shared_target_mat,
    fp_t* shared_parent_mat,
    int debug_sumtable,
    int debug_target_id,
    int debug_sumtable_site,
    int debug_sumtable_rate)
{
    const size_t rate_count = static_cast<size_t>(D.rate_cats);
    const size_t state_count = static_cast<size_t>(D.states);
    const size_t site_span = rate_count * state_count;
    const size_t site_base = static_cast<size_t>(site_idx) * site_span;
    const fp_t* left_clv = left_clv_base + site_base;
    fp_t* sumtable_ptr = sumtable + site_base;
    fp4_t midpoint_rows[RATE_CATS];
    fp4_t mid_base_rows[RATE_CATS];
    fp4_t query_rows[RATE_CATS];
    fp4_t parent_proj_rows[RATE_CATS];
    fp4_t target_proj_rows[RATE_CATS];
    unsigned int midpoint_shifts[RATE_CATS];
    bool rate_has_signal[RATE_CATS];
    unsigned int site_min_shift = 0u;
    bool have_signal = false;

    build_proximal_midpoint_site<RATE_CATS>(
        D,
        target_id,
        site_idx,
        shared_target_mat,
        shared_parent_mat,
        midpoint_rows,
        midpoint_shifts,
        mid_base_rows,
        query_rows,
        parent_proj_rows,
        target_proj_rows);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const size_t rate_offset = static_cast<size_t>(r);
        const size_t rate_base = rate_offset * 4;
        const fp4_t Qclv = reinterpret_cast<const fp4_t*>(left_clv + rate_base)[0];
        const fp4_t Pclv = midpoint_rows[r];
        fp_t* sumtable_row = sumtable_ptr + rate_base;

        const fp4_t piq = make_fp4(
            D.d_frequencies[0] * Qclv.x,
            D.d_frequencies[1] * Qclv.y,
            D.d_frequencies[2] * Qclv.z,
            D.d_frequencies[3] * Qclv.w);
        // Keep the derivative projection aligned with MLIPPER's PMAT basis:
        //   P = V * diag(exp(lambda * t)) * Vinv
        // so the sumtable uses V on the left and Vinv on the right.
        const fp4_t left_proj = matvec4_cols(D.d_V, piq);
        const fp4_t right_proj = matvec4_rows(D.d_Vinv, Pclv);

        sumtable_row[0] = left_proj.x * right_proj.x;
        sumtable_row[1] = left_proj.y * right_proj.y;
        sumtable_row[2] = left_proj.z * right_proj.z;
        sumtable_row[3] = left_proj.w * right_proj.w;

        if (debug_sumtable &&
            (debug_target_id < 0 || target_id == debug_target_id) &&
            static_cast<int>(site_idx) == debug_sumtable_site &&
            (debug_sumtable_rate < 0 || r == debug_sumtable_rate)) {
            const unsigned int inherited_shift =
                scaler_shift_at_site(D, left_scaler_base, site_idx, r);
            const unsigned int mid_base_shift =
                scaler_shift_from_site_ptr(D, mid_base_scaler_ptr(D, target_id, site_idx), r);
            const fp4_t mid_base_vec = mid_base_rows[r];
            const fp4_t query_vec = query_rows[r];
            const fp4_t parent_proj = parent_proj_rows[r];
            const fp4_t target_proj = target_proj_rows[r];
            printf(
                "[midpoint-input-debug] kernel=proximal target=%d site=%u rate=%d "
                "mid_base_shift=%u midpoint_total_shift=%u "
                "mid_base=(%.12e,%.12e,%.12e,%.12e) "
                "parent_proj=(%.12e,%.12e,%.12e,%.12e) "
                "query_clv=(%.12e,%.12e,%.12e,%.12e) "
                "target_proj=(%.12e,%.12e,%.12e,%.12e) "
                "Pclv=(%.12e,%.12e,%.12e,%.12e)\n",
                target_id,
                site_idx,
                r,
                mid_base_shift,
                midpoint_shifts[r],
                static_cast<double>(mid_base_vec.x),
                static_cast<double>(mid_base_vec.y),
                static_cast<double>(mid_base_vec.z),
                static_cast<double>(mid_base_vec.w),
                static_cast<double>(parent_proj.x),
                static_cast<double>(parent_proj.y),
                static_cast<double>(parent_proj.z),
                static_cast<double>(parent_proj.w),
                static_cast<double>(query_vec.x),
                static_cast<double>(query_vec.y),
                static_cast<double>(query_vec.z),
                static_cast<double>(query_vec.w),
                static_cast<double>(target_proj.x),
                static_cast<double>(target_proj.y),
                static_cast<double>(target_proj.z),
                static_cast<double>(target_proj.w),
                static_cast<double>(Pclv.x),
                static_cast<double>(Pclv.y),
                static_cast<double>(Pclv.z),
                static_cast<double>(Pclv.w));
            printf(
                "[sumtable-proj-debug] kernel=proximal left_source=up_clv right_source=midpoint(mid_base,query_clv) target=%d site=%u rate=%d "
                "left_shift=%u midpoint_shift=%u total_shift=%u "
                "Qclv=(%.12e,%.12e,%.12e,%.12e) "
                "Pclv=(%.12e,%.12e,%.12e,%.12e) "
                "left_proj=(%.12e,%.12e,%.12e,%.12e) "
                "right_proj=(%.12e,%.12e,%.12e,%.12e) "
                "sum=(%.12e,%.12e,%.12e,%.12e)\n",
                target_id,
                site_idx,
                r,
                inherited_shift,
                midpoint_shifts[r],
                inherited_shift + midpoint_shifts[r],
                static_cast<double>(Qclv.x),
                static_cast<double>(Qclv.y),
                static_cast<double>(Qclv.z),
                static_cast<double>(Qclv.w),
                static_cast<double>(Pclv.x),
                static_cast<double>(Pclv.y),
                static_cast<double>(Pclv.z),
                static_cast<double>(Pclv.w),
                static_cast<double>(left_proj.x),
                static_cast<double>(left_proj.y),
                static_cast<double>(left_proj.z),
                static_cast<double>(left_proj.w),
                static_cast<double>(right_proj.x),
                static_cast<double>(right_proj.y),
                static_cast<double>(right_proj.z),
                static_cast<double>(right_proj.w),
                static_cast<double>(sumtable_row[0]),
                static_cast<double>(sumtable_row[1]),
                static_cast<double>(sumtable_row[2]),
                static_cast<double>(sumtable_row[3]));
        }

        const fp_t row_max = fp_fmax(
            fp_fmax(sumtable_row[0], sumtable_row[1]),
            fp_fmax(sumtable_row[2], sumtable_row[3]));
        const unsigned int total_shift =
            scaler_shift_at_site(D, left_scaler_base, site_idx, r) +
            midpoint_shifts[r];
        rate_has_signal[r] = (row_max > fp_t(0));
        if (rate_has_signal[r]) {
            if (!have_signal || total_shift < site_min_shift) {
                site_min_shift = total_shift;
            }
            midpoint_shifts[r] = total_shift;
            have_signal = true;
        }
    }

    if (!have_signal) return;

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        if (!rate_has_signal[r]) continue;
        const int diff = (int)midpoint_shifts[r] - (int)site_min_shift;
        if (diff <= 0) continue;
        const size_t rate_base = static_cast<size_t>(r) * 4;
        fp_t* sumtable_row = sumtable_ptr + rate_base;
        sumtable_row[0] = fp_ldexp(sumtable_row[0], -diff);
        sumtable_row[1] = fp_ldexp(sumtable_row[1], -diff);
        sumtable_row[2] = fp_ldexp(sumtable_row[2], -diff);
        sumtable_row[3] = fp_ldexp(sumtable_row[3], -diff);
    }

    if (debug_sumtable &&
        (debug_target_id < 0 || target_id == debug_target_id) &&
        static_cast<int>(site_idx) == debug_sumtable_site) {
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            if (!rate_has_signal[r]) continue;
            if (debug_sumtable_rate >= 0 && r != debug_sumtable_rate) continue;
            const int diff = static_cast<int>(midpoint_shifts[r]) - static_cast<int>(site_min_shift);
            const size_t rate_base = static_cast<size_t>(r) * 4;
            const fp_t* sumtable_row = sumtable_ptr + rate_base;
            printf("[sumtable-common-debug] kernel=proximal left_source=up_clv right_source=midpoint(mid_base,query_clv) target=%d site=%u rate=%d site_min_shift=%u rate_shift=%u diff=%d "
                   "sum_common=(%.12e,%.12e,%.12e,%.12e) "
                   "sum_restored=(%.12e,%.12e,%.12e,%.12e)\n",
                   target_id,
                   site_idx,
                   r,
                   site_min_shift,
                   midpoint_shifts[r],
                   diff,
                   static_cast<double>(sumtable_row[0]),
                   static_cast<double>(sumtable_row[1]),
                   static_cast<double>(sumtable_row[2]),
                   static_cast<double>(sumtable_row[3]),
                   static_cast<double>(fp_ldexp(sumtable_row[0], -static_cast<int>(site_min_shift))),
                   static_cast<double>(fp_ldexp(sumtable_row[1], -static_cast<int>(site_min_shift))),
                   static_cast<double>(fp_ldexp(sumtable_row[2], -static_cast<int>(site_min_shift))),
                   static_cast<double>(fp_ldexp(sumtable_row[3], -static_cast<int>(site_min_shift))));
        }
    }
}

// Reduction and Newton-update helpers.
static __device__ __forceinline__
void reduce_block_derivatives(
    double& local_df,
    double& local_ddf,
    double* warp_df_sums,
    double* warp_ddf_sums,
    double& block_df,
    double& block_ddf)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_count = (blockDim.x + 31) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_df += __shfl_down_sync(0xffffffff, local_df, offset);
        local_ddf += __shfl_down_sync(0xffffffff, local_ddf, offset);
    }
    if (lane == 0) {
        warp_df_sums[warp] = local_df;
        warp_ddf_sums[warp] = local_ddf;
    }
    __syncthreads();
    if (warp == 0) {
        double warp_df_total = (lane < warp_count) ? warp_df_sums[lane] : 0.0;
        double warp_ddf_total = (lane < warp_count) ? warp_ddf_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_df_total += __shfl_down_sync(0xffffffff, warp_df_total, offset);
            warp_ddf_total += __shfl_down_sync(0xffffffff, warp_ddf_total, offset);
        }
        if (lane == 0) {
            block_df = warp_df_total;
            block_ddf = warp_ddf_total;
        }
    }
    __syncthreads();
}

static __device__ __forceinline__
void apply_newton_update(
    double& branch_value,
    int& stop_iterations,
    double& branch_lower_bound,
    double& branch_upper_bound,
    double max_step,
    double block_df,
    double block_ddf)
{
    const double tolerance = OPT_BRANCH_XTOL;

    double branch_delta = 0.0;
    double proposed_branch = branch_value;
    if (block_ddf > 0.0) {
        // Match EPA-ng's bracketed Newton step:
        // update the active bound based on the sign of the first derivative,
        // then use the raw Newton proposal for the step.
        if (fabs(block_df) < tolerance) {
            stop_iterations = 1;
            return;
        }
        if (block_df < 0.0) {
            branch_lower_bound = branch_value;
        } else {
            branch_upper_bound = branch_value;
        }
        const double newton_delta = -block_df / block_ddf;
        const double newton_branch = branch_value + newton_delta;
        const bool outside_bracket =
            (((branch_value - branch_upper_bound) * block_ddf - block_df) *
             ((branch_value - branch_lower_bound) * block_ddf - block_df)) >= 0.0;

        if (outside_bracket) {
            // Fall back to a conservative bracket midpoint when the Newton
            // proposal is not safely bracketed. This mirrors the "safe"
            // behavior of EPA-ng's branch-length optimizer and prevents the
            // step from repeatedly chasing the global cap.
            proposed_branch = branch_lower_bound + 0.5 * (branch_upper_bound - branch_lower_bound);
            branch_delta = proposed_branch - branch_value;
        } else {
            branch_delta = newton_delta;
            proposed_branch = newton_branch;
        }
    } else {
        // Match EPA-ng/libpll's fallback for negative curvature: preserve the
        // ascent direction from the first derivative, but damp the step using
        // the magnitude of the second derivative.
        if (fabs(block_ddf) < tolerance) {
            stop_iterations = 1;
            return;
        }
        branch_delta = -block_df / fabs(block_ddf);
        proposed_branch = branch_value + branch_delta;
    }

    double clipped_delta = max(min(branch_delta, max_step), -max_step);
    if (branch_value + clipped_delta < branch_lower_bound) {
        clipped_delta = branch_lower_bound - branch_value;
    }
    if (branch_value + clipped_delta > branch_upper_bound) {
        clipped_delta = branch_upper_bound - branch_value;
    }

    if (fabs(clipped_delta) < tolerance) {
        stop_iterations = 1;
        return;
    }

    double final_branch = branch_value + clipped_delta;
    const bool hit_lower_cap = (final_branch <= OPT_BRANCH_LEN_MIN);
    const bool hit_upper_cap = (final_branch >= OPT_BRANCH_LEN_MAX);
    if (hit_lower_cap) final_branch = OPT_BRANCH_LEN_MIN;
    if (hit_upper_cap) final_branch = OPT_BRANCH_LEN_MAX;
    branch_value = final_branch;

    // EPA-ng-style behavior: once the Newton update is forced against a hard
    // bound, stop iterating instead of repeatedly chasing the cap. This avoids
    // turning the bound itself into the apparent optimum.
    if (hit_lower_cap || hit_upper_cap) {
        stop_iterations = 1;
    }
}

static __device__ __forceinline__
unsigned int scaler_shift_at_site(
    const DeviceTree& D,
    const unsigned* __restrict__ scaler_base,
    unsigned int site_idx,
    int rate_idx)
{
    if (!scaler_base) return 0u;
    const size_t site_offset = static_cast<size_t>(site_idx);
    if (D.per_rate_scaling) {
        const size_t rate_count = static_cast<size_t>(D.rate_cats);
        const size_t site_base = site_offset * rate_count;
        return scaler_base[site_base + static_cast<size_t>(rate_idx)];
    }
    return scaler_base[site_offset];
}

static __device__ __forceinline__
unsigned int scaler_shift_from_site_ptr(
    const DeviceTree& D,
    const unsigned* __restrict__ scaler_site_ptr,
    int rate_idx)
{
    if (!scaler_site_ptr) return 0u;
    if (D.per_rate_scaling) {
        return scaler_site_ptr[static_cast<size_t>(rate_idx)];
    }
    return scaler_site_ptr[0];
}

static __device__ __forceinline__
fp_t load_pattern_weight_cached(
    const unsigned* __restrict__ pattern_weights,
    size_t site)
{
    return pattern_weights
        ? static_cast<fp_t>(pattern_weights[site])
        : fp_t(1);
}

template<int RATE_CATS>
static __device__ __forceinline__
void rebuild_distal_midpoint_pmat_shared(
    const DeviceTree& D,
    int target_id,
    double total_branch,
    double proximal_branch,
    fp_t* shared_parent_mat)
{
    double proximal_len = 0.0;
    double distal_len = 0.0;
    normalize_split_branch_lengths(
        total_branch,
        proximal_branch,
        static_cast<double>(OPT_BRANCH_LEN_MIN),
        proximal_len,
        distal_len);

    const fp_t branch_length = static_cast<fp_t>(distal_len);
    const unsigned int rate_idx = threadIdx.x;
    if (rate_idx < static_cast<unsigned int>(RATE_CATS)) {
        const size_t state_count = 4;
        const size_t matrix_span = state_count * state_count;
        const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
        const fp_t* rate_lambdas = D.d_lambdas + rate_offset;
        fp_t* out_pmat = shared_parent_mat + static_cast<size_t>(rate_idx) * matrix_span;
        pmatrix_from_triple_device(
            D.d_Vinv,
            D.d_V,
            rate_lambdas,
            fp_t(1.0),
            branch_length,
            fp_t(0.0),
            out_pmat,
            4);
    }
    __syncthreads();
}

// Per-site derivative evaluation helpers.
template<bool WRITE_PLACEMENT_CLV>
static __device__ __forceinline__
void evaluate_site_derivatives_general(
    const DeviceTree D,
    const int*    __restrict__ invariant_site,
    const fp_t* __restrict__ invar_proportion, 
    fp_t invar_scalar,
    const fp_t* __restrict__ sumtable,
    const fp_t* __restrict__ lambdas,  
    const unsigned* __restrict__ pattern_weights,
    size_t site,
    fp_t* placement_clv, 
    double& d1_out,
    double& d2_out)
{
    fp_t site_lk0 = fp_t(0);
    fp_t site_lk1 = fp_t(0);
    fp_t site_lk2 = fp_t(0);

    const int inv = invariant_site ? invariant_site[site] : -1;
    const int states = D.states;
    const int rate_cats = D.rate_cats;
    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_count = static_cast<size_t>(rate_cats);
    const size_t site_span = rate_count * state_count;
    const size_t site_base = site * site_span;
    const fp_t* site_sumtable = sumtable + site_base;
    const fp_t* rate_weights = D.d_rate_weights;
    const fp_t* frequencies = D.d_frequencies;

    for (int i = 0; i < rate_cats; ++i) {
        const size_t rate_base = static_cast<size_t>(i) * state_count;
        const fp_t* sum_row = site_sumtable + rate_base;
        const fp_t* diag_row = lambdas + rate_base * 4;

        fp_t cat0 = fp_t(0);
        fp_t cat1 = fp_t(0);
        fp_t cat2 = fp_t(0);

        for (int j = 0; j < states; ++j) {
            const fp_t sum_value = sum_row[j];
            cat0 = fp_fma(sum_value, diag_row[0], cat0);
            cat1 = fp_fma(sum_value, diag_row[1], cat1);
            cat2 = fp_fma(sum_value, diag_row[2], cat2);
            diag_row += 4;
        }
        const fp_t pinv = invar_proportion ? invar_proportion[i] : invar_scalar;
        if (pinv > fp_t(0)) {
            const fp_t inv_site_lk = (inv < 0) ? fp_t(0) : (frequencies[inv] * pinv);
            const fp_t non_pinv = fp_t(1) - pinv;
            cat0 = cat0 * non_pinv + inv_site_lk;
            cat1 = cat1 * non_pinv;
            cat2 = cat2 * non_pinv;
        }
        const fp_t w = rate_weights ? rate_weights[i] : fp_t(1);
        site_lk0 += cat0 * w;
        site_lk1 += cat1 * w;
        site_lk2 += cat2 * w;
    }

    const fp_t inv_lk0 = (site_lk0 != fp_t(0)) ? (fp_t(1) / site_lk0) : fp_t(0);
    const fp_t d1 = -site_lk1 * inv_lk0;
    const fp_t d2 = d1 * d1 - (site_lk2 * inv_lk0);
    
    const fp_t weight = load_pattern_weight_cached(pattern_weights, site);
    d1_out = static_cast<double>(weight * d1);
    d2_out = static_cast<double>(weight * d2);
    if constexpr (WRITE_PLACEMENT_CLV) {
        placement_clv[site] = fp_log(site_lk0 > FP_EPS ? site_lk0 : FP_EPS) * weight;
    }
}

template<int RATE_CATS, bool WRITE_PLACEMENT_CLV>
static __device__ __forceinline__
void evaluate_site_derivatives_states4_noinv(
    const DeviceTree D,
    const fp_t* __restrict__ sumtable,
    const fp_t* __restrict__ lambdas,
    const unsigned* __restrict__ pattern_weights,
    size_t site,
    fp_t* placement_clv,
    double& d1_out,
    double& d2_out)
{
    const size_t rate_count = static_cast<size_t>(RATE_CATS);
    const size_t site_span = rate_count * 4;
    const size_t site_base = site * site_span;
    const fp_t* site_sumtable = sumtable + site_base;
    const fp_t* rate_weights = D.d_rate_weights;
    fp_t site_lk0 = fp_t(0);
    fp_t site_lk1 = fp_t(0);
    fp_t site_lk2 = fp_t(0);
    bool eq_weights = true;
    fp_t eq_weight = fp_t(1);
    if (rate_weights && RATE_CATS > 1) {
        const fp_t w0 = rate_weights[0];
        eq_weight = w0;
        #pragma unroll
        for (int r = 1; r < RATE_CATS; ++r) {
            if (rate_weights[r] != w0) {
                eq_weights = false;
                break;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const size_t rate_offset = static_cast<size_t>(r);
        const size_t rate_base = rate_offset * 4;
        const fp_t* sum_row = site_sumtable + rate_base;
        const fp_t* diag_row = lambdas + rate_offset * 16;
        const fp_t rw = rate_weights ? rate_weights[r] : fp_t(1);

        const fp_t cat0 = fp_fma(sum_row[0], diag_row[0], fp_fma(sum_row[1], diag_row[4], fp_fma(sum_row[2], diag_row[8], sum_row[3] * diag_row[12])));
        const fp_t cat1 = fp_fma(sum_row[0], diag_row[1], fp_fma(sum_row[1], diag_row[5], fp_fma(sum_row[2], diag_row[9], sum_row[3] * diag_row[13])));
        const fp_t cat2 = fp_fma(sum_row[0], diag_row[2], fp_fma(sum_row[1], diag_row[6], fp_fma(sum_row[2], diag_row[10], sum_row[3] * diag_row[14])));

        if (eq_weights) {
            site_lk0 = fp_fma(cat0, eq_weight, site_lk0);
            site_lk1 = fp_fma(cat1, eq_weight, site_lk1);
            site_lk2 = fp_fma(cat2, eq_weight, site_lk2);
        } else {
            site_lk0 = fp_fma(cat0, rw, site_lk0);
            site_lk1 = fp_fma(cat1, rw, site_lk1);
            site_lk2 = fp_fma(cat2, rw, site_lk2);
        }
    }

    const fp_t inv_lk0 = (site_lk0 != fp_t(0)) ? (fp_t(1) / site_lk0) : fp_t(0);
    const fp_t d1 = -site_lk1 * inv_lk0;
    const fp_t d2 = d1 * d1 - (site_lk2 * inv_lk0);

    const fp_t weight = load_pattern_weight_cached(pattern_weights, site);
    d1_out = static_cast<double>(weight * d1);
    d2_out = static_cast<double>(weight * d2);
    if constexpr (WRITE_PLACEMENT_CLV) {
        placement_clv[site] = fp_log(site_lk0 > FP_EPS ? site_lk0 : FP_EPS) * weight;
    }
}

template<int RATE_CATS, bool WRITE_PLACEMENT_CLV>
static __device__ __forceinline__
void accumulate_site_derivatives_states4_noinv(
    const DeviceTree D,
    const fp_t* __restrict__ sumtable_op,
    const fp_t* __restrict__ diag_shared,
    const unsigned* __restrict__ pattern_weights,
    fp_t* placement_clv,
    unsigned int tid,
    unsigned int step,
    double& df_local,
    double& ddf_local)
{
    df_local = 0.0;
    ddf_local = 0.0;
    for (size_t site = tid; site < D.sites; site += step) {
        double d1 = 0.0;
        double d2 = 0.0;
        evaluate_site_derivatives_states4_noinv<RATE_CATS, WRITE_PLACEMENT_CLV>(
            D,
            sumtable_op,
            diag_shared,
            pattern_weights,
            site,
            placement_clv,
            d1,
            d2);
        df_local += d1;
        ddf_local += d2;
    }
}

template<bool WRITE_PLACEMENT_CLV>
static __device__ __forceinline__
void accumulate_site_derivatives(
    const DeviceTree D,
    const int* __restrict__ invariant_site,
    const fp_t* __restrict__ invar_proportion,
    fp_t invar_scalar,
    const fp_t* __restrict__ sumtable_op,
    const fp_t* __restrict__ diag_shared,
    const unsigned* __restrict__ pattern_weights,
    fp_t* placement_clv,
    unsigned int tid,
    unsigned int step,
    double& df_local,
    double& ddf_local)
{
    df_local = 0.0;
    ddf_local = 0.0;
    const bool specialized_states4_noinv =
        D.states == 4 &&
        invariant_site == nullptr &&
        invar_proportion == nullptr &&
        invar_scalar == fp_t(0);

    if (specialized_states4_noinv) {
        switch (D.rate_cats) {
            case 1:
                accumulate_site_derivatives_states4_noinv<1, WRITE_PLACEMENT_CLV>(
                    D, sumtable_op, diag_shared, pattern_weights, placement_clv, tid, step, df_local, ddf_local);
                return;
            case 4:
                accumulate_site_derivatives_states4_noinv<4, WRITE_PLACEMENT_CLV>(
                    D, sumtable_op, diag_shared, pattern_weights, placement_clv, tid, step, df_local, ddf_local);
                return;
            case 8:
                accumulate_site_derivatives_states4_noinv<8, WRITE_PLACEMENT_CLV>(
                    D, sumtable_op, diag_shared, pattern_weights, placement_clv, tid, step, df_local, ddf_local);
                return;
            default:
                break;
        }
    }

    for (size_t site = tid; site < D.sites; site += step) {
        double d1 = 0.0;
        double d2 = 0.0;
        evaluate_site_derivatives_general<WRITE_PLACEMENT_CLV>(
            D,
            invariant_site,
            invar_proportion,
            invar_scalar,
            sumtable_op,
            diag_shared,
            pattern_weights,
            site,
            placement_clv,
            d1,
            d2);
        df_local += d1;
        ddf_local += d2;
    }
}

// Sumtable builders.
// Walk all sites and dispatch the single-site pendant sumtable builder.
template<int RATE_CATS>
static __device__ __forceinline__
void build_pendant_sumtable(
    DeviceTree D,
    int target_id,
    const fp_t* __restrict__ left_base,
    fp_t* sumtable_op,
    fp_t* shared_target_mat,
    fp_t* shared_parent_mat,
    unsigned int tid,
    unsigned int step,
    int debug_sumtable,
    int debug_target_id,
    int debug_sumtable_site,
    int debug_sumtable_rate)
{
    for (size_t site = tid; site < D.sites; site += step) {
        update_pendant_sumtable_site<RATE_CATS>(
            D,
            target_id,
            left_base,
            static_cast<unsigned int>(site),
            sumtable_op,
            shared_target_mat,
            shared_parent_mat,
            debug_sumtable,
            debug_target_id,
            debug_sumtable_site,
            debug_sumtable_rate);
    }
}

// Walk all sites and dispatch the single-site proximal sumtable builder.
template<int RATE_CATS>
static __device__ __forceinline__
void build_proximal_sumtable(
    DeviceTree D,
    int target_id,
    const fp_t* __restrict__ left_base,
    const unsigned* __restrict__ left_scaler_base,
    fp_t* sumtable_op,
    fp_t* shared_target_mat,
    fp_t* shared_parent_mat,
    unsigned int tid,
    unsigned int step,
    int debug_sumtable,
    int debug_target_id,
    int debug_sumtable_site,
    int debug_sumtable_rate)
{
    for (size_t site = tid; site < D.sites; site += step) {
        update_proximal_sumtable_site<RATE_CATS>(
            D,
            target_id,
            left_base,
            left_scaler_base,
            static_cast<unsigned int>(site),
            sumtable_op,
            shared_target_mat,
            shared_parent_mat,
            debug_sumtable,
            debug_target_id,
            debug_sumtable_site,
            debug_sumtable_rate);
    }
}

// Pendant-side derivative kernel.
__global__ void LikelihoodDerivativePendantKernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int op_idx,
    const int* op_indices,
    const int*    __restrict__ invariant_site,
    const fp_t* __restrict__ invar_proportion,
    fp_t invar_scalar,
    fp_t* __restrict__ sumtable,
    const unsigned* __restrict__ pattern_weights,
    int max_iter,
    fp_t* new_branch_length,
    size_t sumtable_stride,
    const fp_t* prev_branch_lengths,
    const int* active_ops,
    int debug_enabled,
    int debug_all_iters,
    int debug_limit,
    int debug_target_id,
    int debug_sumtable,
    int debug_sumtable_site,
    int debug_sumtable_rate)
{
    if (!sumtable || !ops || op_idx < 0) {
        return;
    }

    const int op_local = op_idx + (int)blockIdx.x;
    const int op_global = op_indices ? op_indices[op_local] : op_local;
    if (op_global < 0 || op_global >= D.N) return;
    if (active_ops && active_ops[op_local] == 0) return;
    const NodeOpInfo op = ops[op_global];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= D.N) return;
    if (!D.d_query_clv || !D.d_clv_mid_base || !D.d_clv_up || !D.d_pmat_mid_dist || !D.d_pmat_mid_prox) {
        return;
    }

    // Per-op output and branch state.
    const size_t op_base = static_cast<size_t>(op_local) * sumtable_stride;
    fp_t* sumtable_op = sumtable + op_base;
    __shared__ double branch_value_shared;
    __shared__ int stop_iterations;
    __shared__ double branch_lower_bound_shared;
    __shared__ double branch_upper_bound_shared;
    __shared__ double max_step_shared;
    double init_branch = OPT_BRANCH_LEN_MIN;
    if (prev_branch_lengths) {
        init_branch = static_cast<double>(prev_branch_lengths[target_id]);
    } else {
        init_branch = DEFAULT_BRANCH_LENGTH;
    }
    if (threadIdx.x == 0) {
        if (init_branch < OPT_BRANCH_LEN_MIN) init_branch = OPT_BRANCH_LEN_MIN;
        if (init_branch > OPT_BRANCH_LEN_MAX) init_branch = OPT_BRANCH_LEN_MAX;
        branch_value_shared = init_branch;
        branch_lower_bound_shared = OPT_BRANCH_LEN_MIN;
        branch_upper_bound_shared = OPT_BRANCH_LEN_MAX;
        max_step_shared = OPT_BRANCH_LEN_MAX / static_cast<double>(max_iter);
        stop_iterations = 0;
        if (debug_enabled &&
            (debug_target_id < 0 || target_id == debug_target_id)) {
            printf(
                "[op-debug] kernel=pendant op=%d target=%d parent=%d left=%d right=%d left_tip=%d right_tip=%d op_type=%d dir=%u init=%.12f\n",
                op_global,
                target_id,
                op.parent_id,
                op.left_id,
                op.right_id,
                op.left_tip_index,
                op.right_tip_index,
                op.op_type,
                static_cast<unsigned>(op.dir_tag),
                init_branch);
        }
    }
    __syncthreads();

    // Shared-memory layout for branch diagonals and midpoint PMATs.
    extern __shared__ fp_t shmem[];
    fp_t* branch_diag_shared = shmem;
    const size_t rate_count = static_cast<size_t>(D.rate_cats);
    const size_t state_count = static_cast<size_t>(D.states);
    const size_t diag_span = rate_count * state_count * 4;
    const size_t midpoint_pmat_span = rate_count * 16;
    fp_t* target_midpoint_pmat_shared = branch_diag_shared + diag_span;
    fp_t* parent_midpoint_pmat_shared = target_midpoint_pmat_shared + midpoint_pmat_span;
    __shared__ double warp_df_sums[32];
    __shared__ double warp_ddf_sums[32];
    __shared__ double block_df_shared;
    __shared__ double block_ddf_shared;

    // Load midpoint PMATs for the current target edge.
    const size_t pmat_base = static_cast<size_t>(target_id) * midpoint_pmat_span;
    const fp_t* target_mat = D.d_pmat_mid_prox + pmat_base;
    const fp_t* parent_mat = D.d_pmat_mid_dist + pmat_base;
    if (!load_midpoint_pmat_pair_dispatch(
            D, target_midpoint_pmat_shared, parent_midpoint_pmat_shared, target_mat, parent_mat)) {
        return;
    }

    // Build the sumtable once for this operation.
    const unsigned int tid = threadIdx.x;
    const unsigned int step = blockDim.x;
    switch (D.rate_cats) {
        case 1:
            build_pendant_sumtable<1>(
                D,
                target_id,
                D.d_query_clv,
                sumtable_op,
                target_midpoint_pmat_shared,
                parent_midpoint_pmat_shared,
                tid,
                step,
                debug_sumtable,
                debug_target_id,
                debug_sumtable_site,
                debug_sumtable_rate);
            break;
        case 4:
            build_pendant_sumtable<4>(
                D,
                target_id,
                D.d_query_clv,
                sumtable_op,
                target_midpoint_pmat_shared,
                parent_midpoint_pmat_shared,
                tid,
                step,
                debug_sumtable,
                debug_target_id,
                debug_sumtable_site,
                debug_sumtable_rate);
            break;
        case 8:
            build_pendant_sumtable<8>(
                D,
                target_id,
                D.d_query_clv,
                sumtable_op,
                target_midpoint_pmat_shared,
                parent_midpoint_pmat_shared,
                tid,
                step,
                debug_sumtable,
                debug_target_id,
                debug_sumtable_site,
                debug_sumtable_rate);
            break;
        default:
            return;
    }
    __syncthreads();

    // Newton iterations over the already-built sumtable.
    double local_df = 0.0;
    double local_ddf = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        if (stop_iterations) break;
        const double branch_before = branch_value_shared;
        const fp_t branch = static_cast<fp_t>(branch_value_shared);
        if (D.states == 4) {
            if (!build_diagtable_states4_dispatch(D, branch, branch_diag_shared)) {
                build_diagtable_for_branch(D, branch, branch_diag_shared);
            }
        } else {
            build_diagtable_for_branch(D, branch, branch_diag_shared);
        }
        __syncthreads();

        accumulate_site_derivatives<false>(
            D,
            invariant_site,
            invar_proportion,
            invar_scalar,
            sumtable_op,
            branch_diag_shared,
            pattern_weights,
            nullptr,
            tid,
            step,
            local_df,
            local_ddf);

        reduce_block_derivatives(local_df, local_ddf, warp_df_sums, warp_ddf_sums, block_df_shared, block_ddf_shared);

        if (threadIdx.x == 0) {
            apply_newton_update(
                branch_value_shared,
                stop_iterations,
                branch_lower_bound_shared,
                branch_upper_bound_shared,
                max_step_shared,
                block_df_shared,
                block_ddf_shared);
            maybe_print_newton_debug(
                debug_enabled,
                debug_all_iters,
                debug_limit,
                debug_target_id,
                "pendant",
                op_global,
                target_id,
                iter,
                branch_before,
                branch_value_shared,
                block_df_shared,
                block_ddf_shared,
                branch_lower_bound_shared,
                branch_upper_bound_shared,
                max_step_shared);
        }
        __syncthreads();
        if (stop_iterations) break;
    }

    if (threadIdx.x == 0) {
        new_branch_length[target_id] = static_cast<fp_t>(branch_value_shared);
    }
}

// Proximal-side derivative kernel.
__global__ void LikelihoodDerivativeProximalKernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int op_idx,
    const int* op_indices,
    const int*    __restrict__ invariant_site,
    const fp_t* __restrict__ invar_proportion,
    fp_t invar_scalar,
    fp_t* __restrict__ sumtable,
    const unsigned* __restrict__ pattern_weights,
    int max_iter,
    fp_t* new_branch_length,
    size_t sumtable_stride,
    const fp_t* prev_branch_lengths,
    const int* active_ops,
    int debug_enabled,
    int debug_all_iters,
    int debug_limit,
    int debug_target_id,
    int debug_sumtable,
    int debug_sumtable_site,
    int debug_sumtable_rate)
{
    if (!sumtable || !ops || op_idx < 0) {
        return;
    }

    const int op_local = op_idx + (int)blockIdx.x;
    const int op_global = op_indices ? op_indices[op_local] : op_local;
    if (op_global < 0 || op_global >= D.N) return;
    if (active_ops && active_ops[op_local] == 0) return;
    const NodeOpInfo op = ops[op_global];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= D.N) return;
    if (!D.d_clv_up || !D.d_clv_mid_base || !D.d_query_clv || !D.d_query_pmat || !D.d_pmat_mid_dist) {
        return;
    }

    // Per-op output and branch state.
    const size_t op_base = static_cast<size_t>(op_local) * sumtable_stride;
    fp_t* sumtable_op = sumtable + op_base;
    __shared__ double branch_value_shared;
    __shared__ int stop_iterations;
    __shared__ double branch_lower_bound_shared;
    __shared__ double branch_upper_bound_shared;
    __shared__ double max_step_shared;
    double init_branch = OPT_BRANCH_LEN_MIN;
    const double total_branch = static_cast<double>(D.d_blen[target_id]);
    const double branch_lower_bound = effective_split_branch_min(total_branch, OPT_BRANCH_LEN_MIN);
    const double branch_upper_bound = scalar_max(branch_lower_bound, total_branch - branch_lower_bound);
    if (prev_branch_lengths) {
        init_branch = static_cast<double>(prev_branch_lengths[target_id]);
    } else {
        init_branch = 0.5 * static_cast<double>(D.d_blen[target_id]);
    }
    if (threadIdx.x == 0) {
        init_branch = clamp_scalar(init_branch, branch_lower_bound, branch_upper_bound);
        branch_value_shared = init_branch;
        branch_lower_bound_shared = branch_lower_bound;
        branch_upper_bound_shared = branch_upper_bound;
        max_step_shared = scalar_max(
            OPT_BRANCH_XTOL,
            (branch_upper_bound_shared - branch_lower_bound_shared) / static_cast<double>(max_iter));
        stop_iterations = 0;
        if (debug_enabled &&
            (debug_target_id < 0 || target_id == debug_target_id)) {
            printf(
                "[op-debug] kernel=proximal op=%d target=%d parent=%d left=%d right=%d left_tip=%d right_tip=%d op_type=%d dir=%u init=%.12f total=%.12f lb=%.12f ub=%.12f\n",
                op_global,
                target_id,
                op.parent_id,
                op.left_id,
                op.right_id,
                op.left_tip_index,
                op.right_tip_index,
                op.op_type,
                static_cast<unsigned>(op.dir_tag),
                init_branch,
                total_branch,
                branch_lower_bound,
                branch_upper_bound);
        }
    }
    __syncthreads();

    // Shared-memory layout for branch diagonals and midpoint PMATs.
    extern __shared__ fp_t shmem[];
    fp_t* branch_diag_shared = shmem;
    const size_t rate_count = static_cast<size_t>(D.rate_cats);
    const size_t state_count = static_cast<size_t>(D.states);
    const size_t diag_span = rate_count * state_count * 4;
    const size_t midpoint_pmat_span = rate_count * 16;
    fp_t* target_midpoint_pmat_shared = branch_diag_shared + diag_span;
    fp_t* parent_midpoint_pmat_shared = target_midpoint_pmat_shared + midpoint_pmat_span;
    __shared__ double warp_df_sums[32];
    __shared__ double warp_ddf_sums[32];
    __shared__ double block_df_shared;
    __shared__ double block_ddf_shared;

    // Resolve the left-side CLV/scaler slice for this target edge.
    if (!D.d_site_scaler_up) return;
    const size_t node_site_span =
        static_cast<size_t>(D.sites) * rate_count * state_count;
    const fp_t* left_base = D.d_clv_up + static_cast<size_t>(target_id) * node_site_span;
    const size_t scaler_span = D.per_rate_scaling
        ? static_cast<size_t>(D.sites) * rate_count
        : static_cast<size_t>(D.sites);
    const unsigned* left_scaler_base = D.d_site_scaler_up + static_cast<size_t>(target_id) * scaler_span;

    // Load midpoint PMATs for the current query/target pair.
    const size_t target_pmat_base = static_cast<size_t>(op_local) * midpoint_pmat_span;
    const size_t parent_pmat_base = static_cast<size_t>(target_id) * midpoint_pmat_span;
    const fp_t* target_mat = D.d_query_pmat + target_pmat_base;
    const fp_t* parent_mat = D.d_pmat_mid_dist + parent_pmat_base;
    if (!load_midpoint_pmat_pair_dispatch(
            D, target_midpoint_pmat_shared, parent_midpoint_pmat_shared, target_mat, parent_mat)) {
        return;
    }
    // Match EPA-ng's contract here: the proximal-side sumtable is prepared
    // once for the current smoothing step, then Newton iterates on that fixed
    // sumtable instead of rebuilding the distal-side PMAT inside the loop.
    const unsigned int tid = threadIdx.x;
    const unsigned int step = blockDim.x;
    switch (D.rate_cats) {
        case 1:
            build_proximal_sumtable<1>(
                D,
                target_id,
                left_base,
                left_scaler_base,
                sumtable_op,
                target_midpoint_pmat_shared,
                parent_midpoint_pmat_shared,
                tid,
                step,
                debug_sumtable,
                debug_target_id,
                debug_sumtable_site,
                debug_sumtable_rate);
            break;
        case 4:
            build_proximal_sumtable<4>(
                D,
                target_id,
                left_base,
                left_scaler_base,
                sumtable_op,
                target_midpoint_pmat_shared,
                parent_midpoint_pmat_shared,
                tid,
                step,
                debug_sumtable,
                debug_target_id,
                debug_sumtable_site,
                debug_sumtable_rate);
            break;
        case 8:
            build_proximal_sumtable<8>(
                D,
                target_id,
                left_base,
                left_scaler_base,
                sumtable_op,
                target_midpoint_pmat_shared,
                parent_midpoint_pmat_shared,
                tid,
                step,
                debug_sumtable,
                debug_target_id,
                debug_sumtable_site,
                debug_sumtable_rate);
            break;
        default:
            return;
    }
    __syncthreads();

    // Newton iterations over the already-built sumtable.
    double local_df = 0.0;
    double local_ddf = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        if (stop_iterations) break;
        const double branch_before = branch_value_shared;
        const fp_t branch = static_cast<fp_t>(branch_value_shared);
        if (D.states == 4) {
            if (!build_diagtable_states4_dispatch(D, branch, branch_diag_shared)) {
                build_diagtable_for_branch(D, branch, branch_diag_shared);
            }
        } else {
            build_diagtable_for_branch(D, branch, branch_diag_shared);
        }
        __syncthreads();

        accumulate_site_derivatives<false>(
            D,
            invariant_site,
            invar_proportion,
            invar_scalar,
            sumtable_op,
            branch_diag_shared,
            pattern_weights,
            nullptr,
            tid,
            step,
            local_df,
            local_ddf);

        reduce_block_derivatives(local_df, local_ddf, warp_df_sums, warp_ddf_sums, block_df_shared, block_ddf_shared);

        if (threadIdx.x == 0) {
            apply_newton_update(
                branch_value_shared,
                stop_iterations,
                branch_lower_bound_shared,
                branch_upper_bound_shared,
                max_step_shared,
                block_df_shared,
                block_ddf_shared);
            maybe_print_newton_debug(
                debug_enabled,
                debug_all_iters,
                debug_limit,
                debug_target_id,
                "proximal",
                op_global,
                target_id,
                iter,
                branch_before,
                branch_value_shared,
                block_df_shared,
                block_ddf_shared,
                branch_lower_bound_shared,
                branch_upper_bound_shared,
                max_step_shared);
        }
        __syncthreads();

        if (stop_iterations) break;
    }

    if (threadIdx.x == 0) {
        new_branch_length[target_id] = static_cast<fp_t>(branch_value_shared);
    }
}
