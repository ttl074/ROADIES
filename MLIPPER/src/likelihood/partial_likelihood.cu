#include <cuda_runtime.h>
#include <cmath>
#include "util/mlipper_util.h"
#include "partial_likelihood.cuh"

__device__ inline void scale_clv_pow2(fp_t &x, int shift) {
    fp_scale_pow2(x, shift);
}

__device__ __forceinline__ unsigned int threshold_scale_shift(fp_t max_val)
{
    const fp_t scale_threshold = fp_ldexp(fp_t(1), SCALE_THRESHOLD_EXPONENT);
    return (max_val < scale_threshold)
        ? static_cast<unsigned int>(-SCALE_THRESHOLD_EXPONENT)
        : 0u;
}

__device__ __forceinline__ unsigned int scaler_slot(
    const DeviceTree& D,
    unsigned int rate_idx)
{
    return D.per_rate_scaling ? rate_idx : 0u;
}

__device__ __forceinline__ unsigned int read_scaler_shift(
    const DeviceTree& D,
    const unsigned int* scaler,
    unsigned int rate_idx)
{
    if (!scaler) return 0u;
    return scaler[scaler_slot(D, rate_idx)];
}

__device__ __forceinline__ void write_scaler_shift(
    const DeviceTree& D,
    unsigned int* scaler,
    unsigned int rate_idx,
    unsigned int value)
{
    if (!scaler) return;
    scaler[scaler_slot(D, rate_idx)] = value;
}

__device__ __forceinline__ unsigned int* scaler_ptr_for_pool(
    const DeviceTree& D,
    uint8_t clv_pool,
    int node_id,
    unsigned int site)
{
    if (clv_pool == static_cast<uint8_t>(CLV_POOL_DOWN)) {
        return down_scaler_ptr(D, node_id, site);
    }
    return up_scaler_ptr(D, node_id, site);
}

__device__ __forceinline__ void add_scaler_shift(
    const DeviceTree& D,
    unsigned int* scaler,
    unsigned int rate_idx,
    unsigned int shift)
{
    if (!scaler || shift == 0) return;
    scaler[scaler_slot(D, rate_idx)] += shift;
}

// ---- Downward per-case helpers (states arbitrary) ----
__device__ __forceinline__ void compute_downward_inner_inner_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id  = target_is_left ? op.left_id  : op.right_id;
    const int sibling_id = target_is_left ? op.right_id : op.left_id;
    if (target_id < 0 || sibling_id < 0) return;

    const unsigned int states    = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)states * (size_t)rate_cats;

    const fp_t* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    const fp_t* sibling_up  = D.d_clv_up   + (size_t)sibling_id * per_node + site_off;
    fp_t*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    fp_t*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    fp_t*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    const fp_t* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !sibling_up || !target_down) return;

    const fp_t* target_mat  = D.d_pmat + (size_t)target_id  * rate_cats * states * states;
    const fp_t* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * rate_cats * states * states)
        : target_mat;
    const fp_t* sibling_mat = D.d_pmat + (size_t)sibling_id * rate_cats * states * states;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, sibling_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    unsigned int* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site);

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const unsigned int parent_shift = read_scaler_shift(D, parent_scaler, r);
        const unsigned int sibling_shift = read_scaler_shift(D, sibling_scaler, r);
        const unsigned int target_up_shift = read_scaler_shift(D, target_up_scaler, r);
        const unsigned int down_inherited = parent_shift + sibling_shift;
        const unsigned int mid_inherited = down_inherited + target_up_shift;
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);
        if (mid_base) write_scaler_shift(D, mid_base_scaler, r, down_inherited);

        const fp_t* Tmat = target_mat  + (size_t)r * states * states;
        const fp_t* Thalf= target_mat_half + (size_t)r * states * states;
        const fp_t* Smat = sibling_mat + (size_t)r * states * states;
        const fp_t* Ppar = parent_down + (size_t)r * states;
        const fp_t* Psib = sibling_up  + (size_t)r * states;
        fp_t*       Pout = target_down + (size_t)r * states;
        fp_t*       Pmid  = (target_mid && target_up) ? (target_mid + (size_t)r * states) : nullptr;
        const fp_t* Pup   = target_up ? (target_up + (size_t)r * states) : nullptr;
        fp_t*       Pbase = mid_base ? (mid_base + (size_t)r * states) : nullptr;

        double sib_to_parent[64];
        for (unsigned int j = 0; j < states; ++j) {
            const fp_t* row = Smat + j * states;
            double acc = 0.0;
            for (unsigned int k = 0; k < states; ++k) acc += row[k] * Psib[k];
            sib_to_parent[j] = acc;
        }

        double col_scale_max_val = 0.0;
        for (unsigned int i = 0; i < states; ++i) {
            const fp_t* Tcol = Tmat + i;
            double acc = 0.0;
            for (unsigned int j = 0; j < states; ++j)
                acc += Tcol[j * states] * (Ppar[j] * sib_to_parent[j]);
            Pout[i] = acc;
            if (acc > col_scale_max_val) col_scale_max_val = acc;
        }

        double pbase_max_val = 0.0;
        double pmid_max_val = 0.0;
        if (Pmid) {
            // Cache parent_down * sibling_up (after sibling branch matrix) per state.
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j) {
                    Pbase[j] = Ppar[j] * sib_to_parent[j];
                    if (Pbase[j] > pbase_max_val) pbase_max_val = Pbase[j];
                }
            }
            for (unsigned int i = 0; i < states; ++i) {
                const fp_t* Throw = Thalf + i * states;
                double par_acc = 0.0, tgt_acc = 0.0;
                for (unsigned int j = 0; j < states; ++j) {
                    const double pj = Pbase ? Pbase[j] : (Ppar[j] * sib_to_parent[j]);
                    par_acc += Throw[j] * pj;
                    tgt_acc += Throw[j] * Pup[j];
                }
                const double val = par_acc * tgt_acc;
                Pmid[i] = val;
                if (val > pmid_max_val) pmid_max_val = val;
            }
        }

        {
            const unsigned int down_shift_local = threshold_scale_shift(col_scale_max_val);
            if (down_shift_local) {
            add_scaler_shift(D, down_scaler, r, down_shift_local);
            for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pout[j], down_shift_local);
            }
            const unsigned int base_shift_local = Pbase ? threshold_scale_shift(pbase_max_val) : 0u;
            if (base_shift_local) {
            add_scaler_shift(D, mid_base_scaler, r, base_shift_local);
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pbase[j], base_shift_local);
            }
            }
            const unsigned int mid_shift_local = Pmid ? threshold_scale_shift(pmid_max_val) : 0u;
            if (mid_shift_local) {
            add_scaler_shift(D, mid_scaler, r, mid_shift_local);
            if (Pmid) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pmid[j], mid_shift_local);
            }
            }
        }
    }
}

__device__ __forceinline__ void compute_downward_inner_tip_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id       = target_is_left ? op.left_id  : op.right_id;
    const int sibling_tip_idx = target_is_left ? op.right_tip_index : op.left_tip_index;
    if (target_id < 0 || sibling_tip_idx < 0) return;

    const unsigned int states    = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)states * (size_t)rate_cats;

    const fp_t* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    fp_t*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    const fp_t* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down) return;

    const fp_t* target_mat  = D.d_pmat + (size_t)target_id * rate_cats * states * states;
    const fp_t* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * rate_cats * states * states)
        : target_mat;
    const fp_t* sibling_mat = D.d_pmat + (size_t)(target_is_left ? op.right_id : op.left_id) * rate_cats * states * states;
    fp_t*         mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, target_is_left ? op.right_id : op.left_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    unsigned int* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site);

    const unsigned char* tipchars = D.d_tipchars + (size_t)sibling_tip_idx * D.sites;

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);
        if (mid_base) write_scaler_shift(D, mid_base_scaler, r, down_inherited);

        const unsigned int mask = D.d_tipmap[tipchars[site]];
        const fp_t* Tmat = target_mat  + (size_t)r * states * states;
        const fp_t* Thalf= target_mat_half + (size_t)r * states * states;
        const fp_t* Smat = sibling_mat + (size_t)r * states * states;
        const fp_t* Ppar = parent_down + (size_t)r * states;
        const fp_t* Pup  = target_up ? (target_up + (size_t)r * states) : nullptr;
        fp_t*       Pout = target_down + (size_t)r * states;
        fp_t*       Pmid = (target_up && D.d_clv_mid)
            ? (D.d_clv_mid + (size_t)target_id * per_node + site_off + (size_t)r * states)
            : nullptr;
        fp_t*       Pbase = mid_base ? (mid_base + (size_t)r * states) : nullptr;

        double sib_to_parent[64];
        for (unsigned int j = 0; j < states; ++j) {
            const fp_t* row = Smat + j * states;
            double acc = 0.0;
            for (unsigned int k = 0; k < states; ++k)
                if (mask & (1u << k)) acc += row[k];
            sib_to_parent[j] = acc;
        }

        double col_scale_max_val = 0.0;
        for (unsigned int i = 0; i < states; ++i) {
            const fp_t* Tcol = Tmat + i;
            double acc = 0.0;
            for (unsigned int j = 0; j < states; ++j)
                acc += Tcol[j * states] * (Ppar[j] * sib_to_parent[j]);
            Pout[i] = acc;
            if (acc > col_scale_max_val) col_scale_max_val = acc;
        }

        double pbase_max_val = 0.0;
        double pmid_max_val = 0.0;
        if (Pmid) {
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j) {
                    Pbase[j] = Ppar[j] * sib_to_parent[j];
                    if (Pbase[j] > pbase_max_val) pbase_max_val = Pbase[j];
                }
            }
            for (unsigned int i = 0; i < states; ++i) {
                const fp_t* Throw = Thalf + i * states;
                double par_acc = 0.0, tgt_acc = 0.0;
                for (unsigned int j = 0; j < states; ++j) {
                    const double pj = Pbase ? Pbase[j] : (Ppar[j] * sib_to_parent[j]);
                    par_acc += Throw[j] * pj;
                    tgt_acc += Throw[j] * Pup[j];
                }
                Pmid[i] = par_acc * tgt_acc;
                if (Pmid[i] > pmid_max_val) pmid_max_val = Pmid[i];
            }
        }

        {
            const unsigned int down_shift_local = threshold_scale_shift(col_scale_max_val);
            if (down_shift_local) {
            add_scaler_shift(D, down_scaler, r, down_shift_local);
            for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pout[j], down_shift_local);
            }
            const unsigned int base_shift_local = Pbase ? threshold_scale_shift(pbase_max_val) : 0u;
            if (base_shift_local) {
            add_scaler_shift(D, mid_base_scaler, r, base_shift_local);
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pbase[j], base_shift_local);
            }
            }
            const unsigned int mid_shift_local = Pmid ? threshold_scale_shift(pmid_max_val) : 0u;
            if (mid_shift_local) {
            add_scaler_shift(D, mid_scaler, r, mid_shift_local);
            if (Pmid) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pmid[j], mid_shift_local);
            }
            }
        }
    }
}

__device__ __forceinline__ void compute_downward_tip_tip_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_tip_idx  = target_is_left ? op.left_tip_index : op.right_tip_index;
    const int sibling_tip_idx = target_is_left ? op.right_tip_index : op.left_tip_index;
    const int target_id       = target_is_left ? op.left_id : op.right_id;
    if (target_tip_idx < 0 || sibling_tip_idx < 0 || target_id < 0) return;

    const unsigned int states    = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)states * (size_t)rate_cats;

    const fp_t* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    fp_t*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    const fp_t* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down) return;

    const fp_t* target_mat  = D.d_pmat + (size_t)target_id * rate_cats * states * states;
    const fp_t* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * rate_cats * states * states)
        : target_mat;
    const fp_t* sibling_mat = D.d_pmat + (size_t)(target_is_left ? op.right_id : op.left_id) * rate_cats * states * states;
    fp_t*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    fp_t*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, target_is_left ? op.right_id : op.left_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    unsigned int* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site);
    const unsigned char* tipchars = D.d_tipchars + (size_t)sibling_tip_idx * D.sites;

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);
        if (mid_base) write_scaler_shift(D, mid_base_scaler, r, down_inherited);

        const unsigned int mask = D.d_tipmap[tipchars[site]];
        const fp_t* Tmat = target_mat  + (size_t)r * states * states;
        const fp_t* Thalf= target_mat_half + (size_t)r * states * states;
        const fp_t* Smat = sibling_mat + (size_t)r * states * states;
        const fp_t* Ppar = parent_down + (size_t)r * states;
        const fp_t* Pup  = target_up ? (target_up + (size_t)r * states) : nullptr;
        fp_t*       Pout = target_down + (size_t)r * states;
        fp_t*       Pmid = (target_up && target_mid) ? (target_mid + (size_t)r * states) : nullptr;
        fp_t*       Pbase = mid_base ? (mid_base + (size_t)r * states) : nullptr;

        double sib_to_parent[64];
        for (unsigned int j = 0; j < states; ++j) {
            const fp_t* row = Smat + j * states;
            double acc = 0.0;
            for (unsigned int k = 0; k < states; ++k)
                if (mask & (1u << k)) acc += row[k];
            sib_to_parent[j] = acc;
        }

        double col_scale_max_val = 0.0;
        for (unsigned int i = 0; i < states; ++i) {
            const fp_t* Tcol = Tmat + i;
            double acc = 0.0;
            for (unsigned int j = 0; j < states; ++j)
                acc += Tcol[j * states] * (Ppar[j] * sib_to_parent[j]);
            Pout[i] = acc;
            if (acc > col_scale_max_val) col_scale_max_val = acc;
        }

        double pbase_max_val = 0.0;
        double pmid_max_val = 0.0;
        if (Pmid) {
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j) {
                    Pbase[j] = Ppar[j] * sib_to_parent[j];
                    if (Pbase[j] > pbase_max_val) pbase_max_val = Pbase[j];
                }
            }
            for (unsigned int i = 0; i < states; ++i) {
                const fp_t* Throw = Thalf + i * states;
                double par_acc = 0.0, tgt_acc = 0.0;
                for (unsigned int j = 0; j < states; ++j) {
                    const double pj = Pbase ? Pbase[j] : (Ppar[j] * sib_to_parent[j]);
                    par_acc += Throw[j] * pj;
                    tgt_acc += Throw[j] * Pup[j];
                }
                Pmid[i] = par_acc * tgt_acc;
                if (Pmid[i] > pmid_max_val) pmid_max_val = Pmid[i];
            }
        }

        {
            const unsigned int down_shift_local = threshold_scale_shift(col_scale_max_val);
            if (down_shift_local) {
            add_scaler_shift(D, down_scaler, r, down_shift_local);
            for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pout[j], down_shift_local);
            }
            const unsigned int base_shift_local = Pbase ? threshold_scale_shift(pbase_max_val) : 0u;
            if (base_shift_local) {
            add_scaler_shift(D, mid_base_scaler, r, base_shift_local);
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pbase[j], base_shift_local);
            }
            }
            const unsigned int mid_shift_local = Pmid ? threshold_scale_shift(pmid_max_val) : 0u;
            if (mid_shift_local) {
            add_scaler_shift(D, mid_scaler, r, mid_shift_local);
            if (Pmid) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pmid[j], mid_shift_local);
            }
            }
        }
    }
}

__device__ __forceinline__ void compute_downward_tip_inner_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_tip_idx  = target_is_left ? op.left_tip_index : op.right_tip_index;
    const int sibling_id      = target_is_left ? op.right_id : op.left_id;
    if (target_tip_idx < 0 || sibling_id < 0) return;

    const unsigned int states    = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)states * (size_t)rate_cats;

    const fp_t* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    const fp_t* sibling_up  = D.d_clv_up   + (size_t)sibling_id * per_node + site_off;
    fp_t*       target_down = D.d_clv_down + (size_t)(target_is_left ? op.left_id : op.right_id) * per_node + site_off;
    const fp_t* target_up   = D.d_clv_up   + (size_t)(target_is_left ? op.left_id : op.right_id) * per_node + site_off;
    if (!parent_down || !sibling_up || !target_down) return;

    const fp_t* target_mat  = D.d_pmat + (size_t)(target_is_left ? op.left_id : op.right_id) * rate_cats * states * states;
    const fp_t* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)(target_is_left ? op.left_id : op.right_id) * rate_cats * states * states)
        : target_mat;
    const fp_t* sibling_mat = D.d_pmat + (size_t)sibling_id * rate_cats * states * states;
    fp_t*       target_mid  = D.d_clv_mid
        ? (D.d_clv_mid + (size_t)(target_is_left ? op.left_id : op.right_id) * per_node + site_off)
        : nullptr;
    fp_t*       mid_base    = D.d_clv_mid_base
        ? (D.d_clv_mid_base + (size_t)(target_is_left ? op.left_id : op.right_id) * per_node + site_off)
        : nullptr;
    const int target_id       = target_is_left ? op.left_id : op.right_id;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, sibling_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    unsigned int* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site);

    const unsigned char* tipchars = D.d_tipchars + (size_t)target_tip_idx * D.sites;
    const unsigned int tmask = D.d_tipmap[tipchars[site]];

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);
        if (mid_base) write_scaler_shift(D, mid_base_scaler, r, down_inherited);

        const fp_t* Tmat = target_mat  + (size_t)r * states * states;
        const fp_t* Thalf= target_mat_half + (size_t)r * states * states;
        const fp_t* Smat = sibling_mat + (size_t)r * states * states;
        const fp_t* Ppar = parent_down + (size_t)r * states;
        const fp_t* Psib = sibling_up  + (size_t)r * states;
        const fp_t* Pup  = target_up ? (target_up + (size_t)r * states) : nullptr;
        fp_t*       Pout = target_down + (size_t)r * states;
        fp_t*       Pmid = (target_up && target_mid) ? (target_mid + (size_t)r * states) : nullptr;
        fp_t*       Pbase = mid_base ? (mid_base + (size_t)r * states) : nullptr;

        double sib_to_parent[64];
        for (unsigned int j = 0; j < states; ++j) {
            const fp_t* row = Smat + j * states;
            double acc = 0.0;
            for (unsigned int k = 0; k < states; ++k) acc += row[k] * Psib[k];
            sib_to_parent[j] = acc;
        }

        double col_scale_max_val = 0.0;
        for (unsigned int i = 0; i < states; ++i) {
            const fp_t* Tcol = Tmat + i;
            double acc = 0.0;
            for (unsigned int j = 0; j < states; ++j)
                acc += Tcol[j * states] * (Ppar[j] * sib_to_parent[j]);
            Pout[i] = (tmask & (1u << i)) ? acc : 0.0;
            if (Pout[i] > col_scale_max_val) col_scale_max_val = Pout[i];
        }

        double pbase_max_val = 0.0;
        double pmid_max_val = 0.0;
        if (Pmid) {
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j) {
                    Pbase[j] = Ppar[j] * sib_to_parent[j];
                    if (Pbase[j] > pbase_max_val) pbase_max_val = Pbase[j];
                }
            }
            for (unsigned int i = 0; i < states; ++i) {
                const fp_t* Throw = Thalf + i * states;
                double par_acc = 0.0, tgt_acc = 0.0;
                for (unsigned int j = 0; j < states; ++j) {
                    const double pj = Pbase ? Pbase[j] : (Ppar[j] * sib_to_parent[j]);
                    par_acc += Throw[j] * pj;
                    tgt_acc += Throw[j] * Pup[j];
                }
                Pmid[i] = (tmask & (1u << i)) ? (par_acc * tgt_acc) : 0.0;
                if (Pmid[i] > pmid_max_val) pmid_max_val = Pmid[i];
            }
        }

        {
            const unsigned int down_shift_local = threshold_scale_shift(col_scale_max_val);
            if (down_shift_local) {
            add_scaler_shift(D, down_scaler, r, down_shift_local);
            for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pout[j], down_shift_local);
            }
            const unsigned int base_shift_local = Pbase ? threshold_scale_shift(pbase_max_val) : 0u;
            if (base_shift_local) {
            add_scaler_shift(D, mid_base_scaler, r, base_shift_local);
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pbase[j], base_shift_local);
            }
            }
            const unsigned int mid_shift_local = Pmid ? threshold_scale_shift(pmid_max_val) : 0u;
            if (mid_shift_local) {
            add_scaler_shift(D, mid_scaler, r, mid_shift_local);
            if (Pmid) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_clv_pow2(Pmid[j], mid_shift_local);
            }
            }
        }
    }
}

// ---- Downward specializations for states=4, templated by rate cats ----
template<int RATE_CATS>
__device__ __forceinline__ void compute_downward_inner_inner_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    if (op.left_tip_index >= 0 || op.right_tip_index >= 0) return;

    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id  = target_is_left ? op.left_id  : op.right_id;
    const int sibling_id = target_is_left ? op.right_id : op.left_id;
    if (target_id < 0 || sibling_id < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;

    const fp_t* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    const fp_t* sibling_up  = D.d_clv_up   + (size_t)sibling_id * per_node + site_off;
    fp_t*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    fp_t*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    fp_t*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    const fp_t* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down || !sibling_up || !target_up) return;

    const fp_t* target_mat  = D.d_pmat + (size_t)target_id  * (size_t)RATE_CATS * 16;
    const fp_t* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16)
        : target_mat;
    const fp_t* sibling_mat = D.d_pmat + (size_t)sibling_id * (size_t)RATE_CATS * 16;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, sibling_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    unsigned int* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const unsigned int parent_shift = read_scaler_shift(D, parent_scaler, r);
        const unsigned int sibling_shift = read_scaler_shift(D, sibling_scaler, r);
        const unsigned int target_up_shift = read_scaler_shift(D, target_up_scaler, r);
        const unsigned int down_inherited = parent_shift + sibling_shift;
        const unsigned int mid_inherited = down_inherited + target_up_shift;
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);
        if (mid_base) write_scaler_shift(D, mid_base_scaler, r, down_inherited);

        const fp_t* Tmat = target_mat  + (size_t)r * 16;
        const fp_t* Thalf= target_mat_half + (size_t)r * 16;
        const fp_t* Smat = sibling_mat + (size_t)r * 16;
        const fp4_t Ppar = reinterpret_cast<const fp4_t*>(parent_down)[r];
        const fp4_t Psib = reinterpret_cast<const fp4_t*>(sibling_up)[r];
        const fp4_t Pup  = reinterpret_cast<const fp4_t*>(target_up)[r];
        fp_t*       Pout = target_down + (size_t)r * 4;

        const fp_t sib0 = fp_dot4(make_fp4(Smat[0], Smat[1], Smat[2], Smat[3]), Psib);
        const fp_t sib1 = fp_dot4(make_fp4(Smat[4], Smat[5], Smat[6], Smat[7]), Psib);
        const fp_t sib2 = fp_dot4(make_fp4(Smat[8], Smat[9], Smat[10], Smat[11]), Psib);
        const fp_t sib3 = fp_dot4(make_fp4(Smat[12], Smat[13], Smat[14], Smat[15]), Psib);

        const fp_t p0 = Ppar.x * sib0;
        const fp_t p1 = Ppar.y * sib1;
        const fp_t p2 = Ppar.z * sib2;
        const fp_t p3 = Ppar.w * sib3;


        Pout[0] = Tmat[0] * p0 + Tmat[4] * p1 + Tmat[8]  * p2 + Tmat[12] * p3;
        Pout[1] = Tmat[1] * p0 + Tmat[5] * p1 + Tmat[9]  * p2 + Tmat[13] * p3;
        Pout[2] = Tmat[2] * p0 + Tmat[6] * p1 + Tmat[10] * p2 + Tmat[14] * p3;
        Pout[3] = Tmat[3] * p0 + Tmat[7] * p1 + Tmat[11] * p2 + Tmat[15] * p3;
        if (mid_base) {
            fp_t* Pbase = mid_base + (size_t)r * 4;
            Pbase[0] = p0;
            Pbase[1] = p1;
            Pbase[2] = p2;
            Pbase[3] = p3;
        }

        if (target_mid) {
            fp_t* Pmid = target_mid + (size_t)r * 4;
            const fp_t par0 = fp_dot4(make_fp4(Thalf[0], Thalf[4], Thalf[8], Thalf[12]), make_fp4(p0, p1, p2, p3));
            const fp_t par1 = fp_dot4(make_fp4(Thalf[1], Thalf[5], Thalf[9], Thalf[13]), make_fp4(p0, p1, p2, p3));
            const fp_t par2 = fp_dot4(make_fp4(Thalf[2], Thalf[6], Thalf[10], Thalf[14]), make_fp4(p0, p1, p2, p3));
            const fp_t par3 = fp_dot4(make_fp4(Thalf[3], Thalf[7], Thalf[11], Thalf[15]), make_fp4(p0, p1, p2, p3));

            const fp_t tgt0 = fp_dot4(make_fp4(Thalf[0], Thalf[4], Thalf[8], Thalf[12]), Pup);
            const fp_t tgt1 = fp_dot4(make_fp4(Thalf[1], Thalf[5], Thalf[9], Thalf[13]), Pup);
            const fp_t tgt2 = fp_dot4(make_fp4(Thalf[2], Thalf[6], Thalf[10], Thalf[14]), Pup);
            const fp_t tgt3 = fp_dot4(make_fp4(Thalf[3], Thalf[7], Thalf[11], Thalf[15]), Pup);

            Pmid[0] = par0 * tgt0;
            Pmid[1] = par1 * tgt1;
            Pmid[2] = par2 * tgt2;
            Pmid[3] = par3 * tgt3;
        }

        const fp_t down_max_val = fp_hmax4(Pout[0], Pout[1], Pout[2], Pout[3]);
        const unsigned int down_shift_local = threshold_scale_shift(down_max_val);
        if (down_shift_local) {
            add_scaler_shift(D, down_scaler, r, down_shift_local);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scale_clv_pow2(Pout[j], down_shift_local);
            }
        }
        if (mid_base) {
            fp_t* Pbase = mid_base + (size_t)r * 4;
            const fp_t base_max_val = fp_hmax4(Pbase[0], Pbase[1], Pbase[2], Pbase[3]);
            const unsigned int base_shift_local = threshold_scale_shift(base_max_val);
            if (base_shift_local) {
                add_scaler_shift(D, mid_base_scaler, r, base_shift_local);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_clv_pow2(Pbase[j], base_shift_local);
                }
            }
        }
        if (target_mid) {
            fp_t* Pmid = target_mid + (size_t)r * 4;
            const fp_t mid_max_val = fp_hmax4(Pmid[0], Pmid[1], Pmid[2], Pmid[3]);
            const unsigned int mid_shift_local = threshold_scale_shift(mid_max_val);
            if (mid_shift_local) {
                add_scaler_shift(D, mid_scaler, r, mid_shift_local);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_clv_pow2(Pmid[j], mid_shift_local);
                }
            }
        }
    }
}

template<int RATE_CATS>
__device__ __forceinline__ void compute_downward_inner_tip_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id       = target_is_left ? op.left_id  : op.right_id;
    const int sibling_tip_idx = target_is_left ? op.right_tip_index : op.left_tip_index;
    if (target_id < 0 || sibling_tip_idx < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;

    const fp_t* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    fp_t*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    fp_t*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    fp_t*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    const fp_t* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down || !target_up) return;

    const fp_t* target_mat  = D.d_pmat + (size_t)target_id * (size_t)RATE_CATS * 16;
    const fp_t* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16)
        : target_mat;
    const fp_t* sibling_mat = D.d_pmat + (size_t)(target_is_left ? op.right_id : op.left_id) * (size_t)RATE_CATS * 16;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, target_is_left ? op.right_id : op.left_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    unsigned int* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site);

    const unsigned char* tipchars = D.d_tipchars + (size_t)sibling_tip_idx * D.sites;

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const unsigned int parent_shift = read_scaler_shift(D, parent_scaler, r);
        const unsigned int sibling_shift = read_scaler_shift(D, sibling_scaler, r);
        const unsigned int target_up_shift = read_scaler_shift(D, target_up_scaler, r);
        const unsigned int down_inherited = parent_shift + sibling_shift;
        const unsigned int mid_inherited = down_inherited + target_up_shift;
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);
        if (mid_base) write_scaler_shift(D, mid_base_scaler, r, down_inherited);

        const unsigned int mask = D.d_tipmap[tipchars[site]];
        const fp_t* Tmat = target_mat  + (size_t)r * 16;
        const fp_t* Thalf= target_mat_half + (size_t)r * 16;
        const fp_t* Smat = sibling_mat + (size_t)r * 16;
        const fp4_t Ppar = reinterpret_cast<const fp4_t*>(parent_down)[r];
        const fp4_t Pup  = reinterpret_cast<const fp4_t*>(target_up)[r];
        fp_t*       Pout = target_down + (size_t)r * 4;

        const fp_t sib0 = ((mask & 1u) ? Smat[0]  : fp_t(0)) + ((mask & 2u) ? Smat[1]  : fp_t(0)) + ((mask & 4u) ? Smat[2]  : fp_t(0)) + ((mask & 8u) ? Smat[3]  : fp_t(0));
        const fp_t sib1 = ((mask & 1u) ? Smat[4]  : fp_t(0)) + ((mask & 2u) ? Smat[5]  : fp_t(0)) + ((mask & 4u) ? Smat[6]  : fp_t(0)) + ((mask & 8u) ? Smat[7]  : fp_t(0));
        const fp_t sib2 = ((mask & 1u) ? Smat[8]  : fp_t(0)) + ((mask & 2u) ? Smat[9]  : fp_t(0)) + ((mask & 4u) ? Smat[10] : fp_t(0)) + ((mask & 8u) ? Smat[11] : fp_t(0));
        const fp_t sib3 = ((mask & 1u) ? Smat[12] : fp_t(0)) + ((mask & 2u) ? Smat[13] : fp_t(0)) + ((mask & 4u) ? Smat[14] : fp_t(0)) + ((mask & 8u) ? Smat[15] : fp_t(0));

        const fp_t p0 = Ppar.x * sib0;
        const fp_t p1 = Ppar.y * sib1;
        const fp_t p2 = Ppar.z * sib2;
        const fp_t p3 = Ppar.w * sib3;

        Pout[0] = Tmat[0] * p0 + Tmat[4] * p1 + Tmat[8]  * p2 + Tmat[12] * p3;
        Pout[1] = Tmat[1] * p0 + Tmat[5] * p1 + Tmat[9]  * p2 + Tmat[13] * p3;
        Pout[2] = Tmat[2] * p0 + Tmat[6] * p1 + Tmat[10] * p2 + Tmat[14] * p3;
        Pout[3] = Tmat[3] * p0 + Tmat[7] * p1 + Tmat[11] * p2 + Tmat[15] * p3;
        if (mid_base) {
            fp_t* Pbase = mid_base + (size_t)r * 4;
            Pbase[0] = p0;
            Pbase[1] = p1;
            Pbase[2] = p2;
            Pbase[3] = p3;
        }

        if (target_mid) {
            fp_t* Pmid = target_mid + (size_t)r * 4;
            const fp_t par0 = fp_dot4(make_fp4(Thalf[0], Thalf[4], Thalf[8], Thalf[12]), make_fp4(p0, p1, p2, p3));
            const fp_t par1 = fp_dot4(make_fp4(Thalf[1], Thalf[5], Thalf[9], Thalf[13]), make_fp4(p0, p1, p2, p3));
            const fp_t par2 = fp_dot4(make_fp4(Thalf[2], Thalf[6], Thalf[10], Thalf[14]), make_fp4(p0, p1, p2, p3));
            const fp_t par3 = fp_dot4(make_fp4(Thalf[3], Thalf[7], Thalf[11], Thalf[15]), make_fp4(p0, p1, p2, p3));

            const fp_t tgt0 = fp_dot4(make_fp4(Thalf[0], Thalf[4], Thalf[8], Thalf[12]), Pup);
            const fp_t tgt1 = fp_dot4(make_fp4(Thalf[1], Thalf[5], Thalf[9], Thalf[13]), Pup);
            const fp_t tgt2 = fp_dot4(make_fp4(Thalf[2], Thalf[6], Thalf[10], Thalf[14]), Pup);
            const fp_t tgt3 = fp_dot4(make_fp4(Thalf[3], Thalf[7], Thalf[11], Thalf[15]), Pup);

            Pmid[0] = par0 * tgt0;
            Pmid[1] = par1 * tgt1;
            Pmid[2] = par2 * tgt2;
            Pmid[3] = par3 * tgt3;
        }

        const fp_t down_max_val = fp_hmax4(Pout[0], Pout[1], Pout[2], Pout[3]);
        const unsigned int down_shift_local = threshold_scale_shift(down_max_val);
        if (down_shift_local) {
            add_scaler_shift(D, down_scaler, r, down_shift_local);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scale_clv_pow2(Pout[j], down_shift_local);
            }
        }
        if (mid_base) {
            fp_t* Pbase = mid_base + (size_t)r * 4;
            const fp_t base_max_val = fp_hmax4(Pbase[0], Pbase[1], Pbase[2], Pbase[3]);
            const unsigned int base_shift_local = threshold_scale_shift(base_max_val);
            if (base_shift_local) {
                add_scaler_shift(D, mid_base_scaler, r, base_shift_local);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_clv_pow2(Pbase[j], base_shift_local);
                }
            }
        }
        if (target_mid) {
            fp_t* Pmid = target_mid + (size_t)r * 4;
            const fp_t mid_max_val = fp_hmax4(Pmid[0], Pmid[1], Pmid[2], Pmid[3]);
            const unsigned int mid_shift_local = threshold_scale_shift(mid_max_val);
            if (mid_shift_local) {
                add_scaler_shift(D, mid_scaler, r, mid_shift_local);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_clv_pow2(Pmid[j], mid_shift_local);
                }
            }
        }
    }
}

template<int RATE_CATS>
__device__ __forceinline__ void compute_downward_tip_inner_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_tip_idx  = target_is_left ? op.left_tip_index : op.right_tip_index;
    const int target_id       = target_is_left ? op.left_id : op.right_id;
    const int sibling_id      = target_is_left ? op.right_id : op.left_id;
    if (target_tip_idx < 0 || sibling_id < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;

    const fp_t* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    const fp_t* sibling_up  = D.d_clv_up   + (size_t)sibling_id * per_node + site_off;
    fp_t*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    fp_t*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    fp_t*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    const fp_t* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down || !sibling_up || !target_up) return;

    const fp_t* target_mat  = D.d_pmat + (size_t)target_id * (size_t)RATE_CATS * 16;
    const fp_t* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16)
        : target_mat;
    const fp_t* sibling_mat = D.d_pmat + (size_t)sibling_id * (size_t)RATE_CATS * 16;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, sibling_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    unsigned int* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site);

    const unsigned char* tipchars = D.d_tipchars + (size_t)target_tip_idx * D.sites;
    const unsigned int tmask = D.d_tipmap[tipchars[site]];

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const unsigned int parent_shift = read_scaler_shift(D, parent_scaler, r);
        const unsigned int sibling_shift = read_scaler_shift(D, sibling_scaler, r);
        const unsigned int target_up_shift = read_scaler_shift(D, target_up_scaler, r);
        const unsigned int down_inherited = parent_shift + sibling_shift;
        const unsigned int mid_inherited = down_inherited + target_up_shift;
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);
        if (mid_base) write_scaler_shift(D, mid_base_scaler, r, down_inherited);

        const fp_t* Tmat  = target_mat  + (size_t)r * 16;
        const fp_t* Thalf = target_mat_half + (size_t)r * 16;
        const fp_t* Smat  = sibling_mat + (size_t)r * 16;
        const fp4_t Ppar = reinterpret_cast<const fp4_t*>(parent_down)[r];
        const fp4_t Psib = reinterpret_cast<const fp4_t*>(sibling_up)[r];
        const fp4_t Pup  = reinterpret_cast<const fp4_t*>(target_up)[r];
        fp_t*       Pout = target_down + (size_t)r * 4;

        const fp_t sib0 = fp_dot4(make_fp4(Smat[0], Smat[1], Smat[2], Smat[3]), Psib);
        const fp_t sib1 = fp_dot4(make_fp4(Smat[4], Smat[5], Smat[6], Smat[7]), Psib);
        const fp_t sib2 = fp_dot4(make_fp4(Smat[8], Smat[9], Smat[10], Smat[11]), Psib);
        const fp_t sib3 = fp_dot4(make_fp4(Smat[12], Smat[13], Smat[14], Smat[15]), Psib);

        const fp_t p0 = Ppar.x * sib0;
        const fp_t p1 = Ppar.y * sib1;
        const fp_t p2 = Ppar.z * sib2;
        const fp_t p3 = Ppar.w * sib3;


        Pout[0] = Tmat[0] * p0 + Tmat[4] * p1 + Tmat[8]  * p2 + Tmat[12] * p3;
        Pout[1] = Tmat[1] * p0 + Tmat[5] * p1 + Tmat[9]  * p2 + Tmat[13] * p3;
        Pout[2] = Tmat[2] * p0 + Tmat[6] * p1 + Tmat[10] * p2 + Tmat[14] * p3;
        Pout[3] = Tmat[3] * p0 + Tmat[7] * p1 + Tmat[11] * p2 + Tmat[15] * p3;
        if (!(tmask & 1u)) Pout[0] = 0.0;
        if (!(tmask & 2u)) Pout[1] = 0.0;
        if (!(tmask & 4u)) Pout[2] = 0.0;
        if (!(tmask & 8u)) Pout[3] = 0.0;

        if (mid_base) {
            fp_t* Pbase = mid_base + (size_t)r * 4;
            Pbase[0] = p0;
            Pbase[1] = p1;
            Pbase[2] = p2;
            Pbase[3] = p3;
        }

        if (target_mid) {
            fp_t* Pmid = target_mid + (size_t)r * 4;
            const fp_t par0 = fp_dot4(make_fp4(Thalf[0], Thalf[4], Thalf[8], Thalf[12]), make_fp4(p0, p1, p2, p3));
            const fp_t par1 = fp_dot4(make_fp4(Thalf[1], Thalf[5], Thalf[9], Thalf[13]), make_fp4(p0, p1, p2, p3));
            const fp_t par2 = fp_dot4(make_fp4(Thalf[2], Thalf[6], Thalf[10], Thalf[14]), make_fp4(p0, p1, p2, p3));
            const fp_t par3 = fp_dot4(make_fp4(Thalf[3], Thalf[7], Thalf[11], Thalf[15]), make_fp4(p0, p1, p2, p3));

            const fp_t tgt0 = fp_dot4(make_fp4(Thalf[0], Thalf[4], Thalf[8], Thalf[12]), Pup);
            const fp_t tgt1 = fp_dot4(make_fp4(Thalf[1], Thalf[5], Thalf[9], Thalf[13]), Pup);
            const fp_t tgt2 = fp_dot4(make_fp4(Thalf[2], Thalf[6], Thalf[10], Thalf[14]), Pup);
            const fp_t tgt3 = fp_dot4(make_fp4(Thalf[3], Thalf[7], Thalf[11], Thalf[15]), Pup);

            Pmid[0] = par0 * tgt0;
            Pmid[1] = par1 * tgt1;
            Pmid[2] = par2 * tgt2;
            Pmid[3] = par3 * tgt3;
            if (!(tmask & 1u)) Pmid[0] = 0.0;
            if (!(tmask & 2u)) Pmid[1] = 0.0;
            if (!(tmask & 4u)) Pmid[2] = 0.0;
            if (!(tmask & 8u)) Pmid[3] = 0.0;
        }

        const fp_t down_max_val = fp_hmax4(Pout[0], Pout[1], Pout[2], Pout[3]);
        const unsigned int down_shift_local = threshold_scale_shift(down_max_val);
        if (down_shift_local) {
            add_scaler_shift(D, down_scaler, r, down_shift_local);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scale_clv_pow2(Pout[j], down_shift_local);
            }
        }
        if (mid_base) {
            fp_t* Pbase = mid_base + (size_t)r * 4;
            const fp_t base_max_val = fp_hmax4(Pbase[0], Pbase[1], Pbase[2], Pbase[3]);
            const unsigned int base_shift_local = threshold_scale_shift(base_max_val);
            if (base_shift_local) {
                add_scaler_shift(D, mid_base_scaler, r, base_shift_local);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_clv_pow2(Pbase[j], base_shift_local);
                }
            }
        }
        if (target_mid) {
            fp_t* Pmid = target_mid + (size_t)r * 4;
            const fp_t mid_max_val = fp_hmax4(Pmid[0], Pmid[1], Pmid[2], Pmid[3]);
            const unsigned int mid_shift_local = threshold_scale_shift(mid_max_val);
            if (mid_shift_local) {
                add_scaler_shift(D, mid_scaler, r, mid_shift_local);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_clv_pow2(Pmid[j], mid_shift_local);
                }
            }
        }
    }
}

// target tip, sibling tip (states=4, rate-specific)
template<int RATE_CATS>
__device__ __forceinline__ void compute_downward_tip_tip_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_tip_idx  = target_is_left ? op.left_tip_index : op.right_tip_index;
    const int sibling_tip_idx = target_is_left ? op.right_tip_index : op.left_tip_index;
    const int target_id       = target_is_left ? op.left_id : op.right_id;
    if (target_tip_idx < 0 || sibling_tip_idx < 0 || target_id < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;

    const fp_t* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    fp_t*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    fp_t*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    fp_t*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    const fp_t* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down || !target_up) return;

    const fp_t* target_mat  = D.d_pmat + (size_t)target_id * (size_t)RATE_CATS * 16;
    const fp_t* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16)
        : target_mat;
    const fp_t* sibling_mat = D.d_pmat + (size_t)(target_is_left ? op.right_id : op.left_id) * (size_t)RATE_CATS * 16;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, target_is_left ? op.right_id : op.left_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    unsigned int* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site);

    const unsigned char* tipchars = D.d_tipchars + (size_t)sibling_tip_idx * D.sites;

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);
        if (mid_base) write_scaler_shift(D, mid_base_scaler, r, down_inherited);

        const unsigned int mask = D.d_tipmap[tipchars[site]];
        const fp_t* Tmat  = target_mat  + (size_t)r * 16;
        const fp_t* Thalf = target_mat_half + (size_t)r * 16;
        const fp_t* Smat  = sibling_mat + (size_t)r * 16;
        const fp4_t Ppar = reinterpret_cast<const fp4_t*>(parent_down)[r];
        const fp4_t Pup  = reinterpret_cast<const fp4_t*>(target_up)[r];
        fp_t*       Pout = target_down + (size_t)r * 4;

        const fp_t sib0 = ((mask & 1u) ? Smat[0]  : fp_t(0)) + ((mask & 2u) ? Smat[1]  : fp_t(0)) + ((mask & 4u) ? Smat[2]  : fp_t(0)) + ((mask & 8u) ? Smat[3]  : fp_t(0));
        const fp_t sib1 = ((mask & 1u) ? Smat[4]  : fp_t(0)) + ((mask & 2u) ? Smat[5]  : fp_t(0)) + ((mask & 4u) ? Smat[6]  : fp_t(0)) + ((mask & 8u) ? Smat[7]  : fp_t(0));
        const fp_t sib2 = ((mask & 1u) ? Smat[8]  : fp_t(0)) + ((mask & 2u) ? Smat[9]  : fp_t(0)) + ((mask & 4u) ? Smat[10] : fp_t(0)) + ((mask & 8u) ? Smat[11] : fp_t(0));
        const fp_t sib3 = ((mask & 1u) ? Smat[12] : fp_t(0)) + ((mask & 2u) ? Smat[13] : fp_t(0)) + ((mask & 4u) ? Smat[14] : fp_t(0)) + ((mask & 8u) ? Smat[15] : fp_t(0));

        const fp_t p0 = Ppar.x * sib0;
        const fp_t p1 = Ppar.y * sib1;
        const fp_t p2 = Ppar.z * sib2;
        const fp_t p3 = Ppar.w * sib3;

        Pout[0] = Tmat[0] * p0 + Tmat[4] * p1 + Tmat[8]  * p2 + Tmat[12] * p3;
        Pout[1] = Tmat[1] * p0 + Tmat[5] * p1 + Tmat[9]  * p2 + Tmat[13] * p3;
        Pout[2] = Tmat[2] * p0 + Tmat[6] * p1 + Tmat[10] * p2 + Tmat[14] * p3;
        Pout[3] = Tmat[3] * p0 + Tmat[7] * p1 + Tmat[11] * p2 + Tmat[15] * p3;

        if (mid_base) {
            fp_t* Pbase = mid_base + (size_t)r * 4;
            Pbase[0] = p0;
            Pbase[1] = p1;
            Pbase[2] = p2;
            Pbase[3] = p3;
        }

        if (target_mid) {
            fp_t* Pmid = target_mid + (size_t)r * 4;
            const fp_t par0 = fp_dot4(make_fp4(Thalf[0], Thalf[4], Thalf[8], Thalf[12]), make_fp4(p0, p1, p2, p3));
            const fp_t par1 = fp_dot4(make_fp4(Thalf[1], Thalf[5], Thalf[9], Thalf[13]), make_fp4(p0, p1, p2, p3));
            const fp_t par2 = fp_dot4(make_fp4(Thalf[2], Thalf[6], Thalf[10], Thalf[14]), make_fp4(p0, p1, p2, p3));
            const fp_t par3 = fp_dot4(make_fp4(Thalf[3], Thalf[7], Thalf[11], Thalf[15]), make_fp4(p0, p1, p2, p3));

            const fp_t tgt0 = fp_dot4(make_fp4(Thalf[0], Thalf[4], Thalf[8], Thalf[12]), Pup);
            const fp_t tgt1 = fp_dot4(make_fp4(Thalf[1], Thalf[5], Thalf[9], Thalf[13]), Pup);
            const fp_t tgt2 = fp_dot4(make_fp4(Thalf[2], Thalf[6], Thalf[10], Thalf[14]), Pup);
            const fp_t tgt3 = fp_dot4(make_fp4(Thalf[3], Thalf[7], Thalf[11], Thalf[15]), Pup);

            Pmid[0] = par0 * tgt0;
            Pmid[1] = par1 * tgt1;
            Pmid[2] = par2 * tgt2;
            Pmid[3] = par3 * tgt3;
        }

        const fp_t down_max_val = fp_hmax4(Pout[0], Pout[1], Pout[2], Pout[3]);
        const unsigned int down_shift_local = threshold_scale_shift(down_max_val);
        if (down_shift_local) {
            add_scaler_shift(D, down_scaler, r, down_shift_local);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scale_clv_pow2(Pout[j], down_shift_local);
            }
        }
        if (mid_base) {
            fp_t* Pbase = mid_base + (size_t)r * 4;
            const fp_t base_max_val = fp_hmax4(Pbase[0], Pbase[1], Pbase[2], Pbase[3]);
            const unsigned int base_shift_local = threshold_scale_shift(base_max_val);
            if (base_shift_local) {
                add_scaler_shift(D, mid_base_scaler, r, base_shift_local);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_clv_pow2(Pbase[j], base_shift_local);
                }
            }
        }
        if (target_mid) {
            fp_t* Pmid = target_mid + (size_t)r * 4;
            const fp_t mid_max_val = fp_hmax4(Pmid[0], Pmid[1], Pmid[2], Pmid[3]);
            const unsigned int mid_shift_local = threshold_scale_shift(mid_max_val);
            if (mid_shift_local) {
                add_scaler_shift(D, mid_scaler, r, mid_shift_local);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_clv_pow2(Pmid[j], mid_shift_local);
                }
            }
        }
    }
}

// ===== Per-site computations =====
template<int RATE_CATS>
__device__ __forceinline__ void compute_tip_tip_site_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    // Compute parent CLV from two tips directly (no lookup).
    const size_t span     = (size_t)4 * RATE_CATS;
    const size_t per_node = per_node_span(D);

    const unsigned char* left_tip  = D.d_tipchars + (size_t)op.left_tip_index  * D.sites;
    const unsigned char* right_tip = D.d_tipchars + (size_t)op.right_tip_index * D.sites;

    const unsigned int lmask = D.d_tipmap[left_tip[site]];
    const unsigned int rmask = D.d_tipmap[right_tip[site]];

    // Ensure tip nodes have CLV-up initialized for downstream use.
    const size_t site_off = (size_t)site * span;
    if (op.left_id >= 0 && D.d_clv_up) {
        fp_t* lclv = D.d_clv_up + (size_t)op.left_id * per_node;
        unsigned int* lscaler = up_scaler_ptr(D, op.left_id, site);
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            write_scaler_shift(D, lscaler, r, 0u);
            fp_t* out = lclv + site_off + (size_t)r * 4;
            out[0] = (lmask & 1u) ? fp_t(1) : fp_t(0);
            out[1] = (lmask & 2u) ? fp_t(1) : fp_t(0);
            out[2] = (lmask & 4u) ? fp_t(1) : fp_t(0);
            out[3] = (lmask & 8u) ? fp_t(1) : fp_t(0);
        }
    }
    if (op.right_id >= 0 && D.d_clv_up) {
        fp_t* rclv = D.d_clv_up + (size_t)op.right_id * per_node;
        unsigned int* rscaler = up_scaler_ptr(D, op.right_id, site);
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            write_scaler_shift(D, rscaler, r, 0u);
            fp_t* out = rclv + site_off + (size_t)r * 4;
            out[0] = (rmask & 1u) ? fp_t(1) : fp_t(0);
            out[1] = (rmask & 2u) ? fp_t(1) : fp_t(0);
            out[2] = (rmask & 4u) ? fp_t(1) : fp_t(0);
            out[3] = (rmask & 8u) ? fp_t(1) : fp_t(0);
        }
    }

    const size_t parent_off = (size_t)op.parent_id * per_node + (size_t)site * span;
    const fp_t* Lbase = D.d_pmat + (size_t)op.left_id  * RATE_CATS * 4 * 4;
    const fp_t* Rbase = D.d_pmat + (size_t)op.right_id * RATE_CATS * 4 * 4;
    fp_t* parent_pool = clv_write_pool_base<fp_t>(D, op);
    if (!parent_pool) return; // placeholder until preorder input logic is defined

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, RATE_CATS);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        write_scaler_shift(D, site_scaler_ptr, r, 0u);
        const fp_t* Lmat = Lbase + (size_t)r * 16;
        const fp_t* Rmat = Rbase + (size_t)r * 16;
        fp_t* pout = parent_pool + parent_off + (size_t)r * 4;

        fp_t maxv = fp_t(0);
        // parent state j
        for (int j = 0; j < 4; ++j) {
            fp_t left_term = fp_t(0);
            fp_t right_term = fp_t(0);
            // sum over allowed tip states
            if (lmask & 1u) left_term  += Lmat[j * 4 + 0];
            if (lmask & 2u) left_term  += Lmat[j * 4 + 1];
            if (lmask & 4u) left_term  += Lmat[j * 4 + 2];
            if (lmask & 8u) left_term  += Lmat[j * 4 + 3];

            if (rmask & 1u) right_term += Rmat[j * 4 + 0];
            if (rmask & 2u) right_term += Rmat[j * 4 + 1];
            if (rmask & 4u) right_term += Rmat[j * 4 + 2];
            if (rmask & 8u) right_term += Rmat[j * 4 + 3];

            fp_t v = left_term * right_term;
            pout[j] = v;
            if (v > maxv) maxv = v;
        }

        if (site_scaler_ptr) {
            unsigned int shift = threshold_scale_shift(maxv);
            if (shift) {
                add_scaler_shift(D, site_scaler_ptr, r, shift);

                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    scale_clv_pow2(pout[s], shift);
                }
            }
        }

    }
}

template<int RATE_CATS>
__device__ __forceinline__ void compute_tip_tip_site_ratecat_nolookup(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    // on-the-fly helper (no lookup table)
    const size_t span     = (size_t)4 * RATE_CATS;
    const size_t per_node = per_node_span(D);

    const unsigned char* left_tip  = D.d_tipchars + (size_t)op.left_tip_index  * D.sites;
    const unsigned char* right_tip = D.d_tipchars + (size_t)op.right_tip_index * D.sites;

    const unsigned int j = (unsigned int)left_tip[site];
    const unsigned int k = (unsigned int)right_tip[site];

    const unsigned int jmask_base = D.d_tipmap[j];
    const unsigned int kmask_base = D.d_tipmap[k];

    const fp_t* __restrict__ jmat_base =
        D.d_pmat + (size_t)op.left_id  * RATE_CATS * 4 * 4;
    const fp_t* __restrict__ kmat_base =
        D.d_pmat + (size_t)op.right_id * RATE_CATS * 4 * 4;

    // Ensure tip nodes have CLV-up initialized for downstream use.
    const size_t site_off = (size_t)site * span;
    if (op.left_id >= 0 && D.d_clv_up) {
        fp_t* lclv = D.d_clv_up + (size_t)op.left_id * per_node;
        unsigned int* lscaler = up_scaler_ptr(D, op.left_id, site);
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            write_scaler_shift(D, lscaler, r, 0u);
            fp_t* out = lclv + site_off + (size_t)r * 4;
            unsigned int m = jmask_base;
            out[0] = (m & 1u) ? fp_t(1) : fp_t(0);
            out[1] = (m & 2u) ? fp_t(1) : fp_t(0);
            out[2] = (m & 4u) ? fp_t(1) : fp_t(0);
            out[3] = (m & 8u) ? fp_t(1) : fp_t(0);
        }
    }
    if (op.right_id >= 0 && D.d_clv_up) {
        fp_t* rclv = D.d_clv_up + (size_t)op.right_id * per_node;
        unsigned int* rscaler = up_scaler_ptr(D, op.right_id, site);
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            write_scaler_shift(D, rscaler, r, 0u);
            fp_t* out = rclv + site_off + (size_t)r * 4;
            unsigned int m = kmask_base;
            out[0] = (m & 1u) ? fp_t(1) : fp_t(0);
            out[1] = (m & 2u) ? fp_t(1) : fp_t(0);
            out[2] = (m & 4u) ? fp_t(1) : fp_t(0);
            out[3] = (m & 8u) ? fp_t(1) : fp_t(0);
        }
    }

    const size_t parent_off = (size_t)op.parent_id * per_node + (size_t)site * span;
    fp_t* parent_pool = clv_write_pool_base<fp_t>(D, op);
    if (!parent_pool) return; // placeholder until preorder input logic is defined
    fp_t* __restrict__ dst = parent_pool + parent_off;

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, RATE_CATS);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        write_scaler_shift(D, site_scaler_ptr, r, 0u);
        const fp_t* __restrict__ jmat = jmat_base + (size_t)r * 4 * 4;
        const fp_t* __restrict__ kmat = kmat_base + (size_t)r * 4 * 4;
        fp_t* __restrict__ Pout = dst + (size_t)r * 4;

        fp_t col_scale_max_val = fp_t(0);

        const fp_t* Lrow = jmat;
        const fp_t* Rrow = kmat;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            fp_t termj = fp_t(0);
            fp_t termk = fp_t(0);
            unsigned int jmask = jmask_base;
            unsigned int kmask = kmask_base;

            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (jmask & 1u) termj += Lrow[m];
                if (kmask & 1u) termk += Rrow[m];
                jmask >>= 1;
                kmask >>= 1;
            }

            Pout[i] = termj * termk;
            if (Pout[i] > col_scale_max_val) col_scale_max_val = Pout[i];

            Lrow += 4;
            Rrow += 4;
        }

        if (site_scaler_ptr) {
            fp_t* pout = Pout;
            fp_t maxv = fp_hmax4(pout[0], pout[1], pout[2], pout[3]);
            unsigned int shift = threshold_scale_shift(maxv);
            if (shift) {
                add_scaler_shift(D, site_scaler_ptr, r, shift);

                #pragma unroll
                for (int s = 0; s < 4; ++s)
                    scale_clv_pow2(pout[s], shift);
            }
        }

    }
}

__device__ __forceinline__ void compute_tip_tip_site_4_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const size_t span     = (size_t)4 * (size_t)D.rate_cats;
    const size_t per_node = per_node_span(D);

    const unsigned char* left_tip  = D.d_tipchars + (size_t)op.left_tip_index  * D.sites;
    const unsigned char* right_tip = D.d_tipchars + (size_t)op.right_tip_index * D.sites;

    const unsigned int j = (unsigned int)left_tip[site];
    const unsigned int k = (unsigned int)right_tip[site];

    const unsigned int jmask_base = D.d_tipmap[j];
    const unsigned int kmask_base = D.d_tipmap[k];

    const fp_t* __restrict__ jmat_base =
        D.d_pmat + (size_t)op.left_id  * D.rate_cats * 4 * 4;
    const fp_t* __restrict__ kmat_base =
        D.d_pmat + (size_t)op.right_id * D.rate_cats * 4 * 4;

    // Ensure tip nodes have CLV-up initialized for downstream use.
    const size_t site_off = (size_t)site * span;
    if (op.left_id >= 0 && D.d_clv_up) {
        fp_t* lclv = D.d_clv_up + (size_t)op.left_id * per_node;
        unsigned int* lscaler = up_scaler_ptr(D, op.left_id, site);
        for (int r = 0; r < D.rate_cats; ++r) {
            write_scaler_shift(D, lscaler, r, 0u);
            fp_t* out = lclv + site_off + (size_t)r * 4;
            unsigned int m = jmask_base;
            out[0] = (m & 1u) ? fp_t(1) : fp_t(0);
            out[1] = (m & 2u) ? fp_t(1) : fp_t(0);
            out[2] = (m & 4u) ? fp_t(1) : fp_t(0);
            out[3] = (m & 8u) ? fp_t(1) : fp_t(0);
        }
    }
    if (op.right_id >= 0 && D.d_clv_up) {
        fp_t* rclv = D.d_clv_up + (size_t)op.right_id * per_node;
        unsigned int* rscaler = up_scaler_ptr(D, op.right_id, site);
        for (int r = 0; r < D.rate_cats; ++r) {
            write_scaler_shift(D, rscaler, r, 0u);
            fp_t* out = rclv + site_off + (size_t)r * 4;
            unsigned int m = kmask_base;
            out[0] = (m & 1u) ? fp_t(1) : fp_t(0);
            out[1] = (m & 2u) ? fp_t(1) : fp_t(0);
            out[2] = (m & 4u) ? fp_t(1) : fp_t(0);
            out[3] = (m & 8u) ? fp_t(1) : fp_t(0);
        }
    }

    const size_t parent_off = (size_t)op.parent_id * per_node + (size_t)site * span;
    fp_t* parent_pool = clv_write_pool_base<fp_t>(D, op);
    if (!parent_pool) return; // placeholder until preorder input logic is defined
    fp_t* __restrict__ dst = parent_pool + parent_off;

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, (unsigned int)D.rate_cats);

    for (int r = 0; r < D.rate_cats; ++r) {
        write_scaler_shift(D, site_scaler_ptr, r, 0u);
        const fp_t* __restrict__ jmat = jmat_base + (size_t)r * 4 * 4;
        const fp_t* __restrict__ kmat = kmat_base + (size_t)r * 4 * 4;
        fp_t* __restrict__ Pout = dst + (size_t)r * 4;

        fp_t col_scale_max_val = fp_t(0);

        const fp_t* Lrow = jmat;
        const fp_t* Rrow = kmat;
        for (int i = 0; i < 4; ++i) {
            fp_t termj = fp_t(0);
            fp_t termk = fp_t(0);
            unsigned int jmask = jmask_base;
            unsigned int kmask = kmask_base;

            for (int m = 0; m < 4; ++m) {
                if (jmask & 1u) termj += Lrow[m];
                if (kmask & 1u) termk += Rrow[m];
                jmask >>= 1;
                kmask >>= 1;
            }

            Pout[i] = termj * termk;
            if (Pout[i] > col_scale_max_val) col_scale_max_val = Pout[i];

            Lrow += 4;
            Rrow += 4;
        }

        if (site_scaler_ptr) {
            fp_t maxv = fp_hmax4(Pout[0], Pout[1], Pout[2], Pout[3]);
            unsigned int shift = threshold_scale_shift(maxv);
            if (shift) {
                site_scaler_ptr[r] += shift;
                for (int s = 0; s < 4; ++s) {
                    scale_clv_pow2(Pout[s], shift);
                }
            }
        }

    }
}

__device__ __forceinline__ void compute_tip_tip_site_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const unsigned int states = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t span     = (size_t)states * rate_cats;
    const size_t per_node = per_node_span(D);

    const unsigned char* left_tip  = D.d_tipchars + (size_t)op.left_tip_index  * D.sites;
    const unsigned char* right_tip = D.d_tipchars + (size_t)op.right_tip_index * D.sites;

    const unsigned int lmask = D.d_tipmap[left_tip[site]];
    const unsigned int rmask = D.d_tipmap[right_tip[site]];

    // Ensure tip nodes have CLV-up initialized for downstream use.
    const size_t site_off = (size_t)site * span;
    if (op.left_id >= 0 && D.d_clv_up) {
        fp_t* lclv = D.d_clv_up + (size_t)op.left_id * per_node;
        unsigned int* lscaler = up_scaler_ptr(D, op.left_id, site);
        for (unsigned int r = 0; r < rate_cats; ++r) {
            write_scaler_shift(D, lscaler, r, 0u);
            fp_t* out = lclv + site_off + (size_t)r * states;
            for (unsigned int s = 0; s < states; ++s) {
                out[s] = (lmask & (1u << s)) ? fp_t(1) : fp_t(0);
            }
        }
    }
    if (op.right_id >= 0 && D.d_clv_up) {
        fp_t* rclv = D.d_clv_up + (size_t)op.right_id * per_node;
        unsigned int* rscaler = up_scaler_ptr(D, op.right_id, site);
        for (unsigned int r = 0; r < rate_cats; ++r) {
            write_scaler_shift(D, rscaler, r, 0u);
            fp_t* out = rclv + site_off + (size_t)r * states;
            for (unsigned int s = 0; s < states; ++s) {
                out[s] = (rmask & (1u << s)) ? fp_t(1) : fp_t(0);
            }
        }
    }

    const fp_t* Lbase = D.d_pmat + (size_t)op.left_id  * rate_cats * states * states;
    const fp_t* Rbase = D.d_pmat + (size_t)op.right_id * rate_cats * states * states;

    const size_t dst_off = (size_t)op.parent_id * per_node + (size_t)site * span;
    fp_t* parent_pool = clv_write_pool_base<fp_t>(D, op);
    if (!parent_pool) return; // placeholder until preorder input logic is defined
    fp_t* Pout = parent_pool + dst_off;

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, rate_cats);

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const fp_t* Lmat = Lbase + (size_t)r * states * states;
        const fp_t* Rmat = Rbase + (size_t)r * states * states;
        fp_t* out_r = Pout + (size_t)r * states;

        fp_t maxv = fp_t(0);
        for (unsigned int j = 0; j < states; ++j) {
            fp_t left_term = fp_t(0);
            fp_t right_term = fp_t(0);
            for (unsigned int k = 0; k < states; ++k) {
                if (lmask & (1u << k)) left_term  += Lmat[j * states + k];
                if (rmask & (1u << k)) right_term += Rmat[j * states + k];
            }
            fp_t v = left_term * right_term;
            out_r[j] = v;
            if (v > maxv) maxv = v;
        }

        if (site_scaler_ptr) {
            unsigned int shift = threshold_scale_shift(maxv);
            if (shift) {
                site_scaler_ptr[r] += shift;
                for (unsigned int s = 0; s < states; ++s) {
                    scale_clv_pow2(out_r[s], shift);
                }
            }
        }
    }
}

template<int RATE_CATS>
__device__ __forceinline__ void compute_tip_inner_site_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const size_t span     = (size_t)4 * RATE_CATS;
    const size_t per_node = per_node_span(D);

    const bool tip_on_left = op.left_tip_index >= 0;
    const int  tip_index   = tip_on_left ? op.left_tip_index  : op.right_tip_index;
    const int  inner_id    = tip_on_left ? op.right_id : op.left_id;
    const int  tip_node_id = tip_on_left ? op.left_id  : op.right_id;

    const unsigned char* d_left_tip = D.d_tipchars + (size_t)tip_index * D.sites;
    const fp_t* d_right_clv = clv_read_ptr_for_node<const fp_t>(D, op, inner_id);
    fp_t* parent_clv = clv_write_ptr_for_node<fp_t>(D, op, op.parent_id);
    if (!d_right_clv || !parent_clv) return; // placeholder until preorder input logic is defined

    const fp_t* d_Lmat = D.d_pmat + (size_t)tip_node_id * RATE_CATS * 4 * 4;
    const fp_t* d_Rmat = D.d_pmat + (size_t)inner_id * RATE_CATS * 4 * 4;

    const size_t site_off = (size_t)site * span;
    const unsigned int tmask = D.d_tipmap[d_left_tip[site]];

    // Write tip CLV into UP pool for downstream use.
    if (D.d_clv_up && tip_node_id >= 0) {
        fp_t* tip_up = D.d_clv_up + (size_t)tip_node_id * per_node;
        unsigned int* tip_scaler = up_scaler_ptr(D, tip_node_id, site);
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            write_scaler_shift(D, tip_scaler, r, 0u);
            fp_t* out = tip_up + site_off + (size_t)r * 4;
            out[0] = (tmask & 1u) ? fp_t(1) : fp_t(0);
            out[1] = (tmask & 2u) ? fp_t(1) : fp_t(0);
            out[2] = (tmask & 4u) ? fp_t(1) : fp_t(0);
            out[3] = (tmask & 8u) ? fp_t(1) : fp_t(0);
        }
    }

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, RATE_CATS);
    unsigned int* inner_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, inner_id, site);

    for (int r = 0; r < RATE_CATS; ++r) {
        write_scaler_shift(D, site_scaler_ptr, r, read_scaler_shift(D, inner_scaler, r));
        const fp_t* Lmat = d_Lmat + (size_t)r * 4 * 4;
        const fp_t* Rmat = d_Rmat + (size_t)r * 4 * 4;
        const fp_t* Rclv = d_right_clv + site_off + (size_t)r * 4;
        fp_t* Pout = parent_clv + site_off + (size_t)r * 4;
        fp_t col_scale_max_val = fp_t(0);

        const fp_t r0 = Rclv[0];
        const fp_t r1 = Rclv[1];
        const fp_t r2 = Rclv[2];
        const fp_t r3 = Rclv[3];

        const fp_t* Lrow = Lmat;
        const fp_t* Rrow = Rmat;
        for (int i = 0; i < 4; ++i) {
            fp_t lefterm = fp_t(0);
            unsigned int lstate = tmask;
            if (lstate & 1u) lefterm += Lrow[0];
            if (lstate & 2u) lefterm += Lrow[1];
            if (lstate & 4u) lefterm += Lrow[2];
            if (lstate & 8u) lefterm += Lrow[3];

            fp_t righterm = fp_dot4(make_fp4(Rrow[0], Rrow[1], Rrow[2], Rrow[3]), make_fp4(r0, r1, r2, r3));
            Pout[i] = lefterm * righterm;
            if (Pout[i] > col_scale_max_val) col_scale_max_val = Pout[i];
            Lrow += 4;
            Rrow += 4;
        }

        if (site_scaler_ptr) {
            unsigned int shift = threshold_scale_shift(col_scale_max_val);
            if (shift) {
                add_scaler_shift(D, site_scaler_ptr, r, shift);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    scale_clv_pow2(Pout[i], shift);
                }
            }
        }

    }
}

__device__ __forceinline__ void compute_tip_inner_site_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const unsigned int states = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t span     = (size_t)states * rate_cats;
    const size_t per_node = per_node_span(D);

    const bool tip_on_left = op.left_tip_index >= 0;
    const int  tip_index   = tip_on_left ? op.left_tip_index  : op.right_tip_index;
    const int  inner_id    = tip_on_left ? op.right_id : op.left_id;
    const int  tip_node_id = tip_on_left ? op.left_id  : op.right_id;

    const unsigned char* d_left_tip = D.d_tipchars + (size_t)tip_index * D.sites;
    const fp_t* d_right_clv = clv_read_ptr_for_node<const fp_t>(D, op, inner_id);
    fp_t* parent_clv = clv_write_ptr_for_node<fp_t>(D, op, op.parent_id);
    if (!d_right_clv || !parent_clv) return; // placeholder until preorder input logic is defined

    const fp_t* d_Lmat = D.d_pmat + (size_t)tip_node_id * D.rate_cats * states * states;
    const fp_t* d_Rmat = D.d_pmat + (size_t)inner_id * D.rate_cats * states * states;

    const size_t site_off = (size_t)site * span;
    const unsigned int tmask = D.d_tipmap[d_left_tip[site]];

    // Write tip CLV into UP pool for downstream use.
    if (D.d_clv_up && tip_node_id >= 0) {
        fp_t* tip_up = D.d_clv_up + (size_t)tip_node_id * per_node;
        unsigned int* tip_scaler = up_scaler_ptr(D, tip_node_id, site);
        for (unsigned int r = 0; r < rate_cats; ++r) {
            write_scaler_shift(D, tip_scaler, r, 0u);
            fp_t* out = tip_up + site_off + (size_t)r * states;
            for (unsigned int s = 0; s < states; ++s) {
                out[s] = (tmask & (1u << s)) ? fp_t(1) : fp_t(0);
            }
        }
    }

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, rate_cats);
    unsigned int* inner_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, inner_id, site);

    for (unsigned int r = 0; r < rate_cats; ++r) {
        write_scaler_shift(D, site_scaler_ptr, r, read_scaler_shift(D, inner_scaler, r));
        fp_t col_scale_max_val = fp_t(0);
        const fp_t* Lmat = d_Lmat + (size_t)r * states * states;
        const fp_t* Rmat = d_Rmat + (size_t)r * states * states;
        const fp_t* Rclv = d_right_clv + site_off + (size_t)r * states;
        fp_t* Pout = parent_clv + site_off + (size_t)r * states;

        const fp_t* Lrow = Lmat;
        const fp_t* Rrow = Rmat;
        for (unsigned int i = 0; i < states; ++i) {
            fp_t lefterm = fp_t(0), righterm = fp_t(0);
            unsigned int lstate = tmask;
            for (unsigned int j = 0; j < states; ++j) {
                if (lstate & 1u) lefterm += Lrow[j];
                righterm += Rrow[j] * Rclv[j];
                lstate >>= 1;
            }
            Pout[i] = lefterm * righterm;
            if (Pout[i] > col_scale_max_val) col_scale_max_val = Pout[i];
            Lrow += states;
            Rrow += states;
        }
        if (site_scaler_ptr) {
            unsigned int shift = threshold_scale_shift(col_scale_max_val);
            if (shift) {
                site_scaler_ptr[r] += shift;
                for (unsigned int i = 0; i < states; ++i) {
                    scale_clv_pow2(Pout[i], shift);
                }
            }
        }
    }
}

template<int RATE_CATS>
__device__ __forceinline__ void compute_inner_inner_site_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const size_t span     = (size_t)RATE_CATS * 4;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * span;

    const fp_t* d_left_clv  = clv_read_ptr_for_node<const fp_t>(D, op, op.left_id);
    const fp_t* d_right_clv = clv_read_ptr_for_node<const fp_t>(D, op, op.right_id);

    fp_t* parent_clv = clv_write_ptr_for_node<fp_t>(D, op, op.parent_id);
    if (!d_left_clv || !d_right_clv || !parent_clv) return; // placeholder until preorder input logic is defined
    const fp_t* d_left_mat  = D.d_pmat + (size_t)op.left_id  * RATE_CATS * 4 * 4;
    const fp_t* d_right_mat = D.d_pmat + (size_t)op.right_id * RATE_CATS * 4 * 4;

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, RATE_CATS);
    unsigned int* left_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, op.left_id, site);
    unsigned int* right_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, op.right_id, site);

    for (int r = 0; r < RATE_CATS; ++r) {
        write_scaler_shift(
            D,
            site_scaler_ptr,
            r,
            read_scaler_shift(D, left_scaler, r) +
            read_scaler_shift(D, right_scaler, r));
        const fp_t* Lclv = d_left_clv  + site_off + (size_t)r * 4;
        const fp_t* Rclv = d_right_clv + site_off + (size_t)r * 4;

        

        const fp_t* Lmat = d_left_mat  + (size_t)r * 4 * 4;
        const fp_t* Rmat = d_right_mat + (size_t)r * 4 * 4;

        fp_t* Pout = parent_clv + site_off + (size_t)r * 4;
        fp_t col_scale_max_val = fp_t(0);

        const fp_t l0 = Lclv[0];
        const fp_t l1 = Lclv[1];
        const fp_t l2 = Lclv[2];
        const fp_t l3 = Lclv[3];
        const fp_t r0 = Rclv[0];
        const fp_t r1 = Rclv[1];
        const fp_t r2 = Rclv[2];
        const fp_t r3 = Rclv[3];

        const fp_t lt0 = fp_dot4(make_fp4(Lmat[0], Lmat[1], Lmat[2], Lmat[3]), make_fp4(l0, l1, l2, l3));
        const fp_t lt1 = fp_dot4(make_fp4(Lmat[4], Lmat[5], Lmat[6], Lmat[7]), make_fp4(l0, l1, l2, l3));
        const fp_t lt2 = fp_dot4(make_fp4(Lmat[8], Lmat[9], Lmat[10], Lmat[11]), make_fp4(l0, l1, l2, l3));
        const fp_t lt3 = fp_dot4(make_fp4(Lmat[12], Lmat[13], Lmat[14], Lmat[15]), make_fp4(l0, l1, l2, l3));

        const fp_t rt0 = fp_dot4(make_fp4(Rmat[0], Rmat[1], Rmat[2], Rmat[3]), make_fp4(r0, r1, r2, r3));
        const fp_t rt1 = fp_dot4(make_fp4(Rmat[4], Rmat[5], Rmat[6], Rmat[7]), make_fp4(r0, r1, r2, r3));
        const fp_t rt2 = fp_dot4(make_fp4(Rmat[8], Rmat[9], Rmat[10], Rmat[11]), make_fp4(r0, r1, r2, r3));
        const fp_t rt3 = fp_dot4(make_fp4(Rmat[12], Rmat[13], Rmat[14], Rmat[15]), make_fp4(r0, r1, r2, r3));

        Pout[0] = lt0 * rt0;
        Pout[1] = lt1 * rt1;
        Pout[2] = lt2 * rt2;
        Pout[3] = lt3 * rt3;

        col_scale_max_val = fp_hmax4(Pout[0], Pout[1], Pout[2], Pout[3]);

        if (site_scaler_ptr) {
            unsigned int shift = threshold_scale_shift(col_scale_max_val);
            if (shift) {
                add_scaler_shift(D, site_scaler_ptr, r, shift);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_clv_pow2(Pout[j], shift);
                }
            }
        }

    }
    
}

template<int RATE_CATS>
__device__ __forceinline__ void load_midpoint_pmat_pair_ratecat(
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

// Midpoint helper for down pass (states=4): parent.down + sibling.up -> mid CLV.
template<int RATE_CATS>
__device__ void compute_midpoint_inner_inner_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site,
    bool proximal_mode,
    int op_pmat_idx,
    bool active_thread,
    fp_t* shared_target_mat,
    fp_t* shared_parent_mat)
{
    if (!D.d_clv_mid) return;
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;

    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id  = target_is_left ? op.left_id  : op.right_id;
    if (op.parent_id < 0 || target_id < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;

    fp_t*         target_mid = D.d_clv_mid + (size_t)target_id * per_node + site_off;
    const fp_t*   mid_base    = D.d_clv_mid_base
        ? D.d_clv_mid_base + (size_t)target_id * per_node + site_off
        : nullptr;
    // proximal_mode uses query CLV as the "upper" side; pendant uses target_up.
    const fp_t* target_up  = proximal_mode
        ? (D.d_query_clv ? (D.d_query_clv + site_off) : nullptr)
        : (D.d_clv_up    ? (D.d_clv_up   + (size_t)target_id * per_node + site_off) : nullptr);
    if (!target_up) return;
    const fp_t* target_mat = nullptr;
    if (proximal_mode && D.d_query_pmat) {
        target_mat = D.d_query_pmat + (size_t)op_pmat_idx * (size_t)RATE_CATS * 16;
    } else if (D.d_pmat_mid_prox) {
        target_mat = D.d_pmat_mid_prox + (size_t)target_id * (size_t)RATE_CATS * 16;
    } else if (D.d_pmat_mid) {
        target_mat = D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16;
    } else {
        // Fall back to half-branch pmats at minimum; avoid full-length pmats here.
        target_mat = D.d_pmat_mid ? (D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16) : nullptr;
    }
    const fp_t* parent_mat = nullptr;
    if (D.d_pmat_mid_dist) {
        parent_mat = D.d_pmat_mid_dist + (size_t)target_id * (size_t)RATE_CATS * 16;
    } else if (D.d_pmat_mid) {
        parent_mat = D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16;
    } else {
        // Avoid using full-length pmats on parent side in proximal mode.
        parent_mat = nullptr;
    }
    if (!target_mat || !parent_mat || !mid_base) return;

    load_midpoint_pmat_pair_ratecat<RATE_CATS>(
        shared_target_mat,
        shared_parent_mat,
        target_mat,
        parent_mat);
    if (!active_thread) return;

    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    unsigned int* mid_base_scaler = mid_base_scaler_ptr(D, target_id, site);
    unsigned int* target_up_scaler = proximal_mode ? nullptr : up_scaler_ptr(D, target_id, site);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        unsigned int inherited_shift = read_scaler_shift(D, mid_base_scaler, r);
        if (target_up_scaler) {
            inherited_shift += read_scaler_shift(D, target_up_scaler, r);
        }
        write_scaler_shift(D, mid_scaler, r, inherited_shift);

        const fp_t* Mtarget = shared_target_mat + (size_t)r * 16;
        const fp_t* Mparent = shared_parent_mat + (size_t)r * 16;
        const fp4_t Pup   = reinterpret_cast<const fp4_t*>(target_up + (size_t)r * 4)[0];
        fp_t*       Pmid  = target_mid + (size_t)r * 4;
        const fp4_t Pbase = reinterpret_cast<const fp4_t*>(mid_base + (size_t)r * 4)[0];

        const fp_t p0 = fp_dot4(make_fp4(Mparent[0], Mparent[1], Mparent[2], Mparent[3]), Pbase) *
                        fp_dot4(make_fp4(Mtarget[0], Mtarget[1], Mtarget[2], Mtarget[3]), Pup);
        const fp_t p1 = fp_dot4(make_fp4(Mparent[4], Mparent[5], Mparent[6], Mparent[7]), Pbase) *
                        fp_dot4(make_fp4(Mtarget[4], Mtarget[5], Mtarget[6], Mtarget[7]), Pup);
        const fp_t p2 = fp_dot4(make_fp4(Mparent[8], Mparent[9], Mparent[10], Mparent[11]), Pbase) *
                        fp_dot4(make_fp4(Mtarget[8], Mtarget[9], Mtarget[10], Mtarget[11]), Pup);
        const fp_t p3 = fp_dot4(make_fp4(Mparent[12], Mparent[13], Mparent[14], Mparent[15]), Pbase) *
                        fp_dot4(make_fp4(Mtarget[12], Mtarget[13], Mtarget[14], Mtarget[15]), Pup);

        Pmid[0] = p0;
        Pmid[1] = p1;
        Pmid[2] = p2;
        Pmid[3] = p3;
        
        fp_t col_scale_max_val = fp_hmax4(Pmid[0], Pmid[1], Pmid[2], Pmid[3]);
        {
            unsigned int shift = threshold_scale_shift(col_scale_max_val);
            if (shift) {
            add_scaler_shift(D, mid_scaler, r, shift);
            Pmid[0] = fp_ldexp(Pmid[0], shift);
            Pmid[1] = fp_ldexp(Pmid[1], shift);
            Pmid[2] = fp_ldexp(Pmid[2], shift);
            Pmid[3] = fp_ldexp(Pmid[3], shift);
            }
        }

    }
}

// Explicit instantiations for placement usage.
template __device__ void compute_midpoint_inner_inner_ratecat<1>(
    const DeviceTree&,
    const NodeOpInfo&,
    unsigned int,
    bool,
    int,
    bool,
    fp_t*,
    fp_t*);
template __device__ void compute_midpoint_inner_inner_ratecat<4>(
    const DeviceTree&,
    const NodeOpInfo&,
    unsigned int,
    bool,
    int,
    bool,
    fp_t*,
    fp_t*);
template __device__ void compute_midpoint_inner_inner_ratecat<8>(
    const DeviceTree&,
    const NodeOpInfo&,
    unsigned int,
    bool,
    int,
    bool,
    fp_t*,
    fp_t*);

__device__ __forceinline__ void compute_inner_inner_site_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const unsigned int states = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t span = (size_t)states * (size_t)rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * span;

    const fp_t* d_left_clv  = clv_read_ptr_for_node<const fp_t>(D, op, op.left_id);
    const fp_t* d_right_clv = clv_read_ptr_for_node<const fp_t>(D, op, op.right_id);
    fp_t* parent_clv = clv_write_ptr_for_node<fp_t>(D, op, op.parent_id);
    if (!d_left_clv || !d_right_clv || !parent_clv) return; // placeholder until preorder input logic is defined
    const fp_t* d_left_mat  = D.d_pmat + (size_t)op.left_id  * D.rate_cats * states * states;
    const fp_t* d_right_mat = D.d_pmat + (size_t)op.right_id * D.rate_cats * states * states;

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, rate_cats);
    unsigned int* left_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, op.left_id, site);
    unsigned int* right_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, op.right_id, site);

    for (unsigned int r = 0; r < rate_cats; ++r) {
        write_scaler_shift(
            D,
            site_scaler_ptr,
            r,
            read_scaler_shift(D, left_scaler, r) +
            read_scaler_shift(D, right_scaler, r));
        const fp_t* Lclv = d_left_clv  + site_off + (size_t)r * states;
        const fp_t* Rclv = d_right_clv + site_off + (size_t)r * states;

        const fp_t* Lmat = d_left_mat  + (size_t)r * states * states;
        const fp_t* Rmat = d_right_mat + (size_t)r * states * states;

        fp_t* Pout = parent_clv + site_off + (size_t)r * states;
        fp_t col_scale_max_val = fp_t(0);

        const fp_t* Lrow = Lmat;
        const fp_t* Rrow = Rmat;
        for (unsigned int j = 0; j < states; ++j) {
            fp_t lt = fp_t(0), rt = fp_t(0);
            #pragma unroll
            for (unsigned int k = 0; k < states; ++k) {
                lt = fp_fma(Lrow[k], Lclv[k], lt);
                rt = fp_fma(Rrow[k], Rclv[k], rt);
            }
            Pout[j] = lt * rt;
            if (Pout[j] > col_scale_max_val) col_scale_max_val = Pout[j];
            Lrow += states;
            Rrow += states;
        }

        if (site_scaler_ptr) {
            unsigned int shift = threshold_scale_shift(col_scale_max_val);
            if (shift) {
                site_scaler_ptr[r] += shift;
                for (unsigned int j = 0; j < states; ++j) {
                    scale_clv_pow2(Pout[j], shift);
                }
            }
        }
    }
}

__global__ void Rtree_Likelihood_Site_Parallel_Upward_Kernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int num_ops
) {
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            const NodeOpInfo& op = ops[i];
            switch (op.op_type) {
                case OP_TIP_TIP:
                    if (D.states == 4) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_tip_tip_site_ratecat_nolookup<1>(D, op, site);
                                break;
                            case 4:
                                compute_tip_tip_site_ratecat_nolookup<4>(D, op, site);
                                break;
                            case 8:
                                compute_tip_tip_site_ratecat_nolookup<8>(D, op, site);
                                break;
                            default:
                                compute_tip_tip_site_4_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_tip_tip_site_generic(D, op, site);
                    }
                    break;
                case OP_TIP_INNER:
                    if (D.states == 4) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_tip_inner_site_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_tip_inner_site_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_tip_inner_site_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_tip_inner_site_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_tip_inner_site_generic(D, op, site);
                    }
                    break;
                case OP_INNER_INNER:
                    if (D.states == 4) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_inner_inner_site_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_inner_inner_site_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_inner_inner_site_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_inner_inner_site_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_inner_inner_site_generic(D, op, site);
                    }
                    break;
            default:
                break;
            
            
        }
    }
}
}
// Downward child kernel: compute target child clv_down from parent.down + sibling.up.
__global__ void Rtree_Likelihood_Site_Parallel_Downward_Kernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int num_ops)
{

    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            const NodeOpInfo& op = ops[i];
            switch (op.op_type) {
                case OP_DOWN_INNER_INNER:
                    if (D.states == 4) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_downward_inner_inner_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_downward_inner_inner_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_downward_inner_inner_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_downward_inner_inner_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_downward_inner_inner_generic(D, op, site);
                    }
                    break;
                case OP_DOWN_INNER_TIP:
                    if (D.states == 4) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_downward_inner_tip_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_downward_inner_tip_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_downward_inner_tip_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_downward_inner_tip_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_downward_inner_tip_generic(D, op, site);
                    }
                    break;
                case OP_DOWN_TIP_INNER:
                    if (D.states == 4) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_downward_tip_inner_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_downward_tip_inner_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_downward_tip_inner_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_downward_tip_inner_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_downward_tip_inner_generic(D, op, site);
                    }
                    break;
                case OP_DOWN_TIP_TIP:
                    if (D.states == 4) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_downward_tip_tip_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_downward_tip_tip_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_downward_tip_tip_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_downward_tip_tip_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_downward_tip_tip_generic(D, op, site);
                    }
                    break;
                default:
                    break;
            }
        }
    }
}
