#pragma once
#include <cstdlib>   // abort
#include <csignal>   // raise, SIGTRAP
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <stdexcept>
#include <cstddef>
#include <vector>
#include "tree/tree.hpp"

// Branch length defaults to match epa-ng constants.
constexpr double DEFAULT_BRANCH_LENGTH = 0.10536051565782628; // -log(0.9)
// Match the EPA-ng / PLLMOD branch-length floor for placement optimization.
constexpr double OPT_BRANCH_LEN_MIN = 1.0e-4;
constexpr double OPT_BRANCH_LEN_MAX = 100.0;  // PLLMOD_OPT_MAX_BRANCH_LEN
constexpr double OPT_BRANCH_EPSILON = 1.0e-1;
constexpr double OPT_BRANCH_XTOL = OPT_BRANCH_LEN_MIN / 10.0;

template <typename T>
__host__ __device__ inline T scalar_min(T lhs, T rhs) {
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
__host__ __device__ inline T scalar_max(T lhs, T rhs) {
    return lhs > rhs ? lhs : rhs;
}

template <typename T>
__host__ __device__ inline T clamp_scalar(T value, T lower, T upper) {
    if (upper < lower) upper = lower;
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}

namespace mlipper {
namespace env {

inline bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] && std::string(value) != "0";
}

inline void set_int_env_if_specified(const char* name, int value) {
    if (value < 0) return;
    setenv(name, std::to_string(value).c_str(), 1);
}

inline void set_double_env_if_specified(const char* name, double value) {
    if (value < 0.0) return;
    setenv(name, std::to_string(value).c_str(), 1);
}

} // namespace env
} // namespace mlipper

__host__ __device__ inline double effective_split_branch_min(
    double total_branch_length,
    double min_branch_length = OPT_BRANCH_LEN_MIN)
{
    if (total_branch_length <= 0.0) return min_branch_length;
    return scalar_min(min_branch_length, 0.5 * total_branch_length);
}

__host__ __device__ inline void normalize_split_branch_lengths(
    double total_branch_length,
    double proposed_proximal_length,
    double min_branch_length,
    double& proximal_length_out,
    double& distal_length_out)
{
    if (total_branch_length <= 0.0) {
        proximal_length_out = min_branch_length;
        distal_length_out = min_branch_length;
        return;
    }

    const double lower_bound = effective_split_branch_min(total_branch_length, min_branch_length);
    const double upper_bound = scalar_max(lower_bound, total_branch_length - lower_bound);
    proximal_length_out = clamp_scalar(proposed_proximal_length, lower_bound, upper_bound);
    distal_length_out = total_branch_length - proximal_length_out;
}

__host__ __device__ inline double sanitize_branch_length(
    double branch_length,
    double min_branch_length = OPT_BRANCH_LEN_MIN,
    double max_branch_length = OPT_BRANCH_LEN_MAX,
    double default_branch_length = DEFAULT_BRANCH_LENGTH)
{
    if (!(branch_length > 0.0)) branch_length = default_branch_length;
    return clamp_scalar(branch_length, min_branch_length, max_branch_length);
}

// Basic tags describing the operation type and CLV buffer selection.
enum NodeOpType : int {
    OP_TIP_TIP = 0,
    OP_TIP_INNER = 1,
    OP_INNER_INNER = 2,
    OP_DOWN_INNER_INNER = 3,
    OP_DOWN_INNER_TIP = 4,
    OP_DOWN_TIP_INNER = 5,
    OP_DOWN_TIP_TIP   = 6
};

enum ClvPool : uint8_t {
    CLV_POOL_UP = 0,
    CLV_POOL_DOWN = 1
};

// Direction tags used by preorder/downward passes.
enum ClvDir : uint8_t {
    CLV_DIR_UNSET      = 0,
    CLV_DIR_UP         = 1, // child -> parent
    CLV_DIR_DOWN_LEFT  = 2, // parent -> left child
    CLV_DIR_DOWN_RIGHT = 3  // parent -> right child
};

struct NodeOpInfo {
    int parent_id = -1;
    int left_id = -1;
    int right_id = -1;
    int left_tip_index = -1;
    int right_tip_index = -1;
    int op_type = OP_TIP_TIP;
    uint8_t clv_pool = static_cast<uint8_t>(CLV_POOL_UP);
    uint8_t dir_tag  = static_cast<uint8_t>(CLV_DIR_UP);
};

// Common CUDA error checking macro used across modules.
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr,                                                       \
                "CUDA ERROR: %s (%d)\n"                                            \
                "  at %s:%d\n"                                                    \
                "  call: %s\n",                                                   \
                cudaGetErrorString(err__), (int)err__,                            \
                __FILE__, __LINE__, #call);                                       \
            raise(SIGTRAP);   /* <<< 讓 gdb 停在這裡 */                            \
            abort();                                                             \
        }                                                                         \
    } while (0)
#endif

#define CHECK_CUDA_LAST()                                                        \
    do {                                                                         \
        cudaError_t err__ = cudaGetLastError();                                  \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr,                                                      \
                "CUDA KERNEL LAUNCH ERROR: %s (%d)\n"                            \
                "  at %s:%d\n",                                                  \
                cudaGetErrorString(err__), (int)err__,                           \
                __FILE__, __LINE__);                                             \
            raise(SIGTRAP);                                                      \
            abort();                                                            \
        }                                                                        \
    } while (0)

// Fast integer log2 ceil for small unsigned values.
inline unsigned int ceil_log2_u32(unsigned int x) {
    if (x <= 1u) return 0u;
    unsigned int v = x - 1u;
    unsigned int r = 0u;
    while (v) { v >>= 1u; ++r; }
    return r;
}

// ===== Device-side CLV helpers (inlined for reuse across CUDA units) =====
__device__ __forceinline__ size_t per_node_span(const DeviceTree& D) {
    return (size_t)D.sites * (size_t)D.rate_cats * (size_t)D.states;
}

__device__ __forceinline__ size_t scaler_span(const DeviceTree& D) {
    if (D.per_rate_scaling) {
        return (size_t)D.sites * (size_t)D.rate_cats;
    }
    return (size_t)D.sites;
}

__device__ __forceinline__ size_t scaler_site_offset(
    const DeviceTree& D,
    unsigned int site)
{
    if (D.per_rate_scaling) {
        return (size_t)site * (size_t)D.rate_cats;
    }
    return (size_t)site;
}

__device__ __forceinline__ unsigned int* scaler_ptr_for_node(
    unsigned int* base,
    const DeviceTree& D,
    int node_id,
    unsigned int site)
{
    if (!base) return nullptr;
    if (node_id < 0 || node_id >= D.capacity_N) return nullptr;
    return base + (size_t)node_id * scaler_span(D) + scaler_site_offset(D, site);
}

__device__ __forceinline__ unsigned int* up_scaler_ptr(
    const DeviceTree& D,
    int node_id,
    unsigned int site)
{
    return scaler_ptr_for_node(D.d_site_scaler_up, D, node_id, site);
}

__device__ __forceinline__ unsigned int* down_scaler_ptr(
    const DeviceTree& D,
    int node_id,
    unsigned int site)
{
    return scaler_ptr_for_node(D.d_site_scaler_down, D, node_id, site);
}

__device__ __forceinline__ unsigned int* mid_scaler_ptr(
    const DeviceTree& D,
    int node_id,
    unsigned int site)
{
    return scaler_ptr_for_node(D.d_site_scaler_mid, D, node_id, site);
}

__device__ __forceinline__ unsigned int* mid_base_scaler_ptr(
    const DeviceTree& D,
    int node_id,
    unsigned int site)
{
    return scaler_ptr_for_node(D.d_site_scaler_mid_base, D, node_id, site);
}

template <typename T>
__device__ __forceinline__ T* clv_write_pool_base(const DeviceTree& D, const NodeOpInfo& op) {
    return (op.clv_pool == static_cast<uint8_t>(CLV_POOL_DOWN))
        ? reinterpret_cast<T*>(D.d_clv_down)
        : reinterpret_cast<T*>(D.d_clv_up);
}

template <typename T>
__device__ __forceinline__ T* clv_read_pool_base(const DeviceTree& D, const NodeOpInfo& op) {
    return (op.clv_pool == static_cast<uint8_t>(CLV_POOL_DOWN))
        ? reinterpret_cast<T*>(D.d_clv_down)
        : reinterpret_cast<T*>(D.d_clv_up);
}

template <typename T>
__device__ __forceinline__ T* clv_write_ptr_for_node(const DeviceTree& D, const NodeOpInfo& op, int node_id) {
    T* base = clv_write_pool_base<T>(D, op);
    return base ? base + (size_t)node_id * per_node_span(D) : nullptr;
}

template <typename T>
__device__ __forceinline__ T* clv_read_ptr_for_node(const DeviceTree& D, const NodeOpInfo& op, int node_id) {
    T* base = clv_read_pool_base<T>(D, op);
    return base ? base + (size_t)node_id * per_node_span(D) : nullptr;
}

// Variant when the pool is implicitly the "up" pool (used in derivative helpers).
template <typename T>
__device__ __forceinline__ T* clv_read_ptr_for_node(const DeviceTree& D, int node_id) {
    T* base = reinterpret_cast<T*>(D.d_clv_up);
    return base ? base + (size_t)node_id * per_node_span(D) : nullptr;
}

__device__ __forceinline__ unsigned int* site_scaler_ptr_base(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site,
    unsigned int rate_cats)
{
    (void)rate_cats;
    if (op.clv_pool == static_cast<uint8_t>(CLV_POOL_DOWN)) {
        return down_scaler_ptr(D, op.parent_id, site);
    }
    return up_scaler_ptr(D, op.parent_id, site);
}
