#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class DispatchPrologueRuntime final : public jit::LaunchRuntime<DispatchPrologueRuntime> {
public:
    struct Args {
        // Templated arguments
        int num_warps;
        int num_ranks;
        int num_max_tokens_per_rank;
        int num_experts, num_topk;

        // Parameters
        topk_idx_t* topk_idx;
        int* rank_count_buffer;
        int* dst_buffer_slot_idx;
        int num_tokens;
        int rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_deterministic_prologue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_deterministic_prologue_impl<{}, {}, {}, {}, {}, {}>);
}}
)",
                           args.launch_args.grid_dim.first,
                           args.num_warps,
                           args.num_ranks,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel,
                                                 config,
                                                 args.topk_idx,
                                                 args.rank_count_buffer,
                                                 args.dst_buffer_slot_idx,
                                                 args.num_tokens,
                                                 args.rank_idx));
    }
};

static void launch_dispatch_deterministic_prologue(topk_idx_t* topk_idx, int* rank_count_buffer,
                                                   int* dst_buffer_slot_idx,
                                                   const int& num_tokens, const int& num_max_tokens_per_rank,
                                                   const int& num_experts, const int& num_topk,
                                                   const int& rank_idx, const int& num_ranks,
                                                   const int& num_sms, const int& num_smem_bytes,
                                                   const at::cuda::CUDAStream& stream) {
    constexpr auto num_warps = 8;
    constexpr auto num_threads = num_warps * 32;
    EP_HOST_ASSERT((2 * num_warps + 1) * num_ranks * sizeof(int) <= num_smem_bytes and
                   "Insufficient shared memory");

    // Generate, build and launch
    const DispatchPrologueRuntime::Args args = {
        .num_warps = num_warps,
        .num_ranks = num_ranks,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk,
        .topk_idx = topk_idx,
        .rank_count_buffer = rank_count_buffer,
        .dst_buffer_slot_idx = dst_buffer_slot_idx,
        .num_tokens = num_tokens,
        .rank_idx = rank_idx,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, true)};
    const auto code = DispatchPrologueRuntime::generate(args);
    const auto runtime = jit::compiler->build("dispatch_deterministic_prologue", code);
    DispatchPrologueRuntime::launch(runtime, args, stream);
}

class DispatchRuntime final : public jit::LaunchRuntime<DispatchRuntime> {
public:
    struct Args {
        // Templated arguments
        bool is_scaleup_nvlink;
        bool do_cpu_sync;
        bool reuse_slot_indices;
        int num_notify_warps;
        int num_dispatch_warps; // For hybrid dispatch
        int num_scaleout_warps, num_forward_warps; // For direct dispatch
        int num_scaleout_ranks, num_scaleup_ranks;
        int num_hidden_bytes, num_sf_packs;
        int num_max_tokens_per_rank;
        int num_experts, num_topk, expert_alignment;
        int num_qps;
        int64_t num_timeout_cycles;

        // Parameters
        void* x; sf_pack_t* sf; topk_idx_t* topk_idx; float* topk_weights;
        topk_idx_t* copied_topk_idx;
        int* cumulative_local_expert_recv_stats;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* psum_num_recv_tokens_per_expert;
        int* dst_buffer_slot_idx;
        int* token_metadata_at_forward;
        int num_tokens;
        int sf_token_stride, sf_hidden_stride;
        ncclDevComm_t nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace; void* mapped_host_workspace;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        std::string header_name, func_name;
        if (args.num_scaleout_ranks == 1) {
            header_name = "dispatch";
            func_name = fmt::format("dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                args.is_scaleup_nvlink,
                args.do_cpu_sync,
                args.reuse_slot_indices,
                args.launch_args.grid_dim.first,
                args.num_notify_warps, args.num_dispatch_warps,
                args.num_scaleup_ranks,
                args.num_hidden_bytes, args.num_sf_packs,
                args.num_max_tokens_per_rank,
                args.num_experts, args.num_topk, args.expert_alignment,
                args.num_qps, args.num_timeout_cycles);
        } else {
            header_name = "hybrid_dispatch";
            func_name = fmt::format("hybrid_dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                args.do_cpu_sync,
                args.reuse_slot_indices,
                args.launch_args.grid_dim.first,
                args.num_notify_warps, args.num_scaleout_warps, args.num_forward_warps,
                args.num_scaleout_ranks, args.num_scaleup_ranks,
                args.num_hidden_bytes, args.num_sf_packs,
                args.num_max_tokens_per_rank,
                args.num_experts, args.num_topk, args.expert_alignment,
                args.num_qps, args.num_timeout_cycles);
        }

        return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        if (args.num_scaleout_ranks == 1) {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
                kernel, config,
                args.x, args.sf, args.topk_idx, args.topk_weights,
                args.copied_topk_idx,
                args.cumulative_local_expert_recv_stats,
                args.psum_num_recv_tokens_per_scaleup_rank,
                args.psum_num_recv_tokens_per_expert,
                args.dst_buffer_slot_idx,
                args.num_tokens,
                args.sf_token_stride, args.sf_hidden_stride,
                args.nccl_dev_comm, args.nccl_window,
                args.buffer,
                args.workspace, args.mapped_host_workspace,
                args.scaleup_rank_idx));
        } else {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
                kernel, config,
                args.x, args.sf, args.topk_idx, args.topk_weights,
                args.copied_topk_idx,
                args.cumulative_local_expert_recv_stats,
                args.psum_num_recv_tokens_per_scaleup_rank,
                args.psum_num_recv_tokens_per_expert,
                args.dst_buffer_slot_idx,
                args.token_metadata_at_forward,
                args.num_tokens,
                args.sf_token_stride, args.sf_hidden_stride,
                args.nccl_dev_comm, args.nccl_window,
                args.buffer,
                args.workspace, args.mapped_host_workspace,
                args.scaleout_rank_idx, args.scaleup_rank_idx
            ));
        }
    }
};

constexpr int kNumNotifyWarps = 4;

static int get_num_notify_smem_bytes(const int& num_ranks, const int& num_experts) {
    return math::align(num_ranks + num_experts, kNumNotifyWarps * 32) * sizeof(int);
}

static layout::TokenLayout get_dispatch_token_layout(
    const int& hidden, const int& elem_size, const int& num_sf_packs, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Dispatch kernel 启动器
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// 整体流程:
//   ┌──────────────────────────────────────────────────────────────┐
//   │ 1. 参数校验 (cached mode 断言)                                │
//   │ 2. Warp 配置计算 (根据 num_scaleout_ranks 分两路)              │
//   │    ├─ Direct mode (scaleout_ranks==1): notify + dispatch      │
//   │    └─ Hybrid mode  (scaleout_ranks>1):  notify + scaleout     │
//   │                                            + forward          │
//   │ 3. 构造 kernel args, JIT 编译, 启动                           │
//   └──────────────────────────────────────────────────────────────┘
//
// 参数说明:
//   x                              : 输入 token hidden states [num_tokens, hidden]
//   sf                             : 输入 scaling factor (FP8 专用, BF16 时为 nullptr)
//   topk_idx                       : MoE top-k 选出的 expert 索引 [num_tokens, num_topk]
//   topk_weights                   : MoE top-k 权重 [num_tokens, num_topk]
//   copied_topk_idx                : topk_idx 的副本 (用于 NCCL all-to-all 通信)
//   cumulative_local_expert_recv_stats : 累计每个 local expert 收到的 token 数 (非 cached)
//   psum_num_recv_tokens_per_scaleup_rank : prefix sum: 每个 scaleup rank 收到的 token 累计
//   psum_num_recv_tokens_per_expert      : prefix sum: 每个 local expert 收到的 token 累计
//   dst_buffer_slot_idx            : 目标 buffer 的 slot 索引 (用于 combine 阶段)
//   token_metadata_at_forward      : hybrid 模式下的 forward 阶段 token 元数据
//   num_tokens                     : 输入 token 数
//   num_max_tokens_per_rank        : 每个 rank 的最大 token 容量
//   hidden                         : hidden 维度大小
//   elem_size                      : 元素字节数 (BF16=2, FP8=1)
//   num_sf_packs                   : scaling factor 的 pack 数 (FP8 专用)
//   sf_token_stride / sf_hidden_stride : sf 张量的 stride
//   num_experts                    : 总 expert 数
//   num_topk                       : 每个 token 选出的 top-k 数
//   expert_alignment               : expert 对齐粒度 (通常 64)
//   nccl_dev_comm / nccl_window    : NCCL 通信子和窗口 (用于 RDMA)
//   buffer                         : 预分配的 RDMA buffer
//   workspace / mapped_host_workspace : GPU/CPU 共享 workspace (用于 low-latency 同步)
//   scaleout_rank_idx / scaleup_rank_idx : 本 rank 在 scaleout/scaleup 维度的索引
//   num_scaleout_ranks / num_scaleup_ranks : scaleout/scaleup 维度的 rank 总数
//   is_scaleup_nvlink              : scaleup 维度是否走 NVLink (影响 buffer 布局)
//   num_sms                        : 使用的 SM 数
//   num_channels_per_sm            : 每个 SM 的 channel 数 (hybrid 模式)
//   num_smem_bytes                 : 每个 SM 可用的 shared memory 大小
//   num_qps                        : QP 数 (IB 连接数)
//   num_timeout_cycles             : 超时周期数
//   cached_mode                    : 是否使用缓存模式 (复用上次 dispatch 的 slot 分配)
//   deterministic                  : 是否确定性模式 (slot 分配可复现)
//   do_cpu_sync                    : 是否用 CPU 同步 (low-latency 模式)
//   stream                         : CUDA stream
//
static void launch_dispatch(void* x,                                    // 输入 token hidden states [num_tokens, hidden]
                            void* sf,                                   // 输入 scaling factor (FP8 专用, BF16 时为 nullptr)
                            topk_idx_t* topk_idx,                       // MoE top-k expert 索引 [num_tokens, num_topk]
                            float* topk_weights,                        // MoE top-k 权重 [num_tokens, num_topk]
                            topk_idx_t* copied_topk_idx,                // topk_idx 的副本 (NCCL all-to-all 通信用)
                            int* cumulative_local_expert_recv_stats,    // 累计每个 local expert 收到的 token 数 (非 cached)
                            int* psum_num_recv_tokens_per_scaleup_rank, // prefix sum: 每个 scaleup rank 收到的 token 累计
                            int* psum_num_recv_tokens_per_expert,       // prefix sum: 每个 local expert 收到的 token 累计
                            int* dst_buffer_slot_idx,                   // 目标 buffer 的 slot 索引 (combine 阶段用)
                            int* token_metadata_at_forward,             // hybrid 模式下 forward 阶段的 token 元数据
                            const int& num_tokens,                      // 输入 token 数
                            const int& num_max_tokens_per_rank,         // 每个 rank 的最大 token 容量
                            const int& hidden,                          // hidden 维度大小
                            const int& elem_size,                       // 元素字节数 (BF16=2, FP8=1)
                            const int& num_sf_packs,                    // scaling factor 的 pack 数 (FP8 专用)
                            const int& sf_token_stride,                 // sf 张量的 token 维度 stride
                            const int& sf_hidden_stride,                // sf 张量的 hidden 维度 stride
                            const int& num_experts,                     // 总 expert 数
                            const int& num_topk,                        // 每个 token 选出的 top-k 数
                            const int& expert_alignment,                // expert 对齐粒度 (通常 64)
                            const ncclDevComm_t& nccl_dev_comm,        // NCCL device 通信子 (all-to-all 用)
                            const ncclWindow_t& nccl_window,           // NCCL RDMA 窗口 (IB 直连用)
                            void* buffer,                               // 预分配的 RDMA buffer
                            void* workspace,                            // GPU workspace (low-latency 同步用)
                            void* mapped_host_workspace,                // CPU 可见 workspace (CPU poll 用)
                            const int& scaleout_rank_idx,               // 本 rank 在 scaleout 维度的索引
                            const int& scaleup_rank_idx,                // 本 rank 在 scaleup 维度的索引
                            const int& num_scaleout_ranks,              // 跨机 rank 总数
                            const int& num_scaleup_ranks,               // 同机 rank 总数
                            const bool& is_scaleup_nvlink,              // scaleup 是否 NVLink (影响 buffer 布局)
                            const int& num_sms,                         // 使用的 SM 数
                            const int& num_channels_per_sm,             // 每个 SM 的 channel 数 (hybrid 模式)
                            const int& num_smem_bytes,                  // 每 SM 可用的 shared memory 大小
                            const int& num_qps,                         // IB QP 数 (连接数)
                            const int64_t& num_timeout_cycles,          // RDMA 超时周期数
                            const bool& cached_mode,                    // 缓存模式 (复用上次 slot 分配)
                            const bool& deterministic,                  // 确定性模式 (slot 分配可复现)
                            const bool& do_cpu_sync,                    // CPU 同步模式 (low-latency)
                            const at::cuda::CUDAStream& stream) {       // CUDA stream

    // ════════════════════════════════════════════════════════════════
    // 1. Cached mode 校验
    //    cached mode 复用上一次 dispatch 的 slot 分配结果,
    //    不需要重新统计 expert token 计数, 所以该指针必须为空
    // ════════════════════════════════════════════════════════════════
    if (cached_mode)
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats == nullptr);

    // ════════════════════════════════════════════════════════════════
    // 2. 基础参数
    // ════════════════════════════════════════════════════════════════
    // 总 rank 数 = scaleout_ranks × scaleup_ranks
    //   例: 2 个 scaleout × 4 个 scaleup = 8 ranks
    const auto num_ranks = num_scaleout_ranks * num_scaleup_ranks;

    // ════════════════════════════════════════════════════════════════
    // 3. Notify warp 配置
    //    Notify warp 负责: 统计每个 expert 收到的 token 数,
    //    并通过 NCCL all-to-all 通知所有 rank
    //
    //    ┌─────────────────────────────────────────────────┐
    //    │ cached_mode=false: kNumNotifyWarps=4 个 warp    │
    //    │   → 4×32=128 线程做 expert 计数 + all-to-all    │
    //    │ cached_mode=true:  0 个 warp (复用上次结果)      │
    //    └─────────────────────────────────────────────────┘
    //
    //    reuse_slot_indices: cached 或 deterministic 时复用 slot 分配
    //      → 保证同一 token 多次 dispatch 分配到相同的 buffer slot
    //      → 对 combine 阶段的正确性至关重要
    // ════════════════════════════════════════════════════════════════
    const int num_notify_warps = cached_mode ? 0 : kNumNotifyWarps;  // kNumNotifyWarps=4
    const bool reuse_slot_indices = cached_mode or deterministic;
    // notify smem 大小 = align(num_ranks + num_experts, 128) * sizeof(int)
    //   存储每个 rank 的 send counter + 每个 expert 的 recv counter
    const int num_notify_smem_bytes = cached_mode ? 0 : get_num_notify_smem_bytes(num_ranks, num_experts);
    EP_HOST_ASSERT(num_notify_warps % 4 == 0);  // 4 的倍数 (warp group 对齐)

    // ════════════════════════════════════════════════════════════════
    // 4. 数据搬运 warp 配置 (根据 scaleout 维度分两路)
    // ════════════════════════════════════════════════════════════════
    int num_dispatch_warps = 0;      // direct 模式: 数据搬运 warp 数
    int num_scaleout_warps = 0, num_forward_warps = 0;  // hybrid 模式: scaleout/forward warp 数
    int num_threads = 0;

    // ── 4a. Direct mode: num_scaleout_ranks == 1 ─────────────────
    //   只有 scaleup 维度 (单机), 无需 scaleout/forward warp
    //
    //   Warp 布局:
    //   ┌──────────────┬──────────────────────────┐
    //   │ notify warps │    dispatch warps         │
    //   │   (4个)      │  (按 smem 容量最大化)      │
    //   └──────────────┴──────────────────────────┘
    //
    //   dispatch warp 数量计算 (三重约束取最小):
    //     ① (smem - notify_smem) / 每个token的smem开销 → smem 容量上限
    //     ② 32 - num_notify_warps → warp 总数上限 (每个 SM 最多 32 个 warp)
    //     ③ ceil(512 / num_sms) → 避免过多 warp 导致调度开销
    if (num_scaleout_ranks == 1) {
        // token_layout: 描述一个 token 在 smem 中的布局 (hidden + sf + topk)
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);
        num_dispatch_warps = std::min<int>(std::min<int>(
            // ① smem 容量: 剩余 smem / 每个 dispatch warp 需要的 smem
            (num_smem_bytes - num_notify_smem_bytes) / token_layout.get_num_bytes<true>(),
            // ② ③: warp 上限
            32 - num_notify_warps),
            math::ceil_div(512, num_sms));
        num_threads = (num_notify_warps + num_dispatch_warps) * 32;

    // ── 4b. Hybrid mode: num_scaleout_ranks > 1 ─────────────────
    //   多机场景, 需要 scaleout (跨机 IB) + forward (同机 NVLink) 两阶段
    //
    //   Warp 布局:
    //   ┌──────────────┬──────────────────┬──────────────────┐
    //   │ notify warps │ scaleout warps   │ forward warps    │
    //   │   (4个)      │  = channels/sm   │  = channels/sm   │
    //   └──────────────┴──────────────────┴──────────────────┘
    //                     ↕ 一一对应 ↕
    //                scaleout_warp_i 和 forward_warp_i 共享 channel_i
    //
    //   num_channels_per_sm 在 buffer.hpp 中计算 (受 smem 和 combine 布局双重约束)
    } else {
        // ⚠️ deterministic 模式在 hybrid 下未实现 (slot 分配涉及跨机 RDMA, 难以确定性复现)
        EP_HOST_ASSERT(not deterministic);

        // scaleout/forward warp 数 = 每SM的channel数 (一一对应, 保证 channel 隔离)
        num_scaleout_warps = num_channels_per_sm;
        num_forward_warps = num_channels_per_sm;
        num_threads = (num_notify_warps + num_scaleout_warps + num_forward_warps) * 32;
    }

    // ════════════════════════════════════════════════════════════════
    // 5. 构造 kernel 参数 & JIT 编译 & 启动
    //
    //   ┌──────────────────────────────────────────────────────────┐
    //   │ DispatchRuntime::generate(args)                          │
    //   │   → 根据 args 生成 CUDA kernel 代码 (运行时 JIT)         │
    //   │                                                          │
    //   │ jit::compiler->build("dispatch", code)                   │
    //   │   → 编译 CUDA 代码, 返回 runtime handle                  │
    //   │                                                          │
    //   │ DispatchRuntime::launch(runtime, args, stream)           │
    //   │   → 启动 kernel                                          │
    //   └──────────────────────────────────────────────────────────┘
    // ════════════════════════════════════════════════════════════════
    const DispatchRuntime::Args args = {
        // ── 通信拓扑 ──
        .is_scaleup_nvlink = is_scaleup_nvlink,    // scaleup 是否 NVLink (影响 TMA/PTX 选择)
        .do_cpu_sync = do_cpu_sync,                // low-latency 模式: CPU poll 而非 GPU wait
        .reuse_slot_indices = reuse_slot_indices,   // 复用 slot 分配 (cached/deterministic)

        // ── Warp 配置 ──
        .num_notify_warps = num_notify_warps,       // notify warp 数 (0 或 4)
        .num_dispatch_warps = num_dispatch_warps,   // direct 模式的 dispatch warp 数
        .num_scaleout_warps = num_scaleout_warps,   // hybrid 模式的 scaleout warp 数
        .num_forward_warps = num_forward_warps,     // hybrid 模式的 forward warp 数

        // ── 逻辑维度 ──
        .num_scaleout_ranks = num_scaleout_ranks,   // 跨机 rank 数
        .num_scaleup_ranks = num_scaleup_ranks,     // 同机 rank 数
        .num_hidden_bytes = hidden * elem_size,     // hidden 维度的字节数
        .num_sf_packs = num_sf_packs,               // scaling factor pack 数
        .num_max_tokens_per_rank = num_max_tokens_per_rank,  // 每个 rank 的最大 token 容量
        .num_experts = num_experts,                 // expert 总数
        .num_topk = num_topk,                       // top-k 数
        .expert_alignment = expert_alignment,       // expert 对齐粒度

        // ── RDMA 配置 ──
        .num_qps = num_qps,                         // IB QP 数
        .num_timeout_cycles = num_timeout_cycles,   // RDMA 超时周期

        // ── 数据指针 ──
        .x = x,                                     // 输入 hidden states
        .sf = static_cast<sf_pack_t*>(sf),          // 输入 scaling factor
        .topk_idx = topk_idx,                       // top-k expert 索引
        .topk_weights = topk_weights,               // top-k 权重
        .copied_topk_idx = copied_topk_idx,         // top-k 索引副本 (NCCL 通信用)
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats,  // expert 计数
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,  // rank prefix sum
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,  // expert prefix sum
        .dst_buffer_slot_idx = dst_buffer_slot_idx,                 // 目标 buffer slot 索引
        .token_metadata_at_forward = token_metadata_at_forward,     // hybrid 模式 forward 元数据
        .num_tokens = num_tokens,                   // 输入 token 数

        // ── SF stride ──
        .sf_token_stride = sf_token_stride,         // sf 的 token 维度 stride
        .sf_hidden_stride = sf_hidden_stride,       // sf 的 hidden 维度 stride

        // ── NCCL 通信 ──
        .nccl_dev_comm = nccl_dev_comm,             // NCCL device comm (用于 all-to-all)
        .nccl_window = nccl_window,                 // NCCL RDMA window (用于 IB 直连)

        // ── Buffer & Workspace ──
        .buffer = buffer,                           // 预分配 RDMA buffer
        .workspace = workspace,                     // GPU workspace (low-latency 同步)
        .mapped_host_workspace = mapped_host_workspace,  // CPU 可见 workspace (CPU poll)

        // ── Rank 索引 ──
        .scaleout_rank_idx = scaleout_rank_idx,     // 本 rank 在 scaleout 维度的索引
        .scaleup_rank_idx = scaleup_rank_idx,       // 本 rank 在 scaleup 维度的索引

        // ── Launch 配置 ──
        //   cluster_dim: 设为 2 以与 clustered 计算kernel重叠
        //     当 num_sms 为偶数时 cluster_dim=2, 奇数时 cluster_dim=1
        //   dynamic_smem=true: 运行时动态分配 smem
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 2 - (num_sms % 2), true)};

    // JIT 代码生成 → 编译 → 启动
    const auto code = DispatchRuntime::generate(args);          // 生成 CUDA kernel 代码
    const auto runtime = jit::compiler->build("dispatch", code); // 编译
    DispatchRuntime::launch(runtime, args, stream);              // 启动
}

class DispatchCopyEpilogueRuntime final : public jit::LaunchRuntime<DispatchCopyEpilogueRuntime> {
public:
    struct Args {
        // Templated arguments
        bool do_expand, cached_mode;
        int num_channels;
        int num_warps;
        int num_scaleout_ranks, num_scaleup_ranks;
        int num_hidden_bytes, num_sf_packs;
        int num_max_tokens_per_rank;
        int num_experts, num_topk;

        // Parameters
        void *buffer, *workspace;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* psum_num_recv_tokens_per_expert;
        void* recv_x; void* recv_sf;
        topk_idx_t* recv_topk_idx; float* recv_topk_weights;
        int* recv_src_metadata;
        int* channel_linked_list;
        int num_recv_tokens;
        int recv_sf_token_stride, recv_sf_hidden_stride;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_copy_epilogue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
                           args.do_expand, args.cached_mode,
                           args.launch_args.grid_dim.first, args.num_channels, args.num_warps,
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.num_hidden_bytes, args.num_sf_packs,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.buffer, args.workspace,
                                                 args.psum_num_recv_tokens_per_scaleup_rank,
                                                 args.psum_num_recv_tokens_per_expert,
                                                 args.recv_x, args.recv_sf, args.recv_topk_idx, args.recv_topk_weights,
                                                 args.recv_src_metadata,
                                                 args.channel_linked_list,
                                                 args.num_recv_tokens,
                                                 args.recv_sf_token_stride, args.recv_sf_hidden_stride,
                                                 args.scaleout_rank_idx, args.scaleup_rank_idx));
    }
};

static void launch_dispatch_copy_epilogue(void* buffer, void* workspace,
                                          int* psum_num_recv_tokens_per_scaleup_rank,
                                          int* psum_num_recv_tokens_per_expert,
                                          void* recv_x, void* recv_sf,
                                          topk_idx_t* recv_topk_idx, float* recv_topk_weights,
                                          int* recv_src_metadata,
                                          int* channel_linked_list,
                                          const int& num_recv_tokens, const int& num_max_tokens_per_rank,
                                          const int& num_hidden_bytes,
                                          const int& num_sf_packs, const int& recv_sf_token_stride, const int& recv_sf_hidden_stride,
                                          const int& num_experts, const int& num_topk,
                                          const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                          const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                          const int& num_sms, const int& num_smem_bytes,
                                          const int& num_channels,
                                          const bool& do_expand, const bool& cached_mode,
                                          const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    const auto token_layout = layout::TokenLayout(num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
    const auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);
    const auto num_threads = num_warps * 32;

    // Generate, build and launch
    const DispatchCopyEpilogueRuntime::Args args = {
        .do_expand = do_expand, .cached_mode = cached_mode,
        .num_channels = num_channels, .num_warps = num_warps,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .num_hidden_bytes = num_hidden_bytes, .num_sf_packs = num_sf_packs,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk,
        .buffer = buffer, .workspace = workspace,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,
        .recv_x = recv_x, .recv_sf = recv_sf,
        .recv_topk_idx = recv_topk_idx, .recv_topk_weights = recv_topk_weights,
        .recv_src_metadata = recv_src_metadata,
        .channel_linked_list = channel_linked_list,
        .num_recv_tokens = num_recv_tokens,
        .recv_sf_token_stride = recv_sf_token_stride, .recv_sf_hidden_stride = recv_sf_hidden_stride,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, false, true)};
    const auto code = DispatchCopyEpilogueRuntime::generate(args);
    const auto runtime = jit::compiler->build("dispatch_copy_epilogue", code);
    DispatchCopyEpilogueRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
