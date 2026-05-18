#pragma once
/**
 * @file dispatch.cuh
 * @brief MoE Dispatch CUDA Kernel - Expert Parallelism 核心通信Kernel
 *
 * ==================== 项目背景 ====================
 *
 *   DeepEP (DeepEveryParallel) - 高性能 MoE 通信库
 *   • 专注 Expert Parallelism (EP), 提供高吞吐低延迟的 all-to-all GPU 内核
 *   • 支持 FP8 低精度, 最高 EP2048
 *   • 调用链: Python ElasticBuffer.dispatch() → C++ JIT 编译 → 本 kernel
 *
 * ==================== 本 Kernel 的通信范围 ====================
 *
 *   ★ 本 kernel 只负责 scaleup (节点内) 通信 ★
 *   • kIsScaleupNVLink=true  → 节点内使用 NVLink (TMA Store 直写)
 *   • kIsScaleupNVLink=false → 节点内使用 RDMA (降级模式, 无 NVLink 时)
 *   • 跨节点 (scaleout) 通信不在此 kernel, 由上层调度或其他 kernel 处理
 *   • 代码中搜索不到任何 scaleout 相关逻辑
 *
 * ==================== 整体架构图 ====================
 *
 *                    ┌─────────────────────────────────────────────────────────┐
 *                    │                     dispatch_impl()                     │
 *                    │                        Grid                              │
 *                    └─────────────────────────────────────────────────────────┘
 *                                        │
 *            ┌──────────────────────────┴──────────────────────────┐
 *            ▼                                                      ▼
 *   ┌─────────────────┐                                   ┌─────────────────┐
 *   │  Notify Warps   │                                   │ Dispatch Warps  │
 *   │ (kNumNotifyWarps)│                                   │(kNumDispatchWarps)│
 *   ├─────────────────┤                                   ├─────────────────┤
 *   │ • 统计专家分布   │                                   │ • TMA 加载数据   │
 *   │ • RDMA 广播计数  │                                   │ • 去重+分配Slot │
 *   │ • 计算前缀和    │                                   │ • NVLink/RDMA发送│
 *   │ • CPU同步(可选) │                                   │                 │
 *   └─────────────────┘                                   └─────────────────┘
 *
 * ==================== 通信模式图 ====================
 *
 *   本 kernel 的通信范围 (scaleup 域内):
 *
 *   rank 0                    rank 1                    rank 2
 *    │                          │                          │
 *    ├────── NVLink (节点内) ───┤├────── NVLink (节点内) ───┤
 *    │  或 RDMA (降级模式)      │  或 RDMA (降级模式)      │
 *    │                          │                          │
 *
 *   gin.put_value 自动选择路径:
 *   ┌──────────────────────────────────────────────────────┐
 *   │ if (dst_ptr != nullptr)                               │
 *   │     NVLink: ptx::st_relaxed_sys(dst_ptr, value)      │  ← 同节点直写
 *   │ else                                                  │
 *   │     RDMA: gin.putValue(dst_rank_idx, sym_ptr, value) │  ← 跨节点协议
 *   └──────────────────────────────────────────────────────┘
 *
 * ==================== Token 分发流程 ====================
 *
 *   Token[0] ──► topk_idx = [3, 7, 2] ──► 分发给 3 个不同的 Expert
 *    │                                        │
 *    ├── Expert 3 ──► rank 0 (本地)  ──► TMA Store (NVLink)
 *    ├── Expert 7 ──► rank 1 (节点内) ──→ RDMA/NVLink Put
 *    └── Expert 2 ──► rank 0 (本地)  ──► 合并到同一 Slot
 *
 *   Deduplicate 逻辑: 同一 Token 的多个 topk 可能选择同一 rank
 *                     只分配一个 slot, 避免重复写入
 */

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>       // ★ comm 命名空间: barrier 原语 + QP 分配 + 超时机制
                                          //   • nvlink_barrier_wo_local_sync  - NVLink 节点内 barrier (red_add_rel_sys)
                                          //   • gin_barrier_wo_local_sync     - RDMA Gin barrier (signal/wait)
                                          //   • scaleup_barrier_wo_local_sync - 按 kIsScaleupNVLink 选择上述两者
                                          //   • scaleout_barrier_wo_local_sync - 跨节点 barrier (ncclTeamTagRail)
                                          //   • gpu_barrier                   - 统一入口,自动处理 scaleup/scaleout 分层
                                          //   • timeout_while                 - 防死锁超时轮询
                                          //   • get_qp_mode                   - QP 分配给不同 SM/warp
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

/**
 * @brief Dispatch Kernel - MoE Expert Parallelism 核心通信原语
 *
 * @tparam kIsScaleupNVLink  true=节点内使用NVLink, false=节点内使用RDMA(降级模式)
 * @tparam kDoCPUSync       是否需要CPU同步获取精确token计数
 * @tparam kReuseSlotIndices 是否复用之前dispatch的slot索引(cached handle模式)
 * @tparam kNumSMs          使用的SM数量(影响并行度)
 * @tparam kNumNotifyWarps  通知线程数(用于全局计数同步)
 * @tparam kNumDispatchWarps 数据分发线程数(实际数据传输)
 * @tparam kNumRanks        总rank数(EP通信域大小)
 * @tparam kNumHiddenBytes  hidden维度字节数(BF16=2, FP8=1)
 * @tparam kNumSFPacks      FP8 scale factor pack数量
 * @tparam kNumMaxTokensPerRank 每个rank最大token数(缓冲区大小)
 * @tparam kNumExperts      总专家数
 * @tparam kNumTopk         每个token选择的专家数
 * @tparam kExpertAlignment expert接收token数的对齐要求
 * @tparam kNumQPs          RDMA queue pair数量
 * @tparam kNumTimeoutCycles 超时周期数
 *
 * ==================== Warp 分工图 ====================
 *
 *   Block 内线程布局:
 *   ┌────────────────────────────────────────────────────┐
 *   │ [Notify Warps (kNumNotifyWarps)] [Dispatch Warps] │
 *   │ warp_idx = 0 ~ kNumNotifyWarps-1   warp_idx >= kNumNotifyWarps │
 *   │                                                    │
 *   │ Notify: 处理元数据同步          Dispatch: 处理数据 │
 *   │ • 统计 expert/rank 分布         • TMA 加载token    │
 *   │ • RDMA 广播计数                 • 去重+分配Slot    │
 *   │ • 前缀和计算                    • 实际数据传输      │
 *   └────────────────────────────────────────────────────┘
 *
 *   例如: kNumNotifyWarps=4, kNumDispatchWarps=2
 *   warp 0-3: Notify Warps
 *   warp 4-5: Dispatch Warps
 */
template <bool kIsScaleupNVLink,
          bool kDoCPUSync,
          bool kReuseSlotIndices,
          int kNumSMs,
          int kNumNotifyWarps, int kNumDispatchWarps,
          int kNumRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumNotifyThreads = kNumNotifyWarps * 32,
          int kNumDispatchThreads = kNumDispatchWarps * 32,
          int kNumThreads = kNumNotifyThreads + kNumDispatchThreads,
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_impl(
    void* x,                                        ///< [in] 输入token tensor, shape [num_tokens, hidden]
    sf_pack_t* sf,                                  ///< [in] FP8 scale factors (如果kNumSFPacks>0)
    topk_idx_t* topk_idx,                           ///< [in] topk专家索引, shape [num_tokens, num_topk]
    float* topk_weights,                            ///< [in] topk权重, shape [num_tokens, num_topk]
    topk_idx_t* copied_topk_idx,                   ///< [out] 复制的topk_idx (用于cached handle)
    int* cumulative_local_expert_recv_stats,      ///< [out,可选] 累计本地expert接收统计
    int* psum_num_recv_tokens_per_scaleup_rank,    ///< [out] rank维前缀和(用于slot分配)
    int* psum_num_recv_tokens_per_expert,           ///< [out] expert维前缀和(用于slot分配)
    int* dst_buffer_slot_idx,                       ///< [out] 目标buffer slot索引
    const int num_tokens,                           ///< token总数
    const int sf_token_stride, const int sf_hidden_stride,  ///< SF的stride
    const ncclDevComm_t nccl_dev_comm,             ///< NCCL设备侧通信器
    const ncclWindow_t nccl_window,                 ///< NCCL窗口(RDMA上下文)
    void* buffer,                                   ///< 通信缓冲区
    void* workspace,                                ///< GPU工作空间(原子计数等)
    void* mapped_host_workspace,                    ///< 映射到CPU的工作空间
    const int rank_idx                              ///< 当前rank索引
) {
    /**
     * ==================== 内部变量图 ====================
     *
     *   smem (shared memory) 布局:
     *   ┌───────────────────────┬─────────────────────┐
     *   │  kNumSmemBytesForNotify │    TMA Buffer       │
     *   │    (Notify计数用)      │  (Dispatch数据用)    │
     *   ├───────────────────────┼─────────────────────┤
     *   │ rank_count[0..kNumRanks-1]                    │
     *   │ expert_count[0..kNumExperts-1]              │
     *   └─────────────────────────────────────────────┘
     */
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "专家数必须能被rank数整除");
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Warp group大小必须是4的倍数");

    // ========== 线程索引计算 ==========
    // sm_idx: 当前处于哪个SM (0 ~ kNumSMs-1)
    // warp_idx: 当前warp在block内的索引
    // lane_idx: warp内线程索引 (0~31)
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();

    // ========== 工作空间布局 ==========
    // workspace_layout: GPU端工作空间,存储各rank的计数、原子计数器等
    // host_workspace_layout: CPU端工作空间,用于CPU同步时写入计数
    //
    // ★ WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts) 参数说明:
    //   num_scaleout_ranks = 1: 本 kernel 按 scaleup 视角工作,无 scaleout 分层
    //   num_scaleup_ranks = kNumRanks: 所有 rank 在同一 scaleup 域
    //   num_experts = kNumExperts: 总专家数
    //
    // ★ WorkspaceLayout 完整内存布局 (固定大小预分配,按 kNumMax* 常量):
    //
    //   Offset 0:
    //   ┌─────────────────────────────────────────────────────────┐
    //   │ Barrier Signals (16B)                                    │
    //   │   nvl_barrier_counter(8B) + nvl_barrier_signal[2](8B)  │
    //   └─────────────────────────────────────────────────────────┘
    //   Offset 16:
    //   ┌─────────────────────────────────────────────────────────┐
    //   │ Notify Reduction Workspace (int64_t × 3072)             │
    //   │   用于跨 SM 计数归约 (red_add)                           │
    //   │   编码: (num_sms << 32) | count                          │
    //   └─────────────────────────────────────────────────────────┘
    //   ┌─────────────────────────────────────────────────────────┐
    //   │ Scaleup Rank/Expert Count (int64_t, Send/Recv双缓冲)    │
    //   │   Send (kIsSendBuffer=true, 偏移+0)                     │
    //   │   Recv (kIsSendBuffer=false, 偏移+kNumMaxRanks+kNumMaxExperts) │
    //   │   ★ false = 偏移量选择器,指向 Recv buffer (peer写入的)  │
    //   │   ★ true  = 偏移+0,指向 Send buffer (本地统计的)        │
    //   └─────────────────────────────────────────────────────────┘
    //   ┌─────────────────────────────────────────────────────────┐
    //   │ Scaleup Atomic Sender Counter (int × kNumMaxRanks)       │
    //   │   每 rank 一个原子计数器,用于 slot 分配 (atomicAdd)       │
    //   └─────────────────────────────────────────────────────────┘
    //   ┌─────────────────────────────────────────────────────────┐
    //   │ Scaleout Rank/Expert Count (int, Send/Recv双缓冲)       │
    //   │   (本 kernel 不使用,为 hybrid 模式预留)                  │
    //   └─────────────────────────────────────────────────────────┘
    //   ┌─────────────────────────────────────────────────────────┐
    //   │ Scaleout Channel Metadata + PP Count + AGRS Signals     │
    //   │   (本 kernel 不使用,为其他功能预留)                       │
    //   └─────────────────────────────────────────────────────────┘
    //
    //   ★ 设计要点: 链式偏移 + 对称内存
    //   - 每个 get_*_ptr() 从上一区域末尾计算偏移
    //   - 所有 rank 的 workspace 布局完全相同
    //   - put_value 通过 sym_ptr 偏移直接定位目标 rank 的对应位置
    //
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, 1, kNumRanks, kNumExperts);

    /**
     * ==================== Shared Memory 图 ====================
     *
     *   smem 分配 (动态shared memory):
     *   ┌──────────────────────────────────────────────────────┐
     *   │                                                    │
     *   │  ┌──────────────┐  ┌────────────────────────────┐ │
     *   │  │ rank_expert_ │  │      TMA Buffer (SMEM)     │ │
     *   │  │ count (Notify)│  │                            │ │
     *   │  │              │  │  • hidden data              │ │
     *   │  │ rank_count   │  │  • scale factors (optional)│ │
     *   │  │ [kNumRanks]  │  │  • topk_idx                 │ │
     *   │  │              │  │  • topk_weights             │ │
     *   │  │ expert_count │  │  • src_token_idx            │ │
     *   │  │ [kNumExperts]│  │                            │ │
     *   │  └──────────────┘  └────────────────────────────┘ │
     *   │                                                    │
     *   │  kNumSmemBytesForNotify                           │
     *   └────────────────────────────────────────────────────┘
     */
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];

    // Notify section 需要对齐到 TMA 对齐字节数
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "TMA对齐失败");

    // Named barrier用于同grid内warp同步
    // ★ kNotifyBarrierIndex = 1 的作用:
    //   CUDA 提供 16 个 named barrier (0-15), idx=1 专属 Notify Warps
    //   这样 Notify Warps 和 Dispatch Warps 可以独立同步,互不干扰:
    //   - Notify Warps → named_barrier(idx=1, kNumNotifyThreads)
    //   - Dispatch Warps → 使用自己的 barrier 同步
    //
    //   named_barrier 实现: asm volatile("bar.sync %0, %1;" :: idx, num_threads)
    //   指定数量的线程到达同一同步点后才继续执行
    constexpr int kNotifyBarrierIndex = 1;

    /**
     * ==================== NCCL Gin 初始化图 ====================
     *
     *   Gin (NCCL Gin backend) - 轻量级RDMA/NVLink通信句柄
     *
     *   QP (Queue Pair) 分配策略:
     *   ┌─────────────────────────────────────────┐
     *   │           kNumSMs = 12, kNumQPs         │
     *   │                                         │
     *   │  SM 0-11 各分配一个 QP context         │
     *   │  每个 SM 的 warp 共享同一个 QP         │
     *   │                                         │
     *   │  sharing_mode: warp间是否共享QP        │
     *   └─────────────────────────────────────────┘
     */
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumDispatchWarps, (kNumNotifyWarps > 0)>(
        sm_idx, warp_idx - kNumNotifyWarps, warp_idx < kNumNotifyWarps);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // ========== 网格同步Barrier ==========
    // 确保所有SM在开始前都就绪,无TMA store flush,无prologue grid sync
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag0, false, false, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    /**
     * ╔═══════════════════════════════════════════════════════════════════════╗
     * ║                        NOTIFY WARPS 分支                              ║
     * ║                    (处理元数据: 计数、广播、前缀和)                     ║
     * ╚═══════════════════════════════════════════════════════════════════════╝
     *
     * ==================== Notify Warps 详细流程图 ====================
     *
     *   Step 1: 清零计数
     *   ┌────────────────────────────────────────────────────┐
     *   │  rank_count[0..kNumRanks-1] = 0                   │
     *   │  expert_count[0..kNumExperts-1] = 0               │
     *   └────────────────────────────────────────────────────┘
     *
     *   Step 2: 遍历所有token,原子增加计数
     *   ┌────────────────────────────────────────────────────┐
     *   │  for each token i:                               │
     *   │    for each top-k j:                             │
     *   │      expert_idx = topk_idx[i][j]                 │
     *   │      expert_count[expert_idx]++  (atomicAdd)     │
     *   │      rank_idx = expert_idx / kNumExpertsPerRank │
     *   │      rank_count[rank_idx]++  (atomicAdd,去重后) │
     *   └────────────────────────────────────────────────────┘
     *
     *   Step 3: 全网格归约
     *   ┌────────────────────────────────────────────────────┐
     *   │  all reduce: rank_count[], expert_count[]        │
     *   │  每个SM的计数合并到一起                           │
     *   └────────────────────────────────────────────────────┘
     *
     *   Step 4: SM 0 执行以下操作:
     *   ├── RDMA 将 rank_count 发送到各 peer rank
     *   ├── RDMA 将 expert_count 发送到各 peer rank
     *   ├── 等待返回值
     *   ├── 计算 prefix sum
     *   └── 写入 host workspace (可选CPU同步)
     */
    if (warp_idx < kNumNotifyWarps) {
        // ========== 分配 Shared Memory ==========
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        // ========== 清零计数数组 ==========
        // 每个线程负责清零一段,利用并行加速
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;
        #pragma unroll
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        /**
         * ==================== Token 遍历 + 计数图 ====================
         *
         *   Thread Layout:
         *   warp 0: 处理 token 0, kNumNotifyWarps*kNumSMs, 2*kNumNotifyWarps*kNumSMs, ...
         *   warp 1: 处理 token 1, 1+kNumNotifyWarps*kNumSMs, ...
         *   ...
         *
         *   每个 lane 处理一个 top-k 选择:
         *   lane 0 -> topk_idx[..., 0]
         *   lane 1 -> topk_idx[..., 1]
         *   ...
         *   lane kNumTopk-1 -> topk_idx[..., kNumTopk-1]
         */
        EP_STATIC_ASSERT(kNumTopk <= 32, "Lane数不足以处理这么多topk");
        const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;

        // 遍历属于本 warp 的 tokens
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
            // ========== 统计 Expert 选择 ==========
            // 每个 lane 读取一个 topk 选择,最多 kNumTopk 个 lane 有有效数据
            // lane_idx >= kNumTopk 时, dst_expert_idx = -1 (无效)
            const auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;

            // 有效选择: expert_idx >= 0
            // ★ atomicAdd_block vs atomicAdd:
            //   atomicAdd_block  = block 级原子操作, 作用于 shared memory
            //   atomicAdd        = 全局原子操作, 作用于 global memory
            //   atomicAdd_system = 系统级原子操作 (CPU+GPU)
            //   这里 expert_count 在 shared memory 中, 只需 block 级原子, 开销更小
            //   Block 间的归约由后续的 ptx::red_add 到全局 workspace 完成
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            /**
             * ==================== Rank 去重图 ====================
             *
             *   场景: 一个 token 的 topk 可能选中同一个 rank 上的多个 expert
             *   例如: topk_idx = [7, 15, 23], kNumExpertsPerRank = 8
             *   - expert 7 -> rank 0 (7/8 = 0)
             *   - expert 15 -> rank 1 (15/8 = 1)
             *   - expert 23 -> rank 2 (23/8 = 2)
             *
             *   假设另一个 token: topk_idx = [8, 16]
             *   - expert 8 -> rank 1 (8/8 = 1)
             *   - expert 16 -> rank 2 (16/8 = 2)
             *   -> rank 1 和 rank 2 被选择两次,但只需分配一个 slot
             */
            // 计算目标 rank 索引
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;

            // deduplicate: 同一 warp 内同一 rank 只计算一次
            // 只有 lane 0 会执行 atomicAdd (基于 lane 0 的值为准)
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        /**
         * ==================== 全网格归约图 ====================
         *
         *   目的: 将所有 SM 的计数合并
         *
         *   ★ 编码机制详解:
         *   每个 SM 写入: counter = (1ll << 32ll) | local_count
         *                 高32位始终是 1 (不是递增!), 低32位是本 SM 的计数
         *   通过 ptx::red_add (原子加法) 累加:
         *     workspace[i] = Σ [(1<<32)|count_0, (1<<32)|count_1, ...]
         *                  = (kNumSMs << 32) | (count_0 + count_1 + ...)
         *   判断完成: (status >> 32) == kNumSMs
         *
         *   示例 (4 个 SM):
         *   ┌────────────────────────────────────────────────┐
         *   │ SM 0 写入: (1<<32) | 100 = 0x1_00000064      │
         *   │ SM 1 写入: (1<<32) | 150 = 0x1_00000096      │
         *   │ SM 2 写入: (1<<32) | 80  = 0x1_00000050      │
         *   │ SM 3 写入: (1<<32) | 70  = 0x1_00000046      │
         *   │                                                │
         *   │ red_add 累加后:                                 │
         *   │ workspace = 0x4_00000190                      │
         *   │            = (4 << 32) | (100+150+80+70)       │
         *   │ status >> 32 = 4 == kNumSMs ✓ 就绪!            │
         *   └────────────────────────────────────────────────┘
         */
        #pragma unroll
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // ========== SM 0: 等待归约完成并处理后续 ==========
        if (sm_idx == 0) {
            /**
             * ==================== 等待 + 写入共享内存图 ====================
             *
             *   轮询 workspace,直到所有SM都完成写入:
             *   ┌─────────────────────────────────────────────────────┐
             *   │ do {                                                │
             *   │   status = *workspace_ptr;                          │
             *   │ } while ((status >> 32) != kNumSMs);  // 等待所有SM  │
             *   │                                                     │
             *   │ decoded_count = status & 0xFFFFFFFF;                │
             *   │ rank_expert_count[i] = decoded_count;               │
             *   │ workspace[i] = 0;  // 清零以便下次使用               │
             *   └─────────────────────────────────────────────────────┘
             *
             *   timeout 机制: 如果超时,打印错误信息但继续执行
             */
            #pragma unroll
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>(true, [=](const bool& is_last_check) {
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    if ((status >> 32) == kNumSMs) {
                        // 编码/解码正数 (用于传输)
                        const auto encoded =
                            math::encode_decode_positive(static_cast<int>(status & 0xffffffffll));
                        rank_expert_count[i] = encoded;

                        // 如果是 NVLink scaleup,写入对应的 workspace
                        if constexpr (not kIsScaleupNVLink)
                            workspace_layout.get_scaleup_rank_expert_count_ptr<true>()[i] = encoded;

                        // 清零为下次使用做准备
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    if (is_last_check) {
                        printf("DeepEP notify (GPU reduction) timeout, rank: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               rank_idx, kNumRanks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }
                    return false;
                });
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            /**
             * ╔═══════════════════════════════════════════════════════════════════════╗
             * ║                    RDMA 广播阶段                                     ║
             * ╠═══════════════════════════════════════════════════════════════════════╣
             * ║                                                                       ║
             * ║   rank_count 分发图:                                                 ║
             * ║                                                                       ║
             * ║   SM 0 计算 rank_count[i] 后,使用 Gin.put_value()                    ║
             * ║   发送到 peer rank i 的 workspace                                    ║
             * ║                                                                       ║
             * ║   ┌─────────┐      put_value(rank_count[0])       ┌─────────┐        ║
             * ║   │ rank 0  │ ──────────────────────────────────► │ rank 0  │        ║
             * ║   │ (SM 0)  │                                     │ workspace│        ║
             * ║   └─────────┘                                     └─────────┘        ║
             * ║        │                                                    │        ║
             * ║        │ put_value(rank_count[1])                          │        ║
             * ║        ▼                                                    ▼        ║
             * ║   ┌─────────┐                                     ┌─────────────────┐║
             * ║   │ rank 1  │                                     │ rank 1 workspace │║
             * ║   │ ...    │                                     │ (最终存储所有rank │║
             * ║   └─────────┘                                     │  的计数)         │║
             * ║                                                     └─────────────────┘║
             * ╚═══════════════════════════════════════════════════════════════════════╝
             */
            // TODO: for further optimization, we can fuse rank and expert counters

            // ========== 发送 rank 计数到 peer ranks ==========
            // ★ put_value 参数: (sym_ptr, value, dst_rank_idx, extra_options)
            //   sym_ptr: 对称内存偏移 (相对于目标 rank 的 workspace)
            //   value: 要写入的值
            //   dst_rank_idx: 目标 rank 索引
            //   extra_options: ncclGinOptFlagsAggregateRequests (聚合请求优化)
            //
            // ★ put_value 内部自动选择通信路径:
            //   if (get_sym_ptr() != nullptr)
            //       NVLink: ptx::st_relaxed_sys(dst_ptr, value)  ← 同节点直写
            //   else
            //       RDMA: gin.putValue(dst_rank_idx, sym_ptr, value, ...)  ← 跨节点
            //
            // ★ dst_rank_counter 使用 Recv buffer (kIsSendBuffer=false):
            //   因为是写入目标 rank 的接收区, 目标 rank 后续从自己的 Recv buffer 读取
            //   false 只是偏移量选择器: true=偏移+0(Send), false=偏移+kNumMaxRanks+kNumMaxExperts(Recv)
            for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                const auto dst_rank_counter =
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + rank_idx;
                gin.put_value<team_t>(dst_rank_counter, static_cast<int64_t>(rank_count[i]), i,
                                      ncclGinOptFlagsAggregateRequests);
            }
            __syncwarp();

            /**
             * ==================== Expert Count 分发图 ====================
             *
             *   ★ 为什么 Expert Count 需要判断 kIsScaleupNVLink 而 Rank Count 不需要?
             *
             *   Rank Count: kNumRanks=8 次 put_value
             *     → 无论 NVLink(st_relaxed_sys) 还是 RDMA(gin.putValue), 8 次都很快
             *     → 不需要优化, put_value 内部自动选路径即可
             *
             *   Expert Count: kNumExperts=256 次 put_value
             *     NVLink 模式: 256 次 st_relaxed_sys 直写 → 每次开销 ~ns → 可接受
             *     RDMA 模式:  256 次 gin.putValue → 每次开销 ~us (QP/ACK) → 太贵!
             *     → RDMA 模式改用 gin.put 批量发送: 8 次 × 每次发 32 个值
             *
             *   NVLink 路径:
             *   ┌────────────────────────────────────────────────────┐
             *   │  for each expert i:                               │
             *   │    expert_rank = i / kNumExpertsPerRank;          │
             *   │    src_idx = i % kNumExpertsPerRank;             │
             *   │    dst_rank = expert_rank;                        │
             *   │    gin.put_value(expert_count[i]) to dst_rank    │
             *   └────────────────────────────────────────────────────┘
             *
             *   RDMA 路径:
             *   ┌────────────────────────────────────────────────────┐
             *   │  for each peer rank i:                           │
             *   │    src = scaleup_expert_count_ptr + i*kNumExpertsPerRank │
             *   │    dst = scaleup_expert_count_ptr + rank_idx*kNumExpertsPerRank │
             *   │    gin.put(dst, src, kNumExpertsPerRank * sizeof(int64_t)) │
             *   └────────────────────────────────────────────────────┘
             */
            if constexpr (kIsScaleupNVLink) {
                // NVLink per-element copy (因为 shared memory 和 global 类型不同,不能用 TMA)
                for (int i = thread_idx; i < kNumExperts; i += kNumNotifyThreads) {
                    const auto idx = kNumExpertsPerRank * rank_idx + (i % kNumExpertsPerRank);
                    gin.put_value<team_t>(
                        workspace_layout.get_scaleup_expert_count_ptr<false>() + idx,
                        static_cast<int64_t>(expert_count[i]), i / kNumExpertsPerRank);
                }
            } else {
                // RDMA bulk copy
                for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                    const auto src_ptr = workspace_layout.get_scaleup_expert_count_ptr<true>() + kNumExpertsPerRank * i;
                    const auto dst_ptr = workspace_layout.get_scaleup_expert_count_ptr<false>() + kNumExpertsPerRank * rank_idx;
                    gin.put<team_t>(dst_ptr, src_ptr, kNumExpertsPerRank * sizeof(int64_t), i);
                }
            }

            // 必须等待, 因为被等待的结果会重写共享内存
            // ★ 这个 named_barrier(idx=1) 确保所有 Notify Warps 完成 RDMA 广播写入后,
            //   再进入"等待远程计数"阶段读取 shared memory
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            /**
             * ==================== 等待远程计数图 ====================
             *
             *   轮询 scaleup_rank_expert_count 的 Recv buffer,
             *   直到所有 peer rank 都写入
             *
             *   ★ encode_decode_positive 编码机制:
             *
             *   math.cuh 定义:
             *     encode_decode_positive(n) = -n - 1    (正数 → 负数)
             *     is_decoded_positive_ready(v) = v >= 0  (非负 = 就绪)
             *
             *   双重编码 = 还原: -(-n-1)-1 = n
             *
             *   ★ 解决 "0 的歧义":
             *     不编码时: recv buffer 初始值=0, peer写入0个token也是0 → 无法区分!
             *     编码后:
             *       初始值 0 → encode_decode_positive(0) = -1 → ready? -1>=0 → false ✓
             *       peer 写 0 → 发送 encode(0)=-1 → recv=-1 → decode(-1)=0 → 0>=0 → true ✓
             *       peer 写 3 → 发送 encode(3)=-4 → recv=-4 → decode(-4)=3 → 3>=0 → true ✓
             *
             *   发送端 (第 370-372 行):
             *     encoded = encode_decode_positive(count)  → 编码为负数
             *     rank_expert_count[i] = encoded
             *     gin.put_value(dst, encoded, peer_rank)
             *
             *   接收端 (本段代码):
             *     count = ld_volatile(recv_buffer + i)     → 读取编码值
             *     decoded = encode_decode_positive(count)  → 还原为正数 (双重编码=解码)
             *     is_decoded_positive_ready(decoded)       → decoded>=0 则就绪
             *
             *   目标 workspace 布局:
             *   ┌─────────────────────────────────────────────────────┐
             *   │ [rank 0 count] [rank 1 count] ... [expert 0 count]│
             *   └─────────────────────────────────────────────────────┘
             */
            const auto start_clock = clock64();
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    const auto count = static_cast<int>(
                        ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + i));
                    const auto decoded = math::encode_decode_positive(count);
                    if (math::is_decoded_positive_ready(decoded)) {
                        workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] = 0;
                        rank_expert_count[i] = decoded;
                        return true;
                    }

                    if (is_last_check)
                        printf("DeepEP notify timeout, rank: %d, thread: %d, count: %d\n", rank_idx, i, decoded);
                    return false;
                }, start_clock);
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            /**
             * ==================== Expert 计数对齐图 ====================
             *
             *   对齐目的: 简化后续 GEMM,避免处理非对齐的情况
             *
             *   对齐公式: aligned_count = align(sum, kExpertAlignment)
             *
             *   示例: kExpertAlignment = 8
             *   - sum = 17 -> aligned = 24
             *   - sum = 8  -> aligned = 8
             *   - sum = 15 -> aligned = 16
             */
            for (int i = thread_idx; i < kNumExpertsPerRank; i += kNumNotifyThreads) {
                int sum = 0;
                #pragma unroll
                for (int j = 0; j < kNumRanks; ++ j)
                    sum += expert_count[j * kNumExpertsPerRank + i];
                expert_count[i] = math::align(sum, kExpertAlignment);

                // 更新可选的统计计数器
                if (cumulative_local_expert_recv_stats != nullptr)
                    atomicAdd(cumulative_local_expert_recv_stats + i, sum);
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            /**
             * ==================== CPU 同步图 ====================
             *
             *   如果 kDoCPUSync=true,写入映射的 host workspace
             *   CPU 端可以读取这些值来获取精确的 token 计数
             *
             *   用途: 在线负载均衡监控、调试信息等
             */
            if constexpr (kDoCPUSync) {
                for (int i = thread_idx; i < kNumRanks + kNumExpertsPerRank; i += kNumNotifyThreads) {
                    host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] =
                        math::encode_decode_positive(rank_expert_count[i]);
                }
                __syncwarp();
            }

            /**
             * ==================== 前缀和计算图 ====================
             *
             *   Inclusive Prefix Sum (前缀和):
             *   输入: [3, 1, 4, 2]
             *   输出: [3, 4, 8, 10]
             *
             *   用于:
             *   - psum_num_recv_tokens_per_scaleup_rank: 计算每个 rank 的起始 slot
             *   - psum_num_recv_tokens_per_expert: 计算每个 expert 的起始 slot
             *
             *   rank_count 例子:
             *   输入: [10, 5, 8, 3] (每个rank的token数)
             *   输出: [10, 15, 23, 26] (每个rank的起始偏移)
             *
             *   expert_count 例子:
             *   输入: [16, 16, 16, 16] (每个local expert的token数,已对齐)
             *   输出: [16, 32, 48, 64] (每个expert的起始偏移)
             */
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                #pragma unroll
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;

                    // Warp 内 inclusive sum
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    // 写入全局内存
                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    // 更新 psum, 使用最后一个 lane 的值
                    psum = ptx::exchange(sum, 31);
                }
            };

            // warp 0: 计算 rank 的 inclusive prefix sum
            if (warp_idx == 0) {
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumRanks, 0);
            }
            // warp 1: 计算 expert 的 exclusive prefix sum (用于 expand 模式)
            else if (warp_idx == 1) {
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
        }
    }

    /**
     * ╔═══════════════════════════════════════════════════════════════════════╗
     * ║                      DISPATCH WARPS 分支                              ║
     * ║                    (处理实际数据: 加载、分发、发送)                       ║
     * ╚═══════════════════════════════════════════════════════════════════════╝
     *
     * ==================== Dispatch Warps 详细流程图 ====================
     *
     *   ┌─────────────────────────────────────────────────────────────────┐
     *   │                                                                 │
     *   │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
     *   │  │  TMA Load     │───►│ Load topk     │───►│ Deduplicate   │  │
     *   │  │  Token Data   │    │ idx/weights   │    │ + Slot Alloc  │  │
     *   │  └───────────────┘    └───────────────┘    └───────────────┘  │
     *   │         │                   │                    │            │
     *   │         ▼                   ▼                    ▼            │
     *   │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
     *   │  │ TMA Load SF   │    │ Store metadata│    │  Issue TMA   │  │
     *   │  │ (if FP8)      │    │ (src rank/idx)│    │  NVLink Store│  │
     *   │  └───────────────┘    └───────────────┘    └───────────────┘  │
     *   │                                                      │        │
     *   │                                                      ▼        │
     *   │                         ┌──────────────────────────────────┐  │
     *   │                         │      Issue RDMA Put              │  │
     *   │                         │   (for non-NVLink ranks)        │  │
     *   │                         └──────────────────────────────────┘  │
     *   └─────────────────────────────────────────────────────────────────┘
     */
    else {
        const int dispatch_warp_idx = warp_idx - kNumNotifyWarps;

        /**
         * ==================== Buffer Layout 图 ====================
         *
         *   通信缓冲布局:
         *   ┌─────────────────────────────────────────────────────────────┐
         *   │                     Recv Buffer                             │
         *   │  ┌─────────┬─────────┬─────────┬─────────┬─────────┐        │
         *   │  │ Rank 0  │ Rank 1  │ Rank 2  │ ...     │ Rank N  │        │
         *   │  │ tokens  │ tokens  │ tokens  │         │ tokens  │        │
         *   │  │ [max]   │ [max]   │ [max]   │         │ [max]   │        │
         *   │  └─────────┴─────────┴─────────┴─────────┴─────────┘        │
         *   └─────────────────────────────────────────────────────────────┘
         *
         *   Send Buffer (紧随 recv buffer):
         *   ┌─────────────────────────────────────────────────────────────┐
         *   │                    Send Buffer                             │
         *   │  ┌─────────────────────────────────────────────────────┐   │
         *   │  │  [kNumMaxTokensPerRank tokens]                       │   │
         *   │  │  用于存储本 warp 发出的 token (可能被 RDMA 读取)     │   │
         *   │  └─────────────────────────────────────────────────────┘   │
         *   └─────────────────────────────────────────────────────────────┘
         *
         *   TMA Buffer (SMEM, 每个 dispatch warp 独立):
         *   ┌─────────────────────────────────────────────────────────────┐
         *   │  • hidden data (kNumHiddenBytes)                          │
         *   │  • scale factors (if FP8)                                  │
         *   │  • topk_idx [kNumTopk]                                     │
         *   │  • topk_weights [kNumTopk]                                 │
         *   │  • src_token_global_idx (用于 combine 阶段)                 │
         *   └─────────────────────────────────────────────────────────────┘
         */
        // Token layout: hidden + SF + topk
        const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);

        // TMA buffer: 指向 shared memory
        const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumDispatchWarps, 1,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(dispatch_warp_idx).get_token_buffer(0);

        // Recv buffer 和 Send buffer
        auto recv_buffer = layout::BufferLayout<false>(token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);
        auto send_buffer = layout::BufferLayout<false>(token_layout, 1, kNumMaxTokensPerRank, recv_buffer.get_buffer_end_ptr());
        recv_buffer = recv_buffer.get_rank_buffer(rank_idx);

        /**
         * ==================== TMA 初始化图 ====================
         *
         *   Multicast Barrier (mbarrier) 用于同步 TMA 操作
         *
         *   流程:
         *   1. elect_one_sync() 选出一个 leader thread
         *   2. Leader 初始化 mbarrier,值为1 (等待一个 arrive)
         *   3. 所有线程 __syncwarp()
         */
        ptx::arrival_phase phase = 0;
        const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
        if (ptx::elect_one_sync())
            ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        __syncwarp();

        /**
         * ==================== Token 迭代图 ====================
         *
         *   每个 dispatch warp 处理一组 tokens,跨所有 SM 交错
         *
         *   示例: kNumDispatchWarps=2, kNumSMs=4
         *   - warp 0: token 0, 8, 16, 24, ...  (dispatch_warp_idx=0)
         *   - warp 1: token 1, 9, 17, 25, ...  (dispatch_warp_idx=1)
         *
         *   这样的交错设计确保:
         *   1. 负载均衡 (所有 SM 处理不同 warp 的 token)
         *   2. 内存访问交错 (减少 bank conflict)
         */
        const auto token_start = dispatch_warp_idx * kNumSMs + sm_idx;
        const auto token_stride = kNumDispatchWarps * kNumSMs;

        /**
         * ╔═══════════════════════════════════════════════════════════════════════╗
         * ║                       TOKEN 主循环                                    ║
         * ╚═══════════════════════════════════════════════════════════════════════╝
         */
        for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {
            const auto token_i64_idx = static_cast<int64_t>(token_idx);

            // ========== 等待上一个 token 的 TMA store 完成 ==========
            ptx::tma_store_wait();
            __syncwarp();

            /**
             * ==================== TMA Load Token Data ====================
             *
             *   使用 TMA 将数据从 global memory 加载到 shared memory
             *
             *   ┌─────────────────────────────────────────────────────────┐
             *   │  x: [num_tokens, hidden]                               │
             *   │       ▲                                               │
             *   │       └── token_i64_idx 偏移                          │
             *   │                                                       │
             *   │  TMA Load: 全量加载 kNumHiddenBytes                   │
             *   │  目的地址: tma_buffer.hidden_ptr (smem)                │
             *   └─────────────────────────────────────────────────────────┘
             */
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            __syncwarp();

            /**
             * ==================== TMA Load Scale Factors (FP8) ====================
             *
             *   FP8 模式: 每个 token 有独立的 scale factor
             *
             *   布局: sf[token, kNumSFPacks, hidden/kNumSFPacks]
             *   加载 stride: sf_token_stride, sf_hidden_stride
             *
             *   使用 cp.async 指令进行异步加载
             */
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "SF元素类型未对齐");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                // 分批次加载,每批32个元素
                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                // 处理剩余元素
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                __syncwarp();
            }

            /**
             * ==================== 加载 Top-k Indices 和 Weights ====================
             *
             *   每个 lane 负责一个 top-k 选择:
             *   lane 0 -> topk_idx[token, 0], topk_weights[token, 0]
             *   lane 1 -> topk_idx[token, 1], topk_weights[token, 1]
             *   ...
             *
             *   同时复制 topk_idx 到 copied_topk_idx (用于返回给 Python)
             */
            EP_STATIC_ASSERT(kNumTopk <= 32, "Lane数不足以加载topk");
            int stored_dst_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
                stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;

                // 写入 TMA buffer 的 metadata
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            __syncwarp();

            /**
             * ==================== 写入源元数据 ====================
             *
             *   用途: combine 阶段需要知道每个 token 的原始来源 (rank + local index)
             *
             *   编码: src_token_global_idx = rank_idx * kNumMaxTokensPerRank + token_idx
             */
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            ptx::tma_store_fence();  // 确保在 TMA store 之前写入
            __syncwarp();

            /**
             * ==================== 去重 + Slot 分配 ====================
             *
             *   问题: 同一 token 的多个 top-k 可能选择同一个 rank
             *   解决: deduplicate + atomicAdd 分配唯一 slot
             *
             *   两种模式:
             *   1. kReuseSlotIndices = false: 新分配 slot
             *   2. kReuseSlotIndices = true: 复用 handle 中的 slot (cached handle)
             *
             *   ┌─────────────────────────────────────────────────────┐
             *   │ 场景: token 0, topk_idx = [3, 11, 19]              │
             *   │       kNumExpertsPerRank = 8                        │
             *   │                                                │
             *   │  expert 3  -> rank 0 (3/8=0)                    │
             *   │  expert 11 -> rank 1 (11/8=1)                    │
             *   │  expert 19 -> rank 2 (19/8=2)                    │
             *   │                                                │
             *   │  Deduplicate 结果: 3 个不同的 rank               │
             *   │  -> 分配 3 个 slot: [0, 1, 2]                   │
             *   └─────────────────────────────────────────────────────┘
             */
            int stored_dst_slot_idx = -1;
            if constexpr (kReuseSlotIndices) {
                // 复用模式: 从 dst_buffer_slot_idx 读取
                if (lane_idx < kNumTopk)
                    stored_dst_slot_idx = __ldg(dst_buffer_slot_idx + token_idx * kNumTopk + lane_idx);
                // 转换为 local slot index (移除 rank 偏移)
                stored_dst_slot_idx = stored_dst_slot_idx >= 0 ?
                    (stored_dst_slot_idx - rank_idx * kNumMaxTokensPerRank) : -1;
            } else {
                // 新分配模式: atomicAdd
                if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
                    stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);

                // 写入全局 slot 索引 (用于后续 combine)
                if (lane_idx < kNumTopk) {
                    const auto value = stored_dst_slot_idx >= 0 ?
                        rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
                    dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = value;
                }
            }
            __syncwarp();

            /**
             * ==================== 等待 TMA Load 完成 ====================
             *
             *   mbarrier 用于确保:
             *   1. TMA load 数据已到达 smem
             *   2. 可以安全地执行 TMA store
             */
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
            }
            __syncwarp();

            /**
             * ==================== NVLink TMA Store ====================
             *
             *   通过 NVLink 发送数据到本地或其他节点
             *   使用 Gin 的对称内存指针获取目标地址
             *
             *   地址计算:
             *   recv_buffer.get_token_buffer(slot).get_base_ptr()
             *   相对于目标 rank 的偏移
             */
            auto send_buffer_ptr = send_buffer.get_token_buffer(token_idx).get_base_ptr();
            if constexpr (not kIsScaleupNVLink) {
                // RDMA 模式: 先存储到 send buffer (RDMA 会读取这个 buffer)
                if (ptx::elect_one_sync())
                    ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                ptx::tma_store_commit();
                __syncwarp();
            }

            // ========== NVLink 发送 ==========
            EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid top-k selection");

            // 获取目标地址 (gin.get_sym_ptr 返回对称内存地址)
            const auto dst_ptr = stored_dst_slot_idx >= 0 ?
                gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), stored_dst_rank_idx) :
                nullptr;

            // 执行 TMA store (如果 dst_ptr != nullptr, 即目标可达)
            if (dst_ptr != nullptr)
                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            ptx::tma_store_commit();
            __syncwarp();

            /**
             * ==================== RDMA Put ====================
             *
             *   对于非 NVLink 可达的 rank (跨节点), 使用 RDMA 发送
             *
             *   条件:
             *   - stored_dst_slot_idx >= 0: 有效 slot
             *   - dst_ptr == nullptr: NVLink 不可达,需要 RDMA
             *
             *   数据来源: send_buffer (在前面 TMA store 时写入)
             */
            if constexpr (not kIsScaleupNVLink) {
                // 等待 send buffer TMA store 完成
                ptx::tma_store_wait<1>();
                __syncwarp();

                // 跨节点 RDMA 发送
                if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
                    gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                    send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
                }
                __syncwarp();
            }
        }
    }

    /**
     * ==================== 网格同步 Barrier ====================
     *
     *   确保所有 SM 都完成数据发送
     *   参数:
     *   - comm::kDispatchTag1: 区分不同阶段的 barrier
     *   - true: prologue grid sync (等待所有 SM)
     *   - true: epilogue grid sync (确保数据可见)
     *   - false: 不 flush TMA store (由下一个 barrier 处理)
     */
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag1, true, true, false>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    /**
     * ==================== 触发 Copy Epilogue ====================
     *
     *   cudaTriggerProgrammaticLaunchCompletion()
     *   通知 CUDA 运行时启动下一个 kernel (copy epilogue)
     *   这是 V2 架构的一部分: dispatch + copy epilogue 分离
     */
    cudaTriggerProgrammaticLaunchCompletion();

    // ========== 清零原子计数器 ==========
    // 为下一次 dispatch 做准备
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "线程数不足以处理所有rank");
    if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumRanks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace deep_ep::elastic