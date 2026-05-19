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
    // get_warp_idx: PTX mov.s32 %laneid + shfl, 获取当前线程在 block 中的 warp 索引
    const auto warp_idx = ptx::get_warp_idx();
    // get_lane_idx: PTX mov.s32 %laneid, 获取当前线程在 warp 内的索引 (0~31)
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

    // ========== 开始前 Barrier ==========
    // gpu_barrier 模板签名:
    //   template<bool kIsScaleupNVLink,
    //            int kNumScaleoutRanks, int kNumScaleupRanks,
    //            int kNumSMs, int kNumThreads, int kNumQPs,
    //            int64_t kNumTimeoutCycles, int kTag,
    //            bool kFlushStores, bool kSyncAtStart, bool kSyncAtEnd>
    //
    //   参数对照:
    //   模板参数                类型      本调用值         含义
    //   ──────────────────     ──────    ──────────       ──────────
    //   kIsScaleupNVLink       bool      运行时           节点内通信走 NVLink 还是 RDMA
    //   kNumScaleoutRanks      int       1                跨节点 rank 数 (dispatch 无跨节点,=1)
    //   kNumScaleupRanks       int       kNumRanks        节点内 rank 数
    //   kNumSMs                int       kNumSMs          使用的 SM 数量
    //   kNumThreads            int       kNumThreads      每 SM 线程数
    //   kNumQPs                int       kNumQPs          QP (Queue Pair) 数量
    //   kNumTimeoutCycles      int64     kNumTimeoutCycles 防死锁超时周期数
    //   kTag                   int       kDispatchTag0    barrier 标签 (区分不同阶段)
    //   kFlushStores           bool      false            不 flush TMA store (还没发数据)
    //   kSyncAtStart           bool      false            barrier 前不 grid sync (各自开始)
    //   kSyncAtEnd             bool      true             barrier 后 grid sync (等所有 SM 一起开始干活)
    //
    //   运行时参数             本调用值         含义
    //   ──────────────────     ──────────       ──────────
    //   gin                    NCCLGin          通信句柄
    //   workspace_layout       workspace        工作区布局
    //   scaleout_rank_idx      0                跨节点 rank 索引 (无跨节点,=0)
    //   scaleup_rank_idx       rank_idx         节点内 rank 索引
    //   sm_idx                 sm_idx           当前 SM 编号
    //   thread_idx             thread_idx       当前线程编号
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

            // deduplicate: warp 内按 value 去重, 只让每个唯一值对应的最高 lane 执行
            //   实现: match(value) 找相同值的 lane 掩码, get_master_lane_idx 取最高位, 与自身 lane_idx 比较
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        // named_barrier: PTX bar.sync, 命名屏障, 指定参与线程数同步
        //   kNotifyBarrierIndex=1 隔离 Notify Warps 和 Dispatch Warps 的同步
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
            // red_add: PTX red.gpu.global.add.u64, 全局内存原子加法 (GPU域, 无内存序保证)
            //   将 counter 累加到 workspace 的归约区域, 多个 SM 的计数通过原子加合并
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
                    // ld_volatile: PTX ld.volatile.global, volatile 加载 (编译器不优化/重排)
                    //   保证每次都从内存读取最新值, 用于轮询检查远程写入的归约状态
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
                    // ld_volatile: PTX ld.volatile.global, volatile 加载, 保证读取最新值
                    //   用于轮询远程 rank 通过 NVLink/RDMA 写入的 count
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
            // do_psum: 用单个 Warp(32 lanes) 以 32 为粒度分块做 prefix sum
            // 核心思路: 分块扫描 + 跨块传递 (block-scan + carry propagation)
            //
            // warp_inclusive_sum: Warp 内 inclusive scan, 用 __shfl_up_sync 实现
            //   例: 4 lanes 输入 [3,1,4,2] → 输出 [3,4,8,10]
            //
            // exchange(sum, 31): 本质是 __shfl_sync(0xffffffff, sum, 31)
            //   即所有 lane 都从 lane 31 获取值 → 拿到当前块的累加总和, 作为下一轮的 psum 基数
            //
            // is_exclusive 巧妙设计: mem_idx = idx - is_exclusive
            //   is_exclusive=0 (inclusive): out[i] = sum(count[0..i])
            //     例: [10,5,8,3] → [10,15,23,26]
            //   is_exclusive=1 (exclusive): out[i] = sum(count[0..i-1]), lane 0 读 mem_idx=-1 → value=0
            //     例: [10,5,8,3] → [0,10,15,23]
            //     同时循环上界变为 ceil_div(n+1, 32), 多输出一个元素
            //
            // 完整流程示例 (n=64, is_exclusive=1):
            //   i=0: lane 0~31, mem_idx=-1~30, warp_inclusive_sum 后加上 psum=0
            //        out[0]=0, out[1]=c0, ..., out[31]=c0+..+c30
            //        psum = exchange(sum, 31) = c0+..+c30 (块累加和)
            //   i=1: lane 0~31, mem_idx=31~62, warp_inclusive_sum 后加上 psum
            //        out[32]=c0+..+c31, ..., out[63]=c0+..+c62
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                #pragma unroll
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;

                    // Warp 内 inclusive sum (使用 __shfl_up_sync 实现)
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    // 写入全局内存
                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    // 更新 psum: 所有 lane 从 lane 31 获取块累加总和, 作为下一轮的基数
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
         *   │                    Send Buffer                              │
         *   │  ┌─────────────────────────────────────────────────────┐    │
         *   │  │  [kNumMaxTokensPerRank tokens]                      │    │
         *   │  │  用于存储本 warp 发出的 token (可能被 RDMA 读取)      │    │
         *   │  └─────────────────────────────────────────────────────┘    │ 
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
        const auto token_layout = layout::TokenLayout(
            kNumHiddenBytes,                     // hidden 字节数
            kNumSFPacks * sizeof(sf_pack_t),     // scale factor 字节数
            kNumTopk,                            // top-k 数量
            true                                 // with_metadata=true, 包含 src_token_global_idx 等
        );

        // TMA buffer: 指向 shared memory
        const auto tma_buffer = layout::BufferLayout<true>( //kWithMBarrier=true → 每个 token 末尾带 mbarrier（TMA 多播同步用）。
            token_layout,          // 用上面的 token 布局
            kNumDispatchWarps,     // num_ranks = dispatch warp 数量
            1,                     // 每个 "rank" 只放 1 个 token
            smem + kNumSmemBytesForNotify  // base = smem 跳过通知区
        ).get_rank_buffer(dispatch_warp_idx)  // 取当前 warp 对应的 rank slice
        .get_token_buffer(0);                // 取第 0 个 token 的 TokenLayout


        // Recv buffer 和 Send buffer
        // recv_buffer（创建时）：整个 recv buffer 区域，包含所有 rank 的 token 槽位
        auto recv_buffer = layout::BufferLayout<false>(token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);
        //send_buffer：紧跟 recv buffer 之后，只有 1 个 "rank"（本 rank），放 kNumMaxTokensPerRank 个 token
        auto send_buffer = layout::BufferLayout<false>(token_layout, 1, kNumMaxTokensPerRank, recv_buffer.get_buffer_end_ptr());
        //将 recv_buffer 缩窄到只看本 rank 对应的分区
        //此时 recv_buffer 和 send_buffer 的 num_ranks 都是 1，结构对称，方便后续统一用 get_token_buffer(idx) 访问。
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
        ptx::arrival_phase phase = 0;  // mbarrier 相位, 每次 wait 后翻转 (phase ^= 1)
        const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
        // elect_one_sync: PTX elect.sync, 从 warp 中选举一个 lane 执行 (硬件随机,低开销)
        if (ptx::elect_one_sync())
            // mbarrier_init_with_fence: PTX mbarrier.init + fence.mbarrier_init.release.cluster
            //   初始化 mbarrier, arrive_count=1 (等待1次arrive即可通过), 并插入cluster级fence
            ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        __syncwarp();

        /**
         * ==================== Token 迭代图 ====================
         *
         *   每个 dispatch warp 处理一组 tokens,跨所有 SM 交错
         *
         *   示例: kNumDispatchWarps=2, kNumSMs=4
         *   - warp 0, SM 0: token 0, 8, 16, 24, ...
         *   - warp 0, SM 1: token 1, 9, 17, 25, ...
         *   - warp 1, SM 0: token 4, 12, 20, 28, ...
         *   - warp 1, SM 1: token 5, 13, 21, 29, ...
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
            // tma_store_wait: PTX cp.async.bulk.wait_group, 等待 TMA 异步 store 完成
            //   参数 0 = 等待所有 pending 的 TMA store 完成
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
            if (ptx::elect_one_sync())  // elect: 选举一个 lane 执行 TMA load
                // tma_load_1d: PTX cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
                //   TMA 硬件加速的 1D 批量异步加载: global→shared, 完成时自动通知 mbarrier
                //   参数: smem_dst, gmem_src, mbarrier(用于同步), num_bytes, cache_hint
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
             *   sf_hidden_stride 通常 = 1 (行优先连续张量, PyTorch 默认),
             *     此时 gmem 也是连续 load (32 个线程读 32 个相邻 sf_pack)
             *   参数化设计是为了支持非连续张量 (transpose 视图等)
             *   对比: 接收端 recv_sf 可能是列优先 (sf_hidden_stride >> 1), 用不同代码路径写回
             *
             *   cp_async_ca: PTX 指令 cp.async.ca.shared::cta.global.L2::128B
             *     - cp.async: 异步拷贝, 发起后线程不等待完成
             *     - .ca: Cache All, 数据可缓存在 L1+L2
             *     - .shared::cta → 目标是当前 CTA 的 shared memory
             *     - .global → 源是 global memory
             *     - .L2::128B → L2 缓存行 128 字节对齐提示
             *     - sizeof(dtype_t) 仅支持 4/8/16 字节
             *
             *   cp_async_mbarrier_arrive: PTX 指令 cp.async.mbarrier.arrive.shared::cta.b64
             *     - "预约"语义: 告诉 mbarrier "之前发起的异步拷贝完成后请通知"
             *     - mbarrier 内部维护计数器, 每个异步拷贝完成时递减, 归零表示全部完成
             *     - 后续 mbarrier_try_wait 会等待所有异步拷贝真正完成
             *
             *   异步加载流程:
             *     1. cp_async_ca(...)       ← 发起异步拷贝 (不等完成)
             *     2. cp_async_ca(...)       ← 发起更多异步拷贝
             *     3. cp_async_mbarrier_arrive ← 注册完成信号
             *     4. __syncwarp()           ← warp 内同步
             *     5. (后续) mbarrier_try_wait ← 等所有异步拷贝完成
             *
             *   每个 lane 搬 1 个 sf_pack_t 元素, 32 个 lane 一轮共搬 32 个:
             *     k=0: lane 0→sf_pack[0], lane 1→sf_pack[1], ..., lane 31→sf_pack[31]
             *     k=1: lane 0→sf_pack[32], lane 1→sf_pack[33], ..., lane 31→sf_pack[63]
             *     ...
             *     smem 端连续 store, gmem 端由 sf_hidden_stride 决定 (通常=1, 也连续)
             */
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "SF元素类型未对齐");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                // 分批次加载,每批32个元素 (每个 lane 搬 1 个 sf_pack_t)
                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                // 处理剩余元素 (只有 lane_idx < 剩余数 的线程工作, 其余空闲)
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                // 通知 mbarrier: 之前发起的异步拷贝完成后请通知 (预约语义)
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
                // 每个lane记录的不一样的dst_rank_idx
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
            if (ptx::elect_one_sync())  // elect: 选举一个 lane 写入 metadata
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;

            // tma_store_fence: PTX fence.proxy.async.shared::cta
            //   确保 shared memory 中的写入在 TMA store 发起前对所有 thread 可见
            ptx::tma_store_fence();
            __syncwarp();

            

            /**
             * ==================== 去重 + Slot 分配 ====================
             *
             *   核心问题: token 要放到接收方 buffer 的哪个位置?
             *   每个 rank 有预分配的 recv_buffer, 大小为 kNumMaxTokensPerRank 个 slot
             *   多个 token 发给同一 rank 时, 必须分配不冲突的 slot 编号
             *
             *   接收方 (rank 1) 的 recv_buffer 示例:
             *     slot 0: [token_来自rank0_topk0]
             *     slot 1: [token_来自rank0_topk2]  ← 不能和 slot 0 冲突
             *     slot 2: [token_来自rank2_topk1]
             *     ...
             *
             *   为什么需要 deduplicate?
             *     一个 token 有 kNumTopk 个 top-k 选择, 可能选到同一 rank 的不同 expert:
             *       token 0 的 top-k: expert 3, expert 5, expert 11
             *                          ↓        ↓        ↓
             *                          rank 0   rank 0   rank 1  ← expert 3 和 5 都在 rank 0!
             *     如果不去重: rank 0 会被发两次同样的 token 0 → 浪费带宽和 slot
             *     去重后:     rank 0 只发一次 token 0 → 节省资源
             *
             *   两种模式的设计动机:
             *     kReuseSlotIndices = false (新分配): 首次 dispatch, 不知哪些 token 发给哪些 rank,
             *       用 atomicAdd(counter[rank], 1) 动态分配 slot, 结果写入 dst_buffer_slot_idx 保存
             *     kReuseSlotIndices = true (复用): 反向 combine 或重复 dispatch, 之前已分配过 slot,
             *       handle 里保存了 dst_buffer_slot_idx, 直接读取跳过原子操作, 更快
             *
             *   典型调用流程:
             *     前向 dispatch (首次):
             *       handle = None → kReuseSlotIndices = false
             *       → atomicAdd 分配 slot
             *       → 返回 handle (包含 dst_buffer_slot_idx)
             *     反向 combine / 重复 dispatch:
             *       handle = 上次的 handle → kReuseSlotIndices = true
             *       → 直接读取 dst_buffer_slot_idx, 省去原子操作开销
             *
             *   dst_buffer_slot_idx 编码:
             *     全局索引 = rank_idx * kNumMaxTokensPerRank + local_slot_idx
             *     例: rank 2 的 slot 5 → 全局索引 = 2 * 256 + 5 = 517
             *     存储: 写全局索引 (便于跨 rank 定位)
             *     使用: 减去 rank 偏移得到 local_slot_idx (用于 recv_buffer 内偏移)
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
                // 复用模式: 直接从 handle 中读取之前分配的 slot (跳过原子操作)
                if (lane_idx < kNumTopk)
                    stored_dst_slot_idx = __ldg(dst_buffer_slot_idx + token_idx * kNumTopk + lane_idx);
                // 全局索引 → local slot index: 减去 rank 偏移
                //   例: 全局索引 517 = 2*256+5 → local_slot_idx = 5
                stored_dst_slot_idx = stored_dst_slot_idx >= 0 ?
                    (stored_dst_slot_idx - rank_idx * kNumMaxTokensPerRank) : -1;
            } else {
                // 新分配模式: 用 atomicAdd 原子递增计数器分配唯一 slot
                if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
                    // deduplicate: warp 内按 value 去重, 只让每个唯一值对应的最高 lane 执行
                    //   实现: match(value) 找相同值的 lane 掩码, get_master_lane_idx 取最高位, 与自身 lane_idx 比较
                    //   例: lane 0 和 lane 2 都要发 rank 0, 只有 lane 2 (最高位) 执行 atomicAdd
                    //   atomicAdd 返回旧值作为 slot 编号, 同时计数器+1
                    //   counter[rank] 初始为 0, 每次 atomicAdd 返回 0,1,2,... 自然递增
                    stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);

                // 写入全局 slot 索引 (保存到 handle, 供后续 combine 复用)
                //   local_slot_idx + rank 偏移 → 全局索引
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
             *   完整 TMA + mbarrier 异步流水线 (5 步):
             *
             *   步骤1: 初始化  (循环开头)
             *     elect_one → mbarrier_init(ptr, 1)
             *     → arrive_count = 1, tx_pending = 0
             *
             *   步骤2: 发起 TMA Load  (异步, 立即返回)
             *     elect_one → tma_load_1d(smem, gmem, mbarrier_ptr, kNumHiddenBytes)
             *     → TMA 硬件在后台搬运, 完成后自动减少 tx_pending
             *
             *   步骤3: Slot 分配  (与 TMA 搬运并行! 关键优化)
             *     deduplicate + atomicAdd / reuse slot
             *     → 此时 TMA 硬件还在搬数据, slot 分配零延迟隐藏!
             *
             *   步骤4: 注册预期字节数 + 等待  ← 本段代码
             *     arrive_and_set_tx + wait_and_flip_phase
             *
             *   步骤5: 安全使用 smem 数据  (TMA store)
             *
             *   ─────────────────────────────────────────────────────
             *   mbarrier 的两个独立计数器:
             *
             *     计数器          初始值             谁修改                     放行条件
             *     ──────────     ──────────        ──────────────             ──────────
             *     arrive_count   init 时设为 1      arrive / arrive_and_set_tx 递减到 0
             *     tx_pending     0                  expect_tx 递增,            递减到 0
             *                                        TMA 完成时硬件递减
             *
             *   mbarrier 放行条件 = (arrive_count == 0) AND (tx_pending == 0)
             *   两者缺一不可, 分别跟踪 "软件就绪" 和 "硬件就绪"
             *
             *   本段代码的精确效果:
             *     mbarrier_arrive_and_set_tx(ptr, kNumHiddenBytes):
             *       → arrive_count -= 1  (1→0 ✅ 软件到达条件满足)
             *       → tx_pending += kNumHiddenBytes  (0→kNumHiddenBytes)
             *       → 此时 tx_pending > 0, 还不能放行
             *     mbarrier_wait_and_flip_phase(ptr, phase):
             *       → 自旋等待直到 arrive_count==0 AND tx_pending==0
             *       → TMA 硬件每搬完一个字节 tx_pending 减 1
             *       → 全部搬完 tx_pending 归零, 两个条件均满足, 放行!
             *       → phase 翻转 (0→1 或 1→0), 避免与下一轮混淆
             *
             *   为什么 arrive_and_set_tx 放在 slot 分配之后?
             *     顺序执行: 总时间 = TMA延迟 + slot分配时间
             *     并行执行: 总时间 ≈ max(TMA延迟, slot分配时间)
             *     arrive_and_set_tx 是 "注册预期", 注册后才开始等,
             *     把它推迟到 slot 分配之后, TMA 有更多时间在后台完成搬运
             *
             *   phase 的作用:
             *     每轮 wait 通过后 phase 翻转 (0↔1), 下一轮等待新相位
             *     防止上一轮的残留信号干扰当前轮的等待判断
             */
            if (ptx::elect_one_sync()) {  // elect: 选举一个 lane 管理 mbarrier
                // mbarrier_arrive_and_set_tx: PTX mbarrier.arrive.expect_tx
                //   arrive_count -= 1 (软件到达), tx_pending += kNumHiddenBytes (注册预期字节数)
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                // mbarrier_wait_and_flip_phase: PTX mbarrier.try_wait.parity (自旋等待)
                //   自旋等待 arrive_count==0 AND tx_pending==0, 通过后翻转 phase
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
            }
            __syncwarp();

            /**
             * ==================== 数据发送: NVLink 直写 / RDMA ====================
             *
             *   ─────────────────────────────────────────────────────
             *   TMA Load vs TMA Store 完成通知机制对比:
             *
             *                    TMA Load                         TMA Store
             *                    ─────────                       ──────────
             *   方向             global → shared                 shared → global
             *   PTX指令          ...mbarrier::complete_tx::bytes ...bulk_group
             *   完成通知         mbarrier (按字节跟踪)            bulk_group (按请求个数排队)
             *   参数             dst, src, mbarrier, bytes       dst, src, bytes
             *
             *   TMA Load 通知链:
             *     tma_load_1d(smem, gmem, mbarrier, bytes)
             *       → TMA 硬件搬完后自动: mbarrier.tx_pending -= bytes
             *     arrive_and_set_tx(mbarrier, bytes)
             *       → arrive_count--, tx_pending += bytes
             *     wait_and_flip_phase(mbarrier, phase)
             *       → 自旋等到 tx_pending==0 (所有字节到位)
             *
             *   TMA Store 通知链:
             *     tma_store_1d(gmem, smem, bytes)
             *       → 请求进入 bulk_group 队列 (FIFO, 按程序顺序)
             *     tma_store_commit()
             *       → PTX: cp.async.bulk.commit_group, 提交当前队列
             *     tma_store_wait<N>()
             *       → PTX: cp.async.bulk.wait_group N
             *       → 等到队列中最多剩 N 个未完成请求
             *       → N=0: 全部完成; N=1: 允许1个overlap (流水线优化)
             *
             *   为什么设计不同?
             *     TMA Load 后要立即使用 smem 数据, 必须等全部字节到位 → 按字节精确跟踪
             *     TMA Store 只需知道"请求完没完", 不关心字节数 → 按请求个数排队
             *
             *   bulk_group 队列是顺序的吗?
             *     是的, FIFO (先入先出), 按程序提交顺序执行和完成
             *     commit_group 提交后, 同一组内的请求保证按序完成
             *     不同 commit_group 之间也保证顺序: 第 N 组全部完成后, 第 N+1 组才开始完成
             *   ─────────────────────────────────────────────────────
             *
             *   两种发送路径 (互斥, 由 kIsScaleupNVLink 编译期决定):
             *
             *   ┌─────────────────────────────────────────────────────────────┐
             *   │  路径A: RDMA 模式 (not kIsScaleupNVLink, 跨节点通信)        │
             *   │                                                             │
             *   │  smem ──TMA store──→ send_buffer (本地 gmem)                │
             *   │                          │                                  │
             *   │              ┌───────────┴───────────┐                      │
             *   │              │                       │                      │
             *   │     NVLink 可达?               NVLink 不可达?               │
             *   │     (dst_ptr != nullptr)       (dst_ptr == nullptr)         │
             *   │              │                       │                      │
             *   │     TMA store 直写             gin.put (RDMA)               │
             *   │     对端 recv_buffer           对端 recv_buffer             │
             *   └─────────────────────────────────────────────────────────────┘
             *                                                             │
             *   ┌─────────────────────────────────────────────────────────────┐
             *   │  路径B: 纯 NVLink 模式 (kIsScaleupNVLink, 节点内通信)       │
             *   │                                                             │
             *   │  smem ──TMA store──→ 对端 recv_buffer (NVLink 直写)         │
             *   │  无 send_buffer 中转, 无 RDMA                               │
             *   └─────────────────────────────────────────────────────────────┘
             *
             *   为什么 RDMA 模式需要 send_buffer 中转?
             *     RDMA (gin.put) 从 global memory 读取源数据,
             *     无法直接从 smem 读取, 所以必须先把 smem 数据 TMA store 到 send_buffer
             */
            auto send_buffer_ptr = send_buffer.get_token_buffer(token_idx).get_base_ptr();
            if constexpr (not kIsScaleupNVLink) {
                // RDMA 模式步骤1: smem → send_buffer (本地 gmem)
                //   RDMA 引擎只能读 gmem, 所以需要先把 smem 数据搬出来
                if (ptx::elect_one_sync())  // elect: 选举一个 lane 执行 TMA store
                
                    // tma_store_1d: PTX cp.async.bulk.global.shared::cta.bulk_group
                    //   TMA 硬件加速的 1D 批量异步存储: shared→global, 加入 bulk_group 队列
                    //   参数: gmem_dst(send_buffer), smem_src(tma_buffer), num_bytes
                    ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());

                // tma_store_commit: PTX cp.async.bulk.commit_group
                //   将当前 bulk_group 中的 TMA store 请求提交执行
                ptx::tma_store_commit();

                __syncwarp();
            }

            // ========== NVLink 直写 ==========
            EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid top-k selection");

            // 获取目标地址 (gin.get_sym_ptr 返回对端对称内存地址)
            //   同一节点内: 返回对端 rank 的 recv_buffer 指针 (NVLink 可达)
            //   跨节点: 返回 nullptr (NVLink 不可达, 需要 RDMA)
            const auto dst_ptr = stored_dst_slot_idx >= 0 ?
                gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), stored_dst_rank_idx) :
                nullptr;

            // NVLink 可达时: smem 直写到对端 recv_buffer (零拷贝)
            // tma_store_1d: PTX cp.async.bulk.global.shared::cta.bulk_group
            //   TMA 硬件加速的 1D 批量异步存储: shared→global (NVLink 直写对端对称内存)
            //
            //   注意: 这里没有用 elect_one_sync(), 是多个 lane 独立发起 TMA store,
            //   但不会重复写入! 因为 deduplicate 已经保证每个目标 rank 只有一个 lane 拥有有效 slot:
            //     例: token 0 的 top-k → [rank 0, rank 0, rank 1]
            //       lane 0: stored_dst_rank_idx=0, slot 有效 → dst_ptr ≠ nullptr → 发 TMA store 到 rank 0
            //       lane 1: stored_dst_rank_idx=0, slot=-1  (被 deduplicate 掉) → dst_ptr = nullptr → 不发
            //       lane 2: stored_dst_rank_idx=1, slot 有效 → dst_ptr ≠ nullptr → 发 TMA store 到 rank 1
            //       lane 3~31: slot=-1 → dst_ptr = nullptr → 不发
            //   每个 lane 写到不同的 dst_ptr (不同 rank), 同一个 tma_buffer 数据复制到多个目标
            //   这正是期望的行为: 一个 token 发给多个 rank (一对多)
            //
            //   对比 RDMA 路径 (line 1219) 用 elect_one_sync(): 那里只写一个 send_buffer_ptr
            //   (单一目标), 只需一个 lane 执行; 而这里需要多个 lane 各自发往不同 rank
            if (dst_ptr != nullptr)
                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            // tma_store_commit: PTX cp.async.bulk.commit_group, 提交当前 bulk_group
            //   所有 lane 都参与 commit, 保证之前所有 TMA store 请求都被提交
            ptx::tma_store_commit();
            __syncwarp();

            /**
             * ==================== RDMA Put (跨节点) ====================
             *
             *   仅 RDMA 模式 (not kIsScaleupNVLink) 需要
             *   对 NVLink 不可达的 rank, 使用 gin.put 做 RDMA 发送
             *
             *   条件:
             *   - stored_dst_slot_idx >= 0: 有效 slot
             *   - dst_ptr == nullptr: NVLink 不可达
             *
             *   数据来源: send_buffer (在前面 smem→send_buffer 的 TMA store 中写入)
             */
            if constexpr (not kIsScaleupNVLink) {
                // 等待 send buffer TMA store 完成
                // tma_store_wait<1>: PTX cp.async.bulk.wait_group, 参数1=等待直到最多1个pending
                //   (确保 send_buffer 写入完成后再 RDMA 读取, 否则 RDMA 读到未完成的数据)
                ptx::tma_store_wait<1>();
                __syncwarp();

                // 跨节点 RDMA 发送: send_buffer → 对端 recv_buffer
                //   条件: stored_dst_slot_idx >= 0 (有效 slot) AND dst_ptr == nullptr (NVLink 不可达)
                //
                //   和 NVLink 直写一样, 这里也是多 lane 独立发起 RDMA, 不会重复:
                //     例: token 0 的 top-k → [rank 0(同节点), rank 0(同节点,被dedup), rank 5(跨节点)]
                //       lane 0: slot有效, dst_ptr≠nullptr → 走上面的 NVLink 直写
                //       lane 1: slot=-1 (deduplicate) → 不发
                //       lane 2: slot有效, dst_ptr==nullptr → 走 RDMA (gin.put 到 rank 5)
                //
                //   多个 lane 可能从同一份 send_buffer_ptr 读数据 gin.put 到不同目标 rank,
                //   这是合法的: gin.put 只读源数据, 不会修改, 多个 lane 并行读不冲突
                if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
                    gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                    send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
                }
                __syncwarp();
            }
        }
    }

    /**
     * ==================== 结束后 Barrier ====================
     *
     *   确保所有 SM 都完成数据发送, 所有 TMA store 数据落盘
     *
     *   与开始前 barrier (kDispatchTag0) 对比:
     *
     *   模板参数                开始前 (Tag0)     结束后 (Tag1)     含义
     *   ──────────────────     ────────────     ────────────     ──────────
     *   kTag                   kDispatchTag0    kDispatchTag1    不同阶段,互不干扰
     *   kFlushStores           false            true             开始前没数据不用flush;
     *                                                            结束后必须flush (等TMA store全部完成)
     *   kSyncAtStart           false            true             开始前各自跑不用sync;
     *                                                            结束后必须sync (等所有SM都干完再barrier)
     *   kSyncAtEnd             true             false            开始后等所有SM一起开工;
     *                                                            结束后不用等 (后面有其他同步机制)
     *
     *   gpu_barrier 内部流程 (kFlushStores=true, kSyncAtStart=true):
     *     1. kFlushStores: tma_store_commit + tma_store_wait → 确保所有 TMA store 数据落盘
     *     2. kSyncAtStart: cooperative_groups::this_grid().sync() → 等所有 SM 到齐
     *     3. scaleup_barrier: NVLink/RDMA barrier → 跨 rank 同步 (确保对端收到数据)
     *     4. kSyncAtEnd=false: 不做额外 grid sync (直接返回)
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