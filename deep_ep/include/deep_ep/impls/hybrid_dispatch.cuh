#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

/**
 * ==================== Hybrid Dispatch Kernel ====================
 *
 * 跨节点 (scale-out, RDMA) + 节点内 (scale-up, NVLink) 混合 All-to-All dispatch
 * 每个 SM 上运行三类 warp, 各司其职:
 *
 *   ┌──────────────────────────────────────────────────────────────────────┐
 *   │                    一个 SM 的 warp 分工                              │
 *   │                                                                      │
 *   │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
 *   │  │ Notify Warps    │  │ Scaleout Warps   │  │ Forward Warps    │    │
 *   │  │ (kNumNotify)    │  │ (kNumScaleout)   │  │ (kNumForward)    │    │
 *   │  ├─────────────────┤  ├──────────────────┤  ├──────────────────┤    │
 *   │  │ 统计每个 rank   │  │ 从本地 x 读取   │  │ 从 recv_buffer  │    │
 *   │  │ 和 expert 应收 │  │ token, 通过 RDMA │  │ 读取跨节点来的  │    │
 *   │  │ token 数量,    │  │ 发送到其他节点   │  │ token, 通过      │    │
 *   │  │ 做 prefix sum  │  │ 的 recv_buffer   │  │ NVLink 转发到   │    │
 *   │  │ 供后续 epilogue│  │ (同时本地 rank   │  │ 同节点其他 rank │    │
 *   │  │ 分配输出行号   │  │ 直接写入)        │  │ 的 recv_buffer   │    │
 *   │  └─────────────────┘  └──────────────────┘  └──────────────────┘    │
 *   └──────────────────────────────────────────────────────────────────────┘
 *
 * 数据流 (以 token T 从 rank A 发到 rank B 为例):
 *
 *   rank A (发送方):                    rank B (接收方):
 *   ┌──────────────┐                   ┌──────────────┐
 *   │ 本地 x[T]    │                   │              │
 *   │    │         │                   │              │
 *   │    ▼ Scaleout Warp              │              │
 *   │ send_buffer[T]──── RDMA ────────▶recv_buffer   │
 *   │    │         │    (跨节点)       │ (scaleout区) │
 *   │    │ 本地    │                   │    │         │
 *   │    ▼ 直写    │                   │    ▼ Forward Warp
 *   │ recv_buffer  │                   │ scaleup_buffer─── NVLink ──▶ 对端 rank
 *   │ (本地slot)   │                   │ (本节点slot)  │
 *   └──────────────┘                   └──────────────┘
 *
 * 三阶段时序:
 *   1) Notify: 统计 → 跨节点通知 → prefix sum (所有 rank 知道各自应收多少 token)
 *   2) Scaleout: 读本地 token → RDMA 发送跨节点 + 本地直写
 *   3) Forward: 从 recv_buffer 读 → NVLink 转发到同节点其他 rank
 *
 * 缓冲区布局 (buffer 内存):
 *   ┌────────────────────────────────┬───────────────────┬──────────────────────┐
 *   │ scaleup_buffer                 │ scaleout_send_buf │ scaleout_recv_buffer │
 *   │ [scaleup_ranks][max_tokens]    │ [1][max_tokens]   │ [scaleout_ranks]     │
 *   │ NVLink 对端直写目标            │ RDMA 发送暂存     │ [channels][max_tok]  │
 *   │                                │                    │ RDMA 接收缓冲        │
 *   └────────────────────────────────┴───────────────────┴──────────────────────┘
 *
 * Channel 机制:
 *   - scaleout/forward warp = channel (一一对应)
 *   - 每个 channel 负责一部分 token (round-robin 分配: token i 归 channel i%kNumChannels)
 *   - 每个 channel 的 recv_buffer 独立, 避免 RDMA 写冲突
 *
 * 模板参数说明:
 *   kDoCPUSync           - 是否 CPU 同步 (精确值 vs worst-case 哨兵)
 *   kReuseSlotIndices    - 是否复用 slot 索引 (低延迟模式跳过 notify, 直接用上次结果)
 *   kNumSMs              - SM 数量
 *   kNumNotifyWarps      - Notify 阶段 warp 数 (必须为 4 的倍数)
 *   kNumScaleoutWarps    - Scaleout 阶段 warp 数 (= kNumForwardWarps)
 *   kNumForwardWarps     - Forward 阶段 warp 数 (= kNumScaleoutWarps)
 *   kNumScaleoutRanks    - 跨节点 rank 数 (节点数)
 *   kNumScaleupRanks     - 节点内 rank 数 (GPU 数)
 *   kNumHiddenBytes      - hidden 维度字节数
 *   kNumSFPacks          - FP8 scale factor pack 数
 *   kNumMaxTokensPerRank - 每 rank 最大 token 数
 *   kNumExperts          - expert 总数
 *   kNumTopk             - top-k 选择数
 *   kExpertAlignment     - expert 对齐粒度
 *   kNumQPs              - NCCL QP 数量
 *   kNumTimeoutCycles    - 超时检测周期数
 */

template <bool kDoCPUSync,
          bool kReuseSlotIndices,
          int kNumSMs,
          int kNumNotifyWarps, int kNumScaleoutWarps, int kNumForwardWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),
          int kNumChannelsPerSM = kNumScaleoutWarps,
          int kNumChannels = kNumScaleoutWarps * kNumSMs,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          int kScaleoutUpdateInterval = 3,
          int kNumSlotsPerForwardChunk = kScaleoutUpdateInterval,
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumNotifyThreads = kNumNotifyWarps * 32,
          int kNumScaleoutSendThreads = kNumScaleoutWarps * 32,
          int kNumForwardThreads = kNumForwardWarps * 32,
          int kNumThreads = kNumNotifyThreads + kNumScaleoutSendThreads + kNumForwardThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,
    topk_idx_t* copied_topk_idx,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* dst_buffer_slot_idx,
    int* token_metadata_at_forward,
    const int num_tokens,
    const int sf_token_stride, const int sf_hidden_stride,
    // TODO(NCCL): so many params, plans to optimize?
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
    void* buffer,
    void* workspace, void* mapped_host_workspace,
    const int scaleout_rank_idx, const int scaleup_rank_idx) {
        
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumScaleupRanks == 0, "Invalid number of experts or ranks");
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");
    EP_STATIC_ASSERT(kNumScaleoutWarps == kNumForwardWarps, "Invalid warp size");

    // ==================== 公共初始化 ====================
    // warp_idx: [0, kNumNotifyWarps) = Notify, [kNumNotifyWarps, kNumNotifyWarps+kNumScaleoutWarps) = Scaleout, 其余 = Forward
    // 每个 scaleout/forward warp 对应一个 channel (不同 channel 可能共享 QP)
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;


    // GPU workspace: 用于跨 SM 聚合 (notify reduction)、链表 tail 指针等
    // Host workspace: CPU 同步模式下, 供 CPU 读取前缀和结果
    //
    // WorkspaceLayout 设计原理:
    //   - 固定大小分配: 使用 kNumMaxRanks=1024, kNumMaxExperts=2048 等编译期常量
    //     运行时 num_scaleout_ranks/num_experts 只影响访问偏移, 不影响分配大小
    //     好处: 同一块 buffer 可复用于不同配置, 无需重新分配
    //
    //   - 内存布局 (8 个连续区域, 详见 layout.cuh):
    //     ┌─────────────────────────────────────────────────────────────────────────┐
    //     │ [0] NVLink barrier signal (16B)                                        │
    //     │     counter(8B) + signal[0](4B) + signal[1](4B)                       │
    //     ├─────────────────────────────────────────────────────────────────────────┤
    //     │ [1] Notify reduction workspace  (kNumMaxRanks + kNumMaxExperts) × 8B  │
    //     │     所有 SM 的 notify warp 做 red_add 聚合到这里                       │
    //     │     编码: (到达SM数 << 32) | 计数值                                    │
    //     ├─────────────────────────────────────────────────────────────────────────┤
    //     │ [2] Scaleup rank+expert count  (send + recv 各一份)                    │
    //     │     send: 仅 dispatch.cuh 非 NVLink 模式使用, 本文件不用               │
    //     │     recv: 对端 put_value / red_add_rel 写入, 本端读取                  │
    //     ├─────────────────────────────────────────────────────────────────────────┤
    //     │ [3] Scaleup atomic sender counter  kNumMaxRanks × 4B                  │
    //     │     forward warp 用 atomicAdd 分配 scaleup_buffer slot                │
    //     ├─────────────────────────────────────────────────────────────────────────┤
    //     │ [4] Scaleout rank+expert count  (send + recv 各一份, int 类型)         │
    //     │     send: encode_decode_positive 编码, 供 RDMA put                    │
    //     │     recv: RDMA 接收, recv_and_reduce 解码求和                          │
    //     ├─────────────────────────────────────────────────────────────────────────┤
    //     │ [5] Scaleout channel signaled tail  [channel][scaleout_rank] → int64  │
    //     │     pack2(finish_flag, tail_count)                                    │
    //     │     scaleout warp → forward warp 的进度通知                            │
    //     ├─────────────────────────────────────────────────────────────────────────┤
    //     │ [6] Channel scaleup tail  [channel][scaleup_rank] → int               │
    //     │     linked list 尾部位置, forward 写入, epilogue 读取                   │
    //     ├─────────────────────────────────────────────────────────────────────────┤
    //     │ [7] PP count + AGRS signals                                           │
    //     └─────────────────────────────────────────────────────────────────────────┘
    //
    //   - Send/Recv 双缓冲模式 (kIsSendBuffer 模板参数):
    //     kIsSendBuffer=true:  写入端 (本节点写)
    //     kIsSendBuffer=false: 读取端 (对端写, 本节点读)
    //     通过 RDMA put / NVLink st_relaxed 实现跨节点/跨 rank 通信
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);

    // smem 布局: [notify 计数区 | TMA buffer 区]
    //   notify 计数区: kNumRanks + kNumExperts 个 int, 供 Notify Warps 做 smem 级聚合
    //   TMA buffer 区: scaleout/forward warp 各一个 TokenLayout (含 mbarrier)
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");

    // named_barrier: 只在 Notify Warps 内部同步 (与 scaleout/forward warp 隔离)
    constexpr int kNotifyBarrierIndex = 1;

    // NCCL Gin handle: 封装 RDMA put / NVLink st_relaxed_sys 等通信原语
    //   qp_idx: QP 编号 (多个 channel 可能共享 QP)
    //   sharing_mode: QP 共享模式
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, (kNumNotifyWarps > 0)>(
        sm_idx, (warp_idx - kNumNotifyWarps) % kNumChannelsPerSM, warp_idx < kNumNotifyWarps);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // ==================== 开始前 Barrier ====================
    // Tag0: syncStart=false, syncEnd=true (等待所有 SM 到齐后才放行)
    // 确保所有 SM 的 workspace 初始化完成, 才开始通信
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag0, false, false, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // ==================== smem TMA buffer 初始化 ====================
    // token_layout: 一个 token 的完整布局 (hidden + sf + metadata, 含 mbarrier)
    // tma_buffer: 当前 warp 在 smem 中的 TMA 缓冲区 (从 smem notify 区之后开始)
    //   layout: [scaleout_warp_0, scaleout_warp_1, ..., forward_warp_0, forward_warp_1, ...]
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumScaleoutWarps + kNumForwardWarps, 1,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(warp_idx - kNumNotifyWarps).get_token_buffer(0);

    // ==================== 全局 buffer 布局 ====================
    // 三段连续内存: scaleup_buffer | scaleout_send_buffer | scaleout_recv_buffer
    //
    // scaleup_buffer: [kNumScaleupRanks][kNumScaleoutRanks * kNumMaxTokensPerRank]
    //   节点内 NVLink 通信的接收缓冲区, 每 rank 一段
    //   对端 rank 通过 NVLink 直写 (TMA store) 到这里
    //
    // scaleout_send_buffer: [1][kNumMaxTokensPerRank]
    //   RDMA 发送暂存区, 本地 token 先 TMA store 到这里, 再 gin.put 发出
    //   只有1个 rank (自己), 因为只暂存自己要发出的 token
    //
    // scaleout_recv_buffer: [kNumScaleoutRanks][kNumChannels * kNumMaxTokensPerChannel]
    //   RDMA 接收缓冲区, 按 (scaleout_rank, channel) 组织
    //   其他节点通过 RDMA put 写到这里
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, 1, kNumMaxTokensPerRank, scaleup_buffer.get_buffer_end_ptr());
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleoutRanks, kNumChannels * kNumMaxTokensPerChannel, scaleout_send_buffer.get_buffer_end_ptr());

    // Init TMA for scale-out and forward warps
    // mbarrier 初始化: arrive_count=1 (配合 elect_one 的单线程 arrive)
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (warp_idx >= kNumNotifyWarps and ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // ==================== Phase 1: Notify Warps ====================
    // 职责: 统计每个 rank/expert 应收多少 token, 跨节点聚合, 计算 prefix sum
    //
    // 流程:
    //   Step 1: 本 SM 内统计 rank/expert 计数 → smem
    //   Step 2: 全 grid reduction → workspace (跨 SM 聚合)
    //   Step 3: SM0 跨节点发送计数 → RDMA put 到其他节点
    //   Step 4: SM0 接收其他节点计数 + 节点内聚合 → NVLink 写到对端 rank
    //   Step 5: SM0 等待本地计数就绪 → 计算 prefix sum
    //
    if (warp_idx < kNumNotifyWarps) {
        // smem 前 kNumRanks+kNumExperts 个 int 用于本地计数
        //   [0..kNumRanks-1]: rank_count (每个 rank 收到多少 token)
        //   [kNumRanks..kNumRanks+kNumExperts-1]: expert_count (每个 expert 收到多少 token)
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        // Step 1: 清零 smem 计数区 (所有 notify 线程参与, 每人写若干个)
        //   注意 thread_idx 从 0 开始, 与 scaleout/forward 的 thread_idx 是同一个命名空间
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;
        #pragma unroll
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Step 2: 遍历分配给本 SM 的 token, 在 smem 中统计 rank/expert 计数
        //   global_warp_idx: 本 warp 在所有 SM 的 notify warp 中的全局编号
        //   每 token 由一个 warp 处理, 每 lane 负责一个 top-k 选择
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        const auto global_warp_idx = sm_idx * kNumNotifyWarps + warp_idx;
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
            // expert_count: 每个 lane 的 top-k 选择对应的 expert, 直接 atomicAdd (无去重)
            //   因为同一 token 的不同 top-k 可能选同一 expert (合法)
            const auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // rank_count: 需要去重! 同一 token 的多个 top-k 可能指向同一 rank
            //   deduplicate: 只让最高位 lane 执行 atomicAdd, 避免重复计数
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Step 3: 全 grid reduction — 把各 SM 的 smem 计数聚合到 workspace
        //   red_add: 原子加法 (高 16 位记录到达 SM 数, 低 32 位记录计数值)
        //   编码格式: (到达SM数 << 32) | 计数值
        #pragma unroll
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // ==================== SM 0 负责后续所有跨节点/跨 rank 通信 ====================
        if (sm_idx == 0) {
            // Step 4: 等待所有 SM 的计数到达 → 解码并写入发送缓冲区
            //   status 高 32 位 = 到达 SM 数, 低 32 位 = 计数值
            //   到达 kNumSMs 时表示所有 SM 都已聚合完毕
            #pragma unroll
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    if ((status >> 32) == kNumSMs) {
                        // encode_decode_positive: 编码为特殊格式 (0→无效, 正数→2*val+1)
                        //   避免 RDMA 写的 "0 值" 与 "未写入" 混淆
                        // ⚠️ 这里写入scaleout的send buffer
                        workspace_layout.get_scaleout_rank_expert_count_ptr<true>()[i] =
                            math::encode_decode_positive<int>(status & 0xffffffffll);

                        // 清理 workspace, 供下次 dispatch 使用
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    if (is_last_check) {
                        printf("DeepEP hybrid notify (GPU reduction) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }
                    return false;
                });
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);


            // Step 5: 跨节点发送 — 把本节点的 rank/expert 计数 RDMA put 到所有其他节点
            //   每个 thread 负责一个 scaleout rank (thread_idx < kNumScaleoutRanks)
            //   gin.put(recv_sym_ptr, send_sym_ptr, ...): RDMA 写到对端节点的 workspace
            //     第1个参数 recv_sym_ptr: 对端节点的接收区 (recv buffer, <false>) → dst
            //     第2个参数 send_sym_ptr: 本节点编码后的计数 (send buffer, <true>) → src
            //   两种计数分开 put:
            //     - rank_count: kNumScaleupRanks 个 int (每 rank 一个)
            //     - expert_count: kNumExpertsPerScaleout 个 int (每 expert 一个)
            EP_STATIC_ASSERT(kReuseSlotIndices or kNumScaleoutRanks <= kNumNotifyThreads,
                             "kNumScaleoutRanks must be less than kNumNotifyThreads");
            // ⚠️ All to All
            if (thread_idx < kNumScaleoutRanks) {
                // 一个thread 负责发送给一个 scaleout rank
                const auto dst_scaleout_rank_idx = thread_idx;
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx),  //recv_sym_ptr：远端接收区地址 → dst
                    workspace_layout.get_scaleout_rank_count_ptr<true>(dst_scaleout_rank_idx),//send_sym_ptr：本地发送区地址 → src
                    // 发送 kNumScaleupRanks 个数据
                    kNumScaleupRanks * sizeof(int), dst_scaleout_rank_idx,
                    ncclGinOptFlagsAggregateRequests);
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_expert_count_ptr<true>(dst_scaleout_rank_idx),
                    // 发送 kNumExpertsPerScaleout 个数据
                    kNumExpertsPerScaleout * sizeof(int), dst_scaleout_rank_idx);
            }
            __syncwarp();


            // Step 6: 接收其他节点的计数 + 节点内聚合
            // recv_and_reduce: 遍历所有 scaleout rank, 等待其 RDMA 数据到达, 解码求和
            //   ld_acquire_sys: 跨节点可见的 load (acquire 语义)
            //   is_decoded_positive_ready: 检查编码值是否有效 (区分 "未写入" 和 "值为0")
            //   累加后清理 (写0), 供下次 dispatch 使用
            const auto recv_and_reduce = [=](const auto& get_ptr_func, const bool& is_expert_reduction = false) -> int {
                int count = 0;
                #pragma unroll
                for (int j = 0; j < kNumScaleoutRanks; ++ j) {
                    const auto ptr = get_ptr_func(j);
                    int decoded;
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check){
                        // -n-1, -(n-1)-1 = -n+1-1 = n
                        decoded = math::encode_decode_positive(ptx::ld_acquire_sys<int>(ptr));
                        if (math::is_decoded_positive_ready(decoded))
                            return true;

                        if (is_last_check) {
                            printf("DeepEP hybrid notify (scale-out %s reduction) timeout, "
                                   "scale-out: %d, scale-up: %d, "
                                   "thread: %d, wait scale-out: %d, decoded: %d\n",
                                   is_expert_reduction ? "expert" : "rank",
                                   scaleout_rank_idx, scaleup_rank_idx, thread_idx, j,
                                   decoded);
                        }
                        return false;
                    });

                    // Add and clean for next usages
                    count += decoded, *ptr = 0;
                }
                return count;
            };

            // Step 6a: 聚合 rank 级计数 → NVLink 写到同节点所有 scaleup rank
            //   对每个 scaleup rank i:
            //     1) 等所有 scaleout rank 的 rank_count[i] 到达 → 求和
            //     2) 编码为 (kNumScaleupRanks << 32 | count) → put_value 到对端 rank 的 scaleup_rank_count
            //   高 32 位 = 到达的 scaleup rank 数 (用于对端判断是否所有 scaleup rank 都写了)
            #pragma unroll
            for (int i = thread_idx; i < kNumScaleupRanks; i += kNumNotifyThreads) {
                // ⚠️ All to All 之后求和，等价于 Reduce-Scatter
                // 每一个thread，各自等待所有scaleout rank的rank_count[i]到达，然后求和
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                    return workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_peer_idx, i);
                });

                // 当前节点的nvl rank i的累加值算好了，
                // ⚠️ 高 32 位是 scaleup rank 总数（用于对端判断是否收齐），低 32 位是聚合后的 token 计数
                // ⚠️ 接收方阻塞等待 高32位 == kNumScaleupRanks, 就意味着该数可用了
                // 🔑 高32位编码对比 (Step 6a vs 6b):
                //   6a rank_count:  put_value 直接写 → 每个 scaleup rank 写不同偏移 → 高32位一次性写 kNumScaleupRanks (就绪标记)
                //   6b expert_count: red_add 原子加 → 所有 scaleup rank 写同一位置 → 高32位每次+1, 收齐后累加 == kNumScaleupRanks
                //   殊途同归: 接收方都等 高32位 == kNumScaleupRanks 即可读
                const int64_t counter = (static_cast<int64_t>(kNumScaleupRanks) << 32ll) | count;

                // ncclTeamTagLsa 就是 NVLink 通信的 tag
                // put_value 内部会通过 ncclTeamTagLsa 获取第 i 个 GPU 的对称地址，然后用 st_relaxed_sys 直接 NVLink 写入
                // ⚠️ 节点内 All to All
                gin.put_value<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx, // sym_ptr: 对端 rank 的接收地址
                    counter,   // value: 要写入的值
                    i); // dst_rank_idx: 目标 scaleup rank 编号
            }
            __syncwarp();

            // Step 6b: 聚合 expert 级计数 → NVLink red_add 到同节点对应 rank
            //   对每个 expert i (属于本节点的 kNumExpertsPerScaleout 个):
            //     1) 等所有 scaleout rank 的 expert_count[i] 到达 → 求和
            //     2) 编码为 (1 << 32 | count) → red_add 到对端 rank 的 scaleup_expert_count
            //     注意: expert 是按 rank 分配的, i/kNumExpertsPerRank = 目标 scaleup rank
            #pragma unroll
            for (int i = thread_idx; i < kNumExpertsPerScaleout; i += kNumNotifyThreads) {
                // ⚠️ All to All 之后求和，等价于 Reduce-Scatter
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                    return workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_peer_idx, i);
                }, true);

                // Write into the remote scale-up peer
                //   red_add_rel: 原子加 + release 语义 (保证对端可见)
                //   counter 高 32 位=1 (表示1个 scaleup rank 的贡献)
                //   dst_scaleup_rank_idx: expert i 所属的 scaleup rank
                // ⚠️ 接收方阻塞等待 高32位 == kNumScaleupRanks, 就意味着该数可用了
                const int64_t counter = (1ll << 32ll) | count;
                const auto dst_scaleup_rank_idx = i / kNumExpertsPerRank;
                const auto expert_idx_in_dst_rank = i % kNumExpertsPerRank;
                // ⚠️ 节点内就不作All to All，直接求sum吧， 相当于直接Reduce Scatter
                gin.red_add_rel<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_expert_count_ptr<false>() + expert_idx_in_dst_rank, // sym_ptr: 对端 rank 的接收地址
                    counter, dst_scaleup_rank_idx);
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Step 7: 等待本地 scaleup rank/expert 计数就绪
            //   现在 smem 的 rank_count/expert_count 区域改为存储本地 (scaleup 级) 计数
            //   接下来只关心本节点内的前缀和 (供后续 epilogue 分配行号)
            expert_count = rank_expert_count + kNumScaleupRanks;

            // 等待所有 scaleup rank 的计数到达 (通过 NVLink put_value / red_add)
            //   status 高 32 位 = 到达的 scaleup rank 数, 期望 = kNumScaleupRanks
            //   status 低 32 位 = 计数值
            //   thread_idx < kNumScaleupRanks: 处理 rank 计数
            //   thread_idx >= kNumScaleupRanks: 处理 expert 计数 (本地 kNumExpertsPerRank 个)
            EP_STATIC_ASSERT(kNumNotifyWarps == 0 or kNumScaleupRanks + kNumExpertsPerRank <= kNumNotifyWarps * 32,
                             "Insufficient notify threads");
            comm::timeout_while<kNumTimeoutCycles>(thread_idx < kNumScaleupRanks + kNumExpertsPerRank,
                [&](const bool& is_last_check) {
                const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + thread_idx);
                if ((status >> 32ull) == kNumScaleupRanks) { // 高32位 == kNumScaleupRanks
                    // status 低 32 位 = 计数值
                    const auto count = static_cast<int>(status & 0xffffffffll);

                    // rank 计数不需要对齐, expert 计数按 kExpertAlignment 对齐
                    //   对齐原因: expand 模式下输出张量按 expert alignment 分配行号
                    const auto aligned_count = math::align<int>(
                        count, thread_idx < kNumScaleupRanks ? 1 : kExpertAlignment);

                    // 清理 GPU workspace
                    workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] = 0;
                    // CPU 同步模式: 写入 host workspace 供 CPU 读取
                    if constexpr (kDoCPUSync) {
                        host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] =
                            // host也要轮询等待device端写入
                            math::encode_decode_positive(aligned_count);
                    }

                    // 只累计统计各个expert的token数 (供外部监控)
                    if (cumulative_local_expert_recv_stats != nullptr and thread_idx >= kNumScaleupRanks)
                        atomicAdd(cumulative_local_expert_recv_stats + (thread_idx - kNumScaleupRanks), count);

                    // 保存到 smem, 供后续 prefix sum 使用
                    // ⚠️ 此时已经是对齐之后的值了
                    rank_expert_count[thread_idx] = aligned_count;
                    return true;
                }

                if (is_last_check) {
                    printf("DeepEP hybrid notify (scale-up reduction) timeout,"
                           "scale-out: %d/%d, scale-up: %d/%d, "
                           "thread: %d, status: %d | %d, expected: %d\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                           static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumScaleupRanks);
                }
                return false;
            });
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);


            // Step 8: 计算 prefix sum
            //   warp 0: inclusive psum of rank_count → psum_num_recv_tokens_per_scaleup_rank
            //     供 epilogue 定位每段 scaleup rank 边界
            //   warp 1: exclusive psum of expert_count → psum_num_recv_tokens_per_expert
            //     供 epilogue expand 模式分配行号 (exclusive 是因为 atomicAdd 从0开始)
            //
            //   原语说明:
            //     warp_inclusive_sum(value, lane_idx):
            //       warp 内 inclusive prefix sum, 每个 lane 输入 value, 输出 lane0..lane_idx 的累加和
            //       例: 输入 [3,1,4,2] → 输出 [3,4,8,10]
            //     exchange(sum, 31):
            //       从 lane 31 广播 sum 到所有 lane, 即获取整批总和, 作为下一批的 carry
            //
            //   分段算法 (n 可能 > 32):
            //     每批 32 元素, 由 warp_inclusive_sum 算批内前缀和
            //     psum (carry) 保存前一批的总和, 加到本批每个结果上
            //     exchange(sum, 31) 取 lane31 的 sum = psum + 本批总和, 作为下一批 carry
            //
            //   Inclusive 例子 (is_exclusive=0, count=[3,1,4,2,5,7,6], n=7):
            //     批0: value=[3,1,4,2,5,7,6,0..], psum=0
            //       warp_inclusive_sum → [3,4,8,10,15,22,28,28..]
            //       sum = 0 + [3,4,8,10,15,22,28,28..] = [3,4,8,10,15,22,28,28..]
            //       out = [3,4,8,10,15,22,28] ✓
            //       psum = exchange(sum,31) = 28
            //
            //   Exclusive 例子 (is_exclusive=1, count=[3,1,4,2,5,7,6], n=7):
            //     巧妙偏移: idx=lane_idx, mem_idx=idx-1, value=count[mem_idx]
            //       lane 0 读 count[-1]→0, lane 1 读 count[0]→3, lane 2 读 count[1]→1, ...
            //     批0: value=[0,3,1,4,2,5,7,6,0..], psum=0
            //       warp_inclusive_sum → [0,3,4,8,10,15,22,28,28..]
            //       sum = 0 + [0,3,4,8,10,15,22,28,28..]
            //       out = [0,3,4,8,10,15,22,28] ✓ (输出比输入多一个元素)
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                #pragma unroll
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    psum = ptx::exchange(sum, 31);
                }
            };

            if (warp_idx == 0) {
                // Inclusive prefix sum
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumScaleupRanks, 0);
            } else if (warp_idx == 1) {
                // Exclusive prefix sum for later expanding
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
        }

    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
        // ==================== Phase 2: Scaleout Warps (跨节点发送) ====================
        // 职责: 从本地 x 读取 token → RDMA 发送到其他节点 + 本地直写
        //
        // 每个 scaleout warp = 一个 channel
        //   channel_idx: 本 warp 的全局 channel 编号
        //   本 channel 负责 token: channel_idx, channel_idx+kNumChannels, channel_idx+2*kNumChannels, ...
        //
        // 数据流:
        //   本地 x[token] ──TMA load──▶ smem tma_buffer
        //     ├─ 本地 rank: TMA store ▶ scaleout_recv_buffer (直写)
        //     ├─ 跨节点:   TMA store ▶ scaleout_send_buffer → gin.put(RDMA) ▶ 对端 recv_buffer
        //     └─ 仅有本地: 跳过 send_buffer (scaleout_rank_mask == 1<<scaleout_rank_idx 时优化)
        const int scaleout_warp_idx = warp_idx - kNumNotifyWarps;
        const int channel_idx = sm_idx * kNumChannelsPerSM + scaleout_warp_idx;
        // recv_buffer 只看本 scaleout rank 的本 channel 段 (本地直写目标)
        scaleout_recv_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_rank_idx);
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);

        // ---------- Scaleout tail 追踪 ----------
        // stored_scaleout_tail: 本 channel 已写入的 slot 数 (对每个 scaleout rank 分别计数)
        //   每处理 kScaleoutUpdateInterval 个 token, 或 finish 时, 通过 red_add_rel 通知对端
        //   对端 forward warp 会轮询这个值来知道有多少 token 可处理
        // signaled_tail 编码: pack2(finish_flag, tail_count) → int64_t
        //   finish_flag: 本 channel 是否已处理完所有 token
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid number of scale-out ranks");
        int stored_scaleout_tail = 0, stored_old_scaleout_tail = 0;
        const auto update_scaleout_tail = [&](const bool& finish_flag = false) {
            if (lane_idx < kNumScaleoutRanks and
                (stored_scaleout_tail >= stored_old_scaleout_tail + kScaleoutUpdateInterval or finish_flag)) {
                const auto signaled_tail = math::pack2<int, int64_t>(finish_flag, stored_scaleout_tail);
                const auto ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx);
                const auto old_signaled_tail = math::pack2<int, int64_t>(0, stored_old_scaleout_tail);

                // red_add_rel: 原子加 + release 语义
                //   跨节点用 RDMA atomic 保证可见性, 本节点用 sys scope (可能走 NVLink)
                gin.red_add_rel<ncclTeamTagRail>(ptr, signaled_tail - old_signaled_tail, lane_idx);
                stored_old_scaleout_tail = stored_scaleout_tail;
            }
            __syncwarp();
        };

        // ---------- 预加载 token ----------
        // TMA load (hidden) + cp.async (SF) 双管道并行:
        //   hidden: TMA load (1D, 异步, mbarrier 通知)
        //   SF: cp.async.ca (cache-all, 异步, mbarrier_arrive 通知)
        //   两者可以重叠执行
        const auto preload_next_token = [&](const int& token_idx) {
            if (token_idx >= num_tokens)
                return;

            // TMA load hidden states: gmem x → smem tma_buffer.hidden
            const auto token_i64_idx = static_cast<int64_t>(token_idx);
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            __syncwarp();

            // cp.async SF packs: gmem sf → smem tma_buffer.sf
            //   每 lane 负责 stride 的 SF pack (与 epilogue 类似的分摊策略)
            //   cp_async_mbarrier_arrive: 注册到 mbarrier 的 tx_pending 计数
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                __syncwarp();
            }
        };

        // ---------- 主循环: 遍历本 channel 的所有 token ----------
        preload_next_token(channel_idx);
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += kNumChannels) {
            // 加载 top-k 索引 + 权重 (直接从 gmem, 不经过 TMA)
            //   stored_dst_scaleout_rank_idx: 本 lane 的 top-k 选择发给哪个 scaleout rank
            //   expert_idx / kNumExpertsPerScaleout = scaleout rank (expert 按 scaleout rank 分段)
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            int stored_dst_scaleout_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
                stored_dst_scaleout_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerScaleout : -1;
                // 写入 smem metadata (供后续 forward/epilogue 读取)
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            __syncwarp();

            // 写入源 metadata: rank_idx * kNumMaxTokensPerRank + token_idx
            //   供 epilogue/combine 识别 token 来源
            //   编码方式: 高位是 rank, 低位是 token 在 rank 内的序号
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            ptx::tma_store_fence();
            __syncwarp();

            // ---------- Slot 分配 + Scaleout tail 更新 ----------
            // deduplicate: 同一 token 的多个 top-k 可能指向同一 scaleout rank
            //   只有 master lane 获得有效 slot 编号
            //   slot 编号 = 该 rank 在本 channel 的当前 tail (exchange 先读后加)
            int stored_dst_slot_idx = -1;
            const auto stored_old_slot_idx = ptx::exchange(
                stored_scaleout_tail, stored_dst_scaleout_rank_idx >= 0 ? stored_dst_scaleout_rank_idx : 0);
            if (ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx) and stored_dst_scaleout_rank_idx >= 0)
                stored_dst_slot_idx = stored_old_slot_idx;

            // 更新 scaleout tail: 每新增一个 scaleout rank 的 token, tail+1
            //   scaleout_rank_mask: bitmap, 哪些 scaleout rank 有新 token
            //   (mask >> lane_idx) & 1: 本 lane 对应的 rank 是否有新 token
            const auto scaleout_rank_mask = ptx::reduce_or(stored_dst_scaleout_rank_idx >= 0 ? (1u << stored_dst_scaleout_rank_idx) : 0u);
            stored_scaleout_tail += (scaleout_rank_mask >> lane_idx) & 1;

            // ---------- 等待 TMA load 完成 + 发送数据 ----------
            if (ptx::elect_one_sync()) {
                // arrive_and_set_tx: 注册 TMA load 的预期字节数
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                // 优化: 如果所有 top-k 都只发给本节点 (无需 RDMA), 跳过 send_buffer
                //   scaleout_rank_mask ^ (1 << scaleout_rank_idx): 去掉本节点后还有其他节点
                //   只有需要跨节点发送时才 TMA store 到 send_buffer
                if (scaleout_rank_mask ^ (1 << scaleout_rank_idx)) {
                    ptx::tma_store_1d(scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                                      tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                }
            }
            __syncwarp();

            // 本地直写: 如果 top-k 选择指向本节点, 直接 TMA store 到 recv_buffer
            //   不经过 send_buffer, 也不需要 RDMA
            //   stored_dst_scaleout_rank_idx == scaleout_rank_idx: 发给本节点
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx == scaleout_rank_idx) {
                ptx::tma_store_1d(scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                  tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            }
            ptx::tma_store_commit();
            ptx::tma_store_wait();
            __syncwarp();

            // 预加载下一个 token (与 IBGDA RDMA 请求重叠, 隐藏延迟)
            preload_next_token(token_idx + kNumChannels);

            // RDMA 发送: 如果 top-k 选择指向其他节点, 通过 gin.put 发送
            //   src: send_buffer 中的数据 (刚才 TMA store 写入的)
            //   dst: 对端节点的 recv_buffer (直接写入对端 gmem)
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx != scaleout_rank_idx) {
                gin.put<ncclTeamTagRail>(
                        scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                        tma_buffer.get_num_bytes<false>(),
                        stored_dst_scaleout_rank_idx,
                        ncclGinOptFlagsAggregateRequests);
            }
            __syncwarp();

            // 定期通知对端 forward warp: 本 channel 新增了多少 token
            update_scaleout_tail();
        }

        // 循环结束后, 刷新未发送的 tail (finish_flag=true 通知对端本 channel 已结束)
        update_scaleout_tail(true);

    // ==================== Phase 3: Forward Warps (节点内 NVLink 转发) ====================
    // 职责: 从 recv_buffer 读取跨节点来的 token → 通过 NVLink 转发到同节点其他 rank
    //
    // 数据流:
    //   其他节点 scaleout warp ──RDMA──▶ recv_buffer (本节点 gmem)
    //                                       │
    //                                       ▼ Forward Warp
    //   本节点 scaleout warp ──直写──▶ recv_buffer (本地 slot)
    //                                       │
    //                                       ▼ TMA load
    //                                  smem tma_buffer
    //                                       │
    //                          ┌────────────┼────────────┐
    //                          ▼            ▼            ▼
    //                    scaleup_buf    scaleup_buf   scaleup_buf
    //                    [rank 0]      [rank 1]     [rank N]
    //                    NVLink        NVLink       NVLink
    //
    // 每个 forward warp = 一个 channel (与 scaleout warp 一一对应)
    //   对端 scaleout warp 写入 recv_buffer → 本 forward warp 读出并转发
    //
    // Channel 映射:
    //   forward_warp_idx = warp_idx - (kNumNotifyWarps + kNumScaleoutWarps)
    //   channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx
    //   与 scaleout warp 的 channel_idx 计算方式相同, 保证一一对应
    } else {
        const int forward_warp_idx = warp_idx - (kNumNotifyWarps + kNumScaleoutWarps);
        const int channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;
        // recv_buffer: 只看本 channel 的数据 (跨所有 scaleout rank)
        //   与 scaleout 不同: scaleout 只看本 rank 的本 channel, forward 看所有 rank 的本 channel
        //   因为 forward 需要轮询所有 scaleout rank 发来的数据
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);
        // scaleup_buffer: 本 scaleup rank 的接收区 (NVLink 直写目标)
        //   对端 rank 通过 NVLink TMA store 直写到这个 buffer 的 slot 中
        scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // ---------- Metadata 布局 ----------
        // token_metadata_at_forward: 全局形状 [kNumChannels][kNumScaleoutRanks * kNumMaxTokensPerChannel + 1][kNumForwardMetadataDims]
        //   +1: 末尾的结束标记 (-1)
        //   每个 channel 的 forward warp 独占一段, 无需跨 channel 同步
        //   kNumForwardMetadataDims = 2 + kNumTopk * 2:
        //     [0]: src_token_global_idx  — token 在源 rank 的全局索引 (rank_idx * max_tokens + token_idx)
        //     [1]: is_last_token_flag    — 是否是本 chunk 的最后一个 token (用于 combine 阶段判断边界)
        //     [2..2+topk-1]: stored_dst_scaleup_rank_idx  — 每个 top-k 选择的目标 scaleup rank (-1 表示无效)
        //     [2+topk..2+2*topk-1]: stored_dst_slot_idx   — 每个 top-k 选择在 scaleup_buffer 中的 slot 编号
        //   供 combine 反向路径使用: 从 scaleup_buffer 读回数据时, 知道每个 token 来自哪个 rank/slot
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // dst_buffer_slot_idx: 全局形状 [kNumChannels, kNumScaleoutRanks, kNumMaxTokensPerChannel, kNumTopk]
        //   记录每个 (channel, scaleout_rank, slot, topk_idx) 的目标 slot 编号
        //   kReuseSlotIndices 模式下, 后续 combine 直接从这里读取 slot, 无需重新计算
        dst_buffer_slot_idx += channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel * kNumTopk);

        // ---------- Linked list 索引变换 ----------
        // 将逻辑 linked list 索引 → 全局物理索引
        //   全局布局: [kNumChannels][kNumTokensInLinkedList][kNumScaleupRanks]
        //     - channel_idx: 本 channel 在全局中的偏移
        //     - idx * kNumScaleupRanks + scaleup_rank_idx: 在 linked list 中的位置
        //   +1 是给 tail 节点 (哨兵) 用的, kNumTokensInLinkedList = kNumMaxTokensPerChannel * kNumScaleoutRanks + 1
        //
        //   举例: channel=2, idx=5, scaleup_rank_idx=3, kNumScaleupRanks=8
        //     → 2 * (N * 8) + 5 * 8 + 3 (N = kNumTokensInLinkedList)
        //     → 本 channel 的第 5 个 linked list 节点中, 属于 scaleup rank 3 的位置
        const auto transform_linked_list_idx = [=](const int& idx) {
            constexpr int kNumTokensInLinkedList = kNumMaxTokensPerChannel * kNumScaleoutRanks + 1;
            return channel_idx * (kNumTokensInLinkedList * kNumScaleupRanks) +
                idx * kNumScaleupRanks + scaleup_rank_idx;
        };

        // ==================== Forward 主循环 ====================
        // 轮询所有 scaleout rank, 从 recv_buffer 读取 token 并转发
        //
        // 核心数据结构 (per lane):
        //   stored_scaleout_old_tail_idx: 每个 scaleout rank 已处理的 slot 数 (lane_idx < kNumScaleoutRanks)
        //   stored_scaleout_tail_idx:     每个 scaleout rank 已到达的 slot 总数 (从 signaled_tail 解码)
        //   stored_finish_flag:           每个 scaleout rank 是否已发完 (0=未完, >0=已完)
        //   stored_scaleup_send_counters: 本 lane 向各 scaleup rank 累计发送的 token 数
        //     用于构建 channel linked list (记录 token 在 scaleup_buffer 中的顺序)
        //
        // 轮询策略: Round-Robin
        //   每次从上次处理的 rank 的下一个开始, 找到第一个有新数据的 rank
        //   避免饥饿: 所有 rank 的数据都会被处理
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Too many scale-out ranks");
        int num_tokens_processed = 0;
        int stored_scaleout_old_tail_idx = 0;   // per lane: 本 rank 已处理的 slot 上界
        int stored_scaleup_send_counters[kNumScaleupRanksPerLane] = {};  // per lane: 向各 scaleup rank 发送的计数 (用于 linked list)
        int stored_finish_flag = lane_idx >= kNumScaleoutRanks;  // per lane: 本 rank 是否已完成 (>0=finish, 0=not yet)
        int stored_scaleout_tail_idx = 0;       // per lane: 本 rank 的最新 tail (从 signaled_tail 解码)
        int recv_scaleout_rank_idx = channel_idx % kNumScaleoutRanks;  // 当前轮询的 scaleout rank

        // wip_mask: "work in progress" 位掩码
        //   每个 lane 贡献 1 bit: (stored_scaleout_tail_idx > stored_scaleout_old_tail_idx) or (stored_finish_flag == 0)
        //   即: 该 lane 对应的 rank 有新数据, 或还没结束
        //   gather: 将所有 lane 的 bit 收集到一个 uint32_t
        //   wip_mask != 0 表示还有工作要做
        uint32_t wip_mask;
        while ((wip_mask = ptx::gather(stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag == 0))) {
            // ---------- Round-Robin 选择下一个有数据的 scaleout rank ----------
            // 从上次处理的 rank 的下一个开始找, 避免总是从 rank 0 开始
            //   offset = (上次 rank + 1) % kNumScaleoutRanks
            //   hi_mask: offset 之后的位 (高位部分)
            //   如果 hi_mask 非零 → 找到高位第一个 1 (ffs)
            //   否则 → 从低位开始找 (wrap-around)
            const auto offset = (recv_scaleout_rank_idx + 1) % kNumScaleoutRanks;
            const auto hi_mask = (wip_mask >> offset) << offset;
            recv_scaleout_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);

            // ---------- 等待选中 rank 的数据就绪 ----------
            // 检查: stored_scaleout_tail_idx > stored_scaleout_old_tail_idx (有新数据)
            //        stored_finish_flag > 0 (已结束, 无需再等)
            // exchange(arrived_or_finished, recv_scaleout_rank_idx):
            //   把本 lane 的状态广播到 recv_scaleout_rank_idx 对应的 lane
            //   如果那个 lane 报告 "有数据" 或 "已结束", 则返回 true
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                const uint32_t arrived_or_finished =
                    stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag > 0;
                if (ptx::exchange(arrived_or_finished, recv_scaleout_rank_idx))
                    return true;

                // 超时打印
                if (is_last_check) {
                    if (lane_idx < kNumScaleoutRanks) {
                        printf("DeepEP hybrid dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, "
                               "channel: %d, lane: %d, old scale-out tail: %d, scale-out tail: (%d, %d)\n",
                               scaleout_rank_idx, scaleup_rank_idx,
                               channel_idx, lane_idx, stored_scaleout_old_tail_idx,
                               stored_finish_flag, stored_scaleout_tail_idx);
                    }
                    return false;
                }

                // 重新读取 signaled_tail (scaleout warp 定期更新)
                //   ld_acquire_sys: 跨节点可见的 load (acquire 语义, 保证看到 scaleout warp 的写入)
                //   unpack2: 解码 pack2(finish_flag, tail_count)
                //   每个 lane 只读自己对应的 scaleout rank 的 tail
                if (lane_idx < kNumScaleoutRanks) {
                    const auto signaled_tail = ptx::ld_acquire_sys<int64_t>(
                        workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx));
                    math::unpack2<int, int64_t>(signaled_tail, stored_finish_flag, stored_scaleout_tail_idx);
                }
                __syncwarp();
                return false;
            });

            // ---------- 处理一个 chunk (最多 kNumSlotsPerForwardChunk 个 slot) ----------
            // 从 start_slot_idx 到 end_slot_idx, 每次 TMA load + NVLink 转发
            // exchange: 把本 lane 的 old_tail/tail 广播给所有 lane, 让每个 lane 都知道当前 rank 的起止位置
            const auto start_slot_idx = ptx::exchange(stored_scaleout_old_tail_idx, recv_scaleout_rank_idx);
            const auto end_slot_idx = std::min(
                ptx::exchange(stored_scaleout_tail_idx, recv_scaleout_rank_idx),
                start_slot_idx + kNumSlotsPerForwardChunk  // 限制 chunk 大小, 避免一次处理太多
            );
            // 更新 old_tail: 本 lane 对应的 rank 已处理到 end_slot_idx
            if (lane_idx == recv_scaleout_rank_idx)
                stored_scaleout_old_tail_idx = end_slot_idx;

            // 遍历本 chunk 的每个 slot
            const auto recv_buffer = scaleout_recv_buffer.get_rank_buffer(recv_scaleout_rank_idx);
            for (int slot_idx = start_slot_idx; slot_idx < end_slot_idx; ++ slot_idx) {
                const auto token_buffer = recv_buffer.get_token_buffer(slot_idx);

                // 等待之前的 TMA store 完成 (保证 smem 可复用)
                ptx::tma_store_wait();
                __syncwarp();

                // ---------- TMA load: recv_buffer → smem tma_buffer ----------
                // elect_one: 只有一个 lane 发起 TMA load (整个 warp 共享 smem)
                // get_num_bytes<false>: 只加载数据部分, 不含 mbarrier (这是 TMA load, 不是 store)
                if (ptx::elect_one_sync()) {
                    ptx::tma_load_1d(tma_buffer.get_base_ptr(), token_buffer.get_base_ptr(),
                                     mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                }
                __syncwarp();

                // ---------- 读取 top-k 索引, 计算目标 scaleup rank ----------
                // 注意: token 的 top-k expert 已经是全局编号, 需要减去本节点的 expert 偏移
                //   dst_expert_idx -= scaleout_rank_idx * kNumExpertsPerScaleout
                //   然后除以 kNumExpertsPerRank 得到目标 scaleup rank
                //
                //   举例: 全局 expert=5, 本节点 scaleout_rank_idx=0, kNumExpertsPerScaleout=8
                //     dst_expert_idx = 5 - 0 = 5, kNumExpertsPerRank=2
                //     stored_dst_scaleup_rank_idx = 5 / 2 = 2 (scaleup rank 2)
                EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
                int stored_dst_scaleup_rank_idx = -1;
                auto dst_expert_idx = lane_idx < kNumTopk ? tma_buffer.get_topk_idx_ptr()[lane_idx] : -1;
                dst_expert_idx -= scaleout_rank_idx * kNumExpertsPerScaleout;
                stored_dst_scaleup_rank_idx = 0 <= dst_expert_idx and dst_expert_idx < kNumExpertsPerScaleout ?
                    dst_expert_idx / kNumExpertsPerRank : -1;

                // ---------- 构建 Channel Linked List ----------
                // linked_list_idx: 本 lane 的 top-k 选择在 linked list 中的位置
                //   = 之前已发给同一 scaleup rank 的 token 数 (即 stored_scaleup_send_counters)
                //   即: 本 token 是该 scaleup rank 收到的第 linked_list_idx 个 token
                //
                // kNumScaleupRanksPerLane: 每 lane 负责的 scaleup rank 数 (ceil(kNumScaleupRanks/32))
                //   因为 kNumScaleupRanks 可能 > 32, 一条 lane 不够放
                //   src_lane_idx: stored_dst_scaleup_rank_idx 相对于当前 j*32 的偏移
                //   valid: 本 lane 是否负责这个 scaleup rank
                //   exchange(stored_scaleup_send_counters[j], ...): 读出当前计数 (作为 linked list 位置)
                int linked_list_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = stored_dst_scaleup_rank_idx - j * 32;
                    const bool valid = 0 <= src_lane_idx and src_lane_idx < 32;
                    const auto exchanged = ptx::exchange(
                        stored_scaleup_send_counters[j], valid ? src_lane_idx : 0);
                    linked_list_idx = valid ? exchanged : linked_list_idx;
                }
                // 将 linked list 位置写入 tma_buffer 的 metadata 区域 (与 TMA 数据一起转发)
                //   transform_linked_list_idx: 逻辑索引 → 全局物理索引
                //   kReuseSlotIndices 模式跳过 (slot 已知, 不需要 linked list)
                if (not kReuseSlotIndices and lane_idx < kNumTopk) {
                    tma_buffer.get_linked_list_idx_ptr()[lane_idx] = transform_linked_list_idx(linked_list_idx);
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // ---------- Slot 分配 (去重) ----------
                // deduplicate: 同一 token 的多个 top-k 可能指向同一 scaleup rank
                //   只有 master lane (最高位) 获得有效 slot 编号
                //   atomicAdd: 从全局计数器中分配一个新 slot (保证不同 channel 的 token 不冲突)
                //
                // kReuseSlotIndices 模式: 直接从 dst_buffer_slot_idx 读取之前分配的 slot
                //   (低延迟模式下, slot 编号在 dispatch 和 combine 之间保持不变)
                int stored_dst_slot_idx = -1;
                const auto dst_slot_idx_ptr = dst_buffer_slot_idx +
                    recv_scaleout_rank_idx * (kNumMaxTokensPerChannel * kNumTopk) + slot_idx * kNumTopk;
                if constexpr (kReuseSlotIndices) {
                    if (lane_idx < kNumTopk)
                        stored_dst_slot_idx = __ldg(dst_slot_idx_ptr + lane_idx);
                } else {
                    // 去重 + atomicAdd 分配 slot
                    if (ptx::deduplicate(stored_dst_scaleup_rank_idx, lane_idx) and stored_dst_scaleup_rank_idx >= 0)
                        stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_scaleup_rank_idx, 1);
                }
                __syncwarp();

                // ---------- TMA store: smem → scaleup_buffer[NVLink 对端 rank] ----------
                // get_sym_ptr: 获取对端 rank 的 gmem 地址 (NVLink 可访问的对称地址)
                //   stored_dst_scaleup_rank_idx: 目标 scaleup rank 编号
                //   TMA store 1D: 整个 token (hidden + sf + metadata) 一次性写入
                //   tma_store_commit + 后续 tma_store_wait: 确保写入完成
                if (stored_dst_slot_idx >= 0) {
                    const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
                        scaleup_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        stored_dst_scaleup_rank_idx);
                    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                __syncwarp();

                // ---------- 更新 per-scaleup 发送计数 ----------
                // scaleup_send_mask: bitmap, 哪些 scaleup rank 收到了本 token 的某个 top-k
                //   reduce_or: warp 级 OR, 汇总所有 lane 的目标 rank
                // stored_scaleup_send_counters[j] += bit: 如果本 lane 负责的 scaleup rank 有新 token, 计数+1
                //   这就是 linked list 的构建过程: 每发一个 token, 计数递增, 下一个 token 的 linked_list_idx 就更大
                EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Invalid number of scale-up peers");
                using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
                const auto scaleup_send_mask = ptx::reduce_or(
                    stored_dst_scaleup_rank_idx >= 0 ?
                    (mask_t(1) << stored_dst_scaleup_rank_idx) : mask_t(0));
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_scaleup_send_counters[j] += (scaleup_send_mask >> (j * 32 + lane_idx)) & 1;

                // ---------- 记录 metadata ----------
                // 写入 token_metadata_at_forward, 供 combine 反向路径使用
                //   kReuseSlotIndices 模式跳过 (metadata 不变)
                if constexpr (not kReuseSlotIndices) {
                    EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of selections");
                    const auto metadata_ptr = token_metadata_at_forward +
                        num_tokens_processed * kNumForwardMetadataDims;

                    // [0]: 源 token 全局索引 (哪个 rank 的哪个 token)
                    // [1]: 是否是本 chunk 最后一个 (slot_idx == end_slot_idx - 1)
                    if (ptx::elect_one_sync()) {
                        metadata_ptr[0] = tma_buffer.get_src_token_global_idx_ptr()[0];
                        metadata_ptr[1] = slot_idx == (end_slot_idx - 1);
                    }

                    // [2..2+topk-1]: 每个 top-k 的目标 scaleup rank (-1 表示无效)
                    // [2+topk..2+2*topk-1]: 每个 top-k 分配的 slot 编号 (-1 表示无效)
                    // 同时写入 dst_buffer_slot_idx (供 kReuseSlotIndices 模式或 combine 读取)
                    if (lane_idx < kNumTopk) {
                        metadata_ptr[2 + lane_idx] = stored_dst_scaleup_rank_idx;
                        metadata_ptr[2 + kNumTopk + lane_idx] = stored_dst_slot_idx;
                        dst_slot_idx_ptr[lane_idx] = stored_dst_slot_idx;
                    }
                }
                num_tokens_processed += 1;
                __syncwarp();
            }
        }

        // ---------- 写入 metadata 结束标记 ----------
        // 在 metadata 数组末尾写入 -1, 表示 token 序列结束
        //   combine 阶段遍历 metadata 时, 遇到 src_token_global_idx == -1 就停止
        if (not kReuseSlotIndices and ptx::elect_one_sync())
            token_metadata_at_forward[num_tokens_processed * kNumForwardMetadataDims] = -1;
        __syncwarp();

        // ---------- 更新 linked list 的 tail 指针 ----------
        // stored_scaleup_send_counters[i]: 本 lane 向第 (i*32+lane_idx) 个 scaleup rank 发送的 token 总数
        //   这个值就是该 scaleup rank 在本 channel 的 linked list 的尾部位置
        //   transform_linked_list_idx: 逻辑位置 → 全局物理索引
        //
        //   epilogue 阶段通过 tail 指针找到 linked list 的最后一个节点, 然后反向遍历
        //
        //   st_relaxed_sys: 写入对端 rank 的 gmem (NVLink 可见, release 语义)
        //   get_sym_ptr<ncclTeamTagLsa>: 获取对端 rank 的对称地址
        if constexpr (not kReuseSlotIndices) {
            const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                    ptx::st_relaxed_sys(
                        gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j),
                        transform_linked_list_idx(stored_scaleup_send_counters[i]));
                }
            }
        }
        __syncwarp();

        // ---------- 清理 scaleout tail 计数器 ----------
        // 写 0 清零, 供下次 dispatch 使用
        //   每个 lane 清理自己对应的 scaleout rank 的 tail
        if (lane_idx < kNumScaleoutRanks)
            *workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx) = 0;
        __syncwarp();
    }

    // ==================== 结束同步 ====================
    // Scale-up barrier: 确保所有 NVLink TMA store 数据到达
    //   只做 scale-up (节点内) barrier, 不做 scale-out barrier
    //   因为 scale-out 的 token 已经被 forward warp 消费完毕, 不需要再同步
    //   Tag1: syncStart=true, syncEnd=true (双向同步)
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag1, true, true, false>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, /* do not scale-out */ false, true);

    // 触发 copy epilogue kernel (programmatic launch, 与本 kernel 流水重叠)
    //   epilogue 在本 kernel 完成后立即启动, 无需 CPU 参与
    cudaTriggerProgrammaticLaunchCompletion();

    // ---------- 清理 scaleup 原子计数器 ----------
    // get_scaleup_atomic_sender_counter: forward 阶段用于分配 slot 的原子计数器
    //   只由 SM0 的线程清理, 避免重复清零
    //   kReuseSlotIndices 模式跳过 (slot 编号跨 dispatch 复用, 不需要清零)
    EP_STATIC_ASSERT(kNumScaleupRanks <= kNumThreads, "Insufficient threads");
    if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumScaleupRanks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace deep_ep::elastic
