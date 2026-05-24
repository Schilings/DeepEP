#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/impls/combine_utils.cuh>

namespace deep_ep::elastic {

// ============================================================================================
//  hybrid_combine_impl —— DeepEP 混合 Combine 内核
// ============================================================================================
//
//  【总览】
//  Combine 是 Dispatch 的逆过程: 将各 expert 处理后的 hidden states 按原始路由权重加权求和,
//  还原为完整的 token 表示。在混合拓扑 (NVLink + RDMA) 下, 数据经历了两次分发:
//
//    Dispatch: 本 rank →(NVLink)→ 同节点其他 rank →(RDMA)→ 跨节点 rank
//    Combine  则反向: 跨节点 rank →(RDMA)→ 同节点 rank →(NVLink)→ 本 rank
//
//  【两阶段 Warp 分工】
//
//  ┌────────────────────────────────────────────────────────────────────────┐
//  │                        hybrid_combine_impl                           │
//  │                                                                      │
//  │  ┌──────────────────────┐     ┌──────────────────────────────────┐  │
//  │  │ Scale-up Warps       │     │ Forward Warps                   │  │
//  │  │ (warp 0..kNumSUp-1)  │     │ (warp kNumSUp..kNumSUp+kNumFwd) │  │
//  │  │                      │     │                                  │  │
//  │  │ 遍历 channel_linked  │     │ 重放 dispatch 元数据             │  │
//  │  │ _list, 从同节点其他  │     │ token_metadata_at_forward       │  │
//  │  │ rank 的 scaleup      │     │                                  │  │
//  │  │ buffer 读取 hidden   │     │ 等 scaleup tail 就绪后:         │  │
//  │  │ states, TMA store 到 │     │   - 从 scaleup_buffer 读数据     │  │
//  │  │ 对端 rank 的         │     │   - 本地 reduce (多 top-k)       │  │
//  │  │ scaleup buffer       │     │   - TMA store → scaleout buffer  │  │
//  │  │                      │     │   - RDMA put → 跨节点 rank       │  │
//  │  │ 同时维护 scaleup     │     │                                  │  │
//  │  │ tail (st.release.sys)│     │ 最后: 清理 tail + scaleout 同步  │  │
//  │  └──────────────────────┘     └──────────────────────────────────┘  │
//  └────────────────────────────────────────────────────────────────────────┘
//
//  【数据流】
//
//  (跨节点 expert 结果)
//        │
//        ▼ RDMA get (由 scaleup warp 从对端 scaleup buffer 读取)
//  ┌─────────────┐   TMA store    ┌──────────────────┐
//  │ 对端 rank的  │ ◄──────────── │ scaleup warp     │
//  │ scaleup_buf │               │ (NVLink TMA)     │
//  └──────┬──────┘               └──────────────────┘
//         │ forward warp 读取
//         ▼
//  ┌──────────────────┐
//  │ 本 rank 的        │    本地 reduce (加权求和)
//  │ scaleup_buf      │ ──────────────────────┐
//  └──────────────────┘                       │
//         │                                   ▼
//         │ TMA store              ┌──────────────────┐
//         ▼                        │ smem (tma_buffer) │
//  ┌──────────────────┐           │ reduce 结果       │
//  │ scaleout_send_buf│           └──────────────────┘
//  │ (跨节点 RDMA)    │
//  └──────────────────┘──RDMA put──▶ 跨节点 rank 的 scaleout_recv_buf
//
//  【三种 Reduce 模式】(Scale-up Warps, 第229-302行)
//    1. no_local_reduce:  不展开 / 仅1个top-k在本rank → 直接 TMA load+store
//    2. local_reduce:     展开且多top-k在本rank → combine_reduce() 加权求和到 smem
//    3. expanded_send:    展开但不允许多次reduce → 每个top-k分别load+store到不同rank buffer
//
//  【Forward Warps 的两种路径】(第447-565行)
//    - kAllowMultipleReduction=false: 逐 top-k TMA load → TMA store → RDMA put (无本地reduce)
//    - kAllowMultipleReduction=true:  combine_reduce() 本地聚合 → TMA store → RDMA put (延迟发射)
//
// ============================================================================================

// ──────────────────────────────────────────────────────────────────────
//  模板参数说明
// ──────────────────────────────────────────────────────────────────────
//  kUseExpandedLayout   : 前向是否展开 (每个 top-k 占独立行), 影响 reduce 路径
//  kAllowMultipleReduction : 是否允许多 top-k 本地聚合 (影响 scaleup/scaleout 布局)
//  kNumSMs              : 使用的 SM 数量
//  kNumScaleupWarps     : 每个 SM 的 scale-up warp 数 (= NVLink 通信 warps)
//  kNumForwardWarps     : 每个 SM 的 forward warp 数 (= RDMA 转发 warps)
//  kNumScaleoutRanks    : 跨节点 rank 数 (RDMA peer 数)
//  kNumScaleupRanks     : 同节点 rank 数 (NVLink peer 数)
//  kHidden              : hidden dimension
//  kNumMaxTokensPerRank : 每个 rank 最多处理的 token 数
//  kNumExperts          : expert 总数
//  kNumTopk             : 每个 token 选的 top-k 数
//  kNumQPs              : RDMA QP 数量
//  kNumTimeoutCycles    : 超时周期数
//
//  派生参数:
//  kNumScaleupRanksPerLane : 每个 lane 负责的 scaleup rank 数 (ceil(kNumScaleupRanks/32))
//  kNumScaleupUpdateInterval : scaleup tail 更新间隔 (st.release.sys 较慢, 批量更新)
//  kNumChannelsPerSM     : 每 SM 的 channel 数 = kNumForwardWarps
//  kNumChannels          : 总 channel 数 = kNumChannelsPerSM * kNumSMs
//  kNumMaxTokensPerChannel : 每 channel 最多 token 数 = ceil(kNumMaxTokensPerRank / kNumChannels)
// ──────────────────────────────────────────────────────────────────────
template <
          // ⚠️ kUseExpandedLayout 默认为 True
          bool kUseExpandedLayout, bool kAllowMultipleReduction,

          int kNumSMs,
          int kNumScaleupWarps, int kNumForwardWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumQPs, int64_t kNumTimeoutCycles,

          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),

          // ⚠️ 通知间隔
          int kNumScaleupUpdateInterval = 3,

          // ⚠️ 还是一个warp一个channel
          int kNumChannelsPerSM = kNumForwardWarps,
          int kNumChannels = kNumChannelsPerSM * kNumSMs,
          // ⚠️ ceil_div(kNumMaxTokensPerRank, kNumChannels)
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),

          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumWarps = kNumScaleupWarps + kNumForwardWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),

          // ⚠️ use_rank_layout: 决定用 "按rank" 还是 "按topk" 的布局
          // not kAllowMultipleReduction → false：固定用 topk 布局
          // kAllowMultipleReduction：rank 数 ≤ topk 数 → true用 rank 布局（更紧凑）
          bool kUseScaleoutRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          bool kUseScaleupRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>(),
          // ⚠️ 返回 use_rank_layout() ? kNumRanks : kNumTopk;
          // get_num_tokens_in_layout: 布局中每组的 token 槽位数
          int kNumTokensInScaleoutLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          int kNumTokensInScaleupLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>()>

__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_combine_impl(
    // ── 输出/输入张量 ──
    nv_bfloat16* x,            // [num_tokens, kHidden] expert 处理后的 hidden states (也是最终 reduce 写回目标)
    float* topk_weights,       // [num_tokens, kNumTopk] top-k 权重 (非展开模式用于加权求和, 展开模式=nullptr)
    // ── 元数据 ──
    int* src_metadata,         // dispatch 阶段写入的源 metadata, scale-up warp 用于定位 token 源地址
    int* psum_num_recv_tokens_per_scaleup_rank, // 前缀和: 每个 scaleup rank 已收到的 token 总数
    int* token_metadata_at_forward,  // dispatch 阶段 forward warp 记录的元数据, 供 combine forward warp 重放
    int* channel_linked_list,  // [kNumChannels, kNumScaleoutRanks*kNumMaxTokensPerChannel+1, kNumScaleupRanks]
                               // dispatch 阶段构建的链表, scale-up warp 遍历此链表找到要处理的 token
    // ── 通信 ──
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
    // ── 缓冲区 ──
    void* buffer,              // 主缓冲区 (包含 scaleup + scaleout recv + scaleout send)
    void* workspace,           // 工作区 (存放 tail 指针、计数等)
    // ── 身份 ──
    const int scaleout_rank_idx,  // 本 rank 在跨节点组中的编号
    const int scaleup_rank_idx,   // 本 rank 在同节点组中的编号
    int num_reduced_tokens        // 需要处理的 token 总数 (如果等于 kNumMaxTokensPerRank*kNumRanks 则从 GPU 读取)
) {
    // ═══════════════════════════════════════════════════════════════════
    //  公共初始化
    // ═══════════════════════════════════════════════════════════════════
    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;
    // ⚠️ kDoExpandedSend: 展开模式 + 不允许多次reduce → 每个 top-k 分别发送到不同 rank 的 buffer

    // Combine 向量类型选择: 按 hidden 对齐选择 int4(16B) 或 longlong4(32B)
    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);

    // 工作区布局 (存放 tail 指针、计数器等跨 SM/跨节点同步数据)
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);

    // 如果没有 CPU 同步, 则从 GPU 读取实际的 received token 数
    // (默认值 kNumMaxTokensPerRank*kNumRanks 是上界, 实际可能更少)
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumScaleupRanks - 1);

    // Token 布局: 描述单个 token 在内存中的排列 (hidden + metadata + topk 等)
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);

    // ── smem (TMA buffer): 每个 warp 独立一块, 用于 TMA load/store 的中转 ──
    //   布局: [warp_0_token_buf, warp_1_token_buf, ...]
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto tma_buffer = layout::BufferLayout<true>(
        token_layout, kNumWarps, 1, smem).get_rank_buffer(warp_idx).get_token_buffer(0);

    // ── gmem buffer 布局 (连续排列) ──
    //  ┌─────────────────────┬──────────────────────┬──────────────────────────┐
    //  │  scaleup_buffer     │  scaleout_recv_buf   │  scaleout_send_buf       │
    //  │  同节点收到的数据    │  跨节点收到的数据     │  跨节点待发送的数据       │
    //  │  (NVLink 可见)       │  (本 rank 最终结果)   │  (RDMA 发往其他节点)      │
    //  └─────────────────────┴──────────────────────┴──────────────────────────┘
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInScaleupLayout, kNumScaleoutRanks * kNumMaxTokensPerRank,
        buffer);
    // ⚠️ scaleout_recv_buffer
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInScaleoutLayout, kNumMaxTokensPerRank,
        scaleup_buffer.get_buffer_end_ptr());

    // ⚠️ scaleout_send_buffer: 跨节点 rank 通过 RDMA put 写入的数据, 本 rank 从这里读取
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, kAllowMultipleReduction ? 1 : kNumTopk, kNumChannels * (kNumScaleoutRanks * kNumMaxTokensPerChannel),
        scaleout_recv_buffer.get_buffer_end_ptr());

    // ── TMA mbarrier 初始化: 用于 TMA 异步 load/store 的完成通知 ──
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // ── NCCL Gin handle: 封装 NVLink/RDMA 通信原语 ──
    //   每个 warp 是一个 channel, 共享 QP (多个 warp 可复用同一 QP)
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM>(sm_idx, warp_idx % kNumChannelsPerSM);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // ── Grid 同步屏障: 确保 dispatch 阶段完成 + scaleup tail 清理完毕 ──
    //   这是 combine 开始前的前置同步, 必须等 dispatch 数据全部就绪
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // ── 寄存器调整: forward warp 需要更多寄存器做 reduce, scaleup warp 少用 ──
    const bool kAdjustRegisters = (kNumChannelsPerSM == 4 or kNumChannelsPerSM == 8) and not kUseExpandedLayout;
    constexpr int kNumRegistersForScaleupWarps = 40;
    constexpr int kNumRegistersForForwardWarps = 256 - kNumRegistersForScaleupWarps;

    // ═══════════════════════════════════════════════════════════════════
    //  Scale-up Warps: 遍历链表, 从同节点其他 rank 读取 hidden states,
    //                  TMA store 到对端 rank 的 scaleup buffer
    // ═══════════════════════════════════════════════════════════════════
    //
    //  整体流程:
    //  ┌──────────────────────────────────────────────────────────────┐
    //  │ while (链表未遍历完) {                                      │
    //  │   1. 从 channel_linked_list 加载 token_idx                 │
    //  │   2. 检查是否所有 rank 都遍历完了 (token_idx < 0)           │
    //  │   3. Round-robin 选择下一个活跃的 dst_scaleup_rank_idx      │
    //  │   4. 从 src_metadata 读取 token 的源信息                    │
    //  │   5. 根据模式:                                              │
    //  │      a) no_local_reduce: 直接 TMA load → TMA store          │
    //  │      b) local_reduce:   combine_reduce → TMA store          │
    //  │      c) expanded_send:  逐 top-k TMA load → TMA store       │
    //  │   6. 更新 tail (st.release.sys, 间隔批量发射)              │
    //  │   7. 推进链表指针                                           │
    //  │ }                                                           │
    //  │ 最后: update_tails(finish=true)                             │
    //  └──────────────────────────────────────────────────────────────┘
    //
    if (warp_idx < kNumScaleupWarps) {
        const auto channel_idx = sm_idx * kNumChannelsPerSM + warp_idx;

        // 释放多余寄存器 (scale-up warp 逻辑简单, 不需要太多寄存器)
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_dealloc<kNumRegistersForScaleupWarps>();

        // rank layout 模式: 切到本 rank 对应的 scaleup buffer 子区
        if constexpr (kUseScaleupRankLayout)
            scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // 展开模式一定是前向 (无 topk_weights), 如果传了 topk_weights 说明调用错误
        // ⚠️ 如果kUseExpandedLayout，那么top weights应该已经融合计算在里面了吧？
        if constexpr (kUseExpandedLayout)
            EP_DEVICE_ASSERT(topk_weights == nullptr);

        // ── Tail 更新器: 批量通知对端 rank "我发了多少 token" ──
        //   st.release.sys 较慢 (~100 cycle), 所以每 kNumScaleupUpdateInterval 个 token 才更新一次
        //   ⚠️ stored_num_tokens_sent[i]: 本 lane 向第 (i*32+lane_idx) 个 scaleup rank 发送的 token 累计
        //   ⚠️ stored_old_num_tokens_sent[i]: 上次通知时的值 (跳过无变化的更新, 节省带宽)
        int update_counter = 0;
        int stored_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        int stored_old_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);

        // ⚠️ update_tails: 批量更新 scaleup tail
        //   当 finish=true 或计数器达到 kNumScaleupUpdateInterval 时触发
        //   使用 st_release_sys 写入对端 rank 的 tail 指针 (NVLink 可见, release 语义)
        const auto update_tails = [&](const bool& finish = false) {
            ++ update_counter;
            if (finish or update_counter == kNumScaleupUpdateInterval) {
                // Wait all TMA stores to finish
                ptx::tma_store_wait();
                __syncwarp();

                // Issue
                #pragma unroll
                for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                    if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                        // NOTES: save some traffic with `stored_old_num_tokens_sent`
                        // Also, we cannot rewrite a finished slot, if the peer is going to clean it
                        if (stored_num_tokens_sent[i] != stored_old_num_tokens_sent[i])
                            ptx::st_release_sys(gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j), stored_num_tokens_sent[i]);
                        stored_old_num_tokens_sent[i] = stored_num_tokens_sent[i];
                    }
                }
                update_counter = 0;
            }
            __syncwarp();
        };

        // ── 遍历 channel_linked_list ──
        // channel_linked_list 形状: [kNumChannels, kNumScaleoutRanks*kNumMaxTokensPerChannel+1, kNumScaleupRanks]
        //   由 dispatch 阶段构建, 每个 (channel, scaleup_rank) 对应一条链表
        //   链表节点值 = token 全局序号, 终止符 = 负数
        //
        // 遍历逻辑:
        //   stored_ll_idx[i]: 当前遍历到第 (i*32+lane_idx) 个 scaleup rank 的链表第几个节点
        //   stored_token_idx[i]: 当前节点对应的 token 全局序号
        //
        //  ┌───────────────────────────────────────────────┐
        //  │ channel_linked_list 布局 (三维数组)           │
        //  │                                               │
        //  │ [channel_idx][ll_idx][scaleup_rank_j]         │
        //  │                                               │
        //  │ 例: channel=0, rank=0 的链表:                 │
        //  │   [0][0][0] → token_5                        │
        //  │   [0][1][0] → token_12                       │
        //  │   [0][2][0] → -1 (终止)                      │
        //  │                                               │
        //  │ 例: channel=0, rank=1 的链表:                 │
        //  │   [0][0][1] → token_3                        │
        //  │   [0][1][1] → -1 (终止)                      │
        //  └───────────────────────────────────────────────┘
        int dst_scaleup_rank_idx = channel_idx;
        int stored_ll_idx[kNumScaleupRanksPerLane] = {}, stored_token_idx[kNumScaleupRanksPerLane] = {};
        // stored_ll_idx[..] = 0, stored_token_idx[..] = -1
        #pragma unroll
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
            stored_token_idx[i] = -1;

        while (true) {
            // ── Step 1: 从链表加载 token_idx ──
            //   每个 lane 负责读取自己对应的 scaleup rank 的链表当前节点
            // 1️⃣ 以warp的视角，分析一个token
            //        1. 这个token属于 channel channel_idx,分channel存储，因此scaleup ranks间不会相互影响
            //        2. 这个token是scaleup rank j 转发过来的
            //        3. 这个token是scaleup rank j 发给我的第 stored_ll_idx[i] 个 token, 第一次遍历 stored_ll_idx[i] = 0，都是第0个token
            //        4. 拿到的值：这个token在转发给我时在我的scaleup buffer中的顺序（不是位置，因为跳过空的slot，例如rank0有10个slot，只要3个有token）
            //        4. 总结而言，就是获取每个scaleup rank j 发给我的 第 stored_ll_idx[i] 个 token 的 位置
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                const auto j = i * 32 + lane_idx;
                stored_token_idx[i] = i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks ?
                    __ldg(channel_linked_list +
                          // channel_idx
                          channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumScaleupRanks +
                          // stored_ll_idx[i] ⚠️ 一开始都是0
                          stored_ll_idx[i] * kNumScaleupRanks 
                          // j
                          + j) : -1;
            }
            __syncwarp();

            // ── Step 2: 检查所有 rank 是否都遍历完了 ──
            //   如果所有 lane 的 token_idx 都是负数, 说明链表全部结束
            // 2️⃣ 拿到的值都是负数，说明是-1，链表的最后一个值token后面会跟placeholder一个-1用于标记结束
            bool exited = true;
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                exited &= ptx::all(stored_token_idx[i] < 0);
            if (exited)
                break;

            // ── Step 3: Round-robin 处理活跃 rank ──
            //   wip_mask: bitmap, bit=1 表示该 scaleup rank 还有 token 需要处理
            //   round-robin 选择下一个 dst_scaleup_rank_idx, 避免饥饿
            // 3️⃣ 现在stored_token_idx 记录了每个scaleup rank 发给我的  各一个token 的 在我这的顺序
            //     那么这个顺序就是唯一的，可以帮我定位到时哪些token,然后把token返还回去
            //     ❶ 如果是no expand模式，那么直接通过顺序就能定位到哪个token
            //     ❷ 如果是expand模式，那么需要recv_metadata来定位到哪个token，顺序--> recv_metadata --> expand后的recv_x
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up ranks for 64-bit mask");
            using mask_t = std::conditional_t<(kNumScaleupRanks <= 32), uint32_t, uint64_t>;
            mask_t wip_mask = 0;
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                wip_mask |= static_cast<mask_t>(ptx::gather(stored_token_idx[j] >= 0)) << (j * 32);

            while (wip_mask) {
                // ── 4️⃣: Round-robin scaleup_rank_idx 一个一个scaleup_rank来
                //   不是逐token，是逐scaleup_rank，是因为上面就是按照 scaleup rank有没有发token来筛选出来的
                //   而且expand模式多个token返还给同一个scaleup_rank
                //   从 dst_scaleup_rank_idx+1 开始找, 找不到则从头找 (ffs = find first set)
                const auto start = (dst_scaleup_rank_idx + 1) % kNumScaleupRanks;
                const auto hi_mask = (wip_mask >> start) << start;
                dst_scaleup_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);
                wip_mask ^= static_cast<mask_t>(1) << dst_scaleup_rank_idx;

                // ── 5️⃣: 从持有 token_idx 的 lane 拿到token_idx值即这个token在我这的顺序，然后广播到所有 lane
                //        这个顺序token_idx，也就是对recv_metadata的索引，也是no expand模式对recv_x的索引
                int token_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = dst_scaleup_rank_idx - j * 32;
                    token_idx = src_lane_idx == lane_idx ? stored_token_idx[j] : token_idx;
                }
                token_idx = ptx::exchange(token_idx, dst_scaleup_rank_idx % 32);


                // ── 6️⃣: 读取源 metadata, 定位 token 在 scaleup_buffer 中的位置 ──  
                //   src_metadata 布局: [src_global_token_idx, slot_info, topk_0, topk_1, ...]
                //   stride = 2 + kNumTopk
                constexpr int kMetadataStride = 2 + kNumTopk;
                // ⚠️src_global_token_idx = rank_idx * kNumMaxTokensPerRank + src_token_idx
                // 可以提取到rank_idx：这个token来自于哪个rank_idx(虽然是dst_scaleup_rank_idx转发给我的)，但是不代表是他给我的
                // 可以提取到src_token_idx： 这个token在src_rank_idx中的顺序 
                // 可以提取到src_scaleout_rank_idx = rank_idx/kNumScaleupRanks
                const auto src_global_token_idx = __ldg(src_metadata + token_idx * kMetadataStride + 0);
                const auto src_token_idx = src_global_token_idx % kNumMaxTokensPerRank;
                const auto src_scaleout_rank_idx = src_global_token_idx / (kNumMaxTokensPerRank * kNumScaleupRanks);
                auto token_buffer = [&]() {
                    // ⚠️ direct： src_metadata[token_idx][1] = current_rank_idx * kNumTopk + master_src_topk_idx
                    // ⚠️ hybrid： src_metadata[token_idx][1] = (i-cur_rank_start)*kNumTopk + master_src_topk_idx
                    if constexpr (kUseScaleupRankLayout) { 
                        // ⚠️ 看 current_rank_idx 或者 (i-cur_rank_start)
                        const auto src_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) / kNumTopk;
                        return scaleup_buffer.get_token_buffer(src_slot_idx);
                    } else { 
                        // ⚠️ 看 master_src_topk_idx
                        const auto master_topk_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) % kNumTopk;
                        return scaleup_buffer
                            .get_rank_buffer(master_topk_idx)
                            .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                    }
                }();
                token_buffer.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(token_buffer.get_base_ptr(), dst_scaleup_rank_idx));
                // ⚠️ 将 token_buffer 的基地址转换为对端 rank 的对称地址
                //   TMA store 需要写到的目标是对端 rank 的 gmem, 使用 NVLink 对称地址


                
                EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");

                // ── 7️⃣: 读取展开模式的 top-k slot 索引 ，expand 模式才不为-1，
                //         recv_metadata --> 多个stored_topk_slot_idx是这个token分开计算后的在recv_x的位置
                // Read source indices for expand mode
                int stored_topk_slot_idx = -1;
                if constexpr (kUseExpandedLayout) { 
                    if (lane_idx < kNumTopk)
                        stored_topk_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + (2 + lane_idx));
                    __syncwarp();
                }

                // ────────────── 三种 Reduce 模式 ──────────────
                //
                //  ┌──────────────────────────────────────────────────────────────────┐
                //  │ Mode 1: no_local_reduce (不展开 / 仅1个top-k在本rank)          │
                //  │   最简单: TMA load x[token] → smem → TMA store → 对端 scaleup   │
                //  │                                                                │
                //  │ Mode 2: local_reduce (展开 + 多top-k在本rank)                   │
                //  │   combine_reduce(): 从多个 x[topk_slot] 加权求和 → smem         │
                //  │   → TMA store → 对端 scaleup                                   │
                //  │                                                                │
                //  │ Mode 3: expanded_send (展开 + 不允许多次reduce)                 │
                //  │   逐 top-k: TMA load x[slot_k] → smem → TMA store → rank_k    │
                //  │   每个 top-k 分别写入不同 rank 的 buffer                        │
                //  └──────────────────────────────────────────────────────────────────┘
                // ⚠️ 
                auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
                // ⚠️ no_local_reduce 判断:
                //   不展开模式: 直接 no_local_reduce=true
                //   展开模式: 仅当只有1个 top-k 在本 rank 时才跳过 reduce
                // 📌 正常训练(DeepSeek-V3): kUseExpandedLayout=true, kAllowMultipleReduction=true
                //   → 只有1个 top-k 在本 rank → no_local_reduce=true → 走 Mode 1 (最常见)
                //   → 多个 top-k 在本 rank → no_local_reduce=false → 走 Mode 2
                //   → Mode 3 在 kAllowMultipleReduction=false 时才进入, 正常训练不会走
                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
                if (no_local_reduce) {
                    // ── Mode 1: 直接 load+store, 无本地 reduce ──
                    //   展开: token_idx_in_tensor = topk_slot_idx (从 master lane 广播)
                    //   不展开: token_idx_in_tensor = token_idx
                    // 📌 正常训练最常走的路径: 大多数 token 只有1个 top-k 落在本 rank, 无需 reduce
                    int token_idx_in_tensor = token_idx;
                    if constexpr (kUseExpandedLayout)
                        token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

                    // TMA load: x[token_idx_in_tensor] → smem (tma_buffer)
                    if (ptx::elect_one_sync()) {
                        const auto load_ptr = math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                        ptx::tma_store_wait();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    }
                    __syncwarp();

                } else if constexpr (kAllowMultipleReduction) {

                    // ── Mode 2: 本地 reduce (展开 + 多 top-k 在本 rank) ──
                    // 📌 正常训练: 本 token 有多个 top-k 落在同一 rank 时走此路径, 需要加权求和
                    //   1. compute_topk_slots: 将有效的 top-k slot 排序到数组前部
                    //   2. combine_reduce: 从 x[topk_slot_k] 加权求和 → smem
                    
                    // ⚠️ 把 reduce_valid_mask 中的有效 lane 的 stored_topk_slot_idx值 收集到 topk_slot_idx 数组中
                    // 📌 compute_topk_slots (定义在 combine_utils.cuh):
                    //   输入: mask=reduce_valid_mask, fetch_func=exchange(stored_topk_slot_idx, lane_idx)
                    //   逻辑: 逐个提取 mask 中最低位的1 → 得到 lane 编号 → exchange 获取该 lane 的 slot → 存入数组 → 清除该 bit
                    //   举例: reduce_valid_mask=0b10110 (lane1,2,4有效)
                    //     k=0: lowest=1, fetch lane1的slot → topk_slot_idx[0], mask=0b10100
                    //     k=1: lowest=2, fetch lane2的slot → topk_slot_idx[1], mask=0b10000
                    //     k=2: lowest=4, fetch lane4的slot → topk_slot_idx[2], mask=0b00000
                    //   本质: 把 bitmap 展开成有序数组, 供后续 combine_reduce 逐个 load 并加权求和
                    int topk_slot_idx[kNumTopk];
                    compute_topk_slots(
                        topk_slot_idx, reduce_valid_mask,
                        [=](const int& idx) {
                            return ptx::exchange(stored_topk_slot_idx, idx);
                        }
                    );

                    // Reduce 结果写入 smem (tma_buffer)
                    // 📌 combine_reduce (定义在 combine_utils.cuh): 多 top-k 加权求和
                    //   模板参数:
                    //     kHiddenVec   = hidden维度的向量数 (kNumHiddenBytes / sizeof(vec_t))
                    //     kUnrollFactor= 循环展开因子 (由 get_max_unroll_factor 计算, 最大4)
                    //     kNumExpectedTopk = ceil(kNumTopk / kNumRanks), 预期每个rank平均收到的top-k数
                    //   函数参数:
                    //     lane_idx              : 当前 lane 编号, 用于计算 smem 写入偏移
                    //     topk_slot_idx         : 有效 top-k 的 slot 索引数组 (compute_topk_slots 的输出)
                    //     dst_buffer_ptr        : smem 目标地址 (tma_buffer), reduce 结果写入此处
                    //     get_src_buffer_ptr_func: 回调, 给定 slot_idx → 返回 x 中对应 token 的指针
                    //       此处实现: advance_ptr(x, slot_idx * kNumHiddenBytes)
                    //     wait_buffer_func      : 回调, 等待 smem buffer 释放 (tma_store_wait + syncwarp)
                    //   核心逻辑:
                    //     对每个 hidden 维度的 chunk:
                    //       1. 遍历 topk_slot_idx, 从 x[slot_k] load 每个 top-k 的 hidden state
                    //       2. 累加到 reduced[] (FP32 精度), 可选加 bias
                    //       3. FP32→BF16 转换, 写入 dst_buffer_ptr (smem)
                    //     优化: 若只有 ≤2 个 top-k 且无 bias, 走 hadd_bypass 路径, 直接用 BF16 加法避免 FP32 转换
                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
                    combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumRanks)>(
                        lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                        /* Get source base */ [=](const int& slot_idx) {
                            return math::advance_ptr<combine_vec_t>(
                                x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                        },
                        /* Wait buffer release */ [=]() {
                            ptx::tma_store_wait();
                            __syncwarp();
                        }
                    );
                    ptx::tma_store_fence();
                    __syncwarp();
                    
                } else {

                    // ── Mode 3: 展开发送 (展开 + 不允许多次 reduce) ──
                    //   每个 top-k 分别: TMA load x[slot_k] → smem → TMA store → rank_k 的 buffer
                    // 📌 正常训练不会走此路径: 仅在 kAllowMultipleReduction=false 时进入
                    //   每个 top-k 对应不同的目标 rank
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        int topk_slot_idx = ptx::exchange(stored_topk_slot_idx, k);
                        if (topk_slot_idx < 0)
                            continue;

                        if (ptx::elect_one_sync()) {
                            // Load
                            const auto load_ptr = math::advance_ptr(x, static_cast<int64_t>(kDoExpandedSend ? topk_slot_idx : token_idx) * kNumHiddenBytes);
                            ptx::tma_store_wait();
                            ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                            // NOTES: We don't need to care about `topk_weights` since we are in expand mode

                            // Store
                            const auto dst_token_buffer = scaleup_buffer
                                .get_rank_buffer(k)
                                .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                            ptx::tma_store_1d(
                                gin.get_sym_ptr<ncclTeamTagLsa>(dst_token_buffer.get_base_ptr(), dst_scaleup_rank_idx),
                                tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                            ptx::tma_store_commit();
                        }
                        __syncwarp();
                    }
                }


                // ── 写入 top-k 权重 (仅非展开模式) ──
                //   展开: 不需要权重 (每个 top-k 独立一行)
                //   非展开: 需要将权重随数据一起发送, 供对端 reduce
                if (not kUseExpandedLayout and topk_weights != nullptr and lane_idx < kNumTopk) {
                    const float value = __ldg(topk_weights + (token_idx * kNumTopk + lane_idx));
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = value;
                    ptx::tma_store_fence();
                }
                __syncwarp();

                
                // ── 发起 TMA store: smem → 对端 rank 的 scaleup buffer ──
                //   Mode 3 (expanded_send) 已经在上面逐 top-k 发出了
                //   Mode 1/2: 这里统一发起一次 TMA store
                if (not kDoExpandedSend and ptx::elect_one_sync()) {
                    // Wait TMA arrival (only for non-reduced cases)
                    if (no_local_reduce) {
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    }

                    // Issue stores
                    ptx::tma_store_1d(
                        token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                        token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                // ⚠️ 更新本 lane 发送到 dst_scaleup_rank_idx 的 token 计数
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                __syncwarp();
            }

            // ⚠️ 每轮链表遍历结束后, 批量更新 tail
            update_tails();

            // ┚─ 推进链表指针: 仅当该 rank 的 token 有效时才前进一步 ──
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                stored_ll_idx[i] += (stored_token_idx[i] >= 0);
        }

        // ⚠️ 链表遍历结束, 强制更新所有未发出的 tail (finish=true)
        update_tails(true);




    } else {





        // ═══════════════════════════════════════════════════════════════════
        //  Forward Warps: 重放 dispatch 元数据, 等待 scaleup 数据就绪,
        //                 从 scaleup_buffer 读取 → reduce → RDMA 转发到跨节点 rank
        // ═══════════════════════════════════════════════════════════════════
        //
        //  整体流程:
        //  ┌──────────────────────────────────────────────────────────────────┐
        //  │ for each token in token_metadata_at_forward:                    │
        //  │   1. 读取 src_token_global_idx, src_scaleup_rank_idx 等        │
        //  │   2. 等待相关 scaleup rank 的 tail 就绪 (数据到位)             │
        //  │   3. 根据模式:                                                  │
        //  │      a) kAllowMultipleReduction=false:                          │
        //  │         逐 top-k TMA load → TMA store → RDMA put               │
        //  │      b) kAllowMultipleReduction=true:                           │
        //  │         combine_reduce() 本地聚合 → TMA store → 延迟 RDMA put  │
        //  │   4. 最后: 清理 scaleup tail + scaleout 同步                    │
        //  └──────────────────────────────────────────────────────────────────┘
        //

        // ⚠️ 一个warp负责一个token
        const auto forward_warp_idx = warp_idx - kNumScaleupWarps;
        const auto channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;

        // 分配更多寄存器 (forward warp 需要 reduce, 寄存器需求大)
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_alloc<kNumRegistersForForwardWarps>();

        // ⚠️ 切到本 channel 对应的 scaleout send buffer 子区
        scaleout_send_buffer = scaleout_send_buffer.get_channel_buffer<kNumScaleoutRanks * kNumMaxTokensPerChannel>(channel_idx);

        // ⚠️token_metadata_at_forward 布局:
        //   [kNumChannels, kNumScaleoutRanks*kNumMaxTokensPerChannel+1, kNumForwardMetadataDims]
        //   kNumForwardMetadataDims = 2 + kNumTopk*2
        //     [0]: src_token_global_idx
        //     [1]: is_token_last_in_chunk
        //     [2..2+kNumTopk-1]: src_scaleup_rank_idx[k] (每个 top-k 的来源 scaleup rank)
        //     [2+kNumTopk..2+2*kNumTopk-1]: src_slot_idx[k] (每个 top-k 在 scaleup buffer 的 slot)
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // ── 延迟 RDMA 发射器: 重叠 TMA store 和 RDMA put ──
        //   目的: TMA store 完成后再发起 RDMA, 但延迟一个 token
        //   这样当前 token 的 TMA store 和上一个 token 的 RDMA 可以并行
        //
        //   流水线示意:
        //   token 0: [TMA store]
        //   token 1:            [TMA store]  +  [RDMA put token 0]   ← 并行
        //   token 2:                        [TMA store]  +  [RDMA put token 1]
        //
        //   flush_last_tma_and_issue_rdma 逻辑:
        //     1. tma_store_wait(): 等待上一个 TMA store (smem → scaleout send buffer) 完成
        //     2. 若非本 rank (last_src_scaleout_rank_idx != scaleout_rank_idx): 发起 RDMA put
        //     3. 若是本 rank: 不需要 RDMA, 数据已在本地
        //
        //   调用时机:
        //     a) combine_reduce 内部: reduce 第一次写入前调用, 确保上一 token TMA 完成 + RDMA 发射
        //     b) 链表遍历结束后: 强制发射最后一个 token 的 RDMA (延迟队列中还剩一个未发射)
        int last_src_scaleout_rank_idx = -1;
        int last_is_token_last_in_chunk = 0;
        void* last_recv_token_buffer_ptr = nullptr;
        void* last_send_token_buffer_ptr = nullptr;
        const auto flush_last_tma_and_issue_rdma = [&]() {
            if (last_src_scaleout_rank_idx >= 0 and ptx::elect_one_sync()) {
                ptx::tma_store_wait();

                // Issue only if not local rank
                if (last_src_scaleout_rank_idx != scaleout_rank_idx) {
                    gin.put<ncclTeamTagRail>(
                        last_recv_token_buffer_ptr,
                        last_send_token_buffer_ptr,
                        token_layout.get_num_bytes<false>(),
                        last_src_scaleout_rank_idx,
                        last_is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests
                    );
                }
            }
            __syncwarp();
        };

        // ── 重放 dispatch: 遍历 token_metadata_at_forward ──
        // ⚠️ 监控所有scaleup rank的进度，等待接收
        //    stored_cached_scaleup_tail: 当前scaleup rank通知的进度
        //    stored_num_tokens_recv: 当前处理的进度
        int stored_num_tokens_recv[kNumScaleupRanksPerLane] = {}, stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};
        // ⚠️ 遍历这个channel的所有token
        for (int i = 0; ; ++ i) {
            // ⚠️ token的源信息
            const auto src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
            const auto is_token_last_in_chunk = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 1);
            const auto src_rank_idx = src_token_global_idx / kNumMaxTokensPerRank;
            const auto src_scaleout_rank_idx = src_rank_idx / kNumScaleupRanks;
            const auto src_token_idx = src_token_global_idx % kNumMaxTokensPerRank;

            // ⚠️token的Forward分发信息：目的rank，目的rank的buffer的slot位置
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + lane_idx) : -1;
            auto stored_src_slot_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + kNumTopk + lane_idx) : -1;
            if (src_token_global_idx < 0)
                break;

            // ⚠️ 需要等待分发的目的scaleup rank都返还了该toekn，才能进行reduce和发送
            // ── 构建 scaleup rank mask: 哪些 scaleup rank 需要等待 ──
            //   reduce_or 聚合所有 lane 的 scaleup_rank_idx → bitmap
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up peers");
            using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
            const auto scaleup_mask = ptx::reduce_or(
                stored_src_scaleup_rank_idx >= 0 ?
                (mask_t(1) << stored_src_scaleup_rank_idx) : mask_t(0));
            // ⚠️ bool向量记录是哪些scaleup rank
            bool stored_is_scaleup_rank_needed[kNumScaleupRanksPerLane];
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_is_scaleup_rank_needed[j] = (scaleup_mask >> (j * 32 + lane_idx)) & 1;

            // ⚠️ 需要等待分发的目的scaleup rank都返还了该toekn，才能进行reduce和发送
            // ── 等待 scaleup tail 就绪: 确保数据已写入 scaleup_buffer ──
            //   stored_num_tokens_recv[j] < stored_cached_scaleup_tail[j]:
            //     已收到的 token 数 < scaleup warp 通知的 tail 位置
            //   使用 ld_acquire_sys 读取 tail (保证看到 scaleup warp 的 st_release_sys 写入)
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                bool arrived = true;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    arrived &= not stored_is_scaleup_rank_needed[j] or stored_num_tokens_recv[j] < stored_cached_scaleup_tail[j];
                
                // ⚠️ 全 1 代表：要等待的scaleup rank已经更新tail，都通知我可以去消费了
                if (ptx::all(arrived))
                    return true;

                // Reload cached
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto k = j * 32 + lane_idx;
                    stored_cached_scaleup_tail[j] = j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks ?
                        ptx::ld_acquire_sys(workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k)) : -1;
                }

                // Timeout
                if (is_last_check) {
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                        printf("DeepEP combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, lane: %d, recv: %d, tail: %d (wait=%d)\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                               channel_idx, j * 32 + lane_idx,
                               stored_num_tokens_recv[j],
                               stored_cached_scaleup_tail[j],
                               stored_is_scaleup_rank_needed[j]);
                    }
                }
                return false;
            });

            // ⚠️ 更新 当前处理的进度  + 1，每个等待的scaleup rank的值都+1
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_num_tokens_recv[j] += static_cast<int>(stored_is_scaleup_rank_needed[j]);
            

            if constexpr (not kAllowMultipleReduction) {
                // ────────────── Forward Warp 两种路径 ──────────────
                //
                //  ┌───────────────────────────────────────────────────────────────────┐
                //  │ Path A: kAllowMultipleReduction=false                            │
                //  │   逐 top-k: TMA load scaleup_buf → TMA store scaleout_buf       │
                //  │   → RDMA put 到跨节点 rank (非本地)                              │
                //  │   不做本地 reduce, 每个 top-k 独立发送                             │
                //  │                                                                 │
                //  │ Path B: kAllowMultipleReduction=true                             │
                //  │   combine_reduce() 从 scaleup_buf 聚合多个 top-k → smem         │
                //  │   → TMA store scaleout_buf → 延迟 RDMA put                      │
                //  │   本地 reduce 后只发一份, 节省 RDMA 带宽                          │
                //  └───────────────────────────────────────────────────────────────────┘
                //
                // ── Path A: 逐 top-k 转发, 无本地 reduce ──
                const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                auto topk_valid_mask = kUseExpandedLayout ?
                    ptx::gather(stored_src_scaleup_rank_idx >= 0) :
                    ptx::gather(ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);  // Deduplicate w.r.t. scaleup rank index if expanded mode is disabled
                if (ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        if ((topk_valid_mask & (1u << k)) == 0u)
                            continue;

                        // ⚠️ 逐 top-k: TMA load from scaleup_buf → smem
                        ptx::tma_load_1d(
                            tma_buffer.get_base_ptr(), scaleup_buffer.get_rank_buffer(k).get_token_buffer(src_slot_idx).get_base_ptr(),
                            mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);


                        // ⚠️ TMA store from smem → scaleout_buf (recv or send)
                        //   本地 rank: 直接写到 recv_buffer (无需 RDMA)
                        //   跨节点 rank: 写到 send_buffer (后续 RDMA put)
                        // ⚠️ （token_layout, kNumTokensInScaleoutLayout, kNumMaxTokensPerRank）
                        const auto recv_buffer_ptr = scaleout_recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx).get_base_ptr();
                        // ⚠️ （kAllowMultipleReduction ? 1 : kNumTopk, kNumChannels * (kNumScaleoutRanks * kNumMaxTokensPerChannel)）
                        const auto send_buffer_ptr = src_scaleout_rank_idx == scaleout_rank_idx ?
                            recv_buffer_ptr : scaleout_send_buffer.get_rank_buffer(k).get_token_buffer(i).get_base_ptr();
                        ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                        ptx::tma_store_commit();
                        ptx::tma_store_wait();

                        // ⚠️ RDMA put: 仅当目标不是本地 rank 时才发
                        //   AggregateRequests: 聚合多个 RDMA 请求 (最后一个 token 时 flush)
                        topk_valid_mask ^= 1u << k;
                        if (src_scaleout_rank_idx != scaleout_rank_idx) {
                            gin.put<ncclTeamTagRail>(
                                recv_buffer_ptr,
                                send_buffer_ptr,
                                token_layout.get_num_bytes<false>(),
                                src_scaleout_rank_idx,
                                topk_valid_mask == 0 and is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests
                            );
                        }
                    }
                }
                __syncwarp();
            
            
            } else {
                    // ────────────── Forward Warp 两种路径 ──────────────
                //
                //  ┌───────────────────────────────────────────────────────────────────┐
                //  │                                                                   │
                //  │ Path B: kAllowMultipleReduction=true                              │
                //  │   combine_reduce() 从 scaleup_buf 聚合多个 top-k → smem            │
                //  │   → TMA store scaleout_buf → 延迟 RDMA put                        │
                //  │   本地 reduce 后只发一份, 节省 RDMA 带宽                            │
                //  └───────────────────────────────────────────────────────────────────┘
                //
                // ── Path B: 本地 reduce + RDMA 转发 ──
                //   去重: 多个 top-k 可能来自同一 scaleup rank, 只需读一次
                //   reduce: combine_reduce() 聚合多个 top-k 的 hidden states
                auto reduce_valid_mask = ptx::gather(
                    ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);

                // ⚠️ 计算源 buffer 索引: top-k 数据在 scaleup_buffer 中的位置
                int stored_src_buffer_idx = 0;
                if constexpr (kUseScaleupRankLayout) {
                    stored_src_buffer_idx =
                        stored_src_scaleup_rank_idx * scaleup_buffer.num_max_tokens_per_rank + stored_src_slot_idx;
                } else {
                    const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                    stored_src_buffer_idx = stored_src_slot_idx == -1 ? -1 :
                        lane_idx * scaleup_buffer.num_max_tokens_per_rank + src_slot_idx;
                }
                
                // ⚠️ 收集每个stored_src_buffer_idx的信息到topk_slot_idx ===> 因为需要整个warp一起运数据
                int topk_slot_idx[kNumTokensInScaleupLayout];
                compute_topk_slots(
                    topk_slot_idx, reduce_valid_mask,
                    [=](const int& idx) {
                        return ptx::exchange(stored_src_buffer_idx, idx);
                    }
                );

                
                //
                // get_max_unroll_factor<kHiddenVec, kAdjustRegisters ? 8 : 4>():
                //   找最大的展开因子 i, 使得 kHiddenVec % (32 * i) == 0
                //   kAdjustRegisters=true  → 上限4, 节省寄存器 (forward warp 需要更多)
                //   kAdjustRegisters=false → 上限8, 最大展开以提升吞吐
                //   例: kHiddenVec=448, 上限8 → 448%(32*7=224)=0 → 返回7; 上限4 → 448%(32*4=128)=0 → 返回4
                //
                // combine_reduce 模板参数:
                //   kHiddenVec        : hidden 维度的向量数 (kHidden * sizeof(bf16) / sizeof(vec_t))
                //   kUnrollFactor     : 循环展开因子, 每个 lane 一次迭代处理 kUnrollFactor 个向量
                //   kNumExpectedTopk  : ceil(kNumTopk / kNumScaleoutRanks), 优化分支
                //
                // combine_reduce 入参:
                //   lane_idx          : 当前 lane 索引
                //   topk_slot_idx     : 每个 top-k 对应的 buffer 槽位索引 (-1=无效)
                //   dst_buffer_ptr    : 目标 buffer (TMA buffer, smem)
                //   get_src_buffer_ptr_func: slot_idx → scaleup_buffer 中对应 token 的指针
                //   wait_buffer_func : 等待 buffer 就绪 (flush_last_tma_and_issue_rdma)
                //
                // 核心逻辑分两条路径:
                //   enable_hadd_bypass (≤2 个有效 top-k 且无 bias): 直接 BF16 加法, 无需转 float, 更快
                //   常规路径: 先转 float 累加 (精度更好), 再转回 BF16 写入 dst
                // ⚠️ combine_reduce: 从 scaleup_buffer 多个 top-k 位置加权求和 → smem
                constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, kAdjustRegisters ? 8 : 4>();
                combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumScaleoutRanks)>(
                    lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                    /* Get source base */ [=](const int& slot_idx) {
                        // global=true: slot_idx 是跨 rank 全局索引, 绕过 num_ranks==1 断言
                        return static_cast<combine_vec_t*>(scaleup_buffer.get_token_buffer(slot_idx, true).get_base_ptr());
                    },
                    /* Wait buffer release */ [=]() {
                        flush_last_tma_and_issue_rdma();
                    }
                );

                // 1. ptx::match(stored_src_scaleup_rank_idx) — 返回一个 bitmap，标记哪些 lane 的 stored_src_scaleup_rank_idx 与当前 lane 值相同
                // 2. ptx::get_master_lane_idx(...) — 从 bitmap 中取最高位索引（31 - __clz(mask)），即相同 rank 的 lane 中编号最大的那个作为 "master lane"
                // 3. ptx::exchange(stored_src_buffer_idx, master_lane_idx) — 从 master lane 广播 stored_src_buffer_idx 到所有同 rank 的 lane
                // ⚠️效果：同一个 scaleup rank 的多个 lane，统一采用 master lane 的 stored_src_buffer_idx，确保后续访问 scaleup buffer 时同一 rank 的所有 lane 读同一个 slot。
                // ⚠️为什么需要：一个 token 可能有多个 top-k 来自同一个 scaleup rank，但 reduce 时这些 top-k 已被合并，只需要读一个 slot。master lane 的 stored_src_buffer_idx 是 combine_reduce 产出的统一 slot 索引，其他同 rank 的 lane 必须与之对齐。
                // 并非多次一举，是因为Hybrid dispatch中对于非master lane保留的stored_src_slot_idx=-1
                stored_src_buffer_idx = ptx::exchange(
                    stored_src_buffer_idx, ptx::get_master_lane_idx(ptx::match(stored_src_scaleup_rank_idx)));
                if (not kUseExpandedLayout and stored_src_scaleup_rank_idx >= 0) {
                    tma_buffer.get_topk_weights_ptr()[lane_idx] =
                        scaleup_buffer.get_token_buffer(stored_src_buffer_idx, true)
                                    .get_topk_weights_ptr()[lane_idx];
                }
                ptx::tma_store_fence();
                __syncwarp(); // Necessary to let the leader lane see the writes


                // ⚠️ 确定 send/recv buffer: reduce 结果写入哪里?
                //   kUseScaleoutRankLayout: 按 scaleout rank 索引选 buffer
                //   否则: 按 master top-k 索引选 buffer
                //   本地 rank: send = recv (无需 RDMA, 直接写到接收区)
                int scaleout_recv_buffer_rank_idx;
                if constexpr (kUseScaleoutRankLayout) {
                    scaleout_recv_buffer_rank_idx = scaleout_rank_idx;
                } else {
                    const int src_topk_idx = ptx::get_master_lane_idx(ptx::gather(stored_src_scaleup_rank_idx >= 0));
                    scaleout_recv_buffer_rank_idx = src_topk_idx;
                }
                // ⚠️ （token_layout, kNumTokensInScaleoutLayout, kNumMaxTokensPerRank）
                const auto recv_token_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_recv_buffer_rank_idx).get_token_buffer(src_token_idx);
                // ⚠️ （kAllowMultipleReduction ? 1 : kNumTopk, kNumChannels * (kNumScaleoutRanks * kNumMaxTokensPerChannel)）
                const auto send_token_buffer = src_scaleout_rank_idx == scaleout_rank_idx ?
                    recv_token_buffer :
                    scaleout_send_buffer.get_token_buffer(i);

                // ⚠️ TMA store: smem → scaleout send/recv buffer
                if (ptx::elect_one_sync()) {
                    ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                                    token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                __syncwarp();

                // ⚠️ 记录 RDMA 信息: 下一个 token 处理时再发射 (延迟一个 token, 重叠计算与通信)
                last_src_scaleout_rank_idx = src_scaleout_rank_idx;
                last_is_token_last_in_chunk = is_token_last_in_chunk;
                last_recv_token_buffer_ptr = recv_token_buffer.get_base_ptr();
                last_send_token_buffer_ptr = send_token_buffer.get_base_ptr();
            }
        }

        // ⚠️ 发射最后一个 token 的 RDMA (延迟队列中还有未发射的)
        if constexpr (kAllowMultipleReduction)
            flush_last_tma_and_issue_rdma();

        // ── 清理 scaleup tail: 归零, 供下次 dispatch/combine 使用 ──
        #pragma unroll
        for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
            const auto k = j * 32 + lane_idx;
            if (j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks)
                *workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k) = 0;
        }
        __syncwarp();

        // ── Scale-out 同步: 通知所有跨节点 rank "本 channel 处理完毕" ──
        //   1. red_add_rel: 原子加 pack2(1, 0) 到对端 rank 的 signaled_tail
        //      → 通知对端: 本 channel 的 forward warp 已完成
        //   2. 等待所有 scaleout rank 的 signal 到达
        //      → 确保所有跨节点 rank 都处理完毕
        //   3. 清零 signaled_tail, 供下次使用
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid ranks");
        if (lane_idx < kNumScaleoutRanks) {
            // Update remote tails
            const auto expected_signal = math::pack2<int, int64_t>(1, 0);
            gin.red_add_rel<ncclTeamTagRail>(
                workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx),
                expected_signal, lane_idx);

            // Wait tail arrival
            const auto wait_ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx);
            comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal = ptx::ld_acquire_sys<int64_t>(wait_ptr);
                if (signal == expected_signal) {
                    // Clean for next usages
                    *wait_ptr = 0;
                    return true;
                }

                if (is_last_check) {
                    printf("DeepEP combine (scale-out wait all) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                           "channel: %d, lane: %d, signal: %lld, expected: %lld\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                           channel_idx, lane_idx,
                           signal, expected_signal);
                }
                return false;
            });
        }
        __syncwarp();
    }

    // ⚠️ 无需 epilogue barrier: 各 warp 独立完成, 数据一致性已由 tail/signal 机制保证
}

}  // namespace deep_ep::elastic
