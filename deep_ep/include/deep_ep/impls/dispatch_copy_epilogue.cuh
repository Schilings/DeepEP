#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

/**
 * ==================== Dispatch Copy Epilogue 算子 ====================
 *
 *   作用: dispatch 主 kernel 把 token 数据通过 NVLink/RDMA 发送到各 rank 的 recv_buffer 后,
 *         本 kernel 负责把 recv_buffer 中的数据**搬出来**到最终的输出张量
 *         (recv_x, recv_sf, recv_topk_idx, recv_topk_weights, recv_src_metadata)
 *
 *   为什么需要这个 epilogue?
 *     dispatch 主 kernel 直接写入 recv_buffer (对称内存布局, 按 rank×slot 组织),
 *     但用户需要的是按 expert 或按 token 顺序排列的连续张量,
 *     所以需要一次 "搬出 + 重排" 的后处理
 *
 *   两种模式:
 *     kDoExpand=true  (前向 dispatch): 一个 token 可能发给多个 expert → 需要展开 (expand)
 *       例: token 0 发给 expert 3 和 expert 5 → 输出张量中占两行
 *       dst_tensor_idx 用 atomicAdd(psum_num_recv_tokens_per_expert) 分配
 *
 *     kDoExpand=false (反向 combine): 一个 token 只需一行输出 → 不展开
 *       dst_tensor_idx = i (直接用接收序号)
 *
 *   数据流:
 *     recv_buffer (按 rank×slot 组织)
 *       → TMA load 到 smem (tma_buffer)
 *       → 从 smem 分发到多个输出张量:
 *           recv_x           : hidden states (TMA store)
 *           recv_sf          : scale factors (普通 store, 非 TMA, 因为 stride 不连续)
 *           recv_topk_idx    : top-k expert 索引 (普通 store)
 *           recv_topk_weights: top-k 权重 (普通 store)
 *           recv_src_metadata: 源 token 元数据 (普通 store)
 *
 *   与 hybrid_dispatch 的关系:
 *     本 kernel 是纯 scale-up (节点内) 的 epilogue,
 *     hybrid_dispatch 有自己的 epilogue 处理 scale-out (跨节点) 的部分
 *
 *   模板参数:
 *     kDoExpand           - true=前向展开模式, false=反向不展开
 *     kCachedMode         - true=缓存模式, 跳过链表创建
 *     kNumSMs             - 使用的 SM 数量
 *     kNumChannels        - scale-out 通道数 (仅跨节点场景)
 *     kNumWarps           - 每 SM 的 warp 数
 *     kNumScaleoutRanks   - 跨节点 rank 数 (1=纯节点内)
 *     kNumScaleupRanks    - 节点内 rank 数
 *     kNumHiddenBytes     - hidden 维度字节数
 *     kNumSFPacks         - scale factor pack 数
 *     kNumMaxTokensPerRank- 每 rank 最大 token 数
 *     kNumExperts         - 总 expert 数
 *     kNumTopk            - top-k 选择数
 *     kDoCreateLinkedList - 自动推导: 跨节点+非缓存时创建链表 (用于后续 RDMA 通道清理)
 */
template <bool kDoExpand,                  //【核心开关】true = expand 模式, 一行一个 (token, expert) 副本
          bool kCachedMode,                // 是否是缓存 dispatch(略过链表构建)
          // NOTES: this channel concept only applies for scale-out ranks
          int kNumSMs,                     // 参与 kernel 的 SM 数
          int kNumChannels,                // 每 rank 的通信通道数(仅 scaleout 生效)
          int kNumWarps,                   // 每 block 的 warp 数
          int kNumScaleoutRanks,           // scaleout 域 rank 数(通常走 RDMA)
          int kNumScaleupRanks,            // scaleup 域 rank 数(通常走 NVLink)
          int kNumHiddenBytes,             // 单 token hidden 维度的字节数(已含 dtype)
          int kNumSFPacks,                 // FP8 scale factor pack 数；0 表示无 SF
          int kNumMaxTokensPerRank,        // 每 rank 最大 token 容量(worst case 的占位)
          int kNumExperts,                 // 全局 expert 总数
          int kNumTopk,                    // 每 token 选几个 expert
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumThreads = kNumWarps * 32,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          bool kDoCreateLinkedList = (kNumScaleoutRanks > 1 and not kCachedMode)>
          //                         ↑ 仅 hybrid 非 cached 模式需要构建 per-channel 链表
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_copy_epilogue_impl(
    void* buffer,                                   // 【输入】对称通信缓冲, 主 kernel 填充；本 kernel 只读
    void* workspace,                                // 【输入/输出】per-rank 元数据工作区(链表 tail 指针等)
    int*  psum_num_recv_tokens_per_scaleup_rank,    // 【输入】inclusive psum, 用于定位每段 scaleup rank 边界
    int*  psum_num_recv_tokens_per_expert,          // 【输入/输出】exclusive+align psum
                                                    //    expand 模式下本 kernel 会 atomicAdd 在此计数
    void* recv_x,                                   // 【输出】用户可见 token 张量 [num_recv|num_expanded, hidden]
    sf_pack_t*   recv_sf,                           // 【输出】FP8 scale factor(strided 布局)
    topk_idx_t*  recv_topk_idx,                     // 【输出-非 Expand 专用】[num_recv, topk] 命中 local expert id
    float*       recv_topk_weights,                 // 【输出】非 Expand: [num_recv, topk]；Expand: [num_expanded]
    int*         recv_src_metadata,                 // 【输出】[num_recv, 2+topk], 供 combine 反向路由
    int*         channel_linked_list,               // 【输出】hybrid 非 cached: per-channel 链表体
    int          num_recv_tokens,                   // 【输入】CPU-sync 下是精确值；否则为 worst-case 哨兵
    constint    recv_sf_token_stride,               // 【输入】recv_sf 在 token 维的 stride(packs 数)
    constint    recv_sf_hidden_stride,              // 【输入】recv_sf 在 hidden 维的 stride
    constint    scaleout_rank_idx,                  // 【输入】本 rank 的 scaleout 编号
    constint    scaleup_rank_idx) {                 // 【输入】本 rank 的 scaleup 编号

    // ==================== 基础索引 ====================
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();

    // global_warp_idx: 跨所有 SM 的全局 warp 编号, 用于交错遍历 token
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;

    // ==================== Expert 索引映射 ====================
    // expert_idx → rank_idx: dst_expert_idx / kNumExpertsPerRank
    // rank_idx → expert 范围: [expert_start_idx, expert_end_idx)
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;
    const auto expert_start_idx = kNumExpertsPerRank * rank_idx, expert_end_idx = kNumExpertsPerRank * (rank_idx + 1);

    // ==================== Buffer 布局 ====================
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    // token_layout: 描述单个 token 的内存布局 (hidden + sf + metadata + mbarrier)
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);

    // tma_buffer: smem 中每个 warp 的 TMA 中转区, 一次搬一个 token 的全部数据
    //   kWithMBarrier=true → 包含 mbarrier (TMA load 同步用)
    //   kNumWarps 个 "rank", 每个 1 个 token → 每个 warp 独立工作
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);

    // scaleup_buffer: recv_buffer (gmem), 按 scaleup rank 组织
    //   kNumScaleupRanks 个 rank, 每个最多 kNumScaleoutRanks * kNumMaxTokensPerRank 个 token
    //   注意: kNumScaleoutRanks 个跨节点 rank 各自向同一 scaleup rank 发送,
    //         所以每 rank 的容量要乘 kNumScaleoutRanks
    const auto scaleup_buffer = layout::BufferLayout<false>(token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);

    // ==================== TMA 初始化 ====================
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // ==================== 等待 dispatch 主 kernel 完成 ====================
    // PDL (Programmatic Dependent Launch): 依赖 dispatch kernel 完成后才开始执行
    // 此时 recv_buffer 中数据已全部可见
    // 注意: 使用 PDL 时不能用 __ldg (只读缓存可能看到旧数据)
    cudaGridDependencySynchronize();

    // ==================== 获取实际接收 token 数 ====================
    // 如果 num_recv_tokens == kNumMaxTokensPerRank * kNumRanks (占位最大值),
    // 说明 Python 层没传实际值, 需要从 GPU 前缀和数组中读取
    // psum_num_recv_tokens_per_scaleup_rank[kNumScaleupRanks-1] = 所有 rank 的总接收数
    if (num_recv_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[kNumScaleupRanks - 1];

    // ==================== 主循环: 遍历所有接收到的 token ====================
    // current_rank_idx: 当前 token 所属的 scaleup rank
    // stored_psum_num_recv_tokens: 通过 warp shuffle 广播的前缀和值
    // current_rank_start/end: 当前 rank 在全局 token 序列中的范围 [start, end)
    //   用于将全局序号 i 映射回 rank 内的 slot 索引
    int current_rank_idx = -1, stored_psum_num_recv_tokens;
    int current_rank_start = 0, current_rank_end = 0;
    #pragma unroll
    for (int i = global_warp_idx; i < num_recv_tokens; i += kNumWarps * kNumSMs) {

        // ────────────── Step 1: 定位 token 在 recv_buffer 中的位置 ──────────────
        // psum_num_recv_tokens_per_scaleup_rank 存储前缀和:
        //   [rank0_count, rank0+rank1_count, rank0+rank1+rank2_count, ...]
        // 全局序号 i 落在哪个 rank 区间? 用 while 逐 rank 推进
        //
        // 例: psum=[3, 5, 9], i=6
        //   rank 0: 区间 [0, 3),  rank 1: 区间 [3, 5),  rank 2: 区间 [5, 9)
        //   6>=0→推进 rank0, 6>=3→推进 rank1, 6>=5→推进 rank2, 6<9→停!
        //   结果: rank=2, slot=i-current_rank_start=6-5=1
        //
        // 优化: warp 协同加载前缀和 (32 个 rank 一批)
        //   stored_lane_idx = current_rank_idx % 32
        //   当 stored_lane_idx==0 时: 32 个 lane 同时从 gmem 批量加载 psum[0..31]
        //     (coalesced, 1 次内存事务, 每个 lane 暂存到自己的寄存器)
        //   当 stored_lane_idx!=0 时: 跳过 gmem 加载 (值已在寄存器中),
        //     用 exchange (shfl) 从对应 lane 广播即可
        //   效果: 32 个 rank 只需 1 次 gmem 访问 + 31 次 shfl (寄存器级, 零开销)
        //
        // 例: kNumScaleupRanks=3
        //   current_rank_idx=0: stored_lane_idx=0 → 批量加载 psum[0..2]
        //     lane0=psum[0], lane1=psum[1], lane2=psum[2]
        //     exchange(s, 0) → 所有 lane 得到 lane0 的值 (psum[0])
        //   current_rank_idx=1: stored_lane_idx=1 → 跳过加载
        //     exchange(s, 1) → 所有 lane 得到 lane1 的值 (psum[1])
        //   current_rank_idx=2: stored_lane_idx=2 → 跳过加载
        //     exchange(s, 2) → 所有 lane 得到 lane2 的值 (psum[2])
        while (i >= current_rank_end) {
            current_rank_idx += 1;
            EP_DEVICE_ASSERT(current_rank_idx < kNumScaleupRanks);
            const auto stored_lane_idx = current_rank_idx % 32;
            // stored_lane_idx==0 时批量加载: 32 lane 并行读 gmem (coalesced)
            if (stored_lane_idx == 0 and current_rank_idx + lane_idx < kNumScaleupRanks)
                stored_psum_num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[current_rank_idx + lane_idx];
            current_rank_start = current_rank_end;
            // exchange: 从 stored_lane_idx 号 lane 广播给所有 lane (shfl, 寄存器级)
            current_rank_end = ptx::exchange(stored_psum_num_recv_tokens, stored_lane_idx);
        }
        // i - current_rank_start: rank 内的 slot 索引
        const auto buffer_token = scaleup_buffer.get_rank_buffer(current_rank_idx).get_token_buffer(i - current_rank_start);

        
        // ────────────── Step 2: TMA Load 从 recv_buffer 搬到 smem ──────────────
        ptx::tma_store_wait();  // 确保上一轮 TMA store 已完成 (smem 可覆写)
        __syncwarp();

        // 发起 TMA load: gmem → smem (异步)
        // 一次加载整个 token 的所有数据 (hidden + sf + topk metadata)
        if (ptx::elect_one_sync()) {
            ptx::tma_load_1d(tma_buffer.get_base_ptr(), buffer_token.get_base_ptr(),
                             mbarrier_ptr, tma_buffer.get_num_bytes<false>());
            // arrive_and_set_tx: 注册预期字节数, 配合 TMA load 的自动通知机制
            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, tma_buffer.get_num_bytes<false>());
        }
        __syncwarp();

        // ────────────── Step 3: 加载 top-k expert 索引 (直接从 gmem, 不等 TMA) ──────────────
        // 为什么不等 TMA load 完成? 因为 topk_idx 在 gmem 中可以直接用普通 load 读取,
        // 与 TMA load 并行, 隐藏 TMA 延迟
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        int dst_expert_idx = -1;
        if (lane_idx < kNumTopk)
            dst_expert_idx = buffer_token.get_topk_idx_ptr()[lane_idx];
        __syncwarp();

        // ────────────── Step 4: 验证 expert 索引 + 转换为本地索引 ──────────────
        // in_range: 每个 lane 的 top-k 选择是否路由到本 rank
        //   (expert_start_idx <= dst_expert_idx < expert_end_idx)
        // gather(in_range): 收集 warp 内结果 → bitmap, bit=1 表示该 lane 的 expert 属于本 rank
        // get_master_lane_idx: 取 bitmap 中最高位 (= 31 - __clz(mask))
        //
        // master_src_topk_idx: 本 token 路由到本 rank 的那些 lane 中, 编号最大的那个
        //   含义: 一个 token 可能被多个 top-k 选择路由到同一 rank, 但该 rank 只需一个"代表"
        //   选择最高位而非最低位无特殊原因, 关键是确定性地选一个
        //
        //   后续用途:
        //   1) 链表构建: tma_buffer.get_linked_list_idx_ptr()[master_src_topk_idx] 取该 lane 的链表索引
        //   2) metadata 编码: src_peer_info = rank_idx * kNumTopk + master_src_topk_idx
        //      combine 反向时需要知道从哪个 top-k 位置读权重, 所以必须记录具体 lane 编号
        //
        //   例: token T 的 top-k: lane0→expert3(rank0), lane1→expert7(rank1), lane2→expert9(rank1)
        //       本 rank=rank1: gather(in_range)=0b110, master=2 (最高位)
        const auto in_range = expert_start_idx <= dst_expert_idx and dst_expert_idx < expert_end_idx;
        // 这些lane拿到的dst_expert_idx不可能是有重复的，除了-1，master_lane是抽出一个代表
        const auto master_src_topk_idx = ptx::get_master_lane_idx(ptx::gather(in_range));
        // 转换为本地 expert 索引 (减去 rank 起始偏移)， 不在范围的设为 -1
        dst_expert_idx = in_range ? dst_expert_idx - expert_start_idx : -1;
        // 不变量验证: top-k 选出的 expert 互不相同, 转换为本地索引后也应唯一 (除 -1 外)
        // 这些lane拿到的dst_expert_idx不可能是有重复的，除了-1
        EP_DEVICE_ASSERT(ptx::deduplicate(dst_expert_idx, lane_idx) or dst_expert_idx == -1);
        // 非展开模式: 直接写入 recv_topk_idx (本地 expert 索引)
        // 展开模式：不用到 recv_topk_idx
        if (not kDoExpand and lane_idx < kNumTopk)
            recv_topk_idx[i * kNumTopk + lane_idx] = static_cast<topk_idx_t>(dst_expert_idx);
        __syncwarp();

        // ────────────── Step 5: 计算输出张量中的目标位置 dst_tensor_idx ──────────────
        // 两种模式的区别:
        //   kDoExpand=false (反向): dst_tensor_idx = i (一个 token 占一行)
        //     elect_one_sync: 只需一个 lane 写 i, 避免 warp 内重复
        //   kDoExpand=true  (前向): atomicAdd(psum_num_recv_tokens_per_expert[expert])
        //     同一 expert 的多个 token 需要不同行 → 原子递增分配
        //     返回值是该 expert 已有的 token 数 (即分配到的行号)
        int dst_tensor_idx = -1;
        if (not kDoExpand and ptx::elect_one_sync()) {
            dst_tensor_idx = i;
        } else if (kDoExpand and dst_expert_idx >= 0) {
            // ⚠️ align后的真实前缀和，直接在这里加1，那改完就不是前缀和了。
            // ⚠️ expert prefix sum[dst_expert_idx]顺序累加上去没问题，应该算是复用psum_num_recv_tokens_per_expert，后面应该用不到这个了
            dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);
        }
        __syncwarp();

        // ────────────── Step 6: 等待 TMA Load 完成 ──────────────
        // 此时 TMA load 大概率已完成 (因为中间做了 topk_idx 处理和 dst_tensor_idx 计算)
        if (ptx::elect_one_sync())
            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
        __syncwarp();

        
        
        // ────────────── Step 7: 维护 channel 链表 (仅跨节点+非缓存模式) ──────────────
        // 背景: 跨节点 RDMA 场景下, token 按 channel 传输但到达顺序是乱序的
        //   后续需要按逻辑顺序清理 channel, 因此用单向链表把同一 channel 上的 token 串起来
        //
        // channel_linked_list: 全局 gmem 数组, 每个 (channel, scaleup_rank) 对应一条链表
        //   链表节点值 = token 全局序号, 终止符 = -1
        //
        // tma_buffer hidden后 布局 (详见 layout.cuh):
        //   [num_topk]          : topk_idx (expert 索引) 
        //   [num_topk]          : topk_weights
        //  tma_buffer metadata 布局（with_metadata 时的额外部分，共 1 + num_topk 个 int）：
        //   [1]                  : src_token_global_idx
        //   [num_topk]           : linked_list_idx ← get_linked_list_idx_ptr() 指向这里
        //
        //   主 dispatch kernel 在发送阶段已将 linked_list_idx 写入 smem,
        //   值为 "前驱 token 在 channel_linked_list 数组中的 slot 位置"
        //
        // 链接操作 (前驱插入式):
        //   channel_linked_list[前驱slot] = i
        //   即: 让前驱节点的 next 指向当前 token → 把当前 token 链入链表
        //
        // 例: channel_linked_list 数组
        //   slot_0 = token_5, slot_1 = token_3, slot_2 = -1(终止)
        //   token_3.metadata.linked_list_idx = slot_0 (前驱在 slot_0)
        //   token_5.metadata.linked_list_idx = slot_1 (前驱在 slot_1)
        //   遍历链表: slot_0 → token_5, slot_1 → token_3, slot_2 → -1(终止)
        //
        // master_src_topk_idx: 同一 token 的多个 top-k 选择中, 代表本 rank 的那个 lane
        //   (一个 token 在一个 rank 上只需链入一次)
        if constexpr (kDoCreateLinkedList) {
            if (ptx::elect_one_sync())
                channel_linked_list[tma_buffer.get_linked_list_idx_ptr()[master_src_topk_idx]] = i;
            __syncwarp();
        }

        // ────────────── Step 8: TMA Store hidden states → recv_x ──────────────
        // 展开/不展开模式的选择:
        //   kDoExpand=true:  有有效 dst_tensor_idx 的 lane 都执行 (多个 lane 可能写不同行)
        //   kDoExpand=false: 只需一个 lane 执行 (一个 token 写一行)
        if (kDoExpand ? (dst_tensor_idx >= 0) : ptx::elect_one_sync()) {
            ptx::tma_store_1d(math::advance_ptr(recv_x, static_cast<int64_t>(dst_tensor_idx) * kNumHiddenBytes),
                              tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
            ptx::tma_store_commit();
        }
        __syncwarp();


        // ────────────── Step 9: Store Scale Factors → recv_sf ──────────────
        // 为什么 SF 不用 TMA store? 因为 recv_sf 的 stride 不连续:
        //   recv_sf[token_stride * token_idx + hidden_stride * pack_idx]
        //   TMA 要求连续 1D 布局, 所以只能用普通 store + stride 计算
        //
        // 数据布局: kNumSFPacks 个 pack, 32 个 lane 分摊
        //   lane_idx 负责的 pack: lane_idx, lane_idx+32, lane_idx+64, ...
        //   例: kNumSFPacks=100, lane_idx=3 → 负责 pack 3, 35, 67, 99
        //
        // 写入策略 (while-mask 循环):
        //   expand 模式下, warp 内多个 lane 各持有不同的 dst_tensor_idx (不同 expert 的不同行)
        //   但 SF 的写入需要 32 个 lane 协作 (每个 lane 写不同的 pack)
        //   → 必须串行化目标行, 并行化每行的 pack 写入
        //   → while-mask: 逐个处理有效 lane, 每轮 32 lane 一起帮一个 lane 写完
        //
        //   例: lane 1 和 lane 5 有有效 dst_tensor_idx:
        //     mask = 0b00100010
        //     第1轮: ff s→1, exchange 从 lane1 拿 dst_tensor_idx, 32 lane 协作写 lane1 的 SF
        //     第2轮: ffs→5, exchange 从 lane5 拿 dst_tensor_idx, 32 lane 协作写 lane5 的 SF
        //     mask=0, 结束
        //
        //   非 expand 模式: mask=1, 只有 lane 0 (elect_one 的结果) 需要, 循环1次
        if constexpr (kNumSFPacks > 0) {
            constexpr auto kNumFullIters = kNumSFPacks / 32;
            const bool do_last_iter = (kNumSFPacks % 32 != 0) and (kNumFullIters * 32 + lane_idx < kNumSFPacks);
            EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");

            // 从 smem 加载 SF 到寄存器 (coalesced: 每 lane 读不同 pack, 连续访存)
            const auto smem_src_ptr = tma_buffer.get_sf_ptr();
            sf_pack_t reg_src[kNumFullIters + 1];
            #pragma unroll
            for (int k = 0; k < kNumFullIters; ++ k)
                reg_src[k] = smem_src_ptr[k * 32 + lane_idx];
            if (do_last_iter)
                reg_src[kNumFullIters] = smem_src_ptr[kNumFullIters * 32 + lane_idx];

            // 准备 stride (转 int64 避免溢出)
            const auto recv_sf_token_stride_i64 = static_cast<int64_t>(recv_sf_token_stride);
            const auto recv_sf_hidden_stride_i64 = static_cast<int64_t>(recv_sf_hidden_stride);

            // gather: 收集 warp 内哪些 lane 有有效 dst_tensor_idx → bitmap
            // expand 模式: 多个 lane 可能有效; 非 expand: 只有1个 (mask=1)
            auto mask = kDoExpand ? ptx::gather(dst_tensor_idx >= 0) : 1;
            while (mask) {
                // ffs: 找 bitmap 中最低有效位 → 当前要处理的目标 lane
                const int valid_lane_idx = __ffs(mask) - 1;

                // exchange: 所有 32 个 lane 从 valid_lane_idx 获取其 dst_tensor_idx
                //   → 整个 warp 统一知道了当前要写入哪一行
                // advance_ptr: 计算 gmem 目标行起始地址 = recv_sf + dst_tensor_idx * token_stride
                const auto gmem_dst = math::advance_ptr<sf_pack_t>(recv_sf,
                    ptx::exchange(dst_tensor_idx, valid_lane_idx) * (recv_sf_token_stride_i64 * sizeof(sf_pack_t)));

                // 32 个 lane 并行写入各自负责的 SF pack (strided 布局)
                //   gmem_dst[pack_idx * hidden_stride] = reg_src[pack_idx]
                //   pack_idx = k * 32 + lane_idx (每个 lane 写不同的 pack)
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k)
                    gmem_dst[(k * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[k];
                if (do_last_iter)
                    gmem_dst[(kNumFullIters * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[kNumFullIters];

                // 清除已处理的 bit, 移到下一个有效 lane
                mask ^= 1 << valid_lane_idx;
            }
        }

        // ────────────── Step 10: Store top-k weights → recv_topk_weights ──────────────
        // 展开/不展开模式的区别:
        //   kDoExpand=true:  每个 token 按其 dst_tensor_idx 写一行 (权重与 token 一一对应)
        //   kDoExpand=false: 按 top-k 选择写 (i * kNumTopk + lane_idx)
        //     反向 combine 时 weights 可选 (recv_topk_weights 可能为 nullptr)
        if (kDoExpand and recv_topk_weights != nullptr and dst_tensor_idx >= 0) {
            recv_topk_weights[dst_tensor_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        } else if (not kDoExpand and recv_topk_weights != nullptr and lane_idx < kNumTopk) {
            // For backward, weights are optional
            recv_topk_weights[i * kNumTopk + lane_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        }
        __syncwarp();



        // ────────────── Step 11: Store source metadata → recv_src_metadata ──────────────
        // metadata 布局 (每 token 占 kMetadataStride=2+kNumTopk 个 int):
        //   [0]: src_token_global_idx  - 源 token 在发送方的全局编号
        //   [1]: src_peer_info         - 源信息编码 (见下方)
        //   [2..2+kNumTopk-1]: dst_tensor_idx[0..kNumTopk-1] - 展开/归约用
        //
        // src_peer_info 编码:
        //   非混合 (kNumScaleoutRanks==1): current_rank_idx * kNumTopk + master_src_topk_idx
        //     = 源 rank 编号 × topk + 代表 lane 编号
        //   混合 (kNumScaleoutRanks>1): (i - current_rank_start) * kNumTopk + master_src_topk_idx
        //     = rank 内 slot 编号 × topk + 代表 lane 编号
        constexpr int kMetadataStride = 2 + kNumTopk;
        if (ptx::elect_one_sync()) {
            recv_src_metadata[i * kMetadataStride + 0] = *tma_buffer.get_src_token_global_idx_ptr();
            if constexpr (kNumScaleoutRanks == 1) {
                recv_src_metadata[i * kMetadataStride + 1] = current_rank_idx * kNumTopk + master_src_topk_idx;
            } else {
                // 混合模式: rank 内 slot 编号 × topk + 代表 lane 编号
                recv_src_metadata[i * kMetadataStride + 1] = (i - current_rank_start) * kNumTopk + master_src_topk_idx;
            }
        }
        __syncwarp();

        // 展开/归约源索引: 每个 top-k 选择对应的输出行号
        // ⚠️ 用于 combine 阶段的加权归约: 知道每个 expert 结果写回哪一行
        if (kDoExpand and lane_idx < kNumTopk)
            recv_src_metadata[i * kMetadataStride + 2 + lane_idx] = dst_tensor_idx;
        __syncwarp();
    }

    // ==================== 链表尾部标记 ====================
    // 仅跨节点+非缓存模式需要: 为每个 channel 的每个 rank 写入链表终止符 (-1)
    // 同时清理 tail 指针 (归零, 供下次 dispatch 使用)
    //
    // channel_linked_list 结构:
    //   每个 channel + scaleup rank 对应一个链表, 记录该通道上接收到的 token 序号
    //   链表最后一个节点应指向 -1 (终止符)
    //   workspace_layout.get_channel_scaleup_tail_ptr(channel, rank) 指向链表尾节点位置
    //   把 tail 指向的位置写入 -1 → 链表终止
    //   然后把 tail 指针归零 → 清理状态
    if constexpr (kDoCreateLinkedList) {
        constexpr int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32);
        const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
        for (int i = global_warp_idx; i < kNumChannels; i += kNumSMs * kNumWarps) {
            // 每个 lane 负责若干 rank (kNumScaleupRanksPerLane)
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                if (const auto k = j * 32 + lane_idx; j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks) {
                    // 在链表尾节点位置写入 -1 (终止符)
                    channel_linked_list[
                        *workspace_layout.get_channel_scaleup_tail_ptr(i, k)
                    ] = -1;

                    // 清理 tail 指针归零 (供下次 dispatch 使用)
                    *workspace_layout.get_channel_scaleup_tail_ptr(i, k) = 0;
                }
            }
            __syncwarp();
        }
    }
}

}  // namespace deep_ep::elastic
