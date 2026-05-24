#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/layout.cuh>

#include <deep_ep/impls/combine_utils.cuh>


namespace deep_ep::elastic {

// ============================================================================================
//  combine_reduce_epilogue_impl —— Combine 最终 Reduce Epilogue 内核
// ============================================================================================
//
//  【调用时机】
//  在主 combine kernel (combine_impl / hybrid_combine_impl) 完成后作为第二阶段调用。
//  主 combine kernel 负责跨 rank 通信 + 局部 reduce，数据写入 reduce_buffer。
//  本 kernel 从 reduce_buffer 读取，做最终的 top-k 加权求和 + 可选 bias，写回 combined_x。
//
//  【两阶段流水线】
//
//  ┌──────────────────────┐        ┌──────────────────────────────────────────┐
//  │ 主 combine kernel    │        │ combine_reduce_epilogue_impl             │
//  │ (阶段1)               │  之后  │ (阶段2)                                  │
//  │                      │ ────→ │                                          │
//  │ 跨 rank 通信          │        │ 从 reduce_buffer 读取多个 rank 数据       │
//  │ 局部 reduce           │        │ top-k 加权求和 (combine_reduce)          │
//  │ → 写入 reduce_buffer  │        │ 可选 SwiGLU bias                        │
//  │                      │        │ TMA store → combined_x                   │
//  │                      │        │ 写回 topk_weights                        │
//  └──────────────────────┘        └──────────────────────────────────────────┘
//
//  【数据流】
//
//  reduce_buffer (来自阶段1)
//  ┌──────────────────────────────────────────────┐
//  │ rank 0: token_0 | token_1 | ...               │
//  │ rank 1: token_0 | token_1 | ...               │  ← comm_buffer (BufferLayout)
//  │ ...                                          │
//  └──────────────────────────────────────────────┘
//         │
//         │ combine_reduce: 按 topk_slot_idx 读取多个 rank 的 hidden
//         │ 加权求和 (BF16 add 或 float accumulate)
//         ↓
//  smem (tma_buffer, 每个 warp 一个 token slot)
//  ┌──────────────────────────┐
//  │ reduced hidden (1 token) │
//  └──────────────────────────┘
//         │
//         │ TMA store (smem → gmem)
//         ↓
//  combined_x [num_combined_tokens, hidden]       ← 最终输出
//
// ============================================================================================
//
//  【模板参数】
//
// 📌kUseExpandedLayout      : 前向是否展开 (每个 top-k 占独立行), 影响 reduce 路径
//    true  → 每个展开 token 独立一行, 无 topk_weights
//    false → 多 top-k 共享一行, 需要加权求和
//
// 📌kAllowMultipleReduction : 是否允许多次 reduce
//    true  → combine 阶段 token 数与 dispatch 完全一致, 可能在 scaleup + scaleout 各做一次 reduce
//    false → 只做一次 reduce (精度最好, 但可能增加数据传输量)
//
// kNumSMs                : 使用的 SM 数量
//  kNumWarps              : 每个 block 的 warp 数量
//
//  kNumScaleoutRanks       : 跨节点 rank 数 (RDMA 通信的 rank)
//  kNumScaleupRanks        : 同节点 rank 数 (NVLink 通信的 rank)
//
//  kHidden                : hidden 维度大小
//  kNumMaxTokensPerRank   : 每 rank 最大 token 数
//  kNumExperts            : 总 expert 数
//  kNumTopk               : top-k 选择数
//
//  kNumThreads            : = kNumWarps * 32
//  kNumHiddenBytes        : = kHidden * sizeof(nv_bfloat16)
//
//  kNumRanks              : 实际参与 reduce 的 rank 数
//    kNumScaleoutRanks == 1 (非混合模式) → kNumScaleupRanks
//    kNumScaleoutRanks != 1 (混合模式)   → kNumScaleupRanks (只用同节点 rank)
//
// 📌kUseRankLayout          : 是否用 "按 rank" 布局 (而非 "按 topk" 布局)
//    use_rank_layout() 返回值: not kAllowMultipleReduction → false
//    kAllowMultipleReduction 且 kNumRanks ≤ kNumTopk → true (按 rank 更紧凑)
//
// 📌kNumTokensInLayout      : 布局中每组的 token 槽位数
//    kUseRankLayout ? kNumRanks : kNumTopk
//
//  【入参】
//
//  📌combined_x              : [num_combined_tokens, hidden] 输出: reduce 后的 hidden states
//  combined_topk_weights   : [num_combined_tokens, num_topk] 输出: 每个 top-k 的权重
//  combined_topk_idx       : [num_combined_tokens, num_topk] 输入: 每个 token 的 top-k expert 索引
//  📌recv_buffer             : 来自阶段1的 reduce_buffer (按 rank×token 组织)
//  bias_0, bias_1          : 可选 SwiGLU bias
//  num_combined_tokens     : 要 combine 的 token 总数
//  scaleout_rank_idx      : 本 rank 在 scaleout 维度的索引
//  scaleup_rank_idx       : 本 rank 在 scaleup 维度的索引
//
template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps,
          // TODO: merge these two variables into one (ensure the whole file does not contain "scaleup")
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          int kNumRanks = kNumScaleoutRanks == 1 ? kNumScaleupRanks : kNumScaleupRanks,
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>()>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_reduce_epilogue_impl(nv_bfloat16* combined_x,
                             float* combined_topk_weights,
                             topk_idx_t* combined_topk_idx,
                             void* recv_buffer,
                             void* bias_0, void* bias_1,
                             const int num_combined_tokens,
                             const int scaleout_rank_idx, const int scaleup_rank_idx) {
    // ── Expert 索引计算常量 ──
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    constexpr int kNumExpertsPerRank = kNumExperts / (kNumScaleupRanks * kNumScaleoutRanks);
    EP_STATIC_ASSERT(kNumExperts % (kNumScaleupRanks * kNumScaleoutRanks) == 0, "Invalid number of experts or ranks");

    // ── 线程索引 ──
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    // global_warp_idx: 交错分配 warp 到不同 SM, 确保最后一波负载均衡
    //   warp 0 在 SM 0, warp 1 在 SM 1, ..., warp 0 在下一个 SM 循环
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;

    // ── 缓冲区布局 ──
    // comm_buffer (recv_buffer): 按 rank × token 组织, 存放各 rank 发来的 hidden states
    //   ┌─────────────────────────────────────────────────┐
    //   │ slot 0 (rank/topk 0): token_0 | token_1 | ...   │
    //   │ slot 1 (rank/topk 1): token_0 | token_1 | ...   │
    //   │ ...                                             │
    //   └─────────────────────────────────────────────────┘
    //   kNumTokensInLayout = kUseRankLayout ? kNumRanks : kNumTopk
    //   每个槽位包含 hidden + topk_idx + topk_weights (无 SF, with_metadata=false)
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto comm_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    // 📌 (kNumTokensInLayout, kNumMaxTokensPerRank) 各scaleout rank返回来的数据？
    const auto comm_buffer = layout::BufferLayout<false>(
        comm_token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, recv_buffer);

    // output_buffer: 最终输出 combined_x [num_combined_tokens, hidden]
    //   只有 hidden, 无 topk_idx/weights (num_topk=0)
    const auto output_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);
    const auto output_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, combined_x);

    // tma_buffer: smem 中每个 warp 一个 token slot, 用于 TMA 中转
    //   ┌────────────┬────────────┬────────────┐
    //   │ warp 0     │ warp 1     │ warp 2     │ ...
    //   │ 1 token    │ 1 token    │ 1 token    │
    //   └────────────┴────────────┴────────────┘
    const auto tma_buffer = layout::BufferLayout<false>(output_token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);

    // bias 布局: 与 output 相同结构 [num_combined_tokens, hidden]
    const auto bias_0_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_0);
    const auto bias_1_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_1);

    // ── 等待主 combine kernel 完成 (PDL 机制) ──
    // cudaGridDependencySynchronize: 阻塞直到主 combine kernel 信号完成
    // 此时 reduce_buffer 数据已全部可见, 可以安全读取
    // 注意: 使用了 PDL (Programmatic Dependent Launch), 不可用 __ldg
    cudaGridDependencySynchronize();

    // ── 主循环: 每个 warp 交错处理多个 token ──
    // 步长 = kNumWarps * kNumSMs, 确保不同 warp/SM 处理不同 token
    for (int token_idx = global_warp_idx; token_idx < num_combined_tokens; token_idx += kNumWarps * kNumSMs) {

        // ══════════════════════════════════════════════════════════
        //  Step 1: 从 combined_topk_idx 读取每个 lane 的 expert 索引
        // ══════════════════════════════════════════════════════════
        //
        // ⚠️ combined_topk_idx[token_idx * kNumTopk + lane_idx] 存储了该 token 被
        // 哪个 expert 处理。每个 lane (< kNumTopk) 读取自己负责的 top-k 选项。
        //
        // stored_dst_expert_idx: 本 lane 负责的 expert 全局索引
        // stored_dst_rank_idx:   该 expert 所属的 rank 索引
        //   非混合模式 (kNumScaleoutRanks==1): expert_idx / kNumExpertsPerRank
        //   混合模式: expert_idx / kNumExpertsPerScaleout
        int stored_dst_rank_idx = -1, stored_dst_expert_idx = -1;
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        if (lane_idx < kNumTopk) {
            stored_dst_expert_idx = static_cast<int>(combined_topk_idx[token_idx * kNumTopk + lane_idx]);
            stored_dst_rank_idx = stored_dst_expert_idx >= 0 ?
                stored_dst_expert_idx / (kNumScaleoutRanks == 1 ? kNumExpertsPerRank : kNumExpertsPerScaleout) : -1;
        }
        __syncwarp();

        // ══════════════════════════════════════════════════════════
        //  Step 2: 去重 — 同一 rank 的多个 top-k 只需读一次
        // ══════════════════════════════════════════════════════════
        // 一个 token 可能被多个 top-k 选择路由到同一 rank, 但 reduce 时只需读该 rank 一次。
        // 不同模式下去重的粒度不同:
        //
        //   分支1: kUseExpandedLayout && !kAllowMultipleReduction:
        //     → 展开模式, 每个 top-k 独立行, 从未 reduce 过, 不需要去重
        //     should_deduplicate = false
        //
        //   分支2: kNumScaleoutRanks != 1 && !kUseExpandedLayout && !kAllowMultipleReduction:
        //     → 混合模式, 非 expanded, 非多次 reduce
        //     → 按 expert 所属的完整 rank 去重 (scaleup + scaleout 组合)
        //     deduplicate_key = expert_idx / kNumExpertsPerRank
        //
        //   分支3: 其他 (含 kAllowMultipleReduction=true, 即默认模式):
        //     → 按 dst_rank_idx 去重 (非混合模式按 rank, 混合模式按 scaleup rank)
        //     deduplicate_key = stored_dst_rank_idx
        //
        //   ⚠️ 默认模式 (allow_multiple_reduction=True): 走分支3, 按 stored_dst_rank_idx 去重
        const auto [should_deduplicate, deduplicate_key] = [&]() -> std::pair<bool, int> {
            if constexpr (kUseExpandedLayout and not kAllowMultipleReduction) {
                return {false, 0};
            } else if constexpr (kNumScaleoutRanks != 1 and not kUseExpandedLayout and not kAllowMultipleReduction) {
                return {true, stored_dst_expert_idx >= 0 ? stored_dst_expert_idx / kNumExpertsPerRank : -1};
            } else {
                return {true, stored_dst_rank_idx};
            }
        }();

        // reduce_valid_mask: 哪些 top-k 位置需要参与 reduce
        //   去重模式: 只有 master lane (同值中编号最大的) 参与且 rank_idx >= 0
        //   非去重模式: 所有 rank_idx >= 0 的 lane 参与
        auto reduce_valid_mask = should_deduplicate ?
            ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0) :
            ptx::gather(stored_dst_rank_idx >= 0);

        // ══════════════════════════════════════════════════════════
        //  Step 3: 计算每个有效 top-k 对应的 comm_buffer 槽位索引
        // ══════════════════════════════════════════════════════════
        // compute_topk_slots: 从 reduce_valid_mask 中提取有效 lane (按 bit 从低到高),
        //   为每个有效 lane 调用 fetch_func 获取值, 存入 topk_slot_idx[]
        //
        //   fetch_func(idx):
        //     kUseRankLayout=true  → exchange(stored_dst_rank_idx, idx):
        //       从 lane idx 广播其 stored_dst_rank_idx
        //       因为 rank layout 下, buffer 按 rank 分组, slot_idx 就是 rank_idx
        //     kUseRankLayout=false → 直接返回 idx (即 lane_idx = topk 序号):
        //       因为 topk layout 下, buffer 按 topk 分组, slot_idx 就是 topk 编号
        //
        //   后续用途: comm_buffer.get_rank_buffer(topk_slot_idx[k]) 定位到对应槽位
        int topk_slot_idx[kNumTokensInLayout];
        compute_topk_slots(
            topk_slot_idx, reduce_valid_mask,
            [=](const int& idx) {
                return kUseRankLayout ? ptx::exchange(stored_dst_rank_idx, idx) : idx;
            }
        );

        // ══════════════════════════════════════════════════════════
        //  Step 4: combine_reduce — 从多个 rank 读取 hidden 并加权求和
        // ══════════════════════════════════════════════════════════
        //
        // 向量类型选择: SM100+ 用 longlong4_t (32B), 否则 int4 (16B)
        // kHiddenVec: hidden 维度拆分为多少个向量
        // kUnrollFactor: 循环展开因子, 最大4 (此 kernel 无需调整寄存器)
        using combine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;
        constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);
        constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();

        // combine_reduce 参数:
        //   lane_idx          : 当前 lane
        //   topk_slot_idx     : 每个 top-k 对应的 buffer 槽位 (-1=无效)
        //   tma_buffer        : 目标 (smem, reduce 结果写入此处)
        //   get_src_buffer_ptr: slot_idx → comm_buffer 中对应 rank 的 token 指针
        //   wait_buffer_func  : 等待 TMA store 完成 
        //   bias_0/1          : 可选 SwiGLU bias
        combine_reduce<kHiddenVec, kUnrollFactor, kNumTokensInLayout>(
            lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
            /* Get source base */ [=](const int& slot_idx) {
                // slot_idx → comm_buffer 中对应 rank 分区 → 对应 token 的 base ptr
                return static_cast<combine_vec_t*>(
                    // 📌 确实已经按照scaleout rank划分好了
                    // 📌 但是好像统一 token_idx 了token在不同scaleout rank的buffer上的位置诶，那么说明空了蛮多空间？也不一定，说不定会都发给了所有scaleout rank
                    comm_buffer.get_rank_buffer(slot_idx).get_token_buffer(token_idx).get_base_ptr());
            },
            /* Wait buffer release */ [=]() {
                ptx::tma_store_wait();
                __syncwarp();
            },
            /* Bias 0 */ bias_0 == nullptr ?
                nullptr : static_cast<combine_vec_t*>(bias_0_buffer.get_token_buffer(token_idx).get_base_ptr()),
            /* Bias 1 */ bias_1 == nullptr ?
                nullptr : static_cast<combine_vec_t*>(bias_1_buffer.get_token_buffer(token_idx).get_base_ptr())
        );
        // TMA store fence: 确保 smem 写入对 TMA controller 可见
        ptx::tma_store_fence();
        __syncwarp();

        // ══════════════════════════════════════════════════════════
        //  Step 5: TMA store — smem → combined_x
        // ══════════════════════════════════════════════════════════
        //   elect_one_sync: 选一个 leader lane 发起 TMA
        //   tma_store_1d: 1D 异步拷贝, smem → gmem
        //   tma_store_commit: 提交 TMA 事务
        if (ptx::elect_one_sync()) {
            ptx::tma_store_1d(output_buffer.get_token_buffer(token_idx).get_base_ptr(),
                              tma_buffer.get_base_ptr(), kNumHiddenBytes);
            ptx::tma_store_commit();
        }
        __syncwarp();

        // ══════════════════════════════════════════════════════════
        //  Step 6: 写回 top-k 权重
        // ══════════════════════════════════════════════════════════
        // 非展开模式需要写回 topk_weights, 用于后续加权求和:
        //   combined_topk_weights[token_idx * kNumTopk + lane_idx] = weight
        //
        // 权重来源: comm_buffer 中对应 rank 的 token 的 topk_weights
        //   kUseRankLayout=true  → 按 stored_dst_rank_idx 选 rank 分区
        //   kUseRankLayout=false → 按 master_lane_idx 选 (同 rank 的 master lane 编号)
        if (combined_topk_weights != nullptr) {
            const auto master_lane_idx = ptx::get_master_lane_idx(ptx::match(stored_dst_rank_idx));
            if (lane_idx < kNumTopk) {
                float value = 0;
                if (stored_dst_rank_idx >= 0) {
                    const auto dst_ptr = comm_buffer
                        .get_rank_buffer(kUseRankLayout ? stored_dst_rank_idx : master_lane_idx)
                        .get_token_buffer(token_idx).get_topk_weights_ptr() + lane_idx;
                    value = *dst_ptr;
                }
                combined_topk_weights[token_idx * kNumTopk + lane_idx] = value;
            }
            __syncwarp();
        }
    }
}

}  // deep_ep::elastic
