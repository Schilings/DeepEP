#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic::layout {

/**
 * ==================== WorkspaceLayout ====================
 *
 * Workspace 是 GPU 端的临时工作区, 用于 All-to-All dispatch/combine 期间的
 * 跨 SM 聚合、跨节点通信、链表追踪等中间状态。
 *
 * 内存布局 (固定大小, 用 kNumMax* 常量分配):
 *
 *   offset  区域                                          大小
 *   ──────  ────────────────────────────────────────────  ──────────────────────────────
 *   0       NVLink barrier signal                         16 bytes (kNumBarrierSignalBytes)
 *           [0..7]  counter (uint64)
 *           [8..11] signal[0] (int)
 *           [12..15] signal[1] (int)
 *
 *   16      Notify reduction workspace                    (kNumMaxRanks + kNumMaxExperts) * 8 bytes
 *           所有 SM 的 notify warp 把 rank/expert 计数
 *           red_add 到这里, 高 32 位=到达 SM 数, 低 32 位=计数值
 *
 *   16 + N  Scaleup rank+expert count (send buffer)      (kNumMaxRanks + kNumMaxExperts) * 8 bytes
 *           仅 dispatch.cuh (非 NVLink 模式) 使用:
 *           本节点编码后的 rank/expert 计数, 供 RDMA put 发送
 *           hybrid_dispatch.cuh 不使用此区域 (put_value 直接写对端 recv)
 *           编码: (到达 scaleup rank 数 << 32) | 计数值
 *
 *   16+2N   Scaleup rank+expert count (recv buffer)      (kNumMaxRanks + kNumMaxExperts) * 8 bytes
 *           对端 scaleup rank 通过 NVLink put_value / red_add_rel 写入
 *           本节点读取后解码, 存入 smem 供 prefix sum 使用
 *
 *   16+4N   Scaleup atomic sender counter                 kNumMaxRanks * 4 bytes
 *           forward warp 用 atomicAdd 分配 scaleup_buffer slot
 *           每次 +1, 保证不同 channel 的 token 分到不同 slot
 *
 *   ...     Scaleout rank+expert count (send buffer)      (kNumMaxRanks + kNumMaxExperts) * 4 bytes
 *           编码后的 (encode_decode_positive) rank/expert 计数
 *           供 RDMA put 发送到其他节点
 *
 *   ...     Scaleout rank+expert count (recv buffer)      (kNumMaxRanks + kNumMaxExperts) * 4 bytes
 *           从其他节点 RDMA 收到的 rank/expert 计数
 *           recv_and_reduce 读取后解码求和
 *
 *   ...     Scaleout channel signaled tail                kNumMaxRanks * kNumMaxChannels * 8 bytes
 *           pack2(finish_flag, tail_count)
 *           scaleout warp 定期通知 forward warp: 本 channel 已写入多少 slot
 *           forward warp 轮询此值来知道有多少 token 可处理
 *
 *   ...     Channel scaleup tail                          kNumMaxRanks * kNumMaxChannels * 4 bytes
 *           linked list 的尾部位置
 *           forward warp 写入, epilogue 读取
 *
 *   ...     PP (pipeline parallelism) send/recv count     2 * 2 * 8 bytes
 *           前后 rank 的发送/接收计数
 *
 *   ...     AGRS (async GPU rendezvous signal) recv       (kNumMaxInflightAGRS + 1) * kNumMaxRanks * 4 bytes
 *           AGRS 接收信号 + 会话信号
 *
 *   总计: 对齐到 32 字节
 *
 * 其中 N = (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t)
 *
 * 关键设计:
 *   - 固定大小: 用 kNumMaxRanks=1024, kNumMaxExperts=2048 等常量分配
 *     运行时 num_ranks/num_experts 只影响访问偏移, 不影响分配大小
 *     好处: 同一块 buffer 可以在不同配置下复用, 无需重新分配
 *   - Send/Recv 双缓冲:
 *     kIsSendBuffer=true  → 写入端 (本节点写)
 *     kIsSendBuffer=false → 读取端 (对端写, 本节点读)
 *     通过 RDMA put / NVLink st_relaxed 实现跨节点/跨 rank 通信
 */
struct WorkspaceLayout {
    void* workspace;    // workspace 基地址 (GPU 端)

    int num_ranks;      // 总 rank 数 = num_scaleout_ranks * num_scaleup_ranks
    int num_scaleout_ranks, num_scaleup_ranks;  // 跨节点 rank 数 / 节点内 rank 数
    int num_experts, num_experts_per_rank;      // expert 总数 / 每 rank 的 expert 数

    // 固定大小常量: 分配时使用最大值, 运行时参数只影响访问偏移
    //   好处: 同一块 buffer 可复用于不同配置, 无需重新分配
    //   代价: 浪费一些内存, 但 workspace 通常不大
    static constexpr int kNumMaxRanks = 1024;          // 最大 rank 数 (全局)
    static constexpr int kNumMaxExperts = 2048;        // 最大 expert 数 (全局)
    static constexpr int kNumMaxExpertsPerRank = 256;  // 每 rank 最大 expert 数
    static constexpr int kNumMaxInflightAGRS = 32;     // AGRS 最大并发信号数

    // NVLink barrier 信号大小: 16 bytes = counter(8) + signal[0](4) + signal[1](4)
    static constexpr int64_t kNumBarrierSignalBytes = 16;

    __forceinline__ __device__ __host__
    WorkspaceLayout(void* workspace,
                    const int& num_scaleout_ranks,
                    const int& num_scaleup_ranks,
                    const int& num_experts):
        workspace(workspace),
        num_ranks(num_scaleout_ranks * num_scaleup_ranks),
        num_scaleout_ranks(num_scaleout_ranks),
        num_scaleup_ranks(num_scaleup_ranks),
        num_experts(num_experts) {
        num_experts_per_rank = num_experts / num_ranks;
        EP_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        EP_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        EP_UNIFIED_ASSERT(num_experts <= kNumMaxExperts);
        EP_UNIFIED_ASSERT(num_experts_per_rank <= kNumMaxExpertsPerRank);
    }

    // ==================== get_num_bytes: 计算总分配大小 ====================
    // 固定大小分配, 用 kNumMax* 常量, 不依赖运行时参数
    //   好处: 同一块 buffer 可复用于不同配置 (不同 num_ranks/num_experts)
    static int64_t get_num_bytes() {
        int64_t num_bytes = 0;

        // [0] NVLink barrier signal: counter(8) + signal[0](4) + signal[1](4) = 16 bytes
        num_bytes += kNumBarrierSignalBytes;

        // [1] Notify reduction workspace: (kNumMaxRanks + kNumMaxExperts) 个 int64_t
        //   每个 int64_t 编码: (到达SM数 << 32) | 计数值
        //   kNumMaxRanks: rank 计数槽, kNumMaxExperts: expert 计数槽
        num_bytes += (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t);

        // [2] Scaleup rank+expert count (send + recv 各一份)
        //   每份: kNumMaxRanks 个 int64_t (rank) + kNumMaxExperts 个 int64_t (expert)
        //   send buffer: 仅 dispatch.cuh 非 NVLink 模式使用 (RDMA put 源)
        //                hybrid_dispatch.cuh 不使用 (put_value 直接写对端 recv)
        //   recv buffer: 对端 scaleup rank 通过 NVLink put_value / red_add_rel 写入
        num_bytes += kNumMaxRanks * sizeof(int64_t) * 2;
        num_bytes += kNumMaxExperts * sizeof(int64_t) * 2;

        // [3] Scaleup atomic sender counter: kNumMaxRanks 个 int
        //   forward warp 用 atomicAdd 分配 scaleup_buffer 的 slot 编号
        num_bytes += kNumMaxRanks * sizeof(int);

        // [4] Scaleout rank+expert count (send + recv 各一份)
        //   每份: kNumMaxRanks 个 int (rank) + kNumMaxExperts 个 int (expert)
        //   send buffer: 编码后的计数 (encode_decode_positive), 供 RDMA put
        //   recv buffer: 从 RDMA 接收, 供 recv_and_reduce 解码求和
        num_bytes += kNumMaxRanks * sizeof(int) * 2;
        num_bytes += kNumMaxExperts * sizeof(int) * 2;

        // [5] Scaleout channel signaled tail: [channel][scaleout_rank] → int64_t
        //   编码: pack2(finish_flag, tail_count)
        //   scaleout warp 定期通知 forward warp: 本 channel 已写入多少 slot
        //   kNumMaxChannels 定义在 compiled.cuh 中
        num_bytes += kNumMaxRanks * kNumMaxChannels * sizeof(int64_t);

        // [6] Channel scaleup tail: [channel][scaleup_rank] → int
        //   linked list 的尾部位置, forward warp 写入, epilogue 读取
        num_bytes += kNumMaxRanks * kNumMaxChannels * sizeof(int);

        // [7] PP (pipeline parallelism) send/recv count: 2 个方向 × 2 个 rank × int64_t
        num_bytes += 2 * 2 * sizeof(int64_t);

        // [8] AGRS (async GPU rendezvous signal) recv + session
        //   recv: [slot][rank] → int, kNumMaxInflightAGRS 个 slot
        //   session: [rank] → int, 1 个 slot (额外的 +1)
        num_bytes += (kNumMaxInflightAGRS + 1) * kNumMaxRanks * sizeof(int);

        // 对齐到 32 字节, 保证 LDG.256 (128-bit load) 可用
        return math::align<int64_t>(num_bytes, 32);
    }

    // ==================== NVLink Barrier ====================
    // counter: barrier 计数器 (多少个 rank 已到达)
    // signal[phase]: barrier 信号 (0 或 1, 交替翻转)
    __forceinline__ __device__ __host__ unsigned long long* get_nvl_barrier_counter_ptr() const {
        return static_cast<unsigned long long*>(workspace);
    }
    __forceinline__ __device__ __host__ int* get_nvl_barrier_signal_ptr(const int& phase) const {
        return math::advance_ptr<int>(workspace, (2 + phase) * sizeof(int));
    }

    // ==================== Notify Reduction Workspace ====================
    // 位置: 紧接在 barrier signal 之后 (offset = kNumBarrierSignalBytes)
    // 用途: 所有 SM 的 notify warp 做 red_add 聚合到这里
    //   [0..kNumMaxRanks-1]: rank 计数 (每个 rank 收到多少 token)
    //   [kNumMaxRanks..kNumMaxRanks+kNumMaxExperts-1]: expert 计数
    //   每个 int64_t = (到达SM数 << 32) | 计数值
    __forceinline__ __device__ __host__ int64_t* get_notify_reduction_workspace_ptr() const {
        return math::advance_ptr<int64_t>(workspace, kNumBarrierSignalBytes);
    }

    // ==================== Scaleup Rank/Expert Count ====================
    // 位置: 紧接在 notify reduction 之后
    // kIsSendBuffer=true:  send buffer (本节点写入, 供 RDMA put 发送)
    //   注意: hybrid_dispatch.cuh 不使用此区域!
    //         它用 put_value/red_add_rel 直接写对端的 recv buffer,
    //         不需要本地 send buffer 做暂存。
    //         仅 dispatch.cuh 的非 NVLink 模式 (kIsScaleupNVLink=false) 使用。
    // kIsSendBuffer=false: recv buffer (对端通过 NVLink 写入, 本节点读取)
    //
    // 内存结构 (send 和 recv 布局相同):
    //   [0..num_scaleup_ranks-1]: rank_count (每 scaleup rank 一个 int64_t)
    //   [num_scaleup_ranks..num_scaleup_ranks+num_experts_per_rank-1]: expert_count
    //
    //   rank_count 编码: (到达 scaleup rank 数 << 32) | count
    //     高 32 位用于判断是否所有 scaleup rank 都已写入
    //   expert_count 编码: 同上
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int64_t>(get_notify_reduction_workspace_ptr(), (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
    }

    // rank_count 子区域: 前 num_scaleup_ranks 个 int64_t
    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>();
    }

    // expert_count 子区域: 从 num_scaleup_ranks 开始, 共 num_experts_per_rank 个 int64_t
    //   注意: num_experts_per_rank 是运行时值, 不是 kNumMaxExpertsPerRank
    //   但分配用的是 kNumMaxExperts, 所以偏移不会越界
    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_expert_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>() + num_scaleup_ranks;
    }

    // ==================== Scaleup Atomic Sender Counter ====================
    // 位置: 紧接在 scaleup rank+expert count (send+recv) 之后
    // 用途: forward warp 用 atomicAdd 从这里分配 scaleup_buffer 的 slot
    //   [0..num_scaleup_ranks-1]: 每个 scaleup rank 一个原子计数器
    //   每次 atomicAdd +1, 返回值就是分配的 slot 编号
    //   不同 channel 的 forward warp 共享此计数器, 保证 slot 不冲突
    __forceinline__ __device__ __host__ int* get_scaleup_atomic_sender_counter() const {
        return math::advance_ptr<int>(
            get_scaleup_rank_expert_count_ptr<true>(), 2 * (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
    }

    // ==================== Scaleout Rank/Expert Count ====================
    // 位置: 紧接在 scaleup atomic sender counter 之后
    // 与 scaleup 版本类似, 但数据类型是 int (不是 int64_t)
    //   因为 encode_decode_positive 编码后的值是 int, 直接 RDMA put
    //
    // kIsSendBuffer=true:  send buffer (编码后的计数, 供 RDMA put 发送)
    // kIsSendBuffer=false: recv buffer (RDMA 接收, 供 recv_and_reduce 解码)
    //
    // 内存结构:
    //   [0..num_ranks-1]: rank_count (按 scaleout_rank * scaleup_ranks + scaleup_rank 排列)
    //   [num_ranks..num_ranks+num_experts_per_scaleout-1]: expert_count
    //     (按 scaleout_rank * experts_per_scaleout + expert_idx 排列)
    __forceinline__ __device__ __host__ int* get_scaleout_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int>(get_scaleup_atomic_sender_counter(), kNumMaxRanks * sizeof(int));
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
    }

    // rank_count: 按 (scaleout_rank, scaleup_rank) 二维排列
    //   偏移 = scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx
    //   每 scaleout rank 有一组 scaleup_ranks 个 rank 计数
    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_count_ptr(
        const int& scaleout_rank_idx = 0, const int& scaleup_rank_idx = 0) const {
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>();
        return base_ptr + scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx;
    }

    // expert_count: 按 (scaleout_rank, expert_idx) 二维排列
    //   偏移 = num_ranks + scaleout_rank_idx * (num_scaleup_ranks * num_experts_per_rank) + expert_idx
    //   注意: expert 是按 scaleout rank 分段的, 每个 scaleout rank 负责一部分 expert
    //   expert_idx 是在该 scaleout rank 内的局部编号
    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_expert_count_ptr(
        const int& scaleout_rank_idx = 0, const int& expert_idx = 0) const {
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>() + num_ranks;
        return base_ptr + scaleout_rank_idx * (num_scaleup_ranks * num_experts_per_rank) + expert_idx;
    }

    // ==================== Scaleout Channel Signaled Tail ====================
    // 位置: 紧接在 scaleout rank+expert count 之后
    // 用途: scaleout warp 定期通知 forward warp 已写入的 slot 数
    //   编码: pack2(finish_flag, tail_count) → int64_t
    //     finish_flag: 本 channel 是否已处理完所有 token
    //     tail_count: 已写入的 slot 数
    //
    // 索引: [channel_idx * num_scaleout_ranks + scaleout_rank_idx]
    //   每个 (channel, scaleout_rank) 组合一个 int64_t
    //   forward warp 轮询此值来知道有多少 token 可处理
    __forceinline__ __device__ __host__ int64_t* get_scaleout_channel_signaled_tail_ptr(
        const int& channel_idx, const int& scaleout_rank_idx) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_scaleout_rank_expert_count_ptr<true>(),
            (kNumMaxRanks + kNumMaxExperts) * sizeof(int) * 2);
        return base_ptr + (channel_idx * num_scaleout_ranks + scaleout_rank_idx);
    }

    // ==================== Channel Scaleup Tail ====================
    // 位置: 紧接在 scaleout channel signaled tail 之后
    // 用途: linked list 的尾部位置
    //   forward warp 处理完所有 token 后, 写入每个 scaleup rank 的 tail
    //   tail = 本 channel 向该 scaleup rank 发送的 token 总数
    //   transform_linked_list_idx(tail) 得到全局物理索引
    //   epilogue 从 tail 开始反向遍历 linked list
    //
    // 索引: [channel_idx * num_scaleup_ranks + scaleup_rank_idx]
    __forceinline__ __device__ __host__ int* get_channel_scaleup_tail_ptr(
        const int& channel_idx, const int& scaleup_rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_scaleout_channel_signaled_tail_ptr(0, 0),
            kNumMaxRanks * kNumMaxChannels * sizeof(int64_t));
        return base_ptr + (channel_idx * num_scaleup_ranks + scaleup_rank_idx);
    }

    // ==================== Pipeline Parallelism Count ====================
    // PP prev/next rank 的发送/接收计数
    //   get_pp_send_count_ptr: 发送计数, offset=0 或 1
    //   get_pp_recv_count_ptr: 接收计数, offset=0 或 1
    __forceinline__ __device__ __host__ int64_t* get_pp_send_count_ptr(const int& offset) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_channel_scaleup_tail_ptr(0, 0),
            kNumMaxRanks * kNumMaxChannels * sizeof(int));
        return base_ptr + offset;
    }
    __forceinline__ __device__ __host__ int64_t* get_pp_recv_count_ptr(const int& offset) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_pp_send_count_ptr(0), 2 * sizeof(int64_t));
        return base_ptr + offset;
    }

    // ==================== AGRS Signals ====================
    // AGRS = Async GPU Rendezvous Signal (异步 GPU 汇合信号)
    //   用于 GPU 间的同步/通知机制
    //
    // recv_signal: [slot][rank] → int
    //   slot ∈ [0, kNumMaxInflightAGRS)
    //   kNumMaxInflightAGRS: 最大并发 AGRS 请求数
    __forceinline__ __device__ __host__ int* get_agrs_recv_signal_ptr(const int& slot, const int& rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_pp_recv_count_ptr(0), 2 * sizeof(int64_t));
        return base_ptr + slot * kNumMaxRanks + rank_idx;
    }

    // session_signal: [rank] → int
    //   额外的一个 slot, 用于 AGRS 会话管理
    __forceinline__ __device__ __host__ int* get_agrs_session_signal_ptr(const int& rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_agrs_recv_signal_ptr(0, 0), kNumMaxInflightAGRS * kNumMaxRanks * sizeof(int));
        return base_ptr + rank_idx;
    }
};

/**
 * ==================== TokenLayout ====================
 *
 * 描述单个 token 在内存中的布局, 用于 TMA load/store 的 1D 传输。
 *
 * 内存布局 (每个 token):
 *   ┌──────────────────────────┬───────────────┬─────────────────────────────────────────────┬─────────────┐
 *   │ hidden states            │ SF packs      │ metadata                                    │ mbarrier    │
 *   │ [num_hidden_bytes]       │ [num_sf_bytes]│ [num_metadata_bytes]                        │ (可选)       │
 *   │                          │               │ ┌─────────┬──────────┬──────────┬──────────┐│             │
 *   │                          │               │ │topk_idx │topk_wts  │src_global│linked_lst││             │
 *   │                          │               │ │[num_topk]│[num_topk]│[1]       │[num_topk]││             │
 *   │                          │               │ └─────────┴──────────┴──────────┴──────────┘│             │
 *   └──────────────────────────┴───────────────┴─────────────────────────────────────────────┴─────────────┘
 *
 *   hidden:   模型 hidden states, 字节数 = kNumHiddenBytes
 *   SF:       FP8 scale factor packs, 字节数 = kNumSFPacks * sizeof(sf_pack_t)
 *   metadata: top-k 索引 + 权重 + 源 token 信息 + linked list 索引
 *     - topk_idx:    [0..num_topk-1], int, 每个 top-k 选择的 expert 全局编号 (-1=无效)
 *     - topk_weights:[0..num_topk-1], float, 每个 top-k 选择的权重
 *     - src_global:  [0], int, 源 token 全局索引 (rank_idx * max_tokens + token_idx)
 *                    (epilogue/combine 用于识别 token 来源)
 *     - linked_lst:  [0..num_topk-1], int, 每个 top-k 的 linked list 位置
 *                    (epilogue 用此构建 channel linked list, 仅在 with_metadata=true 时存在)
 *   mbarrier: TMA 同步原语, 仅 kWithMBarrier=true 时分配
 *
 * 所有字段按 ptx::kNumTMAAlignBytes (16 bytes) 对齐, 保证 TMA 1D 传输的要求
 */
struct TokenLayout {
    int num_hidden_bytes, num_sf_bytes;
    // with_metadata: 是否包含额外的 metadata (src_token_global_idx + linked_list_idx)
    //   true:  dispatch/forward 路径 (需要完整 metadata)
    //   false: 部分 copy 路径 (只需要 top-k + weight)
    bool with_metadata;
    int num_topk, num_metadata_bytes;
    void* base;

    __forceinline__ __device__ __host__
    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,
                const int& num_topk, const bool& with_metadata, void* base = nullptr) :
        num_hidden_bytes(num_hidden_bytes),
        num_sf_bytes(num_sf_bytes),
        with_metadata(with_metadata),
        num_topk(num_topk),
        // metadata 字节计算:
        //   top-k 索引: num_topk * sizeof(int)
        //   top-k 权重: num_topk * sizeof(float)
        //   源 token + linked list: (1 + num_topk) * sizeof(int), 仅 with_metadata=true
        //     src_token_global_idx: 1 个 int
        //     linked_list_idx: num_topk 个 int (每个 top-k 一个)
        num_metadata_bytes(num_topk * (sizeof(int) + sizeof(float)) +
                           (with_metadata ? (1 + num_topk) * sizeof(int) : 0)),
        base(base) {
        EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");
        EP_UNIFIED_ASSERT(num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);
    }

    // ==================== get_num_bytes: 计算单个 token 的总字节数 ====================
    // kWithMBarrier=true:  包含 mbarrier 空间 (用于 TMA store 的同步)
    //   用途: TMA store 写入时的缓冲区大小
    // kWithMBarrier=false: 不含 mbarrier (纯数据大小)
    //   用途: TMA load 读取、RDMA 传输的数据大小
    template <bool kWithMBarrier, typename dtype_t = int>
    __forceinline__ __device__ __host__ dtype_t get_num_bytes() const {
        const auto num_bytes = math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_sf_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes) +
                               math::align<int>(kWithMBarrier ? sizeof(ptx::mbarrier) : 0, ptx::kNumTMAAlignBytes);
        return static_cast<dtype_t>(num_bytes);
    }

    // ==================== 各字段的指针 ====================
    __forceinline__ __device__ __host__ void* get_base_ptr() const { return base; }
    __forceinline__ __device__ __host__ void set_base_ptr(void* ptr) { base = ptr; }

    // hidden: 起始位置 = base
    __forceinline__ __device__ __host__ void* get_hidden_ptr() const { return get_base_ptr(); }

    // SF: 紧接在 hidden 之后, 按 TMA 对齐
    __forceinline__ __device__ __host__ sf_pack_t* get_sf_ptr() const {
        return math::advance_ptr<sf_pack_t>(base, math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes));
    }

    // metadata: 紧接在 SF 之后, 按 TMA 对齐
    __forceinline__ __device__ __host__ int* get_metadata_ptr() const {
        return math::advance_ptr<int>(get_sf_ptr(), math::align(num_sf_bytes, ptx::kNumTMAAlignBytes));
    }

    // top-k 索引: metadata 的起始位置 (前 num_topk 个 int)
    __forceinline__ __device__ __host__ int* get_topk_idx_ptr() const { return get_metadata_ptr(); }

    // top-k 权重: 紧接在 top-k 索引之后 (num_topk 个 float)
    __forceinline__ __device__ __host__ float* get_topk_weights_ptr() const {
        return math::advance_ptr<float>(get_metadata_ptr(), num_topk * sizeof(int));
    }

    // 源 token 全局索引: 紧接在 top-k 权重之后 (1 个 int)
    //   编码: rank_idx * kNumMaxTokensPerRank + token_idx
    //   epilogue/combine 用此识别 token 来自哪个 rank
    __forceinline__ __device__ __host__ int* get_src_token_global_idx_ptr() const {
        return math::advance_ptr<int>(get_topk_weights_ptr(), num_topk * sizeof(float));
    }

    // linked list 索引: src_token_global_idx 之后 (num_topk 个 int)
    //   与 src_token_global_idx 共享同一个 int 的高/低位
    //   实际上 get_linked_list_idx_ptr() = get_src_token_global_idx_ptr() + 1
    //   即: [src_token_global_idx, linked_list_idx[0], linked_list_idx[1], ...]
    //   (因为 with_metadata 时, 额外的 (1+num_topk) 个 int 中, 第1个是 src, 后 num_topk 个是 linked_list)
    __forceinline__ __device__ __host__ int* get_linked_list_idx_ptr() const {
        return get_src_token_global_idx_ptr() + 1;
    }

    // mbarrier: 紧接在 metadata 之后, 按 TMA 对齐
    //   用于 TMA load/store 的异步完成通知
    __forceinline__ __device__ ptx::mbarrier* get_mbarrier_ptr() const {
        return math::advance_ptr<ptx::mbarrier>(get_metadata_ptr(), math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes));
    }
};

/**
 * ==================== BufferLayout ====================
 *
 * 描述一组 token 的缓冲区布局, 按 (rank, token) 二维组织。
 *
 * 内存布局:
 *   ┌───────────────────────┬───────────────────────┬─────┬───────────────────────┐
 *   │ rank 0                │ rank 1                │ ... │ rank N-1              │
 *   │ [max_tokens_per_rank] │ [max_tokens_per_rank] │     │ [max_tokens_per_rank] │
 *   │ token_0, token_1, ... │ token_0, token_1, ... │     │ token_0, token_1, ... │
 *   └───────────────────────┴───────────────────────┴─────┴───────────────────────┘
 *
 * kWithMBarrier:
 *   true:  每个 token 包含 mbarrier (用于 TMA store 的 smem 缓冲区)
 *   false: 每个 token 不含 mbarrier (用于 gmem 缓冲区, 如 scaleup/recv/send buffer)
 *
 * 三种使用场景:
 *   1. smem TMA buffer (kWithMBarrier=true):
 *      num_ranks = kNumScaleoutWarps + kNumForwardWarps
 *      max_tokens_per_rank = 1
 *      每个 warp 一个 token 的 smem 缓冲区
 *
 *   2. scaleup_buffer (kWithMBarrier=false):
 *      num_ranks = kNumScaleupRanks
 *      max_tokens_per_rank = kNumScaleoutRanks * kNumMaxTokensPerRank
 *      NVLink 对端直写目标
 *
 *   3. scaleout_send/recv_buffer (kWithMBarrier=false):
 *      send: num_ranks=1, max_tokens=kNumMaxTokensPerRank
 *      recv: num_ranks=kNumScaleoutRanks, max_tokens=kNumChannels*kNumMaxTokensPerChannel
 *      RDMA 发送暂存 / 接收缓冲区
 */
template <bool kWithMBarrier>
struct BufferLayout {
    TokenLayout token_layout;
    int num_ranks;                  // rank 数 (buffer 的第一维)
    int num_max_tokens_per_rank;    // 每 rank 的最大 token 数 (buffer 的第二维)

    void* base;

    __forceinline__ __device__ __host__
    BufferLayout(const TokenLayout& token_layout,
                 const int& num_ranks,
                 const int& max_num_tokens_per_rank,
                 void* base = nullptr) :
        token_layout(token_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    // ==================== 大小计算 ====================
    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_token() const {
        return token_layout.get_num_bytes<kWithMBarrier, int64_t>();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * get_num_bytes_per_token();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    // buffer 结束地址 (用于构建连续的多个 buffer)
    __forceinline__ __device__ __host__
    void* get_buffer_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // ==================== 子缓冲区访问 ====================

    // 获取指定 rank 的子缓冲区
    //   返回一个新的 BufferLayout, num_ranks=1, base 指向该 rank 的起始位置
    //   常用: scaleup_buffer.get_rank_buffer(i) → 第 i 个 scaleup rank 的缓冲区
    __forceinline__ __device__ __host__
    BufferLayout get_rank_buffer(const int& rank_idx) const {
        return BufferLayout(token_layout,
                            1, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_rank() * rank_idx);
    }

    // 获取指定 channel 的子缓冲区
    //   注意: channel 维度是 token 维度内的切分, 不是 rank 维度
    //   kNumTokensPerChannel = kNumMaxTokensPerRank / kNumChannels
    //   返回的 BufferLayout 仍保留 num_ranks, 但 base 偏移到本 channel 的起始位置
    //
    //   举例: recv_buffer 的布局 [kNumScaleoutRanks][kNumChannels * kNumMaxTokensPerChannel]
    //     get_channel_buffer<max_per_ch>(ch_idx):
    //       base += ch_idx * max_per_ch * bytes_per_token
    //       num_ranks 保持不变 (仍可按 rank 访问)
    __forceinline__ __device__ __host__
    BufferLayout get_channel_buffer(const int& channel_idx) const {
        EP_UNIFIED_ASSERT(num_max_tokens_per_rank % kNumTokensPerChannel == 0);
        return BufferLayout(token_layout,
                            // 保持 num_ranks 不变 (不是 num_max_tokens_per_rank / kNumTokensPerChannel)
                            // 因为 channel 切的是 token 维度, 不是 rank 维度
                            num_ranks, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_token() * kNumTokensPerChannel * channel_idx);
    }

    // 获取指定 token 的 TokenLayout (定位到单个 token 的内存位置)
    //   前提: num_ranks == 1 (已通过 get_rank_buffer 选择了特定 rank)
    //   或者 global=true (使用绝对 token 索引, 忽略 rank 维度)
    __forceinline__ __device__ __host__
    TokenLayout get_token_buffer(const int& token_idx, const bool& global = false) const {
        EP_UNIFIED_ASSERT(num_ranks == 1 or global);
        return TokenLayout(token_layout.num_hidden_bytes, token_layout.num_sf_bytes, token_layout.num_topk, token_layout.with_metadata,
                           static_cast<int8_t*>(base) + token_layout.get_num_bytes<kWithMBarrier, int64_t>() * token_idx);
    }
};

}  // namespace deep_ep::elastic
