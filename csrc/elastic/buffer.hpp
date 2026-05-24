#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <pybind11/functional.h>

#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/compiled.cuh>

#include "../kernels/backend/api.cuh"
#include "../kernels/elastic/api.hpp"
#include "../utils/event.hpp"
#include "utils.hpp"

namespace deep_ep::elastic {

class ElasticBuffer {
    // Buffer registered for both scaleout and scaleup
    int64_t num_buffer_bytes;
    void* buffer;

    // Destructor settings
    bool explicitly_destroy;
    bool destroyed = false;

    // Workspace
    // NOTES: for all workspace, we must keep them as zeros
    void *workspace;
    void *host_workspace, *mapped_host_workspace;
    std::shared_ptr<layout::WorkspaceLayout> workspace_layout_wo_expert;

    // CUDA streams
    at::cuda::CUDAStream comm_stream;

    // Whether to use deterministic algorithms
    bool deterministic;

    // Whether to use hybrid mode (scale-out with scale-up)
    bool allow_hybrid_mode;

    // Whether to allow multiple reductions
    bool allow_multiple_reduction;

    // Whether to prefer overlapping communication with compute (use more SMs and channels if false)
    bool prefer_overlap_with_compute;

    // Timeout settings
    int num_cpu_timeout_secs;
    int64_t num_gpu_timeout_cycles;

    // NCCL context
    std::shared_ptr<nccl::NCCLSymmetricMemoryContext> nccl_context;

    // Some EP hybrid mode settings
    static constexpr int kNumMaxChannelsPerSM = 8;
    static constexpr int kNumMaxSMs = 160;
    static constexpr int kNumMaxChannels = kNumMaxChannelsPerSM * kNumMaxSMs;

    // Some Engram storage settings
    int num_engram_entries = 0, engram_hidden = 0;
    int64_t num_engram_storage_bytes = 0;
    int64_t num_engram_recv_bytes = 0;

    // PP settings
    int prev_rank_idx = 0, next_rank_idx = 0;
    int64_t num_max_pp_tensor_bytes = 0;
    int num_max_pp_inflight_tensors = 0;

    // AGRS session settings
    int64_t num_max_agrs_session_bytes = 0;
    int num_max_agrs_per_session = 0;
    int agrs_session_idx = 0;
    bool agrs_in_session = false;

    // AGRS in-session settings
    int64_t agrs_buffer_offset = 0;
    int agrs_buffer_slot_idx = 0;

public:
    ElasticBuffer(const int& rank_idx, const int& num_ranks,
                  const int64_t& nccl_comm,
                  const int64_t& num_buffer_bytes,
                  const bool& deterministic,
                  const bool& allow_hybrid_mode,
                  const bool& allow_multiple_reduction,
                  const bool& prefer_overlap_with_compute,
                  const int& sl_idx, const int& num_allocated_qps,
                  const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
                  const bool& explicitly_destroy):
        num_buffer_bytes(num_buffer_bytes),
        explicitly_destroy(explicitly_destroy),
        comm_stream(get_global_comm_stream()),
        deterministic(deterministic),
        allow_hybrid_mode(allow_hybrid_mode),
        allow_multiple_reduction(allow_multiple_reduction),
        prefer_overlap_with_compute(prefer_overlap_with_compute) {
        // Init NCCL runtime
        static constexpr int kBufferAlignment = 16;
        this->nccl_context = std::make_shared<nccl::NCCLSymmetricMemoryContext>(
            nccl_comm, num_ranks, rank_idx,
            layout::WorkspaceLayout::get_num_bytes() + num_buffer_bytes, kBufferAlignment,
            allow_hybrid_mode, sl_idx, num_allocated_qps);

        // Timeout
        this->num_cpu_timeout_secs = num_cpu_timeout_secs;
        this->num_gpu_timeout_cycles = static_cast<int64_t>(num_gpu_timeout_secs);
        this->num_gpu_timeout_cycles *= jit::device_runtime->get_clock_rate();

        // Assign workspaces and buffers
        workspace = this->nccl_context->mapped_window_ptr;
        workspace_layout_wo_expert = std::make_shared<layout::WorkspaceLayout>(
            workspace, nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks, 0);
        buffer = static_cast<uint8_t*>(workspace) + layout::WorkspaceLayout::get_num_bytes();
        CUDA_RUNTIME_CHECK(cudaMemset(workspace, 0, layout::WorkspaceLayout::get_num_bytes()));

        // Allocate host workspaces
        CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, layout::WorkspaceLayout::get_num_bytes(), cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
        std::memset(host_workspace, 0, layout::WorkspaceLayout::get_num_bytes());

        // We should call a barrier at the end
        // The barrier should be called by Python `dist.barrier`
        // NOTES: do not call our barrier, as the workspace is not ready yet
    }

    ~ElasticBuffer() noexcept(false) {
        if (not explicitly_destroy)
            destroy();

        if (not destroyed) {
            printf("`destroy()` is not called before DeepEP elastic buffer destruction, which can leak resources.\n");
            fflush(stdout);
        }
    }

    void destroy() {
        EP_HOST_ASSERT(not destroyed);

        // Finish all works on all GPUs
        barrier(true, true);

        // Deallocate host workspaces
        CUDA_RUNTIME_CHECK(cudaFreeHost(host_workspace));

        // Destroy NCCL context
        nccl_context->finalize();

        // Cannot use anymore
        destroyed = true;
    }

    torch::Stream get_comm_stream() const {
        return comm_stream;
    }

    std::tuple<int, int> get_physical_domain_size() const {
        return {nccl_context->num_rdma_ranks, nccl_context->num_nvl_ranks};
    }

    std::tuple<int, int> get_logical_domain_size() const {
        return {nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks};
    }

    // ReSharper disable once CppMemberFunctionMayBeStatic
    void barrier(const bool& use_comm_stream, const bool& with_cpu_sync) const {
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        const auto stream = use_comm_stream ? comm_stream : compute_stream;
        if (use_comm_stream)
            stream_wait(comm_stream, compute_stream);

        // Wait all streams to finish on this GPU
        if (with_cpu_sync)
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        // Launch GPU barrier
        launch_barrier(nccl_context->dev_comm, nccl_context->window,
                       workspace,
                       nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                       num_gpu_timeout_cycles,
                       nccl_context->is_scaleup_nvlink,
                       stream);

        // Let CPU wait
        if (with_cpu_sync)
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        // Compute stream should also wait for the barrier
        if (use_comm_stream)
            stream_wait(compute_stream, comm_stream);
    }

    void engram_write(const torch::Tensor& storage) {
        // Ensure previous fetch are finished
        barrier(false, true);

        const auto compute_stream = at::cuda::getCurrentCUDAStream();

        // Check storage
        const auto [num_entries, hidden] = get_shape<2>(storage);
        EP_HOST_ASSERT(storage.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(storage.is_cuda() and storage.is_contiguous());
        num_engram_entries = num_entries, engram_hidden = hidden;

        // Write storage and ensure the received buffer is aligned
        EP_HOST_ASSERT(storage.nbytes() <= num_buffer_bytes);
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            buffer, storage.data_ptr(), storage.nbytes(),
            cudaMemcpyDeviceToDevice, compute_stream));
        num_engram_storage_bytes = math::align<int64_t>(storage.nbytes(), 32);
        num_engram_recv_bytes = num_buffer_bytes - num_engram_storage_bytes;

        // Ensure data is visible for all ranks
        barrier(false, true);
    }

    std::function<torch::Tensor()> engram_fetch(const torch::Tensor& indices, int num_qps) const {
        const auto [num_tokens] = get_shape<1>(indices);
        EP_HOST_ASSERT(indices.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(indices.is_cuda() and indices.is_contiguous());
        EP_HOST_ASSERT(num_tokens * engram_hidden * sizeof(nv_bfloat16) <= num_engram_recv_bytes);

        // Calculate a QP count
        if (num_qps == 0)
            num_qps = nccl_context->num_allocated_qps;

        // Return tensor from the raw buffer
        EP_HOST_ASSERT(num_engram_entries > 0);
        const auto fetched = torch::from_blob(
            math::advance_ptr(buffer, num_engram_storage_bytes),
            {num_tokens, engram_hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA)
        );

        // Last issued Gin requests
        const auto last_gin_requests = torch::empty(
            {nccl_context->num_ranks * num_qps, sizeof(ncclGinRequest_t)},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        );

        // Launch the fetch kernel
        launch_engram_fetch(
            nccl_context->dev_comm, nccl_context->window,
            buffer,
            fetched.data_ptr(),
            indices.data_ptr<int>(),
            static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),
            num_engram_entries, engram_hidden,
            num_tokens,
            nccl_context->num_ranks, num_qps,
            at::cuda::getCurrentCUDAStream()
        );

        return [=, this]() {
            // Wait for all RDMA gets to complete
            launch_engram_fetch_wait(
                static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),
                nccl_context->dev_comm,
                nccl_context->window,
                nccl_context->num_ranks, num_qps,
                at::cuda::getCurrentCUDAStream()
            );
            return fetched;
        };
    }

    void pp_set_config(const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors) {
        // Flush previous operations
        barrier(false, true);

        EP_HOST_ASSERT(num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0);
        EP_HOST_ASSERT(num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2 <= num_buffer_bytes);
        this->prev_rank_idx = (nccl_context->rank_idx + nccl_context->num_ranks - 1) % nccl_context->num_ranks;
        this->next_rank_idx = (nccl_context->rank_idx + 1) % nccl_context->num_ranks;
        this->num_max_pp_tensor_bytes = math::align<int64_t>(num_max_tensor_bytes, 32);
        this->num_max_pp_inflight_tensors = num_max_inflight_tensors;
    }

    void pp_send(const torch::Tensor& x, const int& dst_rank_idx, const int& num_sms) const {
        EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() <= num_max_pp_tensor_bytes);
        EP_HOST_ASSERT(dst_rank_idx == prev_rank_idx or dst_rank_idx == next_rank_idx);

        launch_pp_send(
            nccl_context->dev_comm, nccl_context->window,
            x.data_ptr(), x.nbytes(),
            buffer, workspace,
            nccl_context->rank_idx, dst_rank_idx, nccl_context->num_ranks,
            num_max_pp_tensor_bytes,
            num_max_pp_inflight_tensors,
            num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,
            num_gpu_timeout_cycles,
            jit::device_runtime->get_num_smem_bytes(),
            at::cuda::getCurrentCUDAStream()
        );
    }

    void pp_recv(const torch::Tensor& x, const int& src_rank_idx, const int& num_sms) const {
        EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() <= num_max_pp_tensor_bytes);
        EP_HOST_ASSERT(src_rank_idx == prev_rank_idx or src_rank_idx == next_rank_idx);

        launch_pp_recv(
            nccl_context->dev_comm, nccl_context->window,
            x.data_ptr(), x.nbytes(),
            buffer, workspace,
            nccl_context->rank_idx, src_rank_idx, nccl_context->num_ranks,
            num_max_pp_tensor_bytes,
            num_max_pp_inflight_tensors,
            num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,
            num_gpu_timeout_cycles,
            jit::device_runtime->get_num_smem_bytes(),
            at::cuda::getCurrentCUDAStream()
        );
    }

    void agrs_set_config(const int64_t& num_max_session_bytes,
                         const int& new_num_max_agrs_per_session) {
        // Flush previous operations
        barrier(true, true);

        EP_HOST_ASSERT(nccl_context->num_ranks > 1);
        EP_HOST_ASSERT(num_max_session_bytes > 0 and new_num_max_agrs_per_session > 0);
        EP_HOST_ASSERT(num_max_session_bytes <= num_buffer_bytes);
        EP_HOST_ASSERT(new_num_max_agrs_per_session <= layout::WorkspaceLayout::kNumMaxInflightAGRS);
        EP_HOST_ASSERT(nccl_context->num_nvl_ranks == nccl_context->num_ranks);
        this->num_max_agrs_session_bytes = math::align<int64_t>(num_max_session_bytes, 32);
        this->num_max_agrs_per_session = new_num_max_agrs_per_session;
    }

    void create_agrs_session() {
        EP_HOST_ASSERT(not agrs_in_session);
        agrs_in_session = true;
        agrs_buffer_offset = 0;
        agrs_buffer_slot_idx = 0;
        agrs_session_idx += 1;
    }

    void destroy_agrs_session() {
        // Must be in a session
        EP_HOST_ASSERT(agrs_in_session);
        agrs_in_session = false;

        // Wait compute stream
        stream_wait(comm_stream, at::cuda::getCurrentCUDAStream());

        // Notify that the buffer is now available & Wait for the buffer to be ready
        // NOTES: self-wait is guaranteed by in-stream order
        std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);
        std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);
        for (int i = 0; i < nccl_context->num_ranks - 1; ++ i) {
            const auto dst_rank_idx = (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;
            write_ptrs[i] = static_cast<int*>(
                nccl_context->get_sym_ptr(workspace_layout_wo_expert->get_agrs_session_signal_ptr(nccl_context->rank_idx), dst_rank_idx));
            wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_session_signal_ptr(dst_rank_idx);
        }
        cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, agrs_session_idx);
    }

    std::vector<torch::Tensor> agrs_get_inplace_tensor(const std::vector<int64_t>& num_bytes_list) const {
        EP_HOST_ASSERT(num_bytes_list.size() >= 1);
        EP_HOST_ASSERT(num_max_agrs_session_bytes > 0 and num_max_agrs_per_session > 0 and agrs_in_session);

        std::vector<torch::Tensor> out;
        out.reserve(num_bytes_list.size());
        int64_t offset = agrs_buffer_offset;
        for (const auto& num_bytes: num_bytes_list) {
            EP_HOST_ASSERT(num_bytes % 32 == 0);
            EP_HOST_ASSERT(offset + num_bytes * nccl_context->num_ranks <= num_max_agrs_session_bytes and
                           agrs_buffer_slot_idx < num_max_agrs_per_session and
                           "Not enough session buffer size. Did you forget to flush session?");
            out.push_back(torch::from_blob(math::advance_ptr(buffer, offset + num_bytes * nccl_context->rank_idx),
                          {num_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)));
            offset += num_bytes * nccl_context->num_ranks;
        }
        return out;
    }

    std::pair<std::vector<torch::Tensor>, std::function<void()>>
    all_gather(const std::vector<torch::Tensor>& tensors) {
        const int num_tensors = tensors.size();
        EP_HOST_ASSERT(num_max_agrs_session_bytes > 0 and num_max_agrs_per_session > 0 and agrs_in_session);
        EP_HOST_ASSERT(num_tensors >= 1);

        int num_copies = 0;
        std::vector<int64_t> offset(num_tensors);
        for (int i = 0; i < num_tensors; ++ i) {
            const auto& x = tensors[i];
            EP_HOST_ASSERT(x.is_contiguous());
            EP_HOST_ASSERT(x.is_cuda() and x.nbytes() % 32 == 0);

            const auto x_offset = math::ptr_diff(x.data_ptr(), buffer);
            const bool is_inplace = 0 <= x_offset and x_offset < num_max_agrs_session_bytes;
            offset[i] = agrs_buffer_offset;
            num_copies += nccl_context->num_ranks - is_inplace;
            agrs_buffer_offset += x.nbytes() * nccl_context->num_ranks;
            EP_HOST_ASSERT(not is_inplace or x.data_ptr() == math::advance_ptr(buffer, offset[i] + x.nbytes() * nccl_context->rank_idx));
        }
        EP_HOST_ASSERT(agrs_buffer_offset <= num_max_agrs_session_bytes and
                       agrs_buffer_slot_idx < num_max_agrs_per_session and
                       "Not enough session buffer size. Did you forget to flush session?");

        // Wait compute stream
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        stream_wait(comm_stream, compute_stream);

        // Send data to all ranks
        std::vector<size_t> sizes(num_copies);
        std::vector<void*> dst_ptrs(num_copies), src_ptrs(num_copies);
        int count = 0;
        for (int i = 0; i < nccl_context->num_ranks; ++ i) {
            for (int j = 0; j < num_tensors; ++ j) {
                const auto& x = tensors[j];
                const auto dst_rank_idx = (nccl_context->rank_idx + i) % nccl_context->num_ranks;
                void* src_ptr = x.data_ptr();
                void* dst_ptr =
                    nccl_context->get_sym_ptr(math::advance_ptr(buffer, offset[j] + x.nbytes() * nccl_context->rank_idx), dst_rank_idx);
                if (src_ptr != dst_ptr) {
                    src_ptrs[count] = src_ptr;
                    dst_ptrs[count] = dst_ptr;
                    sizes[count] = x.nbytes();
                    count += 1;
                }
            }
        }
        cudaMemcpyAttributes attrs = {
            .srcAccessOrder = cudaMemcpySrcAccessOrderStream,
            .flags = cudaMemcpyFlagPreferOverlapWithCompute
        };
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
        CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), num_copies, attrs, comm_stream));
#else
        CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), num_copies, attrs, nullptr, comm_stream));
#endif

        // Wait for data from other ranks
        const int current_session = agrs_session_idx;
        const int slot_idx = agrs_buffer_slot_idx;
        agrs_buffer_slot_idx += 1;
        std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);
        std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);
        for (int i = 0; i < nccl_context->num_ranks - 1; ++ i) {
            const auto dst_rank_idx = (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;
            write_ptrs[i] = nccl_context->get_sym_ptr(
                workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, nccl_context->rank_idx), dst_rank_idx);
            wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, dst_rank_idx);
        }
        cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, current_session);

        // Build output tensors eagerly
        std::vector<torch::Tensor> out(num_tensors);
        for (int i = 0; i < num_tensors; ++ i) {
            auto shape = tensors[i].sizes().vec();
            shape.insert(shape.begin(), nccl_context->num_ranks);
            out[i] = torch::from_blob(math::advance_ptr(buffer, offset[i]), shape, tensors[i].options());
        }

        // Return tensors and a handle to wait for data arrival
        const auto event = EventHandle(comm_stream);
        auto handle = [=, this]() {
            EP_HOST_ASSERT(compute_stream == at::cuda::getCurrentCUDAStream());
            EP_HOST_ASSERT(agrs_in_session and current_session == this->agrs_session_idx);
            stream_wait(compute_stream, event);
        };
        return {std::move(out), std::move(handle)};
    }

    torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event,
                                                    const bool& allocate_on_comm_stream,
                                                    const bool& async_with_compute_stream) const {
        // Allocate all tensors on communication stream if set
        // NOTES: do not allocate tensors upfront!
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(comm_stream);

        // Assertion for safety
        // `previous_event` implicitly means the overlapping computation kernels are launched first,
        // in order not to use the memory on the compute stream, we must allocate on the communication stream.
        // If you launch the communication kernels firstly, then `previous_event` must be unnecessary.
        if (previous_event.has_value())
            EP_HOST_ASSERT(allocate_on_comm_stream);

        // Wait previous tasks to finish
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }
        return compute_stream;
    }

    void stream_control_before_epilogue(const std::optional<EventHandle>& previous_event_before_epilogue) const {
        if (previous_event_before_epilogue.has_value())
            stream_wait(comm_stream, previous_event_before_epilogue.value());
    }

    std::optional<EventHandle> stream_control_epilogue(const std::vector<std::optional<torch::Tensor>>& tensors,
                                                       const at::cuda::CUDAStream& compute_stream,
                                                       const bool& allocate_on_comm_stream,
                                                       const bool& async_with_compute_stream) const {
        // Ensure memory access safety between two streams
        std::optional<EventHandle> event;
        if (async_with_compute_stream) {
            event = EventHandle(comm_stream);

            // NOTES: this environment only applies to V2 APIs
            if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {
                event->tensors_to_record = tensors;
            } else {
                for (auto& t: tensors) if (t.has_value()) {
                    t->record_stream(compute_stream);
                    t->record_stream(comm_stream);
                }
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        // The CUDA event marking the finishing
        return event;
    }

    static int64_t get_dispatch_buffer_size(const int& num_max_tokens_per_rank,
                                            const int& hidden, const int& num_sf_packs, const int& num_topk,
                                            const int& elem_size,
                                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                            const bool& is_scaleup_nvlink) {
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);

        if (num_scaleout_ranks == 1) {
            // Direct dispatch
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : 1, num_max_tokens_per_rank);
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_ranks, num_max_tokens_per_rank);
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
        } else {
            // Hybrid dispatch
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleup_ranks, num_scaleout_ranks * num_max_tokens_per_rank);
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, 1, num_max_tokens_per_rank);
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleout_ranks,
                /* kNumChannels * kNumMaxTokensPerChannel */ num_max_tokens_per_rank + kNumMaxChannels);
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();
        }
    }

    static int64_t get_combine_buffer_size(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const bool& is_scaleup_nvlink,
                                           const bool& allow_multiple_reduction) {
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
        const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);

        if (num_scaleout_ranks == 1) {
            // Direct combine
            const auto num_tokens_in_layout = allow_multiple_reduction ? std::min(num_ranks, num_topk) : num_topk;
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : num_ranks,
                // For single reduction cases, the maximum number of received tokens is
                // `num_ranks * num_topk * num_max_tokens_per_rank` (we assume the bad case of `do_expand=True`)
                num_max_tokens_per_rank * (allow_multiple_reduction ? 1 : num_topk));
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_tokens_in_layout, num_max_tokens_per_rank);
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
        } else {
            // Hybrid combine
            const int num_tokens_in_scaleup_layout = allow_multiple_reduction ? std::min(num_scaleup_ranks, num_topk) : num_topk;
            const int num_tokens_in_scaleout_layout = allow_multiple_reduction ? std::min(num_scaleout_ranks, num_topk) : num_topk;
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank);
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank);
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, allow_multiple_reduction ? 1 : num_topk,
                /* kNumChannels * num_scaleout_ranks * kNumMaxTokensPerChannel */
                num_scaleout_ranks * (num_max_tokens_per_rank + kNumMaxChannels));
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();
        }
    }

    static int64_t calculate_buffer_size(const int64_t& nccl_comm,
                                         const int& num_max_tokens_per_rank, const int& hidden,
                                         int num_topk, const bool& use_fp8_dispatch,
                                         const bool& allow_hybrid_mode,
                                         const bool& allow_multiple_reduction) {
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);

        // The worst case SF bytes must be less than the main part
        EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);

        // NOTES: there are lots of `kNumTopk <= 32` restrictions, so we use 32 to calculate token size
        num_topk = num_topk == 0 ? 32 : num_topk;

        // Topology
        const auto [num_rdma_ranks, num_nvl_ranks] = nccl::get_physical_domain_size(nccl_comm);
        const auto [num_scaleout_ranks, num_scaleup_ranks] = nccl::get_logical_domain_size(nccl_comm, allow_hybrid_mode);
        const auto is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

        // Dispatch size
        const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
        const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0; // An approximation for number of SF packs
        const auto num_dispatch_bytes = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size,
            num_scaleout_ranks, num_scaleup_ranks,
            is_scaleup_nvlink);

        // Combine layout
        const auto num_combine_bytes = get_combine_buffer_size(
            num_max_tokens_per_rank, hidden, num_topk,
            num_scaleout_ranks, num_scaleup_ranks,
            is_scaleup_nvlink, allow_multiple_reduction);

        // Return the maximum of those layouts
        return std::max(num_dispatch_bytes, num_combine_bytes);
    }

    // ============================================================================================
    //  ElasticBuffer::dispatch —— Token 分发入口
    // ============================================================================================
    //
    //  【总览】
    //  将 token 按 top-k 路由分发到各 rank 的 buffer 中, 供 expert 计算。
    //  支持单节点 (NVLink) 和多节点 (NVLink + RDMA) 混合拓扑。
    //
    //  【整体流程】
    //
    //  ┌──────────────────────────────────────────────────────────────────────────────────┐
    //  │ 1. 参数校验 + 张量分配                                                           │
    //  │ 2. (确定性模式) launch_dispatch_deterministic_prologue                            │
    //  │ 3. launch_dispatch / launch_hybrid_dispatch (主通信 kernel)                       │
    //  │    → 各 rank 将 token 写入对端 buffer, 并记录 rank_count / expert_count          │
    //  │ 4. (do_cpu_sync) CPU 轮询 host_workspace 等待所有 rank 数据就绪                   │
    //  │    (cached_mode) 直接使用缓存的计数值                                             │
    //  │    (否则) 分配最坏情况大小                                                       │
    //  │ 5. 分配输出张量 (recv_x, recv_sf, recv_topk_idx, ...)                           │
    //  │ 6. launch_dispatch_copy_epilogue (PDL epilogue, 从 buffer 拷贝到输出张量)        │
    //  │ 7. stream_control_epilogue (流同步)                                             │
    //  └──────────────────────────────────────────────────────────────────────────────────┘
    //
    //  【返回值】
    //  ┌───────────────────────────────────────────────────────────────────┐
    //  │ recv_x          : [num_allocated_tokens, hidden] 接收到的 hidden  │
    //  │ recv_sf         : (可选) [num_allocated_tokens, num_sf_packs] SF   │
    //  │ recv_topk_idx   : (可选) [num_allocated_tokens, num_topk] topk索引│
    //  │ recv_topk_weights: (可选) topk权重                               │
    //  │ copied_topk_idx : (可选) topk_idx副本 (防止用户修改)              │
    //  │ num_recv_tokens_per_expert_list: 每个本地expert收到的token数      │
    //  │ psum_num_recv_tokens_per_scaleup_rank: [num_scaleup_ranks] 前缀和 │
    //  │ psum_num_recv_tokens_per_expert: [num_local_experts] 前缀和       │
    //  │ recv_src_metadata: [num_recv_tokens, num_topk+2] 源元数据        │
    //  │ dst_buffer_slot_idx: 目标buffer slot索引                         │
    //  │ token_metadata_at_forward: (混合模式) forward元数据               │
    //  │ channel_linked_list: (混合模式) 通道链表                          │
    //  │ event           : 流同步事件                                     │
    //  └───────────────────────────────────────────────────────────────────┘
    //
    //  【入参详解】
    //  x                              : [num_tokens, hidden] 输入token, BF16或FP8
    //  sf                             : (FP8模式) [num_tokens, num_sf_packs] scale factors
    //  topk_idx                       : [num_tokens, num_topk] 每个token选择的expert索引
    //  topk_weights                   : [num_tokens, num_topk] 每个top-k的权重 (训练backward可选)
    //  cumulative_local_expert_recv_stats: (可选) [num_local_experts] 累计接收计数
    //  cached_num_recv_tokens         : (缓存模式) 缓存的总接收token数
    //  cached_num_recv_tokens_per_expert_list: (缓存模式) 缓存的每expert接收数
    //  cached_psum_num_recv_tokens_per_scaleup_rank: (缓存模式) 缓存的scaleup前缀和
    //  cached_psum_num_recv_tokens_per_expert: (缓存模式) 缓存的expert前缀和
    //  cached_dst_buffer_slot_idx     : (缓存模式) 缓存的目标slot索引
    //  cached_token_metadata_at_forward: (缓存模式/混合) 缓存的forward元数据
    //  cached_channel_linked_list     : (缓存模式/混合) 缓存的通道链表
    //  num_max_tokens_per_rank        : 每rank最大token容量
    //  num_experts                    : 总expert数
    //  expert_alignment               : expert对齐要求 (expand模式用)
    //  num_sms                        : 使用的SM数
    //  num_qps                        : RDMA QP数
    //  previous_event                 : (流控制) 前一个事件
    //  previous_event_before_epilogue  : (流控制) epilogue前的事件
    //  async_with_compute_stream      : 是否与compute流异步
    //  allocate_on_comm_stream        : 是否在comm流上分配
    //  do_handle_copy                 : 是否拷贝topk_idx (防止用户修改)
    //  do_cpu_sync                    : CPU是否同步等待dispatch完成
    //  do_expand                      : 是否展开 (每个top-k占独立行)
    //  use_tma_aligned_col_major_sf   : 是否用TMA对齐的列主序SF布局
    //
    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::vector<int>,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    dispatch(const torch::Tensor& x,
             const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const std::optional<int>& cached_num_recv_tokens,
             const std::optional<std::vector<int>>& cached_num_recv_tokens_per_expert_list,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_scaleup_rank,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_dst_buffer_slot_idx,
             const std::optional<torch::Tensor>& cached_token_metadata_at_forward,
             const std::optional<torch::Tensor>& cached_channel_linked_list,
             const int& num_max_tokens_per_rank,
             const int& num_experts, const int& expert_alignment,
             const int& num_sms, const int& num_qps,
             const std::optional<EventHandle>& previous_event,
             const std::optional<EventHandle>& previous_event_before_epilogue,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream,
             const bool& do_handle_copy, const bool& do_cpu_sync, const bool& do_expand,
             const bool& use_tma_aligned_col_major_sf) const {
        // ── 1. 基本校验 ──
        EP_HOST_ASSERT(num_sms > 0);

        // ── 2. 缓存模式校验 ──
        //   cached_mode: 重用上一轮 dispatch 的路由信息, 跳过 GPU 计数 + CPU 同步
        //   要求: 所有 cached_* 张量都已提供 (混合模式还需 token_metadata + linked_list)
        const bool cached_mode = cached_num_recv_tokens.has_value();
        if (cached_mode) {
            EP_HOST_ASSERT(cached_num_recv_tokens.has_value());
            EP_HOST_ASSERT(cached_num_recv_tokens_per_expert_list.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_scaleup_rank.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_expert.has_value());
            EP_HOST_ASSERT(cached_dst_buffer_slot_idx.has_value());

            // Hybrid kernels require more
            if (nccl_context->num_scaleout_ranks > 1) {
                EP_HOST_ASSERT(cached_token_metadata_at_forward.has_value());
                EP_HOST_ASSERT(cached_channel_linked_list.has_value());
            }
        }

        // ── 3. 输入张量校验 ──
        //   x: [num_tokens, hidden], 必须CUDA + contiguous + int4对齐
        //   num_local_experts = 总expert / rank数 (每个rank负责的expert数)
        const auto [num_tokens, hidden] = get_shape<2>(x);
        const auto num_hidden_bytes = hidden * static_cast<int>(x.element_size());
        const auto num_local_experts = num_experts / nccl_context->num_ranks;
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
        EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

        // ── 4. Scale Factor (FP8模式) 校验 ──
        //   sf: [num_tokens, num_sf_packs], 类型为 sf_pack_t (packed UE8M0x4)
        //   sf_token_stride / sf_hidden_stride: 支持非连续布局 (如TMA对齐列主序)
        int num_sf_packs = 0;
        void* sf_ptr = nullptr;
        int sf_token_stride = 0, sf_hidden_stride = 0;
        if (sf.has_value()) {
            // SF must be FP32 or packed UE8M0x4
            const auto [num_tokens_, num_sf_packs_] = get_shape<2>(sf.value());
            EP_HOST_ASSERT(num_tokens == num_tokens_);
            EP_HOST_ASSERT(sf->is_cuda());
            EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));
            num_sf_packs = num_sf_packs_;
            sf_ptr = sf->data_ptr();
            sf_token_stride = sf->stride(0);
            sf_hidden_stride = sf->stride(1);
        }

        // ── 5. Top-k 索引 + 权重校验 ──
        //   topk_idx: [num_tokens, num_topk], int类型, 每个lane选择哪个expert
        //   topk_weights: (可选, 训练backward) [num_tokens, num_topk], float
        const auto [num_tokens_, num_topk] = get_shape<2>(topk_idx);
        EP_HOST_ASSERT(num_tokens == num_tokens_);
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());

        // Weights are optional for training backward
        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            const auto [num_tokens__, num_topk_] = get_shape<2>(topk_weights.value());
            EP_HOST_ASSERT(num_tokens == num_tokens__);
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // ── 6. Expert接收计数器校验 ──
        //   cumulative_local_expert_recv_stats: (可选) [num_local_experts]
        //   用于 MoE 负载均衡 loss 的累计统计
        int* cumulative_local_expert_recv_stats_ptr = nullptr;
        if (cumulative_local_expert_recv_stats.has_value()) {
            const auto [num_local_experts_] = get_shape<1>(cumulative_local_expert_recv_stats.value());
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda() and
                           cumulative_local_expert_recv_stats->is_contiguous());
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
        }

        // ── 7. 流控制 prologue ──
        //   确保计算流与通信流的正确同步关系
        //   所有后续张量分配必须在此之后 (避免流竞争)
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);

        // ── 8. 分配 prefix sum 张量 ──
        //
        //   psum_num_recv_tokens_per_expert: [num_local_experts+1] 或 [num_local_experts]
        //     expand模式: exclusive prefix sum (长度=num_local_experts, 同时作为atomic counter)
        //     非expand模式: inclusive prefix sum (长度=num_local_experts)
        //     初始分配 num_local_experts+1 是因为 kernel 写入 inclusive + exclusive 两种
        //
        //   psum_num_recv_tokens_per_scaleup_rank: [num_scaleup_ranks]
        //     inclusive prefix sum: 每个 scaleup rank 收到的 token 累计
        //     供 combine 阶段定位 buffer 边界 (也作为 dispatch handle)
        EP_HOST_ASSERT(num_experts % nccl_context->num_ranks == 0);
        auto psum_num_recv_tokens_per_expert = cached_psum_num_recv_tokens_per_expert.value_or(torch::Tensor());
        if (cached_mode) {
            const auto& [num_local_experts_] = get_shape<1>(psum_num_recv_tokens_per_expert);
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.is_cuda() and psum_num_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.scalar_type() == torch::kInt);
        } else {
            // NOTES: for expand mode, the input is exclusive prefix sum, while for non-expand, it is inclusive
            psum_num_recv_tokens_per_expert = torch::empty(
                {num_local_experts + 1}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }

        // The prefix sum tensor of number of received tokens from each rank
        // Will also be used in combine as the dispatch handle
        auto psum_num_recv_tokens_per_scaleup_rank = cached_psum_num_recv_tokens_per_scaleup_rank.value_or(torch::Tensor());
        if (cached_mode) {
            const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
            EP_HOST_ASSERT(num_scaleup_ranks == nccl_context->num_scaleup_ranks);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        } else {
            psum_num_recv_tokens_per_scaleup_rank = torch::empty(
                {nccl_context->num_scaleup_ranks}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }

        // ── 9. 混合模式: 计算 channel 数 ──
        //
        //   channel = SM × channel_per_SM, 是数据并行的基本调度单位
        //   channel_per_SM 受限于:
        //     a) dispatch token 在 smem 中的大小 → (smem - notify区) / dispatch_token_size
        //     b) combine token 在 smem 中的大小 → smem / combine_token_size
        //     c) warp 种类数 (scaleup + forward = 2种) → channels_per_sm / 2
        //     d) 最大限制 kNumMaxChannelsPerSM=8
        //   prefer_overlap_with_compute=false → 限制最多4 (留SM给计算)
        int num_channels_per_sm = 1, num_channels = 1;
        const int num_smem_bytes = jit::device_runtime->get_num_smem_bytes();
        if (nccl_context->num_scaleout_ranks > 1) {
            const auto dispatch_token_layout = get_dispatch_token_layout(hidden, x.element_size(), num_sf_packs, num_topk);
            const auto combine_token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
            EP_HOST_ASSERT(num_sms <= kNumMaxSMs);
            num_channels_per_sm = std::min<int>(
                (num_smem_bytes - get_num_notify_smem_bytes(nccl_context->num_ranks, num_experts)) / dispatch_token_layout.get_num_bytes<true>(),
                32 - kNumNotifyWarps);
            num_channels_per_sm = std::min<int>(
                num_smem_bytes / combine_token_layout.get_num_bytes<true>(),
                num_channels_per_sm);
            num_channels_per_sm = std::min<int>(
                /* 2 kinds of warps */ num_channels_per_sm / 2, kNumMaxChannelsPerSM);
            if (not prefer_overlap_with_compute)
                num_channels_per_sm = std::min<int>(num_channels_per_sm, 4);
            num_channels = num_sms * num_channels_per_sm;
            if (get_env<int>("EP_BUFFER_DEBUG"))
                printf("Elastic buffer uses %d channels per SM\n", num_channels_per_sm);
        }

        // ── 10. 非混合模式: 分配 dst_buffer_slot_idx ──
        //   形状: [num_tokens, num_topk]
        //   含义: 每个 token 的每个 top-k 在目标 rank buffer 中的 slot 编号
        //   确定性模式: 先运行 prologue kernel 预计算 slot
        //   非确定性: 直接分配, 由 dispatch kernel 内部 atomicAdd 分配
        std::optional<torch::Tensor> deterministic_rank_count_buffer = std::nullopt;
        auto dst_buffer_slot_idx = cached_dst_buffer_slot_idx.value_or(torch::Tensor());
        if (nccl_context->num_scaleout_ranks == 1) {
            if (cached_mode) {
                const auto [num_tokens__, num_topk_] = get_shape<2>(dst_buffer_slot_idx);
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk_);
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            } else if (deterministic) {
                const auto prologue_num_sms = jit::device_runtime->get_num_sms();

                // Allocate new tensors
                deterministic_rank_count_buffer = torch::empty(
                    {prologue_num_sms, nccl_context->num_scaleup_ranks},
                    torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
                dst_buffer_slot_idx = torch::empty(
                    {num_tokens, num_topk}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));

                // Launch a kernel to preprocess the destination slot indices
                launch_dispatch_deterministic_prologue(topk_idx.data_ptr<topk_idx_t>(),
                                                       deterministic_rank_count_buffer->data_ptr<int>(),
                                                       dst_buffer_slot_idx.data_ptr<int>(),
                                                       num_tokens, num_max_tokens_per_rank,
                                                       num_experts, num_topk,
                                                       nccl_context->scaleup_rank_idx, nccl_context->num_scaleup_ranks,
                                                       prologue_num_sms,
                                                       jit::device_runtime->get_num_smem_bytes(),
                                                       comm_stream);
            } else {
                // Allocate a new tensor
                dst_buffer_slot_idx = torch::empty(
                    {num_tokens, num_topk}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
            }
        }

        // ── 11. 混合模式: 分配元数据张量 ──
        //
        //   dst_buffer_slot_idx: [num_channels, num_scaleout_ranks, num_max_tokens_per_channel, num_topk]
        //     每 channel × 每 scaleout peer × 每 token × 每 top-k 的目标 slot
        //
        //   token_metadata_at_forward: [num_channels, num_max_forwarded_tokens, num_forward_metadata_dims]
        //     num_forward_metadata_dims = 2 + num_topk * 2:
        //       [0]: src_token_global_idx (token 在原始 rank 中的全局索引)
        //       [1]: is_token_last_in_chunk (是否是该 scaleout chunk 的最后一个 token)
        //       [2..2+num_topk-1]: src_scaleup_rank_idx[k] (每个 top-k 的来源 scaleup rank)
        //       [2+num_topk..2+2*num_topk-1]: src_slot_idx[k] (每个 top-k 在 scaleup buffer 的 slot)
        //     供 combine forward warp 重放
        //
        //   channel_linked_list: [num_channels, num_scaleout_ranks*num_max_tokens_per_channel+1, num_scaleup_ranks]
        //     每 channel × 每 token 位置 × 每 scaleup rank 的链表指针
        //     [c, 0, r] = 链表头, [c, i, r] = 下一个节点 (0=终止)
        //     供 combine scaleup warp 遍历
        std::optional<torch::Tensor> token_metadata_at_forward, channel_linked_list;
        int *token_metadata_at_forward_ptr = nullptr, *channel_linked_list_ptr = nullptr;
        if (nccl_context->num_scaleout_ranks > 1) {
            EP_HOST_ASSERT(not deterministic);

            // The token destination slot idx during forward
            // `[i, j, k, l]` means: from channel i from scale-out peer k, the j-th token's index in the l-th rank buffer
            // NOTES: Used primarily for cached mode
            // TODO: May make it a linked list to remove the redundant info in `token_metadata_at_forward`
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);
            if (cached_mode) {
                const auto [num_channels_, num_scaleout_ranks_, num_max_tokens_per_channel_, num_topk_] =
                    get_shape<4>(dst_buffer_slot_idx);
                EP_HOST_ASSERT(num_channels == num_channels_ and nccl_context->num_scaleout_ranks == num_scaleout_ranks_ and
                               num_max_tokens_per_channel == num_max_tokens_per_channel_ and num_topk == num_topk_);
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            } else {
                dst_buffer_slot_idx = torch::empty(
                    {num_channels, nccl_context->num_scaleout_ranks, num_max_tokens_per_channel, num_topk},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }

            // The token metadata during forward
            // `[i, j]` means: in channel i, the j-th forwarded token's metadata
            // Info contains:
            //   - Scaleout rank index and source token index in the original rank (0)
            //   - Whether the token is the last one in the chunk (1)
            //   - cached top-k scaleup peer indices (top-k)
            //   - each selections' destination slot indices (top-k)
            const auto num_max_forwarded_tokens = nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1;
            const auto num_forward_metadata_dims = 2 + num_topk * 2;
            if (cached_mode) {
                token_metadata_at_forward = cached_token_metadata_at_forward;
                const auto [num_channels_, num_max_forwarded_tokens_, num_forward_metadata_dims_] = get_shape<3>(token_metadata_at_forward.value());
                EP_HOST_ASSERT(num_channels == num_channels_ and num_max_forwarded_tokens == num_max_forwarded_tokens_
                               and num_forward_metadata_dims == num_forward_metadata_dims_);
                EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());
                EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);
            } else {
                token_metadata_at_forward = torch::empty(
                    {num_channels, num_max_forwarded_tokens, num_forward_metadata_dims},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();

            // Per-scaleup-peer-per-channel linked list
            // `[i, j, k]` means: from channel i from scaleup peer k, the j-th token's index in the combine's input
            if (cached_mode) {
                channel_linked_list = cached_channel_linked_list;
                const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());
                channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
                EP_HOST_ASSERT(num_channels == num_channels__);
                EP_HOST_ASSERT(d1_ == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
                EP_HOST_ASSERT(d2_ == nccl_context->num_scaleup_ranks);
                EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());
                EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
            } else {
                channel_linked_list = torch::empty(
                    // Index 0 of the list means the starting item
                    {num_channels,
                    nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1,
                    nccl_context->num_scaleup_ranks},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
        }

        // ── 12. 拷贝 topk_idx (防止用户在 combine 前修改) ──
        auto copied_topk_idx = std::optional<torch::Tensor>();
        topk_idx_t* copied_topk_idx_ptr = nullptr;
        if (do_handle_copy and not cached_mode) {
            copied_topk_idx = torch::empty_like(topk_idx);
            copied_topk_idx_ptr = copied_topk_idx->data_ptr<topk_idx_t>();
        }

        // ── 13. 校验 buffer 大小是否足够 ──
        EP_HOST_ASSERT(get_dispatch_buffer_size(
                       num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, x.element_size(),
                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                       nccl_context->is_scaleup_nvlink) <= num_buffer_bytes);

        // ── 14. 清零 host workspace ──
        //   host_workspace: CPU 可见的映射内存, GPU kernel 通过 st.release.sys 写入
        //   清零 rank_count 和 expert_count, 为本轮 dispatch 准备
        //   atomic_thread_fence(seq_cst): 确保 CPU 写入在 GPU 读取前可见
        const auto host_workspace_layout = layout::WorkspaceLayout(
            host_workspace,
            nccl_context->num_scaleout_ranks,
            nccl_context->num_scaleup_ranks,
            num_experts);
        std::fill_n(host_workspace_layout.get_scaleup_rank_count_ptr<false>(), nccl_context->num_scaleup_ranks, 0);
        std::fill_n(host_workspace_layout.get_scaleup_expert_count_ptr<false>(), num_local_experts, 0);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        // ── 15. 启动主 dispatch kernel ──
        //   launch_dispatch 内部根据 num_scaleout_ranks 选择:
        //     == 1 → dispatch_impl (单节点, NVLink)
        //     >  1 → hybrid_dispatch_impl (混合, NVLink + RDMA)
        //   kernel 写入对端 buffer + 更新 host_workspace 的 rank/expert 计数
        EP_HOST_ASSERT(num_sms <= jit::device_runtime->get_num_sms());
        launch_dispatch(x.data_ptr(), sf_ptr,
                        topk_idx.data_ptr<topk_idx_t>(), topk_weights_ptr,
                        copied_topk_idx_ptr,
                        cumulative_local_expert_recv_stats_ptr,
                        psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                        psum_num_recv_tokens_per_expert.data_ptr<int>(),
                        dst_buffer_slot_idx.data_ptr<int>(),
                        token_metadata_at_forward_ptr,
                        num_tokens, num_max_tokens_per_rank,
                        hidden, x.element_size(),
                        num_sf_packs, sf_token_stride, sf_hidden_stride,
                        num_experts, num_topk, expert_alignment,
                        nccl_context->dev_comm, nccl_context->window,
                        buffer,
                        workspace, mapped_host_workspace,
                        nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                        nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                        nccl_context->is_scaleup_nvlink,
                        num_sms, num_channels_per_sm,
                        num_smem_bytes,
                        num_qps, num_gpu_timeout_cycles,
                        cached_mode, deterministic, do_cpu_sync,
                        comm_stream);

        // ── 16. 获取接收 token 计数 ──
        //   三种模式:
        //   a) cached_mode: 直接用缓存值 (跳过 CPU 同步)
        //   b) do_cpu_sync: CPU 轮询 host_workspace, 等待 GPU kernel 写入 rank/expert 计数
        //      encode_decode_positive: GPU 写入时编码正值, CPU 解码并检查是否就绪
        //      is_decoded_positive_ready: 值 > 0 表示该 rank/expert 已完成
        //   c) 默认: 分配最坏情况 (num_max_tokens_per_rank * num_ranks)
        int num_recv_tokens = 0, num_expanded_tokens = 0;
        int counter_scaleup_rank_idx = 0, counter_local_expert_idx = 0;
        std::vector<int> num_recv_tokens_per_expert_list;

        // Assign these values according to modes
        if (cached_mode) {
            EP_HOST_ASSERT(not do_expand and "Cannot do expand with cached mode");
            EP_HOST_ASSERT(not do_cpu_sync and "Cannot do CPU sync with cached mode");
            num_recv_tokens = cached_num_recv_tokens.value();
            num_recv_tokens_per_expert_list = cached_num_recv_tokens_per_expert_list.value();
        } else if (do_cpu_sync) {
            // ── CPU 同步模式: 轮询 host_workspace 等待所有数据就绪 ──
            //   host_workspace 由 GPU kernel 通过 st.release.sys 写入 (release 语义保证可见性)
            //   CPU 端循环读取, 直到所有 scaleup rank 和 expert 的计数都已就绪
            //   超时则抛异常
            const auto start_cpu_time = std::chrono::high_resolution_clock::now();
            while (true) {
                bool ready = true;

                // Read number of received tokens from each scaleup rank
                while (counter_scaleup_rank_idx < nccl_context->num_scaleup_ranks and ready) {
                    const auto count = math::encode_decode_positive(
                        host_workspace_layout.get_scaleup_rank_count_ptr<false>()[counter_scaleup_rank_idx]);
                    if ((ready = math::is_decoded_positive_ready(count))) {
                        num_recv_tokens += count;
                        ++ counter_scaleup_rank_idx;
                    }
                }

                // Read expert counts
                while (counter_local_expert_idx < num_local_experts and ready) {
                    const auto count = math::encode_decode_positive(
                        host_workspace_layout.get_scaleup_expert_count_ptr<false>()[counter_local_expert_idx]);
                    if ((ready = math::is_decoded_positive_ready(count))) {
                        num_recv_tokens_per_expert_list.push_back(count);
                        num_expanded_tokens += count;
                        ++ counter_local_expert_idx;
                    }
                }

                // Ready and do next steps
                const auto get_buffer_info = [&]() {
                    std::stringstream ss;
                    ss << "CPU side received count (scaleup: " << nccl_context->scaleup_rank_idx << "): ";
                    for (int i = 0; i < nccl_context->num_scaleup_ranks + num_local_experts; ++ i) {
                        ss << host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i];
                        ss << (i == nccl_context->num_scaleup_ranks - 1 ? " # ": " ");
                    }
                    return ss.str();
                };
                if (ready) {
                    if (get_env<int>("EP_BUFFER_DEBUG"))
                        printf("%s\n", get_buffer_info().c_str());
                    break;
                }

                // Timeout checks
                const auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - start_cpu_time).count() > num_cpu_timeout_secs)
                    throw EPExceptionWithLineInfo("Dispatch CPU wait", get_buffer_info());
            }
        } else {
            // ── 非缓存非同步模式: 按最坏情况分配 ──
            //   无法提前知道实际接收多少 token, 按 max 分配
            //   num_recv_tokens = 所有 rank 各发 max 个
            //   num_expanded_tokens = 所有 rank × min(topk, local_experts) × max_tokens + 对齐
            num_recv_tokens = num_max_tokens_per_rank * nccl_context->num_ranks;
            num_expanded_tokens = nccl_context->num_ranks * num_max_tokens_per_rank * std::min(num_topk, num_local_experts);
            num_expanded_tokens += (expert_alignment - 1) * num_local_experts;
            num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment);
        }

        // ── 17. 分配输出张量 ──
        //
        //   recv_x: [num_allocated_tokens, hidden] 接收到的 hidden states
        //   recv_sf: (FP8) [num_allocated_tokens, num_sf_packs] scale factors
        //     两种布局:
        //       默认: 行主序 (token_stride=num_sf_packs, hidden_stride=1)
        //       TMA对齐列主序: (token_stride=1, hidden_stride=align(tokens, kNumAlignedSFPacks))
        //         → 适配后续 GEMM 的 TMA 加载
        //   recv_topk_idx: (非expand) [num_allocated_tokens, num_topk]
        //   recv_topk_weights: expand→[num_allocated_tokens], 非→[num_allocated_tokens, num_topk]
        //   recv_src_metadata: [num_recv_tokens, num_topk+2]
        //     [0]: src_token_global_idx, [1]: src_rank_idx, [2..+num_topk]: buffer slot indices
        //     供 combine 阶段定位 token 来源
        const auto num_allocated_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;
        auto recv_x = torch::empty({num_allocated_tokens, hidden}, x.options());
        auto recv_sf = std::optional<torch::Tensor>();
        auto recv_topk_idx = std::optional<torch::Tensor>();
        auto recv_topk_weights = std::optional<torch::Tensor>();
        auto recv_src_metadata = torch::empty(
            {num_recv_tokens, num_topk + 2},
            torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));

        // Optional tensors
        void* recv_sf_ptr = nullptr;
        topk_idx_t* recv_topk_idx_ptr = nullptr;
        float* recv_topk_weights_ptr = nullptr;
        int recv_sf_token_stride = 0, recv_sf_hidden_stride = 0;
        if (sf.has_value()) {
            if (not use_tma_aligned_col_major_sf) {
                recv_sf_token_stride = num_sf_packs, recv_sf_hidden_stride = 1;
            } else {
                // TMA-aligned layout for the next GEMM input
                recv_sf_token_stride = 1, recv_sf_hidden_stride = math::align(num_allocated_tokens, kNumAlignedSFPacks);
            }
            recv_sf = torch::empty_strided({num_allocated_tokens, num_sf_packs},
                                           {recv_sf_token_stride, recv_sf_hidden_stride},
                                           sf->options());
            recv_sf_ptr = recv_sf->data_ptr();
        }
        if (not do_expand) {
            recv_topk_idx = torch::empty({num_allocated_tokens, num_topk}, topk_idx.options());
            recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        }
        if (topk_weights.has_value()) {
            recv_topk_weights = do_expand ?
                torch::empty({num_allocated_tokens}, topk_weights->options()) :
                torch::empty({num_allocated_tokens, num_topk}, topk_weights->options());
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
        }

        // ── 18. 处理 prefix sum 切片 ──
        //   expand模式: 使用 exclusive prefix sum (长度=num_local_experts, 作为atomic counter)
        //   非expand非缓存: 使用 inclusive prefix sum (切片去掉第0个元素)
        //   缓存模式: 已由缓存提供, 不动
        if (do_expand) {
            // Slice and exclusive part and do atomic additions into inclusive
            EP_HOST_ASSERT(not cached_mode);
            psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 0, num_local_experts);
        } else if (not cached_mode) {
            // Slice the inclusive part (and will not be used in the epilogue)
            psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 1, num_local_experts + 1);
        }
        EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.size(0) == num_local_experts);

        // ── 19. 启动 dispatch copy epilogue kernel (PDL) ──
        //   从 GPU buffer 拷贝到输出张量 (recv_x, recv_sf, recv_topk_idx, ...)
        //   使用全量 SM (非限制 num_sms, 因为是纯拷贝无通信)
        //   通过 PDL (Programmatic Dependent Launch) 与主 dispatch kernel 流水重叠
        //   stream_control_before_epilogue: 确保在 epilogue 前同步
        stream_control_before_epilogue(previous_event_before_epilogue);
        launch_dispatch_copy_epilogue(buffer, workspace,
                                      psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                                      psum_num_recv_tokens_per_expert.data_ptr<int>(),
                                      recv_x.data_ptr(), recv_sf_ptr,
                                      recv_topk_idx_ptr, recv_topk_weights_ptr,
                                      recv_src_metadata.data_ptr<int>(),
                                      channel_linked_list_ptr,
                                      num_recv_tokens, num_max_tokens_per_rank,
                                      num_hidden_bytes,
                                      num_sf_packs, recv_sf_token_stride, recv_sf_hidden_stride,
                                      num_experts, num_topk,
                                      nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                                      nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                                      jit::device_runtime->get_num_sms(),
                                      jit::device_runtime->get_num_smem_bytes(),
                                      num_channels,
                                      do_expand, cached_mode,
                                      comm_stream);

        // Stream control
        const auto event = stream_control_epilogue(
            {x, sf, topk_idx, topk_weights,
             recv_x, recv_sf, recv_topk_idx, recv_topk_weights,
             cumulative_local_expert_recv_stats,
             copied_topk_idx,
             psum_num_recv_tokens_per_scaleup_rank,
             psum_num_recv_tokens_per_expert,
             recv_src_metadata,
             deterministic_rank_count_buffer,
             dst_buffer_slot_idx,
             token_metadata_at_forward,
             channel_linked_list},
            compute_stream,
            allocate_on_comm_stream, async_with_compute_stream);

        return {recv_x, recv_sf,
                recv_topk_idx, recv_topk_weights,
                copied_topk_idx,
                num_recv_tokens_per_expert_list,
                psum_num_recv_tokens_per_scaleup_rank,
                psum_num_recv_tokens_per_expert,
                recv_src_metadata,
                dst_buffer_slot_idx,
                token_metadata_at_forward,
                channel_linked_list,
                event};
    }

    // ============================================================================================
    //  ElasticBuffer::combine —— Token 合并入口
    // ============================================================================================
    //
    //  【总览】
    //  Combine 是 Dispatch 的逆过程: 将各 expert 处理后的 hidden states 按 top-k 权重加权求和,
    //  还原为完整的 token 表示。
    //
    //  【整体流程】
    //
    //  ┌──────────────────────────────────────────────────────────────────────────────────┐
    //  │ 1. 参数校验 + 张量校验                                                          │
    //  │ 2. launch_combine / launch_hybrid_combine (主通信 kernel, 阶段1)                  │
    //  │    → 各 rank 将 hidden states 推送到对端, 局部 reduce                             │
    //  │    → 数据写入 reduce_buffer                                                      │
    //  │ 3. 分配输出张量 (combined_x, combined_topk_weights)                              │
    //  │ 4. launch_combine_reduce_epilogue (PDL epilogue, 阶段2)                          │
    //  │    → 从 reduce_buffer 读取, 最终 top-k 加权求和 + 可选 SwiGLU bias               │
    //  │    → 写回 combined_x                                                             │
    //  │ 5. stream_control_epilogue (流同步)                                             │
    //  └──────────────────────────────────────────────────────────────────────────────────┘
    //
    //  【返回值】
    //  ┌───────────────────────────────────────────────────────────────────┐
    //  │ combined_x          : [num_combined_tokens, hidden] 合并后的token  │
    //  │ combined_topk_weights: (可选) [num_combined_tokens, num_topk] 权重 │
    //  │ event               : 流同步事件                                  │
    //  └───────────────────────────────────────────────────────────────────┘
    //
    //  【入参详解】
    //  x                              : [num_tokens, hidden] expert处理后的hidden (BF16)
    //  topk_weights                   : (可选) [num_tokens, num_topk] top-k权重 (非展开模式)
    //  bias_0, bias_1                 : (可选) SwiGLU bias [num_combined_tokens, hidden]
    //  src_metadata                   : [num_reduced_tokens, num_topk+2] dispatch阶段写入的源元数据
    //  combined_topk_idx             : [num_combined_tokens, num_topk] 每个token的top-k expert索引
    //  psum_num_recv_tokens_per_scaleup_rank: [num_scaleup_ranks] 来自dispatch的scaleup前缀和
    //  token_metadata_at_forward      : (混合模式) forward元数据
    //  channel_linked_list            : (混合模式) 通道链表
    //  num_experts                    : 总expert数
    //  num_max_tokens_per_rank        : 每rank最大token容量
    //  num_sms, num_qps               : SM数和QP数
    //  previous_event / previous_event_before_epilogue: 流控制事件
    //  async_with_compute_stream / allocate_on_comm_stream: 流控制选项
    //  use_expanded_layout            : 是否展开 (每个top-k占独立行, 配合SwiGLU)
    //
    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    combine(const torch::Tensor& x,
            const std::optional<torch::Tensor>& topk_weights,
            const std::optional<torch::Tensor>& bias_0,
            const std::optional<torch::Tensor>& bias_1,
            const torch::Tensor& src_metadata,
            const torch::Tensor& combined_topk_idx,
            const torch::Tensor& psum_num_recv_tokens_per_scaleup_rank,
            const std::optional<torch::Tensor>& token_metadata_at_forward,
            const std::optional<torch::Tensor>& channel_linked_list,
            const int& num_experts,
            const int& num_max_tokens_per_rank,
            const int& num_sms, const int& num_qps,
            const std::optional<EventHandle>& previous_event,
            const std::optional<EventHandle>& previous_event_before_epilogue,
            const bool& async_with_compute_stream,
            const bool& allocate_on_comm_stream,
            const bool& use_expanded_layout) const {
        // ── 1. 基本校验 ──
        EP_HOST_ASSERT(num_sms > 0);

        // ── 2. 输入张量校验 ──
        //   x: [num_tokens, hidden], BF16, CUDA, contiguous
        //   hidden 必须 int4 对齐 (16B)
        const auto [num_tokens, hidden] = get_shape<2>(x);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);

        // ── 3. dispatch handle 校验 ──
        //   combined_topk_idx: [num_combined_tokens, num_topk]
        //     num_combined_tokens = 本 rank 需要最终 combine 的 token 数
        //     来自 dispatch 阶段的 topk_idx (经过 expert 映射后的本地索引)
        //   psum_num_recv_tokens_per_scaleup_rank: [num_scaleup_ranks]
        //     dispatch 阶段产出的 inclusive prefix sum
        const auto [num_combined_tokens, num_topk] = get_shape<2>(combined_topk_idx);
        const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
        EP_HOST_ASSERT(combined_topk_idx.is_cuda() and combined_topk_idx.is_contiguous());
        EP_HOST_ASSERT(combined_topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(num_scaleup_ranks == nccl_context->num_scaleup_ranks);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(num_combined_tokens <= num_max_tokens_per_rank);

        // ── 4. src_metadata 校验 ──
        //   形状: [num_reduced_tokens, num_topk + 2]
        //   use_expanded_layout=true:  num_reduced_tokens 可以 != num_tokens (每个 top-k 独立行)
        //   use_expanded_layout=false: num_reduced_tokens 必须 == num_tokens
        const auto [num_reduced_tokens, num_topk_p2] = get_shape<2>(src_metadata);
        EP_HOST_ASSERT(num_reduced_tokens == (use_expanded_layout ? num_reduced_tokens : num_tokens));
        EP_HOST_ASSERT(num_topk_p2 == num_topk + 2);
        EP_HOST_ASSERT(src_metadata.is_cuda() and src_metadata.is_contiguous());
        EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);

        // ── 5. 可选张量校验 ──
        //   use_expanded_layout=true → 不能有 topk_weights (reduce 在 SwiGLU 内完成)
        //   bias: [num_combined_tokens, hidden], BF16, contiguous
        if (use_expanded_layout) {
            EP_HOST_ASSERT(not topk_weights.has_value());
        } else if (topk_weights.has_value()) {
            const auto [num_tokens__, num_topk__] = get_shape<2>(topk_weights.value());
            EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk__);
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);
        }

        const auto bias_opts = std::vector({bias_0, bias_1});
        void* bias_ptrs[2] = {nullptr, nullptr};
        for (int i = 0; i < 2; ++ i) {
            if (bias_opts[i].has_value()) {
                auto bias = bias_opts[i].value();
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
                bias_ptrs[i] = bias.data_ptr();
            }
        }

        // ── 6. 流控制 prologue ──
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);

        // ── 7. 校验 buffer 大小 ──
        EP_HOST_ASSERT(get_combine_buffer_size(num_max_tokens_per_rank, hidden, num_topk,
                                               nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                                               nccl_context->is_scaleup_nvlink, allow_multiple_reduction) <= num_buffer_bytes);

        // ── 8. 混合模式: 从 dispatch handle 恢复 channel 元数据 ──
        //   token_metadata_at_forward: [num_channels, num_max_forwarded_tokens, 2+num_topk*2]
        //   channel_linked_list: [num_channels, num_scaleout_ranks*num_max_tokens_per_channel+1, num_scaleup_ranks]
        int num_channels = 1;
        int* token_metadata_at_forward_ptr = nullptr;
        int* channel_linked_list_ptr = nullptr;
        if (nccl_context->num_scaleout_ranks > 1) {
            // The token metadata during forward
            const auto [num_channels_, d1, d2] = get_shape<3>(token_metadata_at_forward.value());
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels_);
            num_channels = num_channels_;
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();
            EP_HOST_ASSERT(d1 == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
            EP_HOST_ASSERT(d2 == 2 + num_topk * 2);
            EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());
            EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);

            // Per-scaleup-peer-per-channel linked list
            const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
            EP_HOST_ASSERT(num_channels == num_channels__);
            EP_HOST_ASSERT(d1_ == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
            EP_HOST_ASSERT(d2_ == nccl_context->num_scaleup_ranks);
            EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());
            EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
        }

        // ── 9. 启动主 combine kernel (阶段1) ──
        //   launch_combine 内部根据 num_scaleout_ranks 选择:
        //     == 1 → combine_impl (单节点, NVLink)
        //     >  1 → hybrid_combine_impl (混合, NVLink + RDMA)
        //   返回 reduce_buffer 指针 (供 epilogue 读取)
        const auto reduce_buffer = launch_combine(
            x.data_ptr(),
            topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
            src_metadata.data_ptr<int>(),
            psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
            token_metadata_at_forward_ptr,
            channel_linked_list_ptr,
            nccl_context->dev_comm, nccl_context->window,
            buffer, workspace,
            num_reduced_tokens, num_max_tokens_per_rank,
            hidden, num_experts, num_topk,
            num_qps, num_gpu_timeout_cycles,
            nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
            nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
            nccl_context->is_scaleup_nvlink,
            num_sms, jit::device_runtime->get_num_smem_bytes(),
            num_channels,
            use_expanded_layout, allow_multiple_reduction,
            comm_stream);

        // ── 10. 分配输出张量 ──
        //   combined_x: [num_combined_tokens, hidden] 最终合并后的 hidden states
        //   combined_topk_weights: (可选) [num_combined_tokens, num_topk] top-k 权重
        auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
        auto combined_topk_weights = std::optional<torch::Tensor>();
        float* combined_topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
            combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
        }

        // ── 11. 启动 combine reduce epilogue kernel (阶段2, PDL) ──
        //   从 reduce_buffer 读取多个 rank 的数据, 做 top-k 加权求和
        //   可选 SwiGLU bias, TMA store 写回 combined_x
        //   通过 PDL 与主 combine kernel 流水重叠
        //   stream_control_before_epilogue: 确保在 epilogue 前同步
        stream_control_before_epilogue(previous_event_before_epilogue);
        launch_combine_reduce_epilogue(combined_x.data_ptr(),
                                       combined_topk_weights_ptr,
                                       combined_topk_idx.data_ptr<topk_idx_t>(),
                                       num_combined_tokens, num_max_tokens_per_rank,
                                       hidden,
                                       num_experts, num_topk,
                                       reduce_buffer,
                                       bias_ptrs[0], bias_ptrs[1],
                                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                                       nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                                       jit::device_runtime->get_num_sms(),
                                       jit::device_runtime->get_num_smem_bytes(),
                                       use_expanded_layout, allow_multiple_reduction,
                                       comm_stream);

        // Stream control
        const auto event = stream_control_epilogue(
            {x, topk_weights, bias_0, bias_1,
             src_metadata,
             combined_topk_idx,
             combined_x, combined_topk_weights,
             psum_num_recv_tokens_per_scaleup_rank,
             token_metadata_at_forward,
             channel_linked_list},
            compute_stream,
            allocate_on_comm_stream, async_with_compute_stream);
        return {combined_x, combined_topk_weights, event};
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool, int, int, int, int, bool>())
        .def("destroy", &ElasticBuffer::destroy)
        .def("get_comm_stream", &ElasticBuffer::get_comm_stream)
        .def("get_physical_domain_size", &ElasticBuffer::get_physical_domain_size)
        .def("get_logical_domain_size", &ElasticBuffer::get_logical_domain_size)
        .def("barrier", &ElasticBuffer::barrier)
        .def("engram_write", &ElasticBuffer::engram_write)
        .def("engram_fetch", &ElasticBuffer::engram_fetch)
        .def("pp_set_config", &ElasticBuffer::pp_set_config)
        .def("pp_send", &ElasticBuffer::pp_send)
        .def("pp_recv", &ElasticBuffer::pp_recv)
        .def("create_agrs_session", &ElasticBuffer::create_agrs_session)
        .def("destroy_agrs_session", &ElasticBuffer::destroy_agrs_session)
        .def("agrs_set_config", &ElasticBuffer::agrs_set_config)
        .def("agrs_get_inplace_tensor", &ElasticBuffer::agrs_get_inplace_tensor)
        .def("all_gather", &ElasticBuffer::all_gather)
        .def("dispatch", &ElasticBuffer::dispatch)
        .def("combine", &ElasticBuffer::combine);
    m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);

    // NCCL communicator handle
    m.def("get_local_nccl_unique_id", &nccl::get_local_unique_id);
    m.def("create_nccl_comm", &nccl::create_nccl_comm);
    m.def("destroy_nccl_comm", &nccl::destroy_nccl_comm);

    // Communication domain utilities
    m.def("get_physical_domain_size", &nccl::get_physical_domain_size);
    m.def("get_logical_domain_size", &nccl::get_logical_domain_size);
}

}  // namespace deep_ep
