#include "llama-moe-cache.h"

#include "llama-model.h"
#include "llama-impl.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>

// A layer is cacheable in either the merged (gate_up + down) or separate
// (gate + up + down) configuration, with no expert biases or per-expert scales.
// This covers Qwen3-MoE / Qwen3.5-MoE and similar and keeps the split path
// numerically equivalent to the baseline. Biased or scaled experts fall back to
// the uncached path.
//
// Fills `src` with the managed host expert tensors and returns the count (0 if
// the layer is not cacheable).
static int layer_expert_tensors(const llama_layer & l, ggml_tensor * src[LLAMA_MOE_MAX_ROLES]) {
    if (!l.ffn_down_exps) return 0;
    if (l.ffn_down_exps_b || l.ffn_up_exps_b || l.ffn_gate_exps_b || l.ffn_gate_up_exps_b) return 0;
    if (l.ffn_down_exps_s || l.ffn_up_exps_s || l.ffn_gate_exps_s) return 0;

    // experts must be host-resident (for the CPU->VRAM promotion copy to read a
    // valid host pointer, and for the cache to actually remove RAM traffic)
    if (!l.ffn_down_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_down_exps->buffer)) return 0;

    int n = 0;
    if (l.ffn_gate_up_exps) {
        // merged gate_up path
        src[n++] = l.ffn_gate_up_exps;
    } else if (l.ffn_gate_exps && l.ffn_up_exps) {
        // separate gate and up path
        src[n++] = l.ffn_gate_exps;
        src[n++] = l.ffn_up_exps;
    } else {
        return 0;
    }
    src[n++] = l.ffn_down_exps;
    return n;
}

llama_moe_cache::llama_moe_cache(const llama_model & model, size_t budget_bytes)
    : model(model), budget_bytes(budget_bytes) {

    if (budget_bytes == 0) {
        return;
    }

    // tuning overrides (also handy for testing on short runs)
    promote_bytes = 512ull << 20;
    if (const char * s = getenv("MOE_CACHE_INTERVAL"))   update_interval = std::max<int64_t>(1, atoll(s));
    if (const char * s = getenv("MOE_CACHE_DECAY"))      decay           = (float) atof(s);
    if (const char * s = getenv("MOE_CACHE_PROMOTE_MB")) promote_bytes   = (size_t) std::max<int64_t>(1, atoll(s)) << 20;

    // pick a GPU device / buffer type to host the cache
    ggml_backend_dev_t dev = nullptr;
    for (const auto & d : model.devices) {
        if (ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { dev = d.dev; break; }
    }
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    }
    if (dev == nullptr) {
        LLAMA_LOG_WARN("%s: no GPU device found; MoE expert cache disabled\n", __func__);
        return;
    }
    buft = ggml_backend_dev_buffer_type(dev);

    // collect cacheable layers
    std::vector<int> candidates;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        ggml_tensor * src[LLAMA_MOE_MAX_ROLES];
        if (layer_expert_tensors(model.layers[il], src) > 0) {
            candidates.push_back((int) il);
        }
    }
    if (candidates.empty()) {
        LLAMA_LOG_WARN("%s: no cacheable MoE layers (need host-resident, unbiased experts); cache disabled\n", __func__);
        return;
    }

    const size_t per_layer_budget = budget_bytes / candidates.size();
    const int    n_expert         = (int) model.hparams.n_expert;
    n_used = std::max(1, (int) model.hparams.n_expert_used);

    for (int il : candidates) {
        ggml_tensor * src[LLAMA_MOE_MAX_ROLES];
        const int n_roles = layer_expert_tensors(model.layers[il], src);

        size_t per_expert_bytes = 0;
        for (int r = 0; r < n_roles; ++r) per_expert_bytes += src[r]->nb[2];
        if (per_expert_bytes == 0) continue;

        // each layer also holds n_used zero-slots (routed misses), so budget those
        int K = (int) (per_layer_budget / per_expert_bytes) - n_used;
        K = std::min(K, n_expert);
        if (K <= 0) continue;

        // guarantee progress even with a tiny promotion budget
        promote_bytes = std::max(promote_bytes, per_expert_bytes);

        register_layer(il, K);
    }

    if (layers.empty()) {
        LLAMA_LOG_WARN("%s: MoE cache budget %zu MiB too small for any layer; cache disabled\n",
                       __func__, budget_bytes >> 20);
        return;
    }

    alloc_tensors();
}

llama_moe_cache::~llama_moe_cache() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx)    ggml_free(ctx);
}

void llama_moe_cache::register_layer(int il, int n_slots) {
    const int n_expert = (int) model.hparams.n_expert;

    llama_moe_cache_layer L;
    L.il       = il;
    L.n_expert = n_expert;
    L.n_slots  = n_slots;
    L.n_roles  = layer_expert_tensors(model.layers[il], L.src);

    L.counts.assign(n_expert, 0.0f);
    L.slot_of.assign(n_expert, -1);
    L.expert_in_slot.assign(n_slots, -1);

    layers.emplace(il, std::move(L));
}

void llama_moe_cache::alloc_tensors() {
    // metadata context: per layer -> up to 3 vram + 2 maps
    const size_t n_tensors = layers.size() * (LLAMA_MOE_MAX_ROLES + 2);
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_tensors + 8),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(ip);
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: failed to create ggml context for MoE cache\n", __func__);
        layers.clear();
        return;
    }

    for (auto & [il, L] : layers) {
        const int K = L.n_slots;
        for (int r = 0; r < L.n_roles; ++r) {
            ggml_tensor * s = L.src[r];
            // [ne0, ne1, K + n_used]; slots K..K+n_used-1 are distinct zero experts
            L.vram[r] = ggml_new_tensor_3d(ctx, s->type, s->ne[0], s->ne[1], K + n_used);
        }
        // shape [1, n_expert]: one lookup value per expert row, indexed on-graph
        // by get_rows(map, selected_experts)
        L.gpu_map = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, L.n_expert);
        L.cpu_map = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, L.n_expert);
    }

    buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        LLAMA_LOG_ERROR("%s: failed to allocate MoE cache VRAM buffer\n", __func__);
        ggml_free(ctx); ctx = nullptr;
        layers.clear();
        return;
    }
    used_bytes = ggml_backend_buffer_get_size(buffer);

    // initialise: zero every vram tensor fully (all zero-slots + any MMQ padding),
    // set maps to "all CPU"
    std::vector<char> zero;
    for (auto & [il, L] : layers) {
        for (int r = 0; r < L.n_roles; ++r) {
            const size_t nbytes = ggml_nbytes(L.vram[r]);
            if (zero.size() < nbytes) zero.assign(nbytes, 0);
            ggml_backend_tensor_set(L.vram[r], zero.data(), 0, nbytes);
        }
        upload_maps(L); // defaults from slot_of == all -1
    }

    log_summary();
}

void llama_moe_cache::promote(llama_moe_cache_layer & L, int expert, int slot) {
    for (int r = 0; r < L.n_roles; ++r) {
        ggml_tensor * s = L.src[r];
        ggml_tensor * d = L.vram[r];
        const size_t nb2 = s->nb[2];
        const char * src_ptr = (const char *) s->data + (size_t) expert * nb2;
        ggml_backend_tensor_set(d, src_ptr, (size_t) slot * d->nb[2], nb2);
    }
}

void llama_moe_cache::upload_maps(llama_moe_cache_layer & L) {
    std::vector<float>   gpu(L.n_expert);
    std::vector<int32_t> cpu(L.n_expert);
    for (int e = 0; e < L.n_expert; ++e) {
        const int slot = L.slot_of[e];
        const bool cached = slot >= 0;
        gpu[e] = cached ? (float) slot + 0.5f : -0.5f; // slot (+0.5 for step); miss handled on-graph
        cpu[e] = cached ? 0 : e;                       // dummy 0 for cached (masked out)
    }
    ggml_backend_tensor_set(L.gpu_map, gpu.data(), 0, gpu.size() * sizeof(float));
    ggml_backend_tensor_set(L.cpu_map, cpu.data(), 0, cpu.size() * sizeof(int32_t));
    L.maps_dirty = false;
}

void llama_moe_cache::observe(int il, const int32_t * ids, int64_t n_used, int64_t n_tokens) {
    auto it = layers.find(il);
    if (it == layers.end()) return;
    llama_moe_cache_layer & L = it->second;

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t u = 0; u < n_used; ++u) {
            const int32_t e = ids[t * n_used + u];
            if (e >= 0 && e < L.n_expert) L.counts[e] += 1.0f;
        }
    }

    // advance the global token clock on the first managed layer only
    if (il == layers.begin()->first) {
        tokens_since_update += n_tokens;
    }
}

void llama_moe_cache::update() {
    if (!enabled()) return;
    if (tokens_since_update < update_interval) return;
    tokens_since_update = 0;

    // bound the synchronous CPU->VRAM copy volume per update; what does not fit
    // is picked up by later updates (spreads the initial fill over many updates)
    size_t copy_budget = promote_bytes;

    for (auto & [il, L] : layers) {
        // decay counters (EMA)
        for (float & c : L.counts) c *= decay;

        const int K = L.n_slots;

        size_t per_expert_bytes = 0;
        for (int r = 0; r < L.n_roles; ++r) per_expert_bytes += L.src[r]->nb[2];

        // desired hot set: top-K experts by count (count > 0)
        const int topk = std::min<int>(K, L.n_expert);
        std::vector<int> order(L.n_expert);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                          [&](int a, int b) { return L.counts[a] > L.counts[b]; });

        std::vector<bool> want(L.n_expert, false);
        for (int i = 0; i < topk; ++i) {
            if (L.counts[order[i]] > 0.0f) want[order[i]] = true;
        }

        // fillable slots: free ones first, then unwanted residents, coldest first.
        // residents are evicted lazily, only when their slot is actually refilled,
        // so they keep serving hits while promotions wait on the copy budget.
        std::vector<int> avail;
        for (int slot = 0; slot < K; ++slot) {
            if (L.expert_in_slot[slot] < 0) avail.push_back(slot);
        }
        std::vector<int> evictable;
        for (int slot = 0; slot < K; ++slot) {
            const int e = L.expert_in_slot[slot];
            if (e >= 0 && !want[e]) evictable.push_back(slot);
        }
        std::sort(evictable.begin(), evictable.end(),
                  [&](int a, int b) { return L.counts[L.expert_in_slot[a]] < L.counts[L.expert_in_slot[b]]; });
        avail.insert(avail.end(), evictable.begin(), evictable.end());

        bool changed = false;

        // promote newly-wanted experts, hottest first
        size_t next = 0;
        for (int i = 0; i < topk && next < avail.size(); ++i) {
            const int e = order[i];
            if (!want[e] || L.slot_of[e] >= 0) continue;
            if (per_expert_bytes > copy_budget) break;

            const int slot = avail[next++];
            const int old  = L.expert_in_slot[slot];
            if (old >= 0) L.slot_of[old] = -1;

            promote(L, e, slot);
            L.slot_of[e] = slot;
            L.expert_in_slot[slot] = e;
            copy_budget -= per_expert_bytes;
            changed = true;
        }

        if (changed) {
            upload_maps(L);
            n_promotions++;
        }
    }

    n_updates++;
    if (getenv("MOE_CACHE_VERBOSE")) {
        // rough occupancy of the first managed layer
        const llama_moe_cache_layer & L0 = layers.begin()->second;
        int filled = 0;
        for (int s = 0; s < L0.n_slots; ++s) filled += (L0.expert_in_slot[s] >= 0);
        LLAMA_LOG_INFO("%s: update #%lld: %lld promotions so far; layer %d cache %d/%d slots filled\n",
                       __func__, (long long) n_updates, (long long) n_promotions,
                       L0.il, filled, L0.n_slots);
    }
}

void llama_moe_cache::log_summary() const {
    int total_slots = 0;
    for (const auto & [il, L] : layers) total_slots += L.n_slots;
    LLAMA_LOG_INFO("%s: MoE expert cache active: %zu layers, %d slots total, %.2f MiB VRAM\n",
                   __func__, layers.size(), total_slots, used_bytes / (1024.0 * 1024.0));
}
