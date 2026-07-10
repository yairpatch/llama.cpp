#pragma once

// Dynamic VRAM expert cache for MoE CPU-offload.
//
// When MoE expert weights are kept in host RAM (via -ot / --n-cpu-moe), token
// generation is bound by the RAM->CPU bandwidth needed to read the selected
// experts every token. Expert activation is heavily skewed and drifts slowly,
// so a small VRAM-resident cache of the hottest experts (chosen online from
// decayed activation counters) can serve a large fraction of expert reads from
// VRAM instead of RAM.
//
// This object owns, per managed layer:
//   - K+n VRAM slots per expert role (gate/up/down); the slots past K are zero
//     experts used as no-op miss targets (n = 1 with MOE_CACHE_DUP_IDS, else
//     one per miss position).
//   - two per-expert lookup maps (updated between decodes) that build_moe_ffn
//     reads on-graph to route each selected expert to VRAM (if cached) or CPU.
//   - decayed per-expert activation counters used to pick the hot set.
//
// Promotion is synchronous: between decodes, newly hot experts are copied
// CPU->VRAM and the maps are refreshed. The host copy stays authoritative, so
// eviction is free and results are bit-identical to the uncached path.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <vector>

struct llama_model;

// A layer has either 2 expert tensors (merged gate_up + down) or 3 (gate + up +
// down). We key VRAM copies by the host source-tensor pointer so both layouts
// work uniformly; build_moe_ffn looks up the copy for whichever tensor it holds.
constexpr int LLAMA_MOE_MAX_ROLES = 3;

struct llama_moe_cache_layer {
    int il         = -1;
    int n_expert   = 0;
    int n_slots    = 0;   // K (number of cacheable experts); VRAM tensors hold K + zero slots
    int n_roles    = 0;   // number of expert projections managed for this layer

    // source (host-resident) expert weight tensors and their VRAM copies (parallel).
    // VRAM copies have shape [.., .., K+1]; slot K is the zero expert.
    ggml_tensor * src [LLAMA_MOE_MAX_ROLES] = { nullptr, nullptr, nullptr };
    ggml_tensor * vram[LLAMA_MOE_MAX_ROLES] = { nullptr, nullptr, nullptr };

    // VRAM copy for a given host source tensor, or nullptr if not managed
    ggml_tensor * vram_for(const ggml_tensor * s) const {
        for (int r = 0; r < n_roles; ++r) {
            if (src[r] == s) return vram[r];
        }
        return nullptr;
    }

    // per-expert maps, read on-graph via get_rows (shared across roles):
    //   gpu_map[e]  = slot(e)+0.5 if cached else -0.5     (F32; +0.5 so step() works for slot 0)
    //   cpu_map[e]  = 0 (dummy) if cached else e          (I32)
    // The CPU-branch mask is derived on-graph from step(gpu_map rows); the GPU
    // branch needs no mask since zero-slot down weights make miss rows zero.
    // A GPU miss is routed on-graph to a distinct zero-slot (K + slot-position) so
    // every id within a token stays distinct, as CUDA mul_mat_id (MMQ) requires.
    ggml_tensor * gpu_map = nullptr;
    ggml_tensor * cpu_map = nullptr;

    // constant [K+0, .., K+n_used-1] (F32), the per-position zero-slot ids
    ggml_tensor * iex = nullptr;

    // experimental single-zero-slot routing (MOE_CACHE_DUP_IDS): plain I32 slot
    // map (miss -> K, duplicated across misses) and F32 miss mask, replacing the
    // distinct-slot id arithmetic. Safe only where CUDA mul_mat_id takes the
    // batch-1 mmvq path (quantized experts), which tolerates duplicate ids; the
    // generic fallback path requires distinct ids per token.
    bool          dup_ids     = false;
    ggml_tensor * gpu_map_i32 = nullptr;
    ggml_tensor * mask_map    = nullptr;

    // row of the cache-wide ids tensor this layer's selected ids are copied to
    ggml_tensor * ids_all     = nullptr;
    int           harvest_row = -1;

    // host state
    std::vector<float> counts;         // decayed activation counters [n_expert]
    std::vector<int>   slot_of;        // expert -> slot, or -1 [n_expert]
    std::vector<int>   expert_in_slot; // slot  -> expert, or -1 [n_slots]
    bool maps_dirty = false;           // maps need re-upload
};

struct llama_moe_cache {
    llama_moe_cache(const llama_model & model, size_t budget_bytes);
    ~llama_moe_cache();

    // whether the layer is managed by the cache
    bool active(int il) const { return layers.find(il) != layers.end(); }

    const llama_moe_cache_layer * get(int il) const {
        auto it = layers.find(il);
        return it == layers.end() ? nullptr : &it->second;
    }

    // ---- runtime (called from llama_context around decode) ----

    // accumulate one micro-batch of selected experts for a layer
    void observe(int il, const int32_t * ids, int64_t n_used, int64_t n_tokens);

    // read the per-layer selected-expert ids (copied on-graph into ids_all) with
    // a single transfer and feed them into the counters; the graph that wrote
    // them must have finished computing
    void harvest();

    // recompute hot sets, promote newly-hot experts (CPU->VRAM), refresh maps.
    // synchronous: blocks until copies + uploads complete. The per-update copy
    // volume is bounded by promote_bytes; the remainder waits for later updates.
    void update();

    bool enabled() const { return buffer != nullptr && !layers.empty(); }

    // whether enough tokens have accumulated to justify a hot-set refresh;
    // updates run more often while free slots remain so the cache warms up fast
    bool due() const {
        const int64_t interval = filling ? std::min<int64_t>(update_interval, 64) : update_interval;
        return enabled() && tokens_since_update >= interval;
    }

    // stats
    size_t vram_bytes() const { return used_bytes; }
    void   log_summary() const;

private:
    void register_layer(int il, int n_slots);
    void alloc_tensors();
    void upload_maps(llama_moe_cache_layer & L);
    void promote(llama_moe_cache_layer & L, int expert, int slot);

    const llama_model & model;

    ggml_context             * ctx    = nullptr;
    ggml_backend_buffer_t      buffer = nullptr;
    ggml_backend_buffer_type_t buft   = nullptr;

    std::map<int, llama_moe_cache_layer> layers;

    // [n_used, n_layers] I32; each managed layer cpy's its selected ids into its
    // row on-graph so the host can read all of them back in one transfer
    ggml_tensor * ids_all = nullptr;
    std::vector<int32_t> ids_host; // scratch for reading ids_all

    int    n_used = 0;   // experts used per token (hparams.n_expert_used); extra zero-slots per layer

    size_t budget_bytes  = 0;
    size_t used_bytes    = 0;
    size_t promote_bytes = 0;  // max CPU->VRAM copy volume per update()

    // exponential decay applied to counters each update() (half-life ~1-2k tokens)
    float  decay = 0.999f;

    // hysteresis: displace a resident expert only when the candidate's count
    // exceeds the resident's by this factor; without it the top-K boundary
    // churns every update when activation is near-uniform
    float  margin = 2.0f;

    bool filling = true;  // free slots remain somewhere; shortens the update interval
    bool dup_ids = false; // single-zero-slot routing (see llama_moe_cache_layer)

    // aggregate hit/total counters since the last update (verbose logging)
    int64_t win_hits  = 0;
    int64_t win_total = 0;
    int64_t tokens_since_update = 0;
    int64_t update_interval     = 256; // recompute hot set at most this often
    int64_t n_updates           = 0;
    int64_t n_promotions        = 0;   // hot-set refreshes that actually changed slots
};
