// tools/moe-stats/moe-stats.cpp
//
// llama-moe-stats: measure per-(layer, expert) activation frequencies for MoE models.
//
// Why: expert placement (--cpu-moe / --n-cpu-moe / -ot) is currently static and
// layer-indexed. Expert activation is typically skewed and workload-dependent,
// but there is no way to observe it. This tool runs a representative prompt set
// through the model and reports which experts are hot, enabling frequency-aware
// placement (and, later, dynamic expert caching in VRAM).
//
// Mechanism: hooks params.cb_eval (same as tools/imatrix). For each
// GGML_OP_MUL_MAT_ID node on the ffn_down_exps weights (one per MoE layer per
// micro-batch, so each token is counted exactly once per layer), reads the
// expert-ids tensor t->src[2] ([n_expert_used, n_tokens], I32, possibly
// non-contiguous) and accumulates counts.
//
// Usage:
//   llama-moe-stats -m model.gguf -f prompts.txt [-ngl 99 -ot "..." --no-mmap ...]
//   -> prints per-layer skew report + coverage curve, writes moe-stats.csv
//
// Build: drop this dir into tools/, add `add_subdirectory(moe-stats)` to
// tools/CMakeLists.txt, rebuild.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct moe_stats {
    // counts[layer][expert] = number of (token, slot) activations (whole run)
    std::map<int, std::vector<int64_t>> counts;
    // window[layer][expert] = activations in the current token window
    std::map<int, std::vector<int64_t>> window;
    int64_t n_tokens_seen = 0;
    std::mutex mutex;
    std::vector<char> ids_host; // scratch for device->host copy

    // temporal windowing: track how the hot set drifts within a run
    int64_t window_size   = 2048;  // tokens per window (see --window note in README)
    int64_t window_tokens = 0;     // tokens accumulated in current window (layer 0 only)
    int     window_idx    = 0;
    std::vector<std::map<int, std::vector<int64_t>>> window_history;
};

static moe_stats g_stats;

// --- debug instrumentation: understand what the callback actually sees ---
static int64_t g_ask_total  = 0;   // total ask-phase invocations
static int64_t g_ask_mmid   = 0;   // asks that were MUL_MAT_ID
static std::map<std::string, int64_t> g_seen_ops;    // op name -> count (ask phase)
static std::map<std::string, int64_t> g_seen_mmid;   // MUL_MAT_ID src0 names -> count

// parse layer index from a tensor name like "blk.17.ffn_down_exps.weight".
// NOTE: when the scheduler offloads a CPU-resident weight's op to GPU (large
// batches / prompt processing), src0 is a scheduler copy named like
// "CUDA0#blk.17.ffn_down_exps.weight#0" - so find "blk." anywhere, not as prefix.
static int layer_from_name(const char * name) {
    const char * p = strstr(name, "blk.");
    if (p == nullptr) return -1;
    return atoi(p + 4);
}

static bool collect_moe_ids(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const struct ggml_tensor * src0 = t->src[0]; // expert weights [cols, rows, n_expert]
    // ask phase: tell the scheduler which nodes we want data for
    if (ask) {
        g_ask_total++;
        if (g_ask_total < 100000) g_seen_ops[ggml_op_name(t->op)]++;
        if (t->op != GGML_OP_MUL_MAT_ID) return false;
        g_ask_mmid++;
        g_seen_mmid[src0 ? src0->name : "(null src0)"]++;
        // count once per layer per token: only the down-projection matmul
        return src0 && strstr(src0->name, "ffn_down_exps") != nullptr;
    }

    std::lock_guard<std::mutex> lock(g_stats.mutex);

    const struct ggml_tensor * ids = t->src[2]; // [n_expert_used, n_tokens] I32
    const int64_t n_expert      = src0->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];

    const int il = layer_from_name(src0->name);
    if (il < 0) return true;

    // ids may live on GPU and may be non-contiguous: copy raw bytes, index by strides
    g_stats.ids_host.resize(ggml_nbytes(ids));
    ggml_backend_tensor_get(ids, g_stats.ids_host.data(), 0, ggml_nbytes(ids));

    auto & cnt = g_stats.counts[il];
    auto & win = g_stats.window[il];
    if (cnt.empty()) cnt.resize(n_expert, 0);
    if (win.empty()) win.resize(n_expert, 0);

    for (int64_t row = 0; row < n_tokens; ++row) {
        for (int64_t idx = 0; idx < n_expert_used; ++idx) {
            const int32_t ex = *(const int32_t *)(g_stats.ids_host.data() + row*ids->nb[1] + idx*ids->nb[0]);
            if (ex >= 0 && ex < n_expert) { cnt[ex]++; win[ex]++; }
        }
    }
    g_stats.n_tokens_seen += n_tokens; // note: counts each layer's pass; normalized later

    // advance the token window once per ubatch (lowest layer index observed)
    if (il == g_stats.counts.begin()->first) {
        g_stats.window_tokens += n_tokens;
        if (g_stats.window_tokens >= g_stats.window_size) {
            g_stats.window_history.push_back(g_stats.window);
            for (auto & [l, w] : g_stats.window) std::fill(w.begin(), w.end(), 0);
            g_stats.window_tokens = 0;
            g_stats.window_idx++;
        }
    }

    return true;
}

static void print_report(const char * csv_path) {
    LOG_INF("\n=== callback debug ===\n");
    LOG_INF("ask invocations: %lld | MUL_MAT_ID asks: %lld\n",
            (long long) g_ask_total, (long long) g_ask_mmid);
    if (g_ask_total == 0) {
        LOG_ERR("cb_eval was NEVER invoked -> your llama.cpp revision is not installing the\n"
                "scheduler eval callback on the decode path. Try: git pull && full rebuild,\n"
                "and/or run with LLAMA_GRAPH_REUSE_DISABLE=1\n");
    }
    int shown = 0;
    for (auto & [name, c] : g_seen_mmid) {
        LOG_INF("  MUL_MAT_ID src0: %-50s x%lld\n", name.c_str(), (long long) c);
        if (++shown >= 12) { LOG_INF("  ... (%zu distinct)\n", g_seen_mmid.size()); break; }
    }
    if (g_ask_total > 0 && g_ask_mmid == 0) {
        LOG_INF("  ops seen at ask phase (no MUL_MAT_ID at all!):\n");
        for (auto & [op, c] : g_seen_ops) LOG_INF("    %-24s x%lld\n", op.c_str(), (long long) c);
    }

    if (g_stats.counts.empty()) {
        LOG_ERR("no MoE activations observed - see callback debug above for the reason\n");
        return;
    }

    std::ofstream csv(csv_path);
    csv << "layer,expert,count\n";

    LOG_INF("\n=== MoE expert activation report ===\n");
    LOG_INF("%-6s %-8s %-10s %-22s %-22s\n", "layer", "experts", "acts", "top-8 share", "top-25%% share");

    // coverage curve accumulators: fraction of activations captured if we pin
    // the top-N hottest experts of each layer to VRAM
    const int Ns[] = {4, 8, 16, 32, 64, 128};
    const int NN = sizeof(Ns)/sizeof(int);
    std::vector<double> cover(NN, 0.0);
    double total_all = 0;

    for (auto & [il, cnt] : g_stats.counts) {
        std::vector<int64_t> sorted = cnt;
        std::sort(sorted.rbegin(), sorted.rend());
        int64_t total = 0; for (auto c : sorted) total += c;
        if (total == 0) continue;
        total_all += (double) total;

        int64_t top8 = 0;  for (int i = 0; i < std::min<size_t>(8, sorted.size()); i++) top8 += sorted[i];
        int64_t topq = 0;  for (size_t i = 0; i < sorted.size()/4; i++) topq += sorted[i];

        LOG_INF("%-6d %-8zu %-10lld %-20.1f%% %-20.1f%%\n",
                il, cnt.size(), (long long) total, 100.0*top8/total, 100.0*topq/total);

        for (int k = 0; k < NN; k++) {
            int64_t s = 0;
            for (int i = 0; i < std::min<int64_t>(Ns[k], (int64_t) sorted.size()); i++) s += sorted[i];
            cover[k] += (double) s;
        }
        for (size_t e = 0; e < cnt.size(); e++) csv << il << "," << e << "," << cnt[e] << "\n";
    }

    LOG_INF("\nCoverage if pinning the top-N hottest experts of EVERY layer to VRAM:\n");
    for (int k = 0; k < NN; k++)
        LOG_INF("  top-%-4d per layer -> %.1f%% of expert reads served from VRAM\n",
                Ns[k], 100.0 * cover[k] / total_all);
    // flush a partial final window if it has meaningful content
    if (g_stats.window_tokens >= g_stats.window_size / 4) {
        g_stats.window_history.push_back(g_stats.window);
    }

    // temporal stability: per consecutive window pair, overlap of top-32 sets
    // and cross-coverage (coverage of window k+1's traffic using window k's top-32)
    const auto & hist = g_stats.window_history;
    if (hist.size() >= 2) {
        LOG_INF("\n=== temporal stability (window = %lld tokens, %zu windows) ===\n",
                (long long) g_stats.window_size, hist.size());
        LOG_INF("%-10s %-18s %-22s\n", "windows", "top32 overlap", "cross-coverage");
        auto top32 = [](const std::vector<int64_t> & c) {
            std::vector<int> idx(c.size());
            for (size_t i = 0; i < c.size(); i++) idx[i] = (int) i;
            std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(32, idx.size()), idx.end(),
                              [&](int a, int b) { return c[a] > c[b]; });
            idx.resize(std::min<size_t>(32, idx.size()));
            return idx;
        };
        for (size_t k = 0; k + 1 < hist.size(); k++) {
            int64_t inter = 0, pinned_n = 0;
            int64_t covered = 0, total = 0;
            for (auto & [il, prev_cnt] : hist[k]) {
                auto it = hist[k+1].find(il);
                if (it == hist[k+1].end()) continue;
                const auto & next_cnt = it->second;
                auto tp = top32(prev_cnt);
                auto tn = top32(next_cnt);
                std::vector<bool> in_prev(prev_cnt.size(), false);
                for (int e : tp) in_prev[e] = true;
                for (int e : tn) if (in_prev[e]) inter++;
                pinned_n += (int64_t) tp.size();
                for (int e : tp) covered += next_cnt[e];
                for (auto c : next_cnt) total += c;
            }
            if (total == 0 || pinned_n == 0) continue;
            LOG_INF("%zu -> %-4zu  %16.1f%% %20.1f%%\n",
                    k, k+1, 100.0*inter/pinned_n, 100.0*covered/total);
        }
        LOG_INF("(cross-coverage ~= hit rate a cache would get if it only refreshed once per window;\n"
                " uniform-router baseline for top-32 of 256 is 12.5%%)\n");

        // windowed CSV for offline analysis
        std::string wpath = std::string(csv_path) + ".windows.csv";
        std::ofstream wcsv(wpath);
        wcsv << "window,layer,expert,count\n";
        for (size_t k = 0; k < hist.size(); k++)
            for (auto & [il, c] : hist[k])
                for (size_t e = 0; e < c.size(); e++)
                    if (c[e] > 0) wcsv << k << "," << il << "," << e << "," << c[e] << "\n";
        LOG_INF("wrote windowed counts to %s\n", wpath.c_str());
    }

    LOG_INF("\nwrote per-expert counts to %s\n", csv_path);
    LOG_INF("NOTE: hotness is workload-dependent. Feed prompts representative of YOUR usage.\n");
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, nullptr)) {
        return 1;
    }
    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    if (const char * ws = getenv("MOE_STATS_WINDOW")) {
        g_stats.window_size = std::max<int64_t>(256, atoll(ws));
    }

    params.cb_eval           = collect_moe_ids;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false; // don't pollute counts with the warmup pass

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) { LOG_ERR("failed to init model/context\n"); return 1; }

    // read prompt text (-f file or -p string)
    std::string text = params.prompt;
    if (text.empty()) { LOG_ERR("provide a prompt with -f prompts.txt or -p \"...\"\n"); return 1; }

    std::vector<llama_token> tokens = common_tokenize(ctx, text, /*add_special*/ true);
    LOG_INF("evaluating %zu tokens...\n", tokens.size());

    const int n_batch = params.n_batch;
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        const int n = std::min<size_t>(n_batch, tokens.size() - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, n);
        if (llama_decode(ctx, batch) != 0) { LOG_ERR("llama_decode failed at %zu\n", i); return 1; }
    }

    const char * genv = getenv("MOE_STATS_GEN");
    const int n_gen = genv ? atoi(genv) : 0;

    if (n_gen <= 0) {
        print_report("moe-stats.csv");
        llama_backend_free();
        return 0;
    }

    // report prefill separately, then reset counters and profile generation
    LOG_INF("\n########## PREFILL PHASE ##########\n");
    print_report("moe-stats-pp.csv");
    {
        std::lock_guard<std::mutex> lock(g_stats.mutex);
        g_stats.counts.clear();
        g_stats.window.clear();
        g_stats.window_history.clear();
        g_stats.window_tokens = 0;
        g_stats.n_tokens_seen = 0;
    }

    LOG_INF("\ngenerating %d tokens (greedy) to profile generation-time routing...\n", n_gen);
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_token tok = llama_sampler_sample(smpl, ctx, -1);
    for (int i = 0; i < n_gen; i++) {
        llama_batch gb = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, gb) != 0) { LOG_ERR("decode failed during generation at %d\n", i); break; }
        tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(llama_model_get_vocab(model), tok)) {
            LOG_INF("hit end-of-generation token at %d; stats cover %d tokens\n", i, i + 1);
            break;
        }
    }
    llama_sampler_free(smpl);

    LOG_INF("\n########## GENERATION PHASE ##########\n");
    print_report("moe-stats-tg.csv");

    llama_backend_free();
    return 0;
}
