//
// Created by Chiheb Boussema on 30/9/25.
//

#include "hpp/SDXLPipeline.h"
#include "hpp/newInferenceHelper.hpp"
#include <unistd.h>
#include <fstream>         // std::ifstream  (fixes “undefined template basic_ifstream”)

#define LOG_TAG "SDXL_PIPELINE"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

void concat_last_dim(
    float* __restrict dst,
    const float* __restrict a, size_t rows, size_t D1,
    const float* __restrict b, size_t D2)
{
    const size_t rowBytes1 = D1 * sizeof(float);
    const size_t rowBytes2 = D2 * sizeof(float);
    for (size_t r = 0; r < rows; ++r) {
        std::memcpy(dst, a, rowBytes1);
        std::memcpy(dst + D1, b, rowBytes2);
        dst += (D1 + D2);
        a   += D1;
        b   += D2;
    }
}

void logNumbers(const float* p, const TensorInfo& t, const std::string& wsName, const size_t& n=8, const char logLevel='i', const char* initLog=""
) {
    std::string vals;
    size_t count = std::min<size_t>(n, t.numel());
    LOGW("TENSOR COUNT %zu", t.numel());
    for (size_t i = 0; i < count; ++i) {
        vals += std::to_string(p[i]);
        if (i + 1 < n) vals += ", ";
    }
    if (logLevel=='w') {
        LOGW("%s\n   Tensor '%s' (workspace='%s', %zu floats): [%s%s]",
             initLog, t.name.c_str(), wsName.c_str(), n,
             vals.c_str(), (n > count ? ", ..." : ""));
    } else {
        LOGI("%s\n   Tensor '%s' (workspace='%s', %zu floats): [%s%s]",
             initLog, t.name.c_str(), wsName.c_str(), n,
             vals.c_str(), (n > count ? ", ..." : ""));
    }
}

bool readFileToBuffer(const std::string& path, void* dst, size_t bytes) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.read(reinterpret_cast<char*>(dst), bytes);
    return static_cast<size_t>(ifs.gcount()) == bytes;
}

// NHWC -> NCHW
static void nhwc_to_nchw(float* dst, const float* src, int N, int H, int W, int C) {
    // N assumed 1 here, but keep N for completeness
    for (int n = 0; n < N; ++n) {
        const float* sN = src + n * H * W * C;
        float*       dN = dst + n * C * H * W;
        for (int c = 0; c < C; ++c) {
            float* dC = dN + c * H * W;
            for (int h = 0; h < H; ++h) {
                const float* sRow = sN + h * W * C + c;
                float* dRow = dC + h * W;
                for (int w = 0; w < W; ++w) {
                    dRow[w] = sRow[w * C];
                }
            }
        }
    }
}

// NCHW -> NHWC
static void nchw_to_nhwc(float* dst, const float* src, int N, int C, int H, int W) {
    for (int n = 0; n < N; ++n) {
        const float* sN = src + n * C * H * W;
        float*       dN = dst + n * H * W * C;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float* dPix = dN + (h * W + w) * C;
                for (int c = 0; c < C; ++c) {
                    dPix[c] = sN[c * H * W + h * W + w];
                }
            }
        }
    }
}


inline PipelineCfg findPipe(const MultiPipelinesCfg& mpcfg, const std::string& n) {
    PipelineCfg out;
    for (const auto& pipe: mpcfg.pipes) {
        if (pipe.name == n) return pipe;
    }
    LOGE("Could not find pipeline %s", n.c_str());
    return out;
}
bool SDXLPipeline::init_networks(char runtime_hint) {
    std::string log;
    bool reset_sessions = true;

    // read config file
    std::string cfgText;
    std::string emsg;
    if (!readAssetToString(mgr_, allModels_config_filename_.c_str(), cfgText, &emsg)) {
        LOGE("Config read failed: %s", emsg.c_str());
        return false;
    }
    // parse config file
    MultiPipelinesCfg mp_cfg;
    {
        std::string emsg;
        if (!ParseMultiConfig(cfgText, mp_cfg, &emsg)) {
            LOGE("Config parse failed: %s", emsg.c_str());
            return false;
        }
        if (mp_cfg.pipes.empty()) {
            LOGE("Config has no pipelines");
            return false;
        }
    }


    // build encoders
    LOGI("Building encoders");
    auto pipe = findPipe(mp_cfg, "text_encoders");
    if (pipe.models.empty()) { LOGE("Missing pipeline 'text_encoders'"); return false; }
    log = buildArbitraryChainFromConfig(mgr_,
                                        g_modelDir_,
                                        pipe,
                                        ws_,
                                        enc_gr_,
                                        runtime_hint,
                                        reset_sessions);

    // build unet networks
    LOGI("Building unet_time_embd");
    pipe = findPipe(mp_cfg, "unet_time_embd");
    if (pipe.models.empty()) { LOGE("Missing pipeline 'unet_time_embd'"); return false; }
    log += buildArbitraryChainFromConfig(mgr_,
                                       g_modelDir_,
                                       pipe,
                                       ws_,
                                       *unet_.time_embd,
                                       runtime_hint,
                                       reset_sessions);

    LOGI("Building unet_aug_emb");
    pipe = findPipe(mp_cfg, "unet_aug_emb");
    if (pipe.models.empty()) { LOGE("Missing pipeline 'unet_aug_emb'"); return false; }
    log += buildArbitraryChainFromConfig(mgr_,
                                       g_modelDir_,
                                       pipe,
                                       ws_,
                                       *unet_.aug_emb,
                                       runtime_hint,
                                       reset_sessions);

    LOGI("Building unet_in_conv");
    pipe = findPipe(mp_cfg, "unet_in_conv");
    if (pipe.models.empty()) { LOGE("Missing pipeline 'unet_in_conv'"); return false; }
    log += buildArbitraryChainFromConfig(mgr_,
                                       g_modelDir_,
                                       pipe,
                                       ws_,
                                       *unet_.inConv,
                                       runtime_hint,
                                       reset_sessions);

    LOGI("Building unet_ublocks");
    pipe = findPipe(mp_cfg, "unet_ublocks");
    if (pipe.models.empty()) { LOGE("Missing pipeline 'unet_ublocks'"); return false; }
    log += buildArbitraryChainFromConfig(mgr_,
                                       g_modelDir_,
                                       pipe,
                                       ws_,
                                       *unet_.unet_blocks,
                                       runtime_hint,
                                       reset_sessions);

    LOGI("Building unet_upblock00");
    pipe = findPipe(mp_cfg, "unet_upblock00");
    if (pipe.models.empty()) { LOGE("Missing pipeline 'unet_upblock00'"); return false; }
    log += buildArbitraryChainFromConfig(mgr_,
                                       g_modelDir_,
                                       pipe,
                                       ws_,
                                       *unet_.unet_upblock00,
                                       runtime_hint,
                                       reset_sessions);

    LOGI("Building unet_upblock00");
    pipe = findPipe(mp_cfg, "unet_upblock01");
    if (pipe.models.empty()) { LOGE("Missing pipeline 'unet_upblock01'"); return false; }
    log += buildArbitraryChainFromConfig(mgr_,
                                       g_modelDir_,
                                       pipe,
                                       ws_,
                                       *unet_.unet_upblock01,
                                       runtime_hint,
                                       reset_sessions);

    LOGI("Building unet_ublocks_2");
    pipe = findPipe(mp_cfg, "unet_ublocks_2");
    if (pipe.models.empty()) { LOGE("Missing pipeline 'unet_ublocks_2'"); return false; }
    log += buildArbitraryChainFromConfig(mgr_,
                                       g_modelDir_,
                                       pipe,
                                       ws_,
                                       *unet_.unet_blocks_2,
                                       runtime_hint,
                                       reset_sessions);

    LOGI("Building unet_out_conv");
    pipe = findPipe(mp_cfg, "unet_out_conv");
    if (pipe.models.empty()) { LOGE("Missing pipeline 'unet_out_conv'"); return false; }
    log += buildArbitraryChainFromConfig(mgr_,
                                       g_modelDir_,
                                       pipe,
                                       ws_,
                                       *unet_.outConv,
                                       runtime_hint,
                                       reset_sessions);

    LOGI("Building VAE decoder");
    pipe = findPipe(mp_cfg, "vae");
    if (pipe.models.empty()) { LOGE("Missing pipeline 'vae'"); return false; }
    log += buildArbitraryChainFromConfig(mgr_,
                                       g_modelDir_,
                                       pipe,
                                       ws_,
                                       *vae_decoder_,
                                       runtime_hint,
                                       reset_sessions);

    LOGI("Building networks complete!");
    return true;
}


bool SDXLPipeline::init_text_encoders(
//        const std::string& te1_dlc_path,
//        const std::string& te2_dlc_path,
        char runtime_hint='D') {

    std::string encoders_log;
    bool reset_sessions = false;

    encoders_log = buildArbitraryChain(mgr_,
                                       g_modelDir_,
                                       encoders_config_filename_,
                                       ws_,
                                       enc_gr_,
                                       runtime_hint,
                                       reset_sessions);



    if (!&enc_gr_) return false;

    std::string input_name = enc_gr_.last().session->inputs().back().name;
    LOGI("[INPUT NAME FOR ENCODER:] %s", input_name.c_str());

    return true;
}

std::string SDXLPipeline::run_encoders(const int32_t* ids1_77,
                                        const int32_t* ids2_77) {

    // --- 1) Copy input ids into workspace (strict zero-copy binding will feed these to TE1/TE2)
    const char* kTe1Ids = "input_ids_1";  // <-- adjust if your config uses different names
    const char* kTe2Ids = "input_ids_2";  // <-- adjust if your config uses different names
//    const char* kTeIds = "input_ids";

    const size_t needBytes = static_cast<size_t>(B_) * static_cast<size_t>(S_) * sizeof(int32_t);

    void* te1_ptr = ws_.data(kTe1Ids);
    void* te2_ptr = ws_.data(kTe2Ids);

    if (!te1_ptr || !te2_ptr) {
        LOGE("encode_prompt: workspace inputs missing (te1='%s' te2='%s')", kTe1Ids, kTe2Ids);
        return "workspace inputs missing"; // empty (null pointers)
    }
    if (ws_.sizeOf(kTe1Ids) != needBytes || ws_.sizeOf(kTe2Ids) != needBytes) {
        LOGE("encode_prompt: workspace input size mismatch (need=%zu, have te1=%zu te2=%zu)",
                needBytes, ws_.sizeOf(kTe1Ids), ws_.sizeOf(kTe2Ids));
        return "workspace input size mismatch";
    }

//    std::memcpy(te1_ptr, ids1_77, needBytes);
    std::memcpy(te1_ptr, ids1_77, sizeof(ids1_77));
//    std::memcpy(te2_ptr, ids2_77, needBytes);
    std::memcpy(te2_ptr, ids2_77, sizeof(ids2_77));

    if (!&enc_gr_) return "Graph not built";
    std::string execution_summary;
    execution_summary = runGraph(enc_gr_, false);
    LOGI("Done with encoders execution!");

    // --- 3) Expose output pointers from workspace
    // Adjust these names to what your config maps the TE outputs to.
    const char* kPrompt1      = "prompt_embeds_1";         // float32 [1,77,768]
    const char* kPrompt2     = "prompt_embeds_2";
    const char* kPromptPool  = "pooled_embeds";  // float32 [1,D_pool]
    const char* kOut = "prompt_embeds";

    float* prompt_embeds1        = reinterpret_cast<float*>(ws_.data(kPrompt1));
    float* prompt_embeds2        = reinterpret_cast<float*>(ws_.data(kPrompt2));
    float* pooled_prompt_embeds  = reinterpret_cast<float*>(ws_.data(kPromptPool));
    if (!prompt_embeds1 || !prompt_embeds2 || !pooled_prompt_embeds) {
        LOGE("encode_prompt: missing output buffers ('%s' or '%s' or '%s')", kPrompt1, kPrompt2, kPromptPool);
//        EncodedOut empty{};
        return "Missing output buffers";
    }

    // read latent from file
    auto pe1_tinfo = ws_.tinfoOf(kPrompt1);
    auto pe2_tinfo = ws_.tinfoOf(kPrompt2);
    auto pooled_pe_tinfo = ws_.tinfoOf(kPromptPool);
    bool read_from_file = readFileToBuffer("/sdcard/Android/data/com.example.snpechainingdemo/files/pe1.bin", prompt_embeds1, pe1_tinfo->bytes());
    if (read_from_file) {
        LOGI("[Prompt Embeddings 1:] Read latent from file!");
    }
    read_from_file = readFileToBuffer("/sdcard/Android/data/com.example.snpechainingdemo/files/pe1.bin", prompt_embeds2, pe2_tinfo->bytes());
    if (read_from_file) {
        LOGI("[Prompt Embeddings 2:] Read latent from file!");
    }
    read_from_file = readFileToBuffer("/sdcard/Android/data/com.example.snpechainingdemo/files/pooled_pe.bin", pooled_prompt_embeds, pooled_pe_tinfo->bytes());
    if (read_from_file) {
        LOGI("[Pooled Prompt Embeddings:] Read latent from file!");
    }

    const size_t BS = static_cast<size_t>(B_) * static_cast<size_t>(S_);
    // ensure output buffer [B,S,D1+D2]
    const size_t D1 = static_cast<size_t>(D1_);
    const size_t D2 = static_cast<size_t>(D2_);
    const size_t outElems = BS * (D1 + D2);
    const size_t outBytes = outElems * sizeof(float);
    std::string emsg;
    TensorInfo tinfo;
    tinfo.name = kOut;
    tinfo.dims = {static_cast<size_t>(B_), static_cast<size_t>(S_), D1 + D2};
    tinfo.elementBytes = 4;
    if (!ensureWorkspaceBuffer(ws_, kOut, tinfo, &emsg)) {
//    if (!ensureWorkspaceBuffer(ws_, kOut, outBytes, &emsg)) {
        LOGE("run_encoders: cannot alloc '%s': %s", kOut, emsg.c_str());
        return execution_summary + "\nalloc prompt_embeds failed";
    }
//    auto* prompt_embeds = static_cast<float*>(ws_.data(kOut));
    auto* prompt_embeds = static_cast<float*>(ws_.data("encoder_hidden_states"));
    // concat with two memcpys per (b,s) row
    for (size_t i = 0; i < BS; ++i) {
        const float* src1 = prompt_embeds1 + i * D1;
        const float* src2 = prompt_embeds2 + i * D2;
        float*       dst  = prompt_embeds + i * (D1 + D2);
        std::memcpy(dst,           src1, D1 * sizeof(float));
        std::memcpy(dst + D1,      src2, D2 * sizeof(float));
    }

    // alias the prompt_embeds tensor into encoder_hidden_states (expected by later networks)
//    const char* prompt_embeddings_tensor_name = "encoder_hidden_states";
//    if (ws_.data(kOut)) {
//        ws_.alias(prompt_embeddings_tensor_name, kOut);
//    }

    // Optional sanity logs (first few floats)
    {
        const float* p = prompt_embeds;
        const int nPreview = 8;
        std::string s = "[prompt_embeds] ";
        for (int i = 0; i < nPreview; ++i) {
            char buf[64];
            snprintf(buf, sizeof(buf), (i==0?"%f":" %f"), p[i]);
            s += buf;
        }
        LOGI("%s ...", s.c_str());
    }
    {
        const float* p = pooled_prompt_embeds;
        const int nPreview = 8;
        std::string s = "[pooled_prompt_embeds] ";
        for (int i = 0; i < nPreview; ++i) {
            char buf[64];
            snprintf(buf, sizeof(buf), (i==0?"%f":" %f"), p[i]);
            s += buf;
        }
        LOGI("%s ...", s.c_str());
    }

    return execution_summary;
}

std::pair<float*, size_t> SDXLPipeline::prepare_latents(const int width,
                                                        const int height,
                                                        const char* wsName /*= "latents"*/,
                                                        std::optional<uint64_t> seed /*= std::nullopt*/,
                                                        bool forceRegen /*= false*/) {

    // 1) Shape
    const int B = 1;
    const int C = net_config_.num_channels_latents;           // typically 4
    const int H = static_cast<int>(height) / net_config_.vae_scale_factor; // e.g. 1024/8=128
    const int W = static_cast<int>(width)  / net_config_.vae_scale_factor; // e.g. 1024/8=128

    if (H <= 0 || W <= 0) {
        LOGE("prepare_latents: invalid H/W after scale (H=%d, W=%d)", H, W);
        return {nullptr, 0};
    }

    const size_t elems = static_cast<size_t>(B) * C * H * W;
    const size_t bytes = elems * sizeof(float);

    const std::string key = (wsName && *wsName) ? std::string(wsName) : std::string("latents");

    // 2) (Re)allocate if missing or wrong size
    bool need_init = forceRegen;
    if (!ws_.has(key)) {
        ws_.allocate(key, bytes);
        need_init = true;
    } else if (ws_.sizeOf(key) != bytes) {
        LOGI("prepare_latents: resizing '%s' from %zu -> %zu bytes", key.c_str(), ws_.sizeOf(key), bytes);
        ws_.release(key);
        ws_.allocate(key, bytes);
        need_init = true;
    }

    auto* ptr = static_cast<float*>(ws_.data(key));
    if (!ptr) {
        LOGE("prepare_latents: workspace returned null for '%s'", key.c_str());
        return {nullptr, 0};
    }

    // scheduler sigma
    const float sigma = scheduler_->init_noise_sigma();

    // 3) If newly allocated (or force), fill with N(0,1) and scale by scheduler sigma
    if (need_init) {
        // rng
        std::mt19937 rng;
        if (seed.has_value()) rng.seed(static_cast<uint64_t>(*seed));
        else                  rng.seed(std::random_device{}());

        std::normal_distribution<float> dist(0.0f, 1.0f);

        // fill
        for (size_t i = 0; i < elems; ++i) {
            ptr[i] = dist(rng) * sigma;
        }

    } else {
        LOGI("prepare_latents: reusing existing '%s' (%zu floats)", key.c_str(), elems);
        // fill
        for (size_t i = 0; i < elems; ++i) {
            ptr[i] *= sigma;
        }
    }

    LOGI("prepare_latents: initialized '%s' with %zu floats (B=%d,C=%d,H=%d,W=%d), sigma=%.6f",
         key.c_str(), elems, B, C, H, W, sigma);

    return {ptr, elems};
}

std::array<int32_t, 6> buildAddTimeIds(
    std::pair<int, int> original_size,          // (H, W) or (W, H) per your convention
    std::pair<int, int> crops_coords_top_left,  // (y, x)
    std::pair<int, int> target_size             // (H, W)
) {
    std::array<int32_t, 6> a;
    // Adjust ordering if your pipeline used (W,H). The snippet assumes (H,W).
    a[0] = static_cast<int32_t>(original_size.first);
    a[1] = static_cast<int32_t>(original_size.second);
    a[2] = static_cast<int32_t>(crops_coords_top_left.first);
    a[3] = static_cast<int32_t>(crops_coords_top_left.second);
    a[4] = static_cast<int32_t>(target_size.first);
    a[5] = static_cast<int32_t>(target_size.second);
    return a;
}

// 2) Validate dims and write a [1,6] float32 buffer into the workspace.
bool SDXLPipeline::add_time_ids_toWorkspace(
        TensorWorkspace& ws,
        const char* wsName,
        std::pair<int,int> original_size,
        std::pair<int,int> crops_coords_top_left,
        std::pair<int,int> target_size,
        const int unet_add_embedding_linear_1_in_features, // net_config_.unet_add_embedding_linear_1_in_features
        const int unet_addition_time_embed_dim,            // net_config_.unet_addition_time_embed_dim
        const int text_encoder_projection_dim,             // net_config_.text_encoder_projection_dim
        std::string* emsg /*=nullptr*/) {

//    LOGI("[time ids] 1");
    const auto vals = buildAddTimeIds(original_size, crops_coords_top_left, target_size);
    constexpr size_t K = vals.size(); // 6
//    LOGI("[time ids] 2");

    // Check the same invariant as Python
    const int passed = unet_addition_time_embed_dim * static_cast<int>(K) + text_encoder_projection_dim;
    const int expected = unet_add_embedding_linear_1_in_features;
    if (expected != passed) {
        if (emsg) {
            *emsg = "add_time_ids dim mismatch: expected "
                    + std::to_string(expected) + ", got "
                    + std::to_string(passed)
                    + " (embed_dim=" + std::to_string(unet_addition_time_embed_dim)
                    + ", K=" + std::to_string(K)
                    + ", proj_dim=" + std::to_string(text_encoder_projection_dim) + ")";
        }
        return false;
    }
//    LOGI("[time ids] 3");
    // Allocate [1,6] as float32 (SNPE user-buffers are best kept float32).
    const size_t bytes = K * sizeof(float);
    const std::string key = (wsName && *wsName) ? wsName : "time_ids";
//    LOGI("[time ids] 4: %s", key.c_str());

    if (!ws.has(key)) {
//        LOGI("[time ids] 5");
        ws.allocate(key, bytes);
    } else if (ws.sizeOf(key) != bytes) {
//        LOGI("[time ids] 6");
        ws.release(key);
        ws.allocate(key, bytes);
    }
//    LOGI("[time ids] 7");
    auto* dst = static_cast<float*>(ws.data(key));
    if (!dst) {
//        LOGI("[time ids] 8");
        if (emsg) *emsg = "workspace returned null for '" + key + "'";
        return false;
    }

//    LOGI("[time ids] 9");
    // Fill as float32 (Python returns float tensor; SNPE float32 UB matches neatly).
    for (size_t i = 0; i < K; ++i) dst[i] = static_cast<float>(vals[i]);
//    LOGI("[time ids] 10");

    // Optional: log for sanity
    // LOGI("add_time_ids [%s]: [%d, %d, %d, %d, %d, %d]  (bytes=%zu)",
    //      key.c_str(), vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], bytes);

    const auto& time_ids_tinfo = ws_.tinfoOf(key);
    logNumbers(dst, *time_ids_tinfo, key.c_str(), 6, 'w', "Time Ids:");

    return true;
}

void SDXLPipeline::get_aug_emb2() {
    std::string aug_emb_summary;

    // inputs: time_ids and text_embeddings
    auto* time_ids = static_cast<float*>(ws_.data("time_ids"));
    if (!time_ids) {
        LOGE("[aug_emb] time_ids tensor does not exist!");
        return;
    }
    auto* text_embeds = static_cast<float*>(ws_.data("pooled_embeds"));
    if (!text_embeds) {
        LOGE("[aug_emb] text_embeds tensor does not exist!");
        return;
    }

    aug_emb_summary = runGraph(*unet_.aug_emb, true);
    LOGI("%s", aug_emb_summary.c_str());
}


static void transpose_inplace_nhwc_nchw(float* data, int N, int C, int H, int W, bool nhwc_to_nchw=true) {
    const size_t total = static_cast<size_t>(N) * C * H * W;
    std::vector<float> tmp(total);

    if (nhwc_to_nchw) {
        // NHWC -> NCHW
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c =0; c < C; ++c)
                    tmp[c*H*W + h*W + w] = data[(h*W + w) * C + c];
    } else {
        // NCHW -> NHWC
        for (int c =0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    tmp[(h*W + w) * C + c] = data[c*H*W + h*W + w];
    }

    std::memcpy(data, tmp.data(), total*sizeof(float));
}

void SDXLPipeline::unet_wrapper2(const float t) {

    std::string unet_execution_summary;
    // 1. time embeddings:
//    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
//    emb = self.time_embedding(t_emb, timestep_cond)
    LOGI("[Unet:] getting time embeds");
    std::string time_embd_input_name = unet_.time_embd->last().session.get()->inputs().back().name;
    auto time_embd_input_wsName = unet_.time_embd->last().inputBinding.at(time_embd_input_name);
    time_embd_input_wsName = !time_embd_input_wsName.empty() ? time_embd_input_wsName : "timestep";
//    auto* timestep = static_cast<int32_t*>(ws_.data(time_embd_input_wsName));
    auto* timestep = static_cast<float*>(ws_.data(time_embd_input_wsName));
    auto* timestep_info = ws_.tinfoOf(time_embd_input_wsName);
//    std::memcpy(timestep, &t, timestep_info->bytes());
    std::memcpy(timestep, &t, sizeof(timestep));
    unet_execution_summary = runGraph(*unet_.time_embd, true);

    // 2. aug_emb:
    // aug_emb = self.get_aug_embed(emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    LOGI("[Unet:] getting aug embs");
    get_aug_emb2();

    // 3. emb = emb + aug_emb if aug_emb is not None else emb
    LOGI("[Unet:] combining emb with aug_emb");
    auto* emb = static_cast<float*>(ws_.data("emb"));
    auto* aug_emb = static_cast<float*>(ws_.data("aug_emb"));
    if (!emb) {
        LOGE("combine embeds: base emb missing");
        return;
    }
    if (!aug_emb) {
        LOGE("combine embeds: aug_emb missing");
        return;
    }
    const size_t embBytes = ws_.sizeOf("emb");
    const size_t augBytes = ws_.sizeOf("aug_emb");
    if (embBytes != augBytes) {
        LOGE("combine_embeds_add: size mismatch emb=%zu aug=%zu", embBytes, augBytes);
        return;
    }
    auto* emb_info = ws_.tinfoOf("emb");
    size_t n;
    if (!emb_info) {
        n = embBytes / sizeof(float);
    } else {
        n = emb_info->numel();
    }
    LOGI("EMB n %lu", n);
    const auto& aug_emb_tinfo = ws_.tinfoOf("aug_emb");
    logNumbers(aug_emb, *aug_emb_tinfo, "aug_emb", 12, 'w', "Aug Emb:");

    float* dst = emb;
    const float* src = aug_emb;
    // simple vector add
    for (size_t i = 0; i < n; ++i) dst[i] += src[i];

    logNumbers(emb, *emb_info, "emb", 8, 'w', "Emb after adding aug_emb:");

    // 4. conv_in: latent --> sample
    LOGI("Running Unet inConv...");
//    check that inputs are present
    std::string conv_in_name = unet_.inConv->last().session->inputs().back().name;
    auto conv_in_wsName = unet_.inConv->last().inputBinding.at(conv_in_name);
    if (!ws_.has(conv_in_wsName) || !ws_.data(conv_in_wsName)) {
        LOGE("[UNET:] inputs to inConv do not exist!");
        return;
    }
    unet_execution_summary += runGraph(*unet_.inConv, true);

    // 5. down-, mid-, and up-blocks: (sample, temb, encoder_hidden_states) ==> one final output
    LOGI("Running Unet cross-attention blocks...");
    const auto& nodes = unet_.unet_blocks->getNodes();
    for (const auto& n : nodes[0].session.get()->inputs()) {
        const auto& wsName = nodes[0].inputBinding.at(n.name);
        if (!ws_.data(wsName)) {
            LOGE("[Unet:] inputs %s (workspace: %s) to u_blocks does not exist", n.name.c_str(), wsName.c_str());
            return;
        }
    }
    unet_execution_summary += runGraph(*unet_.unet_blocks, true);

    LOGW("NODE NAMES ===========");
    for (const auto& node : unet_.unet_blocks_2->getNodes()) {
        LOGW("%s", node.name.c_str());
    }

    auto& tinfos00 = unet_.unet_upblock00->last().session.get()->outputs();
    LOGW("HIDDEN STATES OF UPBLOCK00 DIMENSION:");
    for (auto& tnfo : tinfos00) {
        if (tnfo.name == "output_0") {
            for (auto& d : tnfo.dims) {
                LOGW("%lu", d);
            }
        }
    }

    const auto& tinfos01 = unet_.unet_upblock01->last().session.get()->outputs();
    LOGW("HIDDEN STATES OF UPBLOCK01 DIMENSION:");
    for (auto& tnfo : tinfos01) {
        if (tnfo.name == "output_0") {
            for (auto& d : tnfo.dims) {
                LOGW("%lu", d);
            }
        }
    }

    const auto hidden_states_wsName = unet_.unet_blocks_2->getNode("upblock02").inputBinding.at("hidden_states");
    if (hidden_states_wsName.empty()) {
        LOGE("[Unet:] Did not find workspace tensor name for hidden_states -- likely due to node name mistmatch.");
        return;
    }
    auto* hidden_states = static_cast<float*>(ws_.data(hidden_states_wsName));
    const auto& hidden_states_tinfo = ws_.tinfoOf(hidden_states_wsName);
    LOGW("HIDDEN STATES DIMENSION:");
    for (auto& d : hidden_states_tinfo->dims) {
        LOGW("%lu", d);
    }

    LOGI("Running upblock 00...");
    unet_execution_summary += runGraph(*unet_.unet_upblock00, true);
    LOGI("Transposing...");
//    transpose_inplace_nhwc_nchw(hidden_states, 1, 1280, 32, 32);
    // transpose the output of upblock00 to input upblock01 -----------------------------------
    const auto& out_hidden_00_wsName = unet_.unet_upblock00->last().outputBinding.at("output_0");
    auto* out_hidden_00 = static_cast<float*>(ws_.data(out_hidden_00_wsName));
    const auto& out_hidden_00_tinfo = ws_.tinfoOf(out_hidden_00_wsName);
    LOGI("out hidden 00 numel: %lu", out_hidden_00_tinfo->numel());

    const auto& in_hidden_01_wsName = unet_.unet_upblock00->last().inputBinding.at("hidden_states");
    auto* in_hidden_01 = static_cast<float*>(ws_.data(in_hidden_01_wsName));
    const auto& in_hidden_01_tinfo = ws_.tinfoOf(in_hidden_01_wsName);
    LOGI("in hidden 01 numel: %lu", in_hidden_01_tinfo->numel());

    nhwc_to_nchw(in_hidden_01, out_hidden_00, 1, 32, 32, 1280);


    LOGI("Running upblock 01...");
    unet_execution_summary += runGraph(*unet_.unet_upblock01, true);
    LOGI("Transposing...");
//    transpose_inplace_nhwc_nchw(hidden_states, 1, 1280, 32, 32);
    // transpose the output of upblock00 to input upblock01 -----------------------------------
    const auto& out_hidden_01_wsName = unet_.unet_upblock00->last().outputBinding.at("output_0");
    auto* out_hidden_01 = static_cast<float*>(ws_.data(out_hidden_01_wsName));

    const auto& in_hidden_02_wsName = unet_.unet_upblock00->last().inputBinding.at("hidden_states");
    auto* in_hidden_02 = static_cast<float*>(ws_.data(in_hidden_02_wsName));

    nhwc_to_nchw(in_hidden_02, out_hidden_01, 1, 32, 32, 1280);

    const auto& tinfos = unet_.unet_blocks_2->getNode("upblock02").session.get()->inputs();
    LOGW("HIDDEN STATES OF BLOCK02 DIMENSION:");
    for (auto& tnfo : tinfos) {
        if (tnfo.name == "hidden_states") {
            for (auto& d : tnfo.dims) {
                LOGW("%lu", d);
            }
        }
    }
    LOGI("Running upblock 02 and 1...");
    unet_execution_summary += runGraph(*unet_.unet_blocks_2, true);

    // 6. conv_norm_out, conv_act, conv_out ==> predicted noise
    LOGI("Running Unet outConv...");
    unet_execution_summary += runGraph(*unet_.outConv, true);

    LOGI("Finished running Unet!");
}

void SDXLPipeline::unet_wrapper(const float t) {

    std::string unet_execution_summary;
    // 1. time embeddings:
//    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
//    emb = self.time_embedding(t_emb, timestep_cond)
    LOGI("[Unet:] getting time embeds");
    std::string time_embd_input_name = unet_.time_embd->last().session.get()->inputs().back().name;
    auto time_embd_input_wsName = unet_.time_embd->last().inputBinding.at(time_embd_input_name);
    time_embd_input_wsName = !time_embd_input_wsName.empty() ? time_embd_input_wsName : "timestep";
//    auto* timestep = static_cast<int32_t*>(ws_.data(time_embd_input_wsName));
    auto* timestep = static_cast<float*>(ws_.data(time_embd_input_wsName));
    auto* timestep_info = ws_.tinfoOf(time_embd_input_wsName);
//    std::memcpy(timestep, &t, timestep_info->bytes());
    std::memcpy(timestep, &t, sizeof(timestep));
    unet_execution_summary = runGraph(*unet_.time_embd, true);

    // 2. aug_emb:
    // aug_emb = self.get_aug_embed(emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    LOGI("[Unet:] getting aug embs");
    get_aug_emb2();

    // 3. emb = emb + aug_emb if aug_emb is not None else emb
    LOGI("[Unet:] combining emb with aug_emb");
    auto* emb = static_cast<float*>(ws_.data("emb"));
    auto* aug_emb = static_cast<float*>(ws_.data("aug_emb"));
    if (!emb) {
        LOGE("combine embeds: base emb missing");
        return;
    }
    if (!aug_emb) {
        LOGE("combine embeds: aug_emb missing");
        return;
    }
    const size_t embBytes = ws_.sizeOf("emb");
    const size_t augBytes = ws_.sizeOf("aug_emb");
    if (embBytes != augBytes) {
        LOGE("combine_embeds_add: size mismatch emb=%zu aug=%zu", embBytes, augBytes);
        return;
    }
    auto* emb_info = ws_.tinfoOf("emb");
    size_t n;
    if (!emb_info) {
        n = embBytes / sizeof(float);
    } else {
        n = emb_info->numel();
    }
    LOGI("EMB n %lu", n);
    const auto& aug_emb_tinfo = ws_.tinfoOf("aug_emb");
    logNumbers(aug_emb, *aug_emb_tinfo, "aug_emb", 12, 'w', "Aug Emb:");

    float* dst = emb;
    const float* src = aug_emb;
    // simple vector add
    for (size_t i = 0; i < n; ++i) dst[i] += src[i];

    // 4. conv_in: latent --> sample
    LOGI("Running Unet inConv...");
//    check that inputs are present
    std::string conv_in_name = unet_.inConv->last().session->inputs().back().name;
    auto conv_in_wsName = unet_.inConv->last().inputBinding.at(conv_in_name);
    if (!ws_.has(conv_in_wsName) || !ws_.data(conv_in_wsName)) {
        LOGE("[UNET:] inputs to inConv do not exist!");
        return;
    }
    unet_execution_summary += runGraph(*unet_.inConv, true);

    // 5. down-, mid-, and up-blocks: (sample, temb, encoder_hidden_states) ==> one final output
    LOGI("Running Unet cross-attention blocks...");
    const auto& nodes = unet_.unet_blocks->getNodes();
    for (const auto& n : nodes[0].session.get()->inputs()) {
        const auto& wsName = nodes[0].inputBinding.at(n.name);
        if (!ws_.data(wsName)) {
            LOGE("[Unet:] inputs %s (workspace: %s) to u_blocks does not exist", n.name.c_str(), wsName.c_str());
            return;
        }
    }
    unet_execution_summary += runGraph(*unet_.unet_blocks, true);

    LOGW("NODE NAMES ===========");
    for (const auto& node : unet_.unet_blocks_2->getNodes()) {
        LOGW("%s", node.name.c_str());
    }

    auto& tinfos00 = unet_.unet_upblock00->last().session.get()->outputs();
    LOGW("HIDDEN STATES OF UPBLOCK00 DIMENSION:");
    for (auto& tnfo : tinfos00) {
        if (tnfo.name == "output_0") {
            for (auto& d : tnfo.dims) {
                LOGW("%lu", d);
            }
        }
    }

    const auto& tinfos01 = unet_.unet_upblock01->last().session.get()->outputs();
    LOGW("HIDDEN STATES OF UPBLOCK01 DIMENSION:");
    for (auto& tnfo : tinfos01) {
        if (tnfo.name == "output_0") {
            for (auto& d : tnfo.dims) {
                LOGW("%lu", d);
            }
        }
    }

    const auto hidden_states_wsName = unet_.unet_blocks_2->getNode("upblock02").inputBinding.at("hidden_states");
    if (hidden_states_wsName.empty()) {
        LOGE("[Unet:] Did not find workspace tensor name for hidden_states -- likely due to node name mistmatch.");
        return;
    }
    auto* hidden_states = static_cast<float*>(ws_.data(hidden_states_wsName));
    const auto& hidden_states_tinfo = ws_.tinfoOf(hidden_states_wsName);
    LOGW("HIDDEN STATES DIMENSION:");
    for (auto& d : hidden_states_tinfo->dims) {
        LOGW("%lu", d);
    }

    LOGI("Running upblock 00...");
    unet_execution_summary += runGraph(*unet_.unet_upblock00, true);
    LOGI("Transposing in place");
    transpose_inplace_nhwc_nchw(hidden_states, 1, 1280, 32, 32);

    LOGI("Running upblock 01...");
    unet_execution_summary += runGraph(*unet_.unet_upblock01, true);
    LOGI("Transposing in place");
    transpose_inplace_nhwc_nchw(hidden_states, 1, 1280, 32, 32);


    const auto& tinfos = unet_.unet_blocks_2->getNode("upblock02").session.get()->inputs();
    LOGW("HIDDEN STATES OF BLOCK02 DIMENSION:");
    for (auto& tnfo : tinfos) {
        if (tnfo.name == "hidden_states") {
            for (auto& d : tnfo.dims) {
                LOGW("%lu", d);
            }
        }
    }
    LOGI("Running upblock 02 and 1...");
    unet_execution_summary += runGraph(*unet_.unet_blocks_2, true);

    // 6. conv_norm_out, conv_act, conv_out ==> predicted noise
    LOGI("Running Unet outConv...");
    unet_execution_summary += runGraph(*unet_.outConv, true);

    LOGI("Finished running Unet!");
}

std::vector<float> SDXLPipeline::overall_pipeline(const int32_t* ids1_77,
                                                  const int32_t* ids2_77,
                                                  bool decode_only) {

    std::string log;
    std::vector<float> dummy_out;

    std::string latents_name = vae_decoder_->last().session->inputs().back().name;
    auto latents_wsName = vae_decoder_->last().inputBinding.at(latents_name);
    if (latents_wsName.empty()) latents_wsName = "latents";
    auto latents = static_cast<float*>(ws_.data(latents_wsName));
    const auto& latents_tinfo = ws_.tinfoOf(latents_wsName);
    auto latent_elems = latents_tinfo->numel();

    if (decode_only) {
        bool read_from_file = readFileToBuffer("/sdcard/Android/data/com.example.snpechainingdemo/files/latent_7.bin", latents, latents_tinfo->bytes());
        if (read_from_file) {
            LOGI("[Pipeline:] Read latent from file!");
        } else {
            LOGE("[Pipeline:] Could not read from file!");
        }
    } else {

        log = run_encoders(ids1_77, ids2_77);
        auto *encoder_hidden_states = static_cast<float *>(ws_.data("encoder_hidden_states"));
        const auto &encoder_hidden_states_tinfo = ws_.tinfoOf("encoder_hidden_states");
        logNumbers(encoder_hidden_states, *encoder_hidden_states_tinfo, "encoder_hidden_states", 8,
                   'w', "Encoder Hidden States:");


        std::vector<float> timesteps;
        //    retrieve_timesteps();
        LOGI("[Pipeline:] retrieving timesteps...");
        scheduler_->set_timesteps(inf_config_.num_inference_steps);
        timesteps = scheduler_->timesteps;
        LOGI("[Pipeline:] Obtained timesteps %f", timesteps[0]);
        LOGW("[Pipeline:] scheduler init noise sigma: %f", scheduler_->init_noise_sigma());


        LOGI("[Pipeline:] preparing latents...");

        auto [latents, latent_elems] = prepare_latents(inf_config_.width, inf_config_.height,
                                                       latents_wsName.c_str(), 13540, true);
        if (!latents || latent_elems == 0) {
            LOGE("latents not allocated");
            return dummy_out;
        }
        LOGI("[Pipeline:] latents prepared!");

        logNumbers(latents, *latents_tinfo, latents_wsName, 8, 'w', "Initial Latents");
        //    extra_step_kwargs = {"generator": generator};

        LOGI("[Pipeline:] computing time_ids...");
        bool ok_time_ids = add_time_ids_toWorkspace(ws_,
                                                    "time_ids",
                                                    {inf_config_.height, inf_config_.width},
                                                    {0, 0},
                                                    {inf_config_.height, inf_config_.width},
                                                    net_config_.unet_add_embedding_linear_1_in_features,
                                                    net_config_.unet_addition_time_embed_dim,
                                                    net_config_.text_encoder_projection_dim,
                                                    &log);

        //    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1) // should be 1 so remains
        if (!ok_time_ids) {
            LOGE("Obtaining time_ids was not possible!");
            return dummy_out;
        }

        //    std::vector<float> latent_model_input;
        LOGI("[Pipeline:] getting noise_pred name");
        std::string noise_pred_name = unet_.outConv->last().session.get()->outputs().back().name;
        auto noise_pred_wsName = unet_.outConv->last().outputBinding.at(noise_pred_name);
        noise_pred_wsName = !noise_pred_wsName.empty() ? noise_pred_wsName : "noise_pred";
        const auto &noise_pred_tinfo = ws_.tinfoOf(noise_pred_wsName);
        auto *noise_pred = static_cast<float *>(ws_.data(noise_pred_wsName));
        logNumbers(noise_pred, *noise_pred_tinfo, noise_pred_wsName, 8, 'w', "Initial Noise");

        // construct latent_model_input in workspace
        LOGI("[Pipeline:] constructing latent_model_input if it does not exist...");
        std::string latent_model_input_name = unet_.inConv->last().session.get()->inputs().back().name;
        const auto &latent_model_input_tinfo = unet_.inConv->last().session.get()->inputs().back();
        auto latent_model_input_wsName = unet_.inConv->last().inputBinding.at(
                latent_model_input_name);
        std::string emsg;
        if (!ensureWorkspaceBuffer(ws_, latent_model_input_wsName, latent_model_input_tinfo,
                                   &emsg)) {
            //    if (!ensureWorkspaceBuffer(ws_, latent_model_input_wsName, latent_model_input_tinfo.bytes(), &emsg)) {
            LOGE("Workspace alloc (input) failed for %s : %s", latent_model_input_name.c_str(),
                 emsg.c_str());
            return dummy_out;
        }
        auto *latent_model_input = static_cast<float *>(ws_.data(latent_model_input_wsName));
        logNumbers(latent_model_input, latent_model_input_tinfo, latent_model_input_wsName, 8, 'w',
                   "Initial Latent Model Input");

        LOGI("[Pipeline:] entering the denoising loop...");
        int k = 0;
        int get_latent = 0;//5;
        for (auto t: timesteps) {
            //        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            LOGI("[Pipeline:] Timestep %f", t);
            LOGI("[Pipeline:] scaling model input...");
            if (k == get_latent && get_latent != 0) {
                std::stringstream filename;
                //            filename << "/sdcard/Android/data/com.example.snpechainingdemo/files/latent_" << k-1 << ".bin";
                filename
                        << "/sdcard/Android/data/com.example.snpechainingdemo/files/initial_latent_"
                        << k << ".bin";
                bool read_from_file = readFileToBuffer(filename.str(), latents,
                                                       latents_tinfo->bytes());
                if (read_from_file) {
                    LOGW("[Pipeline:] Read %s from file for step %lu!", filename.str().c_str(), k);
                } else {
                    LOGE("[Pipeline:] Could not read %s from file!", filename.str().c_str());
                }
            }
            scheduler_->scale_model_input_inplace(latents, latent_model_input, latent_elems, t);
            logNumbers(latent_model_input, latent_model_input_tinfo, latent_model_input_wsName, 8,
                       'w', "Latent Model Input after scaling");

            LOGI("[Pipeline:] unet...");
            //        unet_wrapper(static_cast<int32_t>(t));
            if (k >= get_latent) {
                unet_wrapper2(t);
//                std::stringstream filename;
//                filename
//                        << "/sdcard/Android/data/com.example.snpechainingdemo/files/noise_pred_7.bin";
//                bool read_from_file = readFileToBuffer(filename.str(), noise_pred,
//                                                       noise_pred_tinfo->bytes());
//                if (read_from_file) {
//                    LOGW("[Pipeline:] Read %s from file for step %lu!", filename.str().c_str(),
//                         k);
//                } else {
//                    LOGE("[Pipeline:] Could not read %s from file!", filename.str().c_str());
//                }
            }
            logNumbers(noise_pred, *noise_pred_tinfo, noise_pred_wsName, 8, 'w',
                       "Noise Pred after UNet");

            // latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            //        auto* noise_pred = static_cast<float*>(ws_.data(noise_pred_wsName));
            LOGI("[Pipeline:] scheduler step %lu...", scheduler_->step_index());
            scheduler_->step_inplace(noise_pred, t, latents, latent_elems);
            logNumbers(latents, *latents_tinfo, latents_wsName, 8, 'w', "Latents after step");
            k += 1;
            //        if (k >= 7) break;
        }
        LOGI("[Pipeline:] Done with all iterations!");

        //    latents = latents / self.vae.config.scaling_factor // scaling factor 0.13025
        //    size_t n = latent_elems / sizeof(float);

        // read latent from file
        //    bool read_from_file = readFileToBuffer("/sdcard/Android/data/com.example.snpechainingdemo/files/undecoded_latent.bin", latents, latents_tinfo->bytes());
        //    if (read_from_file) {
        //        LOGI("[Pipeline:] Read latent from file!");
        //    }

        //    bool read_from_file = readFileToBuffer("/sdcard/Android/data/com.example.snpechainingdemo/files/latent_7.bin", latents, latents_tinfo->bytes());
        //    if (read_from_file) {
        //        LOGI("[Pipeline:] Read latent from file!");
        //    } else {
        //        LOGE("[Pipeline:] Could not read from file!");
        //    }
    }


    for (size_t i = 0; i < latent_elems; ++i) latents[i] /= net_config_.vae_config_scaling_factor;
    logNumbers(latents, *latents_tinfo, latents_wsName, 8, 'w', "Latents after final scaling");
// should use fp32 VAE decoder
// image = self.vae.decode(latents, return_dict=False)[0]

    auto dumpTI = [&](const char* name){
        const auto* ti = ws_.tinfoOf(name);
        if (!ti) { LOGE("[VAE] no tinfo for '%s'", name); return; }
        size_t elems = 1;
        for (auto d : ti->dims) elems *= d;
        LOGI("[VAE] %s dims=%zu x %zu x %zu x %zu  elemBytes=%zu  -> bytes()=%zu",
             name,
             ti->dims.size()>0?ti->dims[0]:0,
             ti->dims.size()>1?ti->dims[1]:0,
             ti->dims.size()>2?ti->dims[2]:0,
             ti->dims.size()>3?ti->dims[3]:0,
             ti->elementBytes,
             ti->bytes());
        LOGI("[VAE] ws sizeOf('%s')=%zu", name, ws_.sizeOf(name));
    };

    dumpTI(latents_wsName.c_str());   // VAE input
    dumpTI("decoded");         // VAE output

    // keep only the latents; free big encoder/UNet intermediates
    LOGI("[Pipeline:] RELEASING WORSPACE TENSORS ===================");
    const char* toFree[] = {
        "prompt_embeds_1", "prompt_embeds_2", "prompt_embeds",
        "pooled_embeds", "aug_emb", "emb",
        "encoder_hidden_states", "latent_model_input",
        "sample_for_ublocks",
        "db_res0","db_res1","db_res2","db_res3","db_res4","db_res5","db_res6",
        "out_hidden_d2","db1_hidden",
        "out_hidden_m", "out_hidden_u00",
        "in_hidden_u01", "out_hidden_01",
        "in_hidden_02", "up_hidden0"
        // any skip / mid outputs you cached, etc...
    };
//    for (auto n : toFree) { if (ws_.has(n)) ws_.release(n); }

//    usleep(20*1000);
    const int C = 3;
    std::string decoded_name = vae_decoder_->last().session.get()->outputs().back().name;
    LOGI("%s", decoded_name.c_str());
    auto  decoded_wsName = vae_decoder_->last().outputBinding.at(decoded_name);
    if (!ws_.has(decoded_wsName)) {
        LOGE("[Pipeline:] Decoder output not present in workspace!");
        return dummy_out;
    }
    auto decoded_info = vae_decoder_->last().session.get()->outputs().back();// ws_.tinfoOf(decoded_wsName);
    LOGI("decoded latent has shapes:");
    for (const auto& d : decoded_info.dims) {
        LOGI("%lu", d);
    }
    auto* decoded = static_cast<float*>(ws_.data(decoded_wsName));
    logNumbers(decoded, decoded_info, decoded_wsName, 8, 'w', "Initial Decoded");
    LOGI("[Pipeline:] Calling VAE decoder...");
    log += runGraph(*vae_decoder_, false);
    LOGI("[Pipeline:] Decoding complete!");
    logNumbers(decoded, decoded_info, decoded_wsName, 8, 'w', "Decoded");
//
//// image = self.image_processor.postprocess(image, output_type=output_type)
    LOGI("[Pipeline:] Running postprocessing...");
    std::vector<float> img;
    img = postprocess_to_chw255(decoded_wsName.c_str(), C, inf_config_.height, inf_config_.width);

    LOGI("[Pipeline:] Pipeline completed!");
    return img;
//    return dummy_out;
}

std::vector<float> SDXLPipeline::postprocess_to_chw255(const char* wsName, int C, int H, int W) {

    LOGI("[Postprocessing...]");

    // Resolve decoded pointer and element count
    auto info = vae_decoder_->last().session->outputs().back(); //ws_.tinfoOf(wsName);
//    if (!info) {
//        LOGE("[Postprocessing:] no tensor info for '%s'", wsName);
//        return {};
//    }
    const size_t need = static_cast<size_t>(C) * H * W;
    const size_t have = info.numel();
    if (have < need) {
        LOGE("[Postprocessing:] '%s' too small: have=%zu, need=%zu", wsName, have, need);
        return {};
    }

    auto* decoded = static_cast<float*>(ws_.data(wsName));
    std::vector<float> out; out.resize(need);

    LOGI("[Postprocessing:] denormalizing (CHW -> 0..255, C=%d H=%d W=%d)", C, H, W);
    for (size_t i = 0; i < need; ++i) {
        float v = decoded[i] * 0.5f + 0.5f;   // [-1,1] -> [0,1]
        v = std::max(0.0f, std::min(1.0f, v));
        out[i] = std::round(v * 255.0f);
    }
    return out;
}

