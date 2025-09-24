//
// Created by Chiheb Boussema on 8/9/25.
//
#include <jni.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "hpp/inference.h"
#include "SNPE/SNPEFactory.hpp"
#include "ModelSession.hpp"
#include "GraphRunner.hpp"
#include "TensorWorkspace.hpp"
#include "MMapAsset.hpp"
#include "hpp/ParseConfig.hpp"
#include "hpp/MMapFile.h"
#include "hpp/newInferenceHelper.hpp"
#include "hpp/initTensorsHelper.h"

#define LOG_TAG "SNPE_JNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

// Build a RuntimeList from a single preference char (D/G/C).
//static zdl::DlSystem::RuntimeList makeRuntimeOrder(char pref) {
//    using zdl::DlSystem::Runtime_t;
//    zdl::DlSystem::RuntimeList lst;
//    if      (pref == 'D') lst.add(Runtime_t::DSP);
//    else if (pref == 'G') lst.add(Runtime_t::GPU);
//    else                  lst.add(Runtime_t::CPU);
//    return lst;
//}
//
//// Look up a tensor by name in metadata vector.
//static const TensorInfo* findTensor(const std::vector<TensorInfo>& v, const std::string& name) {
//    for (const auto& t : v) if (t.name == name) return &t;
//    return nullptr;
//}
//
//// Convenience: allocate a workspace buffer if not allocated yet.
//static bool ensureWorkspaceBuffer(TensorWorkspace& ws,
//                                  const std::string& wsName,
//                                  size_t bytes,
//                                  std::string* emsg) {
//    // If your TensorWorkspace doesn’t expose a 'has()' method,
//    // add one; if it does, use it here.
//    if (!ws.has(wsName)) {
//        ws.allocate(wsName, bytes);
//    } else {
//        // Validate size matches on reuse
//        const size_t existing = ws.sizeOf(wsName);
//        if (existing != bytes) {
//            if (emsg) *emsg = "Workspace tensor size mismatch for '" + wsName +
//                              "': have " + std::to_string(existing) +
//                              ", need " + std::to_string(bytes);
//            return false;
//        }
//    }
//    return true;
//}

// Keep your workspace / graph state somewhere (or return summaries only)
//static TensorWorkspace* g_ws = nullptr;   // Example: if you want to fetch outputs later
static std::unique_ptr<TensorWorkspace> g_ws;
//
////struct BuildTimes {
////    int64_t openMs=0, build1Ms=0, build2Ms=0, captureMs=0, allocMs=0, totalMs=0;
////};
////static BuildTimes g_buildTimes();
static std::unique_ptr<ModelSession> g_s1, g_s2;
static std::unique_ptr<GraphRunner> g_gr;

// Times:
//static inline int64_t msSince(std::chrono::steady_clock::time_point t0) {
//    using clock = std::chrono::steady_clock;
//    return std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
//}

static std::string g_modelDir;
static void n_setModelDirectory(JNIEnv* env, jclass, jstring jpath) {
    const char* p = env->GetStringUTFChars(jpath, nullptr);
    g_modelDir = p ? p : "";
    env->ReleaseStringUTFChars(jpath, p);
    LOGI("SNPE model base dir set to: %s", g_modelDir.c_str());
}


//std::string buildModelsAndGraph(AAssetManager* mgr, const std::string& configJson, const char defaultRuntimePref,
//                         std::unique_ptr<TensorWorkspace>& outWs,
//                         std::unique_ptr<GraphRunner>& outGraph,
//                         bool reset_session=false) {
//
//    using clock = std::chrono::steady_clock;
//    std::string log;
//
//    // 0) Parse config
//    PipelineCfg cfg;
//    {
//        std::string emsg;
//        if (!ParseConfig(configJson, cfg, &emsg)) {
//            return "Config parse failed: " + emsg;
//        }
//        if (cfg.models.empty()) {
//            return "Config has no models";
//        }
//    }
//
//    // 1) Create workspace & graph
////    auto ws = std::make_unique<TensorWorkspace>();
////    auto gr = std::make_unique<GraphRunner>(*ws);
//
//    int64_t totalAssetMs = 0, totalBuildMs = 0, totalAllocMs = 0, totalGraphMs = 0;
//
//    // 2) Process each model
//    int k = 0;
//    for (const auto &mc: cfg.models)
//    {
//        LOGI("Starting build of Model %s", mc.asset.c_str());
//        const auto tAsset0 = clock::now();
//
//        // mmap DLC
//        std::string emsg;
//        MMapAsset mappedAsset;
//        MMapFile mappedFile;
//        bool mappedOk = false;
//
//        if (g_modelDir.empty() and !cfg.baseDir.empty()) {
//            g_modelDir = cfg.baseDir + "/";
//        }
//        if (!g_modelDir.empty()) {
//            std::string full = g_modelDir + "/" + mc.asset;
//            LOGI("Trying DLC from file: %s", full.c_str());
//            if (mappedFile.openPath(full.c_str(), &emsg)) {
//                mappedOk = true;
//            } else {
//                LOGW("File open failed: %s", emsg.c_str());
//                emsg.clear();
//            }
//        }
//
//        if (!mappedOk) {
//            LOGI("Falling back to APK asset: %s", mc.asset.c_str());
//            if (!mappedAsset.openUncompressed(mgr, mc.asset.c_str(), &emsg)) {
//                return "Failed to mmap asset '" + mc.asset + "': " + emsg;
//            }
//        }
//
//        totalAssetMs += msSince(tAsset0);
//        LOGI("Model %s opened", mc.asset.c_str());
//
//        const void* dlcPtr  = mappedOk ? mappedFile.ptr  : mappedAsset.ptr;
//        size_t      dlcSize = mappedOk ? mappedFile.size : mappedAsset.size;
//
//        // 3) Build ModelSession
//        const auto tBuild0 = clock::now();
//        ModelSession::Options opt;
//        // Runtime order: use per-model pref if present else default
//        const char pref = (mc.runtime == 0 ? defaultRuntimePref : mc.runtime);
//        opt.runtimeOrder = makeRuntimeOrder(pref);
//        opt.perf = zdl::DlSystem::PerformanceProfile_t::BURST;
//        opt.useUserSuppliedBuffers = true;
//        opt.initCache = true; //false;
//
//        std::string buildLog;
//        auto session = ModelSession::Create(static_cast<const uint8_t *>(dlcPtr),
//                                            dlcSize, opt, &buildLog);
//        totalBuildMs += msSince(tBuild0);
//        log += "[Build " + mc.name + "] " + buildLog;
//        LOGI("Session for model %s created", mc.asset.c_str());
//
//        if (!session) {
//            return "SNPE build failed for '" + mc.name + "'";
//        }
//
//        // 4) Validate inputs/outputs exist & allocate workspace for any new names
//        const auto tAlloc0 = clock::now();
//
//        // Inputs
//        for (const auto &kv: mc.inputs)
//        {
//            const std::string &modelTensor = kv.first;
//            const std::string &wsTensor = kv.second;
//
//            const TensorInfo *ti = findTensor(session->inputs(), modelTensor);
//            if (!ti) {
//                return "Model '" + mc.name + "': input tensor not found: " + modelTensor;
//            }
//            // Inputs are bound from workspace too (so they can be fed by earlier models or app)
////            if (!ensureWorkspaceBuffer(*ws, wsTensor, ti->bytes(), &emsg)) {
//            if (!ensureWorkspaceBuffer(*outWs, wsTensor, ti->bytes(), &emsg)) {
//                return "Workspace alloc (input) failed for '" + mc.name + "': " + emsg;
//            }
//        }
//
//        // Outputs
//        for (const auto& kv : mc.outputs) {
//            const std::string& modelTensor = kv.first;
//            const std::string& wsTensor    = kv.second;
//
//            const TensorInfo* ti = findTensor(session->outputs(), modelTensor);
//            if (!ti) {
//                return "Model '" + mc.name + "': output tensor not found: " + modelTensor;
//            }
//            if (!ensureWorkspaceBuffer(*outWs, wsTensor, ti->bytes(), &emsg)) {
//                return "Workspace alloc (output) failed for '" + mc.name + "': " + emsg;
//            }
//        }
//
//        totalAllocMs += msSince(tAlloc0);
//
//        // 5) Add node to graph (strict zero-copy)
//        const auto tGraph0 = clock::now();
//        GraphRunner::Node node;
//        node.name     = mc.name;
//        node.session  = std::move(session);
//        node.inputBinding  = mc.inputs;   // modelTensor -> workspaceTensor
//        node.outputBinding = mc.outputs;  // modelTensor -> workspaceTensor
//
//        if (!outGraph->addNode(std::move(node), /*strictZeroCopy=*/true)) {
//            return "addNode failed for '" + mc.name + "'";
//        }
//        totalGraphMs += msSince(tGraph0);
//        LOGI("Graph node for model %s added", mc.asset.c_str());
//
//        if (reset_session) {
////            LOGI("[BUILDING] Resetting session for node %s", node.name.c_str());
////            node.session.reset();
////            LOGI("[BUILDING] Session reset for node %s", node.name.c_str());
//            if (k >= 2) {
//                LOGI("[BUILDING] Resetting graph!");
//                outGraph->clear();
//                k = 0;
//
////                usleep(20 * 1000);
//            }
//        }
//        k += 1;
//
//    }
//
////    outWs   = std::move(ws);
////    outGraph= std::move(gr);
//
//    char buf[256];
//    snprintf(buf, sizeof(buf),
//             "Build OK. assets=%lld ms, build=%lld ms, alloc=%lld ms, graph=%lld ms",
//             (long long)totalAssetMs, (long long)totalBuildMs,
//             (long long)totalAllocMs, (long long)totalGraphMs);
//    log = std::string(buf) + "\n" + log;
//    return log;
//}

//std::string runGraph(GraphRunner& gr) {
//    auto T0 = std::chrono::steady_clock::now();
//    auto infos = gr.runAll(/*reset_session*/ true);
//    auto T1 = std::chrono::steady_clock::now();
//    auto execMs = std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count();
//    LOGI("Graph Execution time: %lld", execMs);
//
//    // 7) Summarize result
//    std::string summary;
//    for (auto& e : infos) {
//        summary += e.name + " runtime=" + e.runtime + " time=" + std::to_string(e.ms) + "ms "
//                   + (e.ok ? "OK\n" : "FAIL\n");
//    }
//
//    return summary;
//}

static bool readAssetToString(AAssetManager* mgr,
                       const char* filename,
                       std::string& out,
                       std::string* emsg) {
    AAsset* a = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    if (!a) {
        if (emsg) *emsg = std::string("Asset open failed: ") + filename;
        return false;
    }
    size_t len = AAsset_getLength(a);
    out.resize(len);
    int r = AAsset_read(a, out.data(), len);
    AAsset_close(a);
    if (r != (int)len) {
        if (emsg) *emsg = "Asset read truncated";
        return false;
    }
    return true;
}


//static jstring n_executeInference(JNIEnv* env, jclass, jobject assetManager, jchar runtimePref) {
//
//    std::unique_ptr<TensorWorkspace> ws_;
//    std::unique_ptr<GraphRunner> gr_;
//
//    ws_.reset(new TensorWorkspace());
//    gr_.reset(new GraphRunner(*ws_));
//
//    // create asset manager
//    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
//
//    // read config file
//    std::string cfgText;
//    std::string emsg;
//    std::string config_filename = "modelsConfig.json";
//    if (!readAssetToString(mgr, config_filename.c_str(), cfgText, &emsg)) {
//        return env->NewStringUTF(("Config read failed: " + emsg).c_str());
//    }
//
//    // create models, allocate buffers and build graph
//    std::string buildingLog;
//    buildingLog = buildModelsAndGraph(mgr, cfgText, runtimePref, ws_, gr_);
//
//    // execute graph
//    if (!gr_) return env->NewStringUTF("Graph not built");
//    std::string execution_summary;
//    execution_summary = runGraph(*gr_);
//
//    return env->NewStringUTF(execution_summary.c_str());
//
//}

static jstring n_buildArbitrary(JNIEnv* env, jclass, jobject assetManager, jchar runtimePref) {

    g_ws.reset(new TensorWorkspace());
    g_gr.reset(new GraphRunner(*g_ws));

    // create asset manager
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    // read config file
    std::string cfgText;
    std::string emsg;
    std::string config_filename = "modelsConfig.json";
    if (!readAssetToString(mgr, config_filename.c_str(), cfgText, &emsg)) {
        return env->NewStringUTF(("Config read failed: " + emsg).c_str());
    }
    // parse config file
    PipelineCfg cfg;
    {
        std::string emsg;
        if (!ParseConfig(cfgText, cfg, &emsg)) {
            return env->NewStringUTF(("Config parse failed: " + emsg).c_str());
        }
        if (cfg.models.empty()) {
            return env->NewStringUTF("Config has no models");
        }
    }

    // create models, allocate buffers and build graph
    std::string buildingLog;
    int k = 0;
    bool reset_graph = true; //false;
    for (const auto &mc: cfg.models)
    {
//        if (k >=2) {
//            LOGI("[BUILDING] Resetting graph!");
//            reset_graph = true;
//            k = 0;
//        } else {
//            reset_graph = false;
//        }
        buildingLog = buildModelAndGraph(mgr,
                            g_modelDir,
                            cfg,
                            mc,
                            runtimePref,
                            *g_ws,
                            *g_gr,
                            buildingLog,
                            reset_graph);
        k += 1;
    }
    {
        std::string semsg;
        LOGI("Seeding input tensors...");
        if (!seedRequiredInputs(cfg, *g_ws, mgr, &semsg)) {
            return env->NewStringUTF(("Input seeding failed: " + semsg).c_str());
        }
    }

    return env->NewStringUTF(buildingLog.c_str());
}

static jstring n_rebuildArbitrary(JNIEnv* env, jclass) {

    std::string rebuildingLog;
    for (auto& n: g_gr->getNodes())
    {
        LOGI("Rebuilding for node %s", n.name.c_str());
        n.session.get()->reCreate(&rebuildingLog);
        LOGI("REBUILDING SUCCESSFUL! undoing...");
        n.session.get()->reset();
    }
    return env->NewStringUTF(rebuildingLog.c_str());

}

//static jstring n_buildArbitrary(JNIEnv* env, jclass, jobject assetManager, jchar runtimePref) {
//
//    g_ws.reset(new TensorWorkspace());
//    g_gr.reset(new GraphRunner(*g_ws));
//
//    // create asset manager
//    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
//
//    // read config file
//    std::string cfgText;
//    std::string emsg;
//    std::string config_filename = "modelsConfig.json";
//    if (!readAssetToString(mgr, config_filename.c_str(), cfgText, &emsg)) {
//        return env->NewStringUTF(("Config read failed: " + emsg).c_str());
//    }
//
//    // create models, allocate buffers and build graph
//    std::string buildingLog;
//    buildingLog = buildModelsAndGraph(mgr, cfgText, runtimePref, g_ws, g_gr, true);
//    return env->NewStringUTF(buildingLog.c_str());
//}



//static jstring n_buildGraph(JNIEnv* env, jclass /*cls*/, jobject assetManager, jchar runtimePref) {
//    using clock = std::chrono::steady_clock;
//
//    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
//    std::string emsg;
//    // 1) Read DLCs into memory
//    auto readAsset = [&](const char* name, std::vector<uint8_t>& buf) -> bool {
//        AAsset* a = AAssetManager_open(mgr, name, AASSET_MODE_UNKNOWN);
//        if (!a) { LOGE("asset open failed: %s", name); return false; }
//        buf.resize(AAsset_getLength(a));
//        AAsset_read(a, buf.data(), buf.size());
//        AAsset_close(a);
//        return true;
//    };
//
////    std::vector<uint8_t> dlc1, dlc2;
//    const std::string model1_name = "unet_downblock1_8Gen2_prepared.dlc";
//    const std::string model2_name = "unet_downblock2_8Gen2_prepared.dlc";
////    auto T0 = clock::now();
////    if (!readAsset(model1_name.c_str(), dlc1) || !readAsset(model2_name.c_str(), dlc2)) {
////        return env->NewStringUTF("Failed to read DLCs");
////    }
////    auto T1 = clock::now();
////    auto AssetLoadingMs = std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count();
////    LOGI("Asset Loading time: %lld", AssetLoadingMs);
//    MMapAsset dlc1, dlc2;
////    std::vector<uint8_t> dlc1Buf, dlc2Buf; // fallback buffer
////    const uint8_t* bytes = nullptr;
////    size_t         nbytes = 0;
//    auto T0 = clock::now();
//    if (!dlc1.openUncompressed(mgr, model1_name.c_str(), &emsg)) {
//        LOGE("DLC1 map failed: %s", emsg.c_str());
//        return env->NewStringUTF("Map model1 failed");
//    }
//    if (!dlc2.openUncompressed(mgr, model2_name.c_str(), &emsg)) {
//        LOGE("DLC2 map failed: %s", emsg.c_str());
//        return env->NewStringUTF("Map model2 failed");
//    }
//    auto T1 = clock::now();
//    auto AssetLoadingMs = std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count();
//    LOGI("Asset Loading time: %lld", AssetLoadingMs);
//
//    // 2) Build ModelSession options
//    ModelSession::Options opt;
//    // prefer HTP then CPU; or just set order empty and let your Create() fallback fill it
//    if (runtimePref == 'D') { opt.runtimeOrder.add(zdl::DlSystem::Runtime_t::DSP); }
//    else if (runtimePref == 'G') { opt.runtimeOrder.add(zdl::DlSystem::Runtime_t::GPU); }
//    else { opt.runtimeOrder.add(zdl::DlSystem::Runtime_t::CPU); }
//    opt.perf = zdl::DlSystem::PerformanceProfile_t::BURST;
//    opt.useUserSuppliedBuffers = true;
//    opt.initCache = false;
//
//    std::string buildLog;
//    auto tB0 = clock::now();
////    g_s1 = ModelSession::Create(dlc1.data(), dlc1.size(), opt, &buildLog);
//    g_s1 = ModelSession::Create(static_cast<const uint8_t*>(dlc1.ptr), dlc1.size, opt, &buildLog);
//    auto tB1 = clock::now();
//    auto build1Ms = std::chrono::duration_cast<std::chrono::milliseconds>(tB1 - tB0).count();
//    LOGI("Model 1 Building time: %lld", build1Ms);
//
//    auto tB2 = clock::now();
////    g_s2 = ModelSession::Create(dlc2.data(), dlc2.size(), opt, &buildLog);
//    g_s2 = ModelSession::Create(static_cast<const uint8_t*>(dlc2.ptr), dlc2.size, opt, &buildLog);
//    auto tB3 = clock::now();
//    auto build2Ms = std::chrono::duration_cast<std::chrono::milliseconds>(tB3 - tB2).count();
//    LOGI("Model 2 Building time: %lld", build2Ms);
//
//    if (!g_s1 || !g_s2) {
//        LOGE("Build failed:\n%s", buildLog.c_str());
//        return env->NewStringUTF(("Build failed:\n" + buildLog).c_str());
//    }
//
//    // 3) Allocate workspace (names/sizes from tensor metadata)
//    g_ws.reset(new TensorWorkspace());
//    auto findT = [](const std::vector<TensorInfo>& v, const char* name)->const TensorInfo*{
//        for (auto& t: v) if (t.name == name) return &t; return nullptr;
//    };
//    //discover s1.s2 tensors
//    auto T_disc0 = clock::now();
//    for (auto& t : g_s1->inputs())  LOGI("[s1] IN  %s", t.name.c_str());
//    for (auto& t : g_s1->outputs()) LOGI("[s1] OUT %s", t.name.c_str());
//    for (auto& t : g_s2->inputs())  LOGI("[s2] IN  %s", t.name.c_str());
//    for (auto& t : g_s2->outputs()) LOGI("[s2] OUT %s", t.name.c_str());
//
//    const TensorInfo* s1_sample  = findT(g_s1->inputs(),  "sample");
//    const TensorInfo* s1_temb    = findT(g_s1->inputs(),  "temb");
//    const TensorInfo* s1_ehs     = findT(g_s1->inputs(),  "encoder_hidden_states");
//    const TensorInfo* s1_out_h   = findT(g_s1->outputs(), "output_0");
//    const TensorInfo* s1_res0    = findT(g_s1->outputs(), "output_1");
//    const TensorInfo* s1_res1    = findT(g_s1->outputs(), "output_2");
//    const TensorInfo* s1_res2    = findT(g_s1->outputs(), "output_3");
//    const TensorInfo* s1_res3    = findT(g_s1->outputs(), "output_4");
//    const TensorInfo* s1_res4    = findT(g_s1->outputs(), "output_5");
//    const TensorInfo* s1_res5    = findT(g_s1->outputs(), "output_6");
//    const TensorInfo* s1_res6    = findT(g_s1->outputs(), "output_7");
//
//    const TensorInfo* s2_in_h    = findT(g_s2->inputs(),  "hidden_states");
//    const TensorInfo* s2_temb    = findT(g_s2->inputs(),  "temb");
//    const TensorInfo* s2_ehs     = findT(g_s2->inputs(),  "encoder_hidden_states");
//    const TensorInfo* s2_out_h    = findT(g_s2->outputs(), "output_0");
//    const TensorInfo* s2_res0    = findT(g_s2->outputs(), "output_1");
//    const TensorInfo* s2_res1    = findT(g_s2->outputs(), "output_2");
//
//    auto T_disc1 = clock::now();
//    auto discoveryMs = std::chrono::duration_cast<std::chrono::milliseconds>(T_disc1 - T_disc0).count();
//    LOGI("Tensor discovery time: %lld", discoveryMs);
//
//    if (!s1_sample || !s1_temb || !s1_ehs || !s1_out_h || !s1_res0 || !s1_res1 || !s1_res2 ||
//        !s1_res3 || !s1_res4 || !s1_res5 || !s1_res6 ||
//        !s2_in_h   || !s2_temb || !s2_ehs || !s2_out_h || !s2_res0 || !s2_res1) {
//        LOGE("FINDING TENSORS","Missing expected tensor names—adjust bindings.");
//        return env->NewStringUTF("Missing expected tensor names—adjust bindings.");
//    }
//
//    // allocate
//    auto T_alloc0 = clock::now();
//    g_ws->allocate("sample",                s1_sample->bytes());
//    g_ws->allocate("temb",                  s1_temb->bytes());
//    g_ws->allocate("encoder_hidden_states", s1_ehs->bytes());
//
//    g_ws->allocate("out1_hidden", s1_out_h->bytes());
//    g_ws->allocate("res0",       s1_res0->bytes());
//    g_ws->allocate("res1",       s1_res1->bytes());
//    g_ws->allocate("res2",       s1_res2->bytes());
//    g_ws->allocate("res3",       s1_res3->bytes());
//    g_ws->allocate("res4",       s1_res4->bytes());
//    g_ws->allocate("res5",       s1_res5->bytes());
//    g_ws->allocate("res6",       s1_res6->bytes());
//
//    g_ws->allocate("final_out",  s2_out_h->bytes());
//    g_ws->allocate("res7",       s2_res0->bytes());
//    g_ws->allocate("res8",       s2_res1->bytes());
//
//    // 4) Seed inputs (for now fill random / zeros, just to prove the chain works)
//    std::memset(g_ws->data("sample"),                0, g_ws->sizeOf("sample"));
//    std::memset(g_ws->data("temb"),                  0, g_ws->sizeOf("temb"));
//    std::memset(g_ws->data("encoder_hidden_states"), 0, g_ws->sizeOf("encoder_hidden_states"));
//    auto T_alloc1 = clock::now();
//    auto allocMs = std::chrono::duration_cast<std::chrono::milliseconds>(T_alloc1 - T_alloc0).count();
//    LOGI("Tensor allocation time: %lld", allocMs);
//
//    // buid graph runner
//    g_gr.reset(new GraphRunner(*g_ws));
//    auto T_graph0 = clock::now();
//    GraphRunner::Node n1;
//    n1.name = "Model1";
//    n1.session = std::move(g_s1);
//    n1.inputBinding  = {
//            {"sample",                "sample"},
//            {"temb",                  "temb"},
//            {"encoder_hidden_states", "encoder_hidden_states"},
//    };
//    n1.outputBinding = {
//            {"output_0", "out1_hidden"},
//            {"output_1",       "res0"},
//            {"output_2",       "res1"},
//            {"output_3",       "res2"},
//            {"output_4",       "res3"},
//            {"output_5",       "res4"},
//            {"output_6",       "res5"},
//            {"output_7",       "res6"},
//    };
//    if (!g_gr->addNode(std::move(n1), /*strictZeroCopy=*/true)) {
//        return env->NewStringUTF("addNode(Model1) failed");
//    }
//
//    GraphRunner::Node n2;
//    n2.name = "Model2";
//    n2.session = std::move(g_s2);
//    n2.inputBinding  = {
//            {"hidden_states",         "out1_hidden"},
//            {"temb",                  "temb"},
//            {"encoder_hidden_states", "encoder_hidden_states"},
//    };
//    n2.outputBinding = {
//            {"output_0", "final_out"},
//            {"output_1", "res7"},
//            {"output_2", "res8"},
//    };
//    if (!g_gr->addNode(std::move(n2), /*strictZeroCopy=*/true)) {
//        return env->NewStringUTF("addNode(Model2) failed");
//    }
//    auto T_graph1 = clock::now();
//    auto graphMs = std::chrono::duration_cast<std::chrono::milliseconds>(T_graph1 - T_graph0).count();
//    LOGI("Graph building time: %lld", graphMs);
//
//    char msg[256];
//    auto buildMs = AssetLoadingMs + build1Ms + build2Ms + discoveryMs + allocMs + graphMs;
//    snprintf(msg, sizeof(msg), "Build OK in %lld ms. final_out bytes=%zu",
//             (long long)buildMs, g_ws->sizeOf("final_out"));
//    return env->NewStringUTF(msg);
//
//}

static jstring n_runGraphOld(JNIEnv* env, jclass) {
    // 6) Run
    if (!g_gr) return env->NewStringUTF("Graph not built");
    auto T0 = std::chrono::steady_clock::now();
    auto infos = g_gr->runAll(/*reset_session*/ true);
    auto T1 = std::chrono::steady_clock::now();
    auto execMs = std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count();
    LOGI("Graph Execution time: %lld", execMs);

    // 7) Summarize result
    std::string summary;
    for (auto& e : infos) {
        summary += e.name + " runtime=" + e.runtime + " time=" + std::to_string(e.ms) + "ms "
                   + (e.ok ? "OK\n" : "FAIL\n");
    }
    return env->NewStringUTF(summary.c_str());
}

// ------------ Native implementations (static) ------------
//static jstring n_buildTwoModelGraph(JNIEnv* env, jclass /*cls*/,
//                                    jobject assetManager, jchar runtimePref) {
//
//    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
//
//    // 1) Read DLCs into memory
//    auto readAsset = [&](const char* name, std::vector<uint8_t>& buf) -> bool {
//        AAsset* a = AAssetManager_open(mgr, name, AASSET_MODE_UNKNOWN);
//        if (!a) { LOGE("asset open failed: %s", name); return false; }
//        buf.resize(AAsset_getLength(a));
//        AAsset_read(a, buf.data(), buf.size());
//        AAsset_close(a);
//        return true;
//    };
//
//    std::vector<uint8_t> dlc1, dlc2;
//    const std::string model1_name = "unet_downblock1_8Gen2_prepared.dlc";
//    const std::string model2_name = "unet_downblock2_8Gen2_prepared.dlc";
//    if (!readAsset(model1_name.c_str(), dlc1) || !readAsset(model2_name.c_str(), dlc2)) {
//        return env->NewStringUTF("Failed to read DLCs");
//    }
//
//    // 2) Build ModelSession options
//    ModelSession::Options opt;
//    // prefer HTP then CPU; or just set order empty and let your Create() fallback fill it
//    if (runtimePref == 'D') { opt.runtimeOrder.add(zdl::DlSystem::Runtime_t::DSP); }
//    else if (runtimePref == 'G') { opt.runtimeOrder.add(zdl::DlSystem::Runtime_t::GPU); }
//    else { opt.runtimeOrder.add(zdl::DlSystem::Runtime_t::CPU); }
//    opt.perf = zdl::DlSystem::PerformanceProfile_t::BURST;
//    opt.useUserSuppliedBuffers = true;
//    opt.initCache = false;
//
//    std::string buildLog;
//    auto s1 = ModelSession::Create(dlc1.data(), dlc1.size(), opt, &buildLog);
//    auto s2 = ModelSession::Create(dlc2.data(), dlc2.size(), opt, &buildLog);
//    if (!s1 || !s2) {
//        LOGE("Build failed:\n%s", buildLog.c_str());
//        return env->NewStringUTF(("Build failed:\n" + buildLog).c_str());
//    }
//
//    // 3) Workspace (simple helper you already have)
//    TensorWorkspace ws;
//
//    // Figure out sizes for tensors you’ll bind. You can query from each session:
//    auto sizeBytes = [](const TensorInfo& t){ return t.bytes(); };
//
//    // Allocate workspace blocks large enough (once).
//    // Use the sizes from the sessions’ metadata to avoid mismatches.
//    // Inputs that originate outside (e.g., coming from Kotlin later) still need WS blocks
//    // if you want to keep zero-copy end-to-end; you can memcpy into WS for now.
//    auto findT = [](const std::vector<TensorInfo>& v, const char* name)->const TensorInfo*{
//        for (auto& t: v) if (t.name == name) return &t; return nullptr;
//    };
//
////    for (auto& t : s1->inputs())  LOGI_MS("[s1] IN  %s %s", t.name.c_str(), shapeToStr(t.dims).c_str());
////    for (auto& t : s1->outputs()) LOGI_MS("[s1] OUT %s %s", t.name.c_str(), shapeToStr(t.dims).c_str());
//
//    for (auto& t : s1->inputs())  LOGI("[s1] IN  %s", t.name.c_str());
//    for (auto& t : s1->outputs()) LOGI("[s1] OUT %s", t.name.c_str());
//    for (auto& t : s2->inputs())  LOGI("[s2] IN  %s", t.name.c_str());
//    for (auto& t : s2->outputs()) LOGI("[s2] OUT %s", t.name.c_str());
//
//    const TensorInfo* s1_sample  = findT(s1->inputs(),  "sample");
//    const TensorInfo* s1_temb    = findT(s1->inputs(),  "temb");
//    const TensorInfo* s1_ehs     = findT(s1->inputs(),  "encoder_hidden_states");
//    const TensorInfo* s1_out_h   = findT(s1->outputs(), "output_0");
//    const TensorInfo* s1_res0    = findT(s1->outputs(), "output_1");
//    const TensorInfo* s1_res1    = findT(s1->outputs(), "output_2");
//    const TensorInfo* s1_res2    = findT(s1->outputs(), "output_3");
//    const TensorInfo* s1_res3    = findT(s1->outputs(), "output_4");
//    const TensorInfo* s1_res4    = findT(s1->outputs(), "output_5");
//    const TensorInfo* s1_res5    = findT(s1->outputs(), "output_6");
//    const TensorInfo* s1_res6    = findT(s1->outputs(), "output_7");
//
//    const TensorInfo* s2_in_h    = findT(s2->inputs(),  "hidden_states");
//    const TensorInfo* s2_temb    = findT(s2->inputs(),  "temb");
//    const TensorInfo* s2_ehs     = findT(s2->inputs(),  "encoder_hidden_states");
//    const TensorInfo* s2_out_h    = findT(s2->outputs(), "output_0");
//    const TensorInfo* s2_res0    = findT(s2->outputs(), "output_1");
//    const TensorInfo* s2_res1    = findT(s2->outputs(), "output_2");
//
//    if (!s1_sample || !s1_temb || !s1_ehs || !s1_out_h || !s1_res0 || !s1_res1 || !s1_res2 ||
//            !s1_res3 || !s1_res4 || !s1_res5 || !s1_res6 ||
//        !s2_in_h   || !s2_temb || !s2_ehs || !s2_out_h || !s2_res0 || !s2_res1) {
//        LOGE("FINDING TENSORS","Missing expected tensor names—adjust bindings.");
//        return env->NewStringUTF("Missing expected tensor names—adjust bindings.");
//    }
//
//    ws.allocate("sample",                s1_sample->bytes());
//    ws.allocate("temb",                  s1_temb->bytes());
//    ws.allocate("encoder_hidden_states", s1_ehs->bytes());
//
//    ws.allocate("out1_hidden", s1_out_h->bytes());
//    ws.allocate("res0",       s1_res0->bytes());
//    ws.allocate("res1",       s1_res1->bytes());
//    ws.allocate("res2",       s1_res2->bytes());
//    ws.allocate("res3",       s1_res3->bytes());
//    ws.allocate("res4",       s1_res4->bytes());
//    ws.allocate("res5",       s1_res5->bytes());
//    ws.allocate("res6",       s1_res6->bytes());
//
//    ws.allocate("final_out",  s2_out_h->bytes());
//    ws.allocate("res7",       s2_res0->bytes());
//    ws.allocate("res8",       s2_res1->bytes());
//
//    // 4) Seed inputs (for now fill random / zeros, just to prove the chain works)
//    std::memset(ws.data("sample"),                0, ws.sizeOf("sample"));
//    std::memset(ws.data("temb"),                  0, ws.sizeOf("temb"));
//    std::memset(ws.data("encoder_hidden_states"), 0, ws.sizeOf("encoder_hidden_states"));
//
//    // 5) Build GraphRunner and bind
//    GraphRunner gr(ws);
//
//    GraphRunner::Node n1;
//    n1.name = "Model1";
//    n1.session = std::move(s1);
//    n1.inputBinding  = {
//            {"sample",                "sample"},
//            {"temb",                  "temb"},
//            {"encoder_hidden_states", "encoder_hidden_states"},
//    };
//    n1.outputBinding = {
//            {"output_0", "out1_hidden"},
//            {"output_1",       "res0"},
//            {"output_2",       "res1"},
//            {"output_3",       "res2"},
//            {"output_4",       "res3"},
//            {"output_5",       "res4"},
//            {"output_6",       "res5"},
//            {"output_7",       "res6"},
//    };
//    if (!gr.addNode(std::move(n1), /*strictZeroCopy=*/true)) {
//        return env->NewStringUTF("addNode(Model1) failed");
//    }
//
//    GraphRunner::Node n2;
//    n2.name = "Model2";
//    n2.session = std::move(s2);
//    n2.inputBinding  = {
//            {"hidden_states",         "out1_hidden"},
//            {"temb",                  "temb"},
//            {"encoder_hidden_states", "encoder_hidden_states"},
//    };
//    n2.outputBinding = {
//            {"output_0", "final_out"},
//            {"output_1", "res7"},
//            {"output_2", "res8"},
//    };
//    if (!gr.addNode(std::move(n2), /*strictZeroCopy=*/true)) {
//        return env->NewStringUTF("addNode(Model2) failed");
//    }
//
//    // 6) Run
//    auto infos = gr.runAll();
//
//    // 7) Summarize result
//    std::string summary;
//    for (auto& e : infos) {
//        summary += e.name + " runtime=" + e.runtime + " time=" + std::to_string(e.ms) + "ms "
//                   + (e.ok ? "OK\n" : "FAIL\n");
//    }
//    return env->NewStringUTF(summary.c_str());
//}

static jboolean n_getFinalTensor(JNIEnv* env, jclass /*cls*/, jobject dstDirectBuffer) {
    if (!g_ws) return JNI_FALSE;
    void* dst = env->GetDirectBufferAddress(dstDirectBuffer);
    jlong cap = env->GetDirectBufferCapacity(dstDirectBuffer);
    if (!dst || cap <= 0) return JNI_FALSE;

    const char* kOutName = "final_out";  // whatever you allocated/bound
    void* src = g_ws->data(kOutName);
    size_t sz = g_ws->sizeOf(kOutName);
    if (!src || (jlong)sz > cap) return JNI_FALSE;

    std::memcpy(dst, src, sz);
    return JNI_TRUE;
}

static jboolean n_getTensor(JNIEnv* env, jclass /*cls*/, jobject dstDirectBuffer, jstring tensorName) {
    if (!g_ws) return JNI_FALSE;
    void* dst = env->GetDirectBufferAddress(dstDirectBuffer);
    jlong cap = env->GetDirectBufferCapacity(dstDirectBuffer);
    if (!dst || cap <= 0) return JNI_FALSE;

    const char* kOutName = env->GetStringUTFChars(tensorName, nullptr);  // whatever you allocated/bound
    void* src = g_ws->data(kOutName);
    size_t sz = g_ws->sizeOf(kOutName);
    if (!src || (jlong)sz > cap) return JNI_FALSE;

    std::memcpy(dst, src, sz);
    return JNI_TRUE;
}

static jlong n_getTensorSizeBytes(JNIEnv* env, jclass /*cls*/, jstring jname) {
    if (!g_ws) return 0;
    const char* cname = env->GetStringUTFChars(jname, nullptr);
    size_t sz = g_ws->sizeOf(cname ? cname : "");
    if (cname) env->ReleaseStringUTFChars(jname, cname);
    return static_cast<jlong>(sz);
}



static jstring QueryRuntimes(JNIEnv* env, jobject /*thiz*/, jstring native_dir_path) {
    const char *cstr = env->GetStringUTFChars(native_dir_path, nullptr);
    std::string nativeLibPath(cstr);
    env->ReleaseStringUTFChars(native_dir_path, cstr);

    std::string out;
    if (!SetAdspLibraryPath(nativeLibPath)) {
        out = "Failed to set ADSP Library Path";
        return env->NewStringUTF(out.c_str());
    }

    out  = "Querying Runtimes : \n\n";
    out += (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP, zdl::DlSystem::RuntimeCheckOption_t::UNSIGNEDPD_CHECK) ?
            "UnsignedPD DSP runtime : Present\n" : "UnsignedPD DSP runtime : Absent\n");
    out += (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP) ?
            "DSP runtime : Present\n" : "DSP runtime : Absent\n");
    out += (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU) ?
            "GPU runtime : Present\n" : "GPU runtime : Absent\n");
    out += (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::CPU) ?
            "CPU runtime : Present\n" : "CPU runtime : Absent\n");
    return env->NewStringUTF(out.c_str());
}

static jstring InitSNPE(JNIEnv* env, jobject /*thiz*/, jobject asset_manager, jstring jassetName, jchar runtime) {
    const char* asset_cstr = env->GetStringUTFChars(jassetName, nullptr);
    std::string assetName(asset_cstr ? asset_cstr : "");
    env->ReleaseStringUTFChars(jassetName, asset_cstr);

    if (assetName.empty()) {
        std::string r = "No asset name provided";
        return env->NewStringUTF(r.c_str());
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    if (!mgr) {
        std::string r = "AAssetManager_fromJava failed";
        return env->NewStringUTF(r.c_str());
    }

    AAsset* asset = AAssetManager_open(mgr, assetName.c_str(), AASSET_MODE_UNKNOWN);
    if (!asset) {
        std::string r = "Failed to open assets/" + assetName;
        LOGE("%s", r.c_str());
        return env->NewStringUTF(r.c_str());
    }

    const off_t size = AAsset_getLength(asset);
    std::vector<uint8_t> buf(size);
    AAsset_read(asset, buf.data(), size);
    AAsset_close(asset);

    std::string res = "Building DLC: " + assetName + "\n";
    res += build_network_BB(buf.data(), buf.size(), (char)runtime);
    return env->NewStringUTF(res.c_str());
}

static jboolean InferSNPE(JNIEnv* env, jobject /*thiz*/, jobject latentDirectBuffer, jobject outDirectBuffer) {
    if (!latentDirectBuffer || !outDirectBuffer) return JNI_FALSE;

    auto* latentPtr = (float*) env->GetDirectBufferAddress(latentDirectBuffer);
    jlong latentBytes = env->GetDirectBufferCapacity(latentDirectBuffer);

    auto* outPtr = (float*) env->GetDirectBufferAddress(outDirectBuffer);
    jlong outBytes = env->GetDirectBufferCapacity(outDirectBuffer);

    bool ok = executeDLC(latentPtr, (size_t)latentBytes, outPtr, (size_t)outBytes);
    return ok ? JNI_TRUE : JNI_FALSE;
}

static jstring ActiveRuntime(JNIEnv* env, jobject) {
    const std::string& s = getActiveRuntimeName();
    return env->NewStringUTF(s.c_str());
}




// Manual registration
static const JNINativeMethod kMethods[] = {
        {"queryRuntimes", "(Ljava/lang/String;)Ljava/lang/String;", (void*)QueryRuntimes},
//        {"initSNPE", "(Landroid/content/res/AssetManager;Ljava/lang/String;C)Ljava/lang/String;", (void*)InitSNPE},
//        {"inferSNPE", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)Z", (void*)InferSNPE},
////        {"activeRuntime","()Ljava/lang/String;", (void*)Java_com_example_snpedemo_SNPEHelper_activeRuntime}
//        {"activeRuntime", "()Ljava/lang/String;", (void*)ActiveRuntime}
//        {"buildTwoModelGraph", "(Landroid/content/res/AssetManager;C)Ljava/lang/String;",(void *) n_buildTwoModelGraph},
        {"getFinalTensor", "(Ljava/nio/ByteBuffer;)Z", (void*) n_getFinalTensor},
        {"getTensor", "(Ljava/nio/ByteBuffer;Ljava/lang/String;)Z",(void*) n_getTensor},
        {"getTensorSizeBytes", "(Ljava/lang/String;)J",(void*)n_getTensorSizeBytes},
//        {"buildGraph", "(Landroid/content/res/AssetManager;C)Ljava/lang/String;", (void*)n_buildGraph},
        {"runGraph", "()Ljava/lang/String;", (void*)n_runGraphOld},
//        {"executeInference", "(Landroid/content/res/AssetManager;C)Ljava/lang/String;", (void*)n_executeInference},
        {"buildArbitrary", "(Landroid/content/res/AssetManager;C)Ljava/lang/String;", (void*) n_buildArbitrary},
        {"rebuildArbitrary", "()Ljava/lang/String;", (void*) n_rebuildArbitrary},
        {"setModelDirectory", "(Ljava/lang/String;)V", (void*)n_setModelDirectory},
};


//static jstring NativeActiveRuntime(JNIEnv* env, jobject /*thiz*/) {
//    return env->NewStringUTF(g_activeRuntimeName.c_str());
//}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;

    jclass cls = env->FindClass("com/example/snpechainingdemo/SNPEHelper");
    if (!cls) return JNI_ERR;
    if (env->RegisterNatives(cls, kMethods, sizeof(kMethods)/sizeof(kMethods[0])) < 0) return JNI_ERR;

    return JNI_VERSION_1_6;
}
