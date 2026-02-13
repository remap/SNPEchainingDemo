//
// Created by Chiheb Boussema.
//

#include <jni.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "inference.h"
#include "SNPE/SNPEFactory.hpp"
#include "ModelSession.hpp"
#include "GraphRunner.hpp"
#include "TensorWorkspace.hpp"
#include "MMapAsset.hpp"
#include "ParseConfig.hpp"
#include "MMapFile.h"
#include "initTensorsHelper.h"

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "NEW_INFERENCE_HELPER"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)


// Build a RuntimeList from a single preference char (D/G/C).
static zdl::DlSystem::RuntimeList makeRuntimeOrder(char pref) {
    using zdl::DlSystem::Runtime_t;
    zdl::DlSystem::RuntimeList lst;
    if      (pref == 'D') lst.add(Runtime_t::DSP);
    else if (pref == 'G') lst.add(Runtime_t::GPU);
    else                  lst.add(Runtime_t::CPU);
    return lst;
}

// Look up a tensor by name in metadata vector.
static const TensorInfo* findTensor(const std::vector<TensorInfo>& v, const std::string& name) {
    for (const auto& t : v) if (t.name == name) return &t;
    return nullptr;
}

//std::unordered_set<std::string> produced;
//for (auto& m: cfg.models) for (auto& kv: m.outputs) produced.insert(kv.second);
//
//for (auto& m: cfg.models) for (auto& kv: m.inputs) {
//    const std::string& wsName = kv.second;
//    if (!produced.count(wsName) && outWs->has(wsName)) {
//        std::memset(outWs->data(wsName), 0, outWs->sizeOf(wsName));
//    }
//}

// Convenience: allocate a workspace buffer if not allocated yet.
bool ensureWorkspaceBuffer(TensorWorkspace& ws,
                                  const std::string& wsName,
                                  size_t bytes,
                                  std::string* emsg) {
    // If your TensorWorkspace doesn’t expose a 'has()' method,
    // add one; if it does, use it here.
    if (!ws.has(wsName)) {
        void* p = ws.allocate(wsName, bytes);
        if (!p) {
            if (emsg) *emsg = "allocate('" + wsName + "') failed";
            return false;
        }
        std::memset(p, 0, bytes); // zero on first alloc
    } else {
        // Validate size matches on reuse
        const size_t existing = ws.sizeOf(wsName);
        if (existing != bytes) {
            if (emsg) *emsg = "Workspace tensor size mismatch for '" + wsName +
                              "': have " + std::to_string(existing) +
                              ", need " + std::to_string(bytes);
            return false;
        }
    }
    return true;
}

bool ensureWorkspaceBuffer(TensorWorkspace& ws,
                                  const std::string& wsName,
                                  const TensorInfo& tinfo,
                                  std::string* emsg) {
    // If your TensorWorkspace doesn’t expose a 'has()' method,
    // add one; if it does, use it here.
    if (!ws.has(wsName)) {
        void* p = ws.allocate(wsName, tinfo);
        if (!p) {
            if (emsg) *emsg = "allocate('" + wsName + "') failed";
            return false;
        }
        std::memset(p, 0, tinfo.bytes()); // zero on first alloc
    } else {
        // Validate size matches on reuse
        const size_t existing = ws.sizeOf(wsName);
        if (existing != tinfo.bytes()) {
            if (emsg) *emsg = "Workspace tensor size mismatch for '" + wsName +
                              "': have " + std::to_string(existing) +
                              ", need " + std::to_string(tinfo.bytes());
            return false;
        }
    }
    return true;
}

// Times:
static inline int64_t msSince(std::chrono::steady_clock::time_point t0) {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
}

std::string buildModelAndGraph(AAssetManager* mgr,
                                std::string& g_modelDir,
//                                const std::string& configJson,
                                const PipelineCfg& cfg,
                                const ModelCfg& mc,
                                const char defaultRuntimePref,
                                TensorWorkspace& outWs,
                                GraphRunner& outGraph,
                                std::string& log,
                                bool reset_session=false) {

    using clock = std::chrono::steady_clock;

    int64_t totalAssetMs = 0, totalBuildMs = 0, totalAllocMs = 0, totalGraphMs = 0;

    LOGI("Starting build of Model %s", mc.asset.c_str());
    const auto tAsset0 = clock::now();

    // mmap DLC
    std::string emsg;
    MMapAsset mappedAsset;
    MMapFile mappedFile;
    bool mappedOk = false;

    if (g_modelDir.empty() and !cfg.baseDir.empty()) {
        g_modelDir = cfg.baseDir + "/";
    }
    std::string full = g_modelDir + mc.asset;
    if (!g_modelDir.empty()) {
//        std::string full = g_modelDir + "/" + mc.asset;
        LOGI("Trying DLC from file: %s", full.c_str());
        if (mappedFile.openPath(full.c_str(), &emsg)) {
            mappedOk = true;
        } else {
            LOGW("File open failed: %s", emsg.c_str());
            emsg.clear();
        }
    }

    if (!mappedOk) {
        LOGI("Falling back to APK asset: %s", mc.asset.c_str());
        if (!mappedAsset.openUncompressed(mgr, mc.asset.c_str(), &emsg)) {
            return "Failed to mmap asset '" + mc.asset + "': " + emsg;
        }
    }

    totalAssetMs += msSince(tAsset0);
    LOGI("Model %s opened", mc.asset.c_str());

    const void* dlcPtr  = mappedOk ? mappedFile.ptr  : mappedAsset.ptr;
    size_t      dlcSize = mappedOk ? mappedFile.size : mappedAsset.size;
    auto owner = std::make_shared<MMapFile>(std::move(mappedFile));

    // 3) Build ModelSession
    const auto tBuild0 = clock::now();
    ModelSession::Options opt;
    // Runtime order: use per-model pref if present else default
    const char pref = (mc.runtime == 0 ? defaultRuntimePref : mc.runtime);
    opt.runtimeOrder = makeRuntimeOrder(pref);
    opt.perf = zdl::DlSystem::PerformanceProfile_t::BURST;
    opt.useUserSuppliedBuffers = true;
    opt.initCache = true; //false;

    std::string buildLog;
    auto session = ModelSession::Create(static_cast<const uint8_t *>(dlcPtr),
                                        dlcSize, owner, opt, &buildLog,
                                        mc.inputEncodings,
                                        full); //g_modelDir);
    totalBuildMs += msSince(tBuild0);
    log += "[Build " + mc.name + "] " + buildLog;
    LOGI("Session for model %s created", mc.asset.c_str());

    if (!session) {
        return "SNPE build failed for '" + mc.name + "'";
    }

    // 4) Validate inputs/outputs exist & allocate workspace for any new names
    const auto tAlloc0 = clock::now();

    // Inputs
    for (const auto &kv: mc.inputs)
    {
        const std::string &modelTensor = kv.first;
        const std::string &wsTensor = kv.second;

        const TensorInfo *ti = findTensor(session->inputs(), modelTensor);
        if (!ti) {
            return "Model '" + mc.name + "': input tensor not found: " + modelTensor;
        }
        // Inputs are bound from workspace too (so they can be fed by earlier models or app)
//            if (!ensureWorkspaceBuffer(*ws, wsTensor, ti->bytes(), &emsg)) {
//        if (!ensureWorkspaceBuffer(*outWs, wsTensor, ti->bytes(), &emsg)) {
        if (!ensureWorkspaceBuffer(outWs, wsTensor, *ti, &emsg)) {
//        if (!ensureWorkspaceBuffer(outWs, wsTensor, ti->bytes(), &emsg)) {
            return "Workspace alloc (input) failed for '" + mc.name + "': " + emsg;
        }
    }

    // Outputs
    for (const auto& kv : mc.outputs) {
        const std::string& modelTensor = kv.first;
        const std::string& wsTensor    = kv.second;

        const TensorInfo* ti = findTensor(session->outputs(), modelTensor);
        if (!ti) {
            return "Model '" + mc.name + "': output tensor not found: " + modelTensor;
        }
//        if (!ensureWorkspaceBuffer(*outWs, wsTensor, ti->bytes(), &emsg)) {
        if (!ensureWorkspaceBuffer(outWs, wsTensor, *ti, &emsg)) {
//        if (!ensureWorkspaceBuffer(outWs, wsTensor, ti->bytes(), &emsg)) {
            return "Workspace alloc (output) failed for '" + mc.name + "': " + emsg;
        }
    }

    totalAllocMs += msSince(tAlloc0);

    // 5) Add node to graph (strict zero-copy)
    const auto tGraph0 = clock::now();
    GraphRunner::Node node;
    node.name     = mc.name;
    node.session  = std::move(session);
    node.inputBinding  = mc.inputs;   // modelTensor -> workspaceTensor
    node.outputBinding = mc.outputs;  // modelTensor -> workspaceTensor

    if (!outGraph.addNode(std::move(node), /*strictZeroCopy=*/true)) {
        return "addNode failed for '" + mc.name + "'";
    }
    totalGraphMs += msSince(tGraph0);
    LOGI("Graph node for model %s added", mc.asset.c_str());

    if (reset_session) {
//        LOGI("[BUILDING] Resetting graph!");
        LOGI("[BUILDING] Resetting session of last graph node %s!", outGraph.last().name.c_str());
//        outGraph->clear();
        outGraph.last().session.get()->reset();
    }
//    }

//    outWs   = std::move(ws);
//    outGraph= std::move(gr);

    char buf[256];
    snprintf(buf, sizeof(buf),
             "Build OK. assets=%lld ms, build=%lld ms, alloc=%lld ms, graph=%lld ms",
             (long long)totalAssetMs, (long long)totalBuildMs,
             (long long)totalAllocMs, (long long)totalGraphMs);
    log = std::string(buf) + "\n" + log;
    return log;
}

bool readAssetToString(AAssetManager* mgr,
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

std::string buildArbitraryChain(AAssetManager* mgr,
                                std::string& g_modelDir,
                                const std::string config_filename,
                                TensorWorkspace& ws,
                                GraphRunner& gr,
                                const char defaultRuntimePref='D',
                                bool reset_sessions=false) {

//    std::unique_ptr<TensorWorkspace> g_ws; // holds workspace tensors
//    std::unique_ptr<GraphRunner> g_gr; // holds graph runner
////    std::string g_modelDir; // holds the model directory path
//
//    g_ws.reset(new TensorWorkspace());
//    g_gr.reset(new GraphRunner(*g_ws));

    // read config file
    std::string cfgText;
    std::string emsg;
    if (!readAssetToString(mgr, config_filename.c_str(), cfgText, &emsg)) {
        return "Config read failed: " + emsg;
    }
    // parse config file
    PipelineCfg cfg;
    {
        std::string emsg;
        if (!ParseConfig(cfgText, cfg, &emsg)) {
            return "Config parse failed: " + emsg;
        }
        if (cfg.models.empty()) {
            return "Config has no models";
        }
    }

    // create models, allocate buffers and build graph
    std::string buildingLog;
    for (const auto &mc: cfg.models)
    {
        buildingLog = buildModelAndGraph(mgr,
                            g_modelDir,
                            cfg,
                            mc,
                            defaultRuntimePref,
                            ws,
                            gr,
                            buildingLog,
                            reset_sessions);
    }

    {
        std::string semsg;
        if (!seedAllTensors(cfg, ws, mgr, &semsg)) {
//        if (!seedRequiredInputs(cfg, ws, mgr, &semsg)) {
            return "Input seeding failed: " + semsg;
        }
    }

    return buildingLog;
}

std::string buildArbitraryChainFromConfig(AAssetManager* mgr,
                                std::string& g_modelDir,
                                const PipelineCfg cfg,
                                TensorWorkspace& ws,
                                GraphRunner& gr,
                                const char defaultRuntimePref='D',
                                bool reset_sessions=false) {

    // create models, allocate buffers and build graph
    std::string buildingLog;
    for (const auto &mc: cfg.models)
    {
        buildingLog = buildModelAndGraph(mgr,
                            g_modelDir,
                            cfg,
                            mc,
                            defaultRuntimePref,
                            ws,
                            gr,
                            buildingLog,
                            reset_sessions);
    }

    {
        std::string semsg;
        if (!seedAllTensors(cfg, ws, mgr, &semsg)) {
//        if (!seedRequiredInputs(cfg, ws, mgr, &semsg)) {
            return "Input seeding failed: " + semsg;
        }
    }

    return buildingLog;
}

std::string rebuildNodeSession(GraphRunner::Node& node) {

    std::string rebuildingLog;
    node.session.get()->reCreate(&rebuildingLog);

    return rebuildingLog;
}

std::string rebuildMultipleNodes(std::vector<GraphRunner::Node>& nodes) {

    std::string rebuildingLog;
    for (auto& n: nodes) {
        rebuildingLog = rebuildNodeSession(n);
    }

    return  rebuildingLog;
}

std::string rebuildAllGraphNodes(GraphRunner& gr) {

    std::string rebuildingLog;
    for (auto& n: gr.getNodes()) {
        rebuildingLog = rebuildNodeSession(n);
    }

    return  rebuildingLog;
}

std::string runGraph(GraphRunner& gr, bool reset_sessions) {
    auto T0 = std::chrono::steady_clock::now();
    auto infos = gr.runAll(reset_sessions);
    auto T1 = std::chrono::steady_clock::now();
    auto execMs = std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count();
    LOGI("Graph Execution time: %lld", execMs);

    // 7) Summarize result
    std::string summary;
    for (auto& e : infos) {
        summary += e.name + " runtime=" + e.runtime + " time=" + std::to_string(e.ms) + "ms "
                   + (e.ok ? "OK\n" : "FAIL\n");
    }

    return summary;
}

//static bool readAssetToString(AAssetManager* mgr,
//                              const char* filename,
//                              std::string& out,
//                              std::string* emsg) {
//    AAsset* a = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
//    if (!a) {
//        if (emsg) *emsg = std::string("Asset open failed: ") + filename;
//        return false;
//    }
//    size_t len = AAsset_getLength(a);
//    out.resize(len);
//    int r = AAsset_read(a, out.data(), len);
//    AAsset_close(a);
//    if (r != (int)len) {
//        if (emsg) *emsg = "Asset read truncated";
//        return false;
//    }
//    return true;
//}
