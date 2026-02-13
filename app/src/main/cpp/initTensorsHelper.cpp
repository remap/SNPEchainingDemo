//
// Created by Chiheb Boussema on 24/9/25.
//

#include "initTensorsHelper.h"

#define LOG_TAG "INIT_TENSOR_HELPER"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

static std::unordered_set<std::string>
collectProducedNames(const PipelineCfg& cfg) {
    std::unordered_set<std::string> produced;
    for (auto& m : cfg.models) {
        for (auto& kv : m.outputs) produced.insert(kv.second); // workspace names
    }
    return produced;
}

static std::unordered_set<std::string>
collectConsumedNames(const PipelineCfg& cfg) {
    std::unordered_set<std::string> consumed;
    for (auto& m : cfg.models) {
        for (auto& kv : m.inputs) consumed.insert(kv.second); // workspace names
    }
    return consumed;
}

static std::vector<std::string>
computeGraphRoots(const PipelineCfg& cfg) {
    auto produced = collectProducedNames(cfg);
    auto consumed = collectConsumedNames(cfg);

    std::vector<std::string> roots;
    roots.reserve(consumed.size());
    for (auto& wname : consumed) {
        if (!produced.count(wname)) roots.push_back(wname);
    }
    return roots;
}

// Fill with single value
static void fillConst(void* p, size_t bytes, float value) {
    size_t n = bytes / sizeof(float);
    float* f = static_cast<float*>(p);
    for (size_t i = 0; i < n; ++i) f[i] = value;
}

// Gaussian random
void fillRandom(void* p, size_t bytes, float mean, float stddev, uint32_t seed) {
    size_t n = bytes / sizeof(float);
    float* f = static_cast<float*>(p);
    if (seed == 0) {
        seed = static_cast<uint32_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) f[i] = dist(rng);
}

// File (absolute path) -> buffer; expects raw float32 count == bytes/4
static bool readFileToBuffer(const std::string& path, void* dst, size_t bytes) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.read(reinterpret_cast<char*>(dst), bytes);
    return static_cast<size_t>(ifs.gcount()) == bytes;
}

// Asset -> buffer; expects raw float32
static bool readAssetToBuffer(AAssetManager* mgr, const char* asset, void* dst, size_t bytes) {
    if (!mgr) return false;
    AAsset* a = AAssetManager_open(mgr, asset, AASSET_MODE_UNKNOWN);
    if (!a) return false;
    const off_t len = AAsset_getLength(a);
    if (static_cast<size_t>(len) != bytes) {
        AAsset_close(a);
        return false;
    }
    int rd = AAsset_read(a, dst, bytes);
    AAsset_close(a);
    return rd == static_cast<int>(bytes);
}

// Seed one tensor according to spec (or default-zero if spec == nullptr)
static bool seedOneTensor(TensorWorkspace& ws,
                          const std::string& wsName,
                          const InitSpec* spec,
                          AAssetManager* mgr,
                          std::string* emsg) {
    LOGW("SEEDING TENSOR %s", wsName.c_str());
    void* ptr = ws.data(wsName);
//    size_t bytes = ws.sizeOf(wsName);
    size_t bytes = ws.tinfoOf(wsName)->bytes();
    if (!ptr || bytes == 0) {
        if (emsg) *emsg = "Workspace tensor '" + wsName + "' missing or size=0";
        return false;
    }

    if (!spec || spec->kind == InitKind::ZERO || spec->kind == InitKind::UNKNOWN) {
        std::memset(ptr, 0, bytes);
        return true;
    }

    switch (spec->kind) {
        case InitKind::CONST_VALUE:
            fillConst(ptr, bytes, spec->value);
            return true;
        case InitKind::RANDOM:
            fillRandom(ptr, bytes, spec->mean, spec->std, spec->seed);
            return true;
        case InitKind::FILE_PATH:
            if (!readFileToBuffer(spec->path, ptr, bytes)) {
                if (emsg) *emsg = "Failed reading file '" + spec->path + "' for '" + wsName + "'";
                return false;
            }
            return true;
        case InitKind::ASSET_PATH:
            if (!readAssetToBuffer(mgr, spec->path.c_str(), ptr, bytes)) {
                if (emsg) *emsg = "Failed reading asset '" + spec->path + "' for '" + wsName + "'";
                return false;
            }
            return true;
        default:
            if (emsg) *emsg = "Unsupported init kind for '" + wsName + "'";
            return false;
    }
}

bool seedRequiredInputs(const PipelineCfg& cfg,
                               TensorWorkspace& ws,
                               AAssetManager* mgr,
                               std::string* emsg) {
    auto roots = computeGraphRoots(cfg);
    for (auto& wsName : roots) {
        const InitSpec* spec = nullptr;
        auto it = cfg.init.find(wsName);
        if (it != cfg.init.end()) spec = &it->second;

        if (!seedOneTensor(ws, wsName, spec, mgr, emsg)) {
            LOGE("Seeding failed for '%s'%s",
                 wsName.c_str(),
                 (emsg && !emsg->empty()) ? (": " + *emsg).c_str() : "");
            return false;
        }
        LOGI("Seeded root tensor '%s'%s",
             wsName.c_str(),
             spec ? "" : " (default zero)");
    }
    return true;
}

bool seedAllTensors(const PipelineCfg& cfg,
                               TensorWorkspace& ws,
                               AAssetManager* mgr,
                               std::string* emsg) {


    auto produced = collectProducedNames(cfg);
    auto consumed = collectConsumedNames(cfg);

    std::vector<std::string> roots;
    roots.reserve(consumed.size()+produced.size());
    for (auto& wname : consumed) {
        roots.push_back(wname);
    }
    for (auto& wname : produced) {
        if (!consumed.count(wname)) roots.push_back(wname);
    }

    for (auto& wsName : roots) {
        const InitSpec* spec = nullptr;
        auto it = cfg.init.find(wsName);
        if (it != cfg.init.end()) spec = &it->second;

        if (!seedOneTensor(ws, wsName, spec, mgr, emsg)) {
            LOGE("Seeding failed for '%s'%s",
                 wsName.c_str(),
                 (emsg && !emsg->empty()) ? (": " + *emsg).c_str() : "");
            return false;
        }
        LOGI("Seeded root tensor '%s'%s",
             wsName.c_str(),
             spec ? "" : " (default zero)");
    }
    return true;
}
