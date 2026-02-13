//
// Created by Chiheb Boussema on 8/9/25.
//
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstring>

#include "inference.h"
#include "CheckRuntime.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadContainer.hpp"
#include "CreateUserBuffer.hpp"

//#include "DlSystem/RuntimeList.hpp"

static std::mutex g_mtx;
static zdl::DlSystem::RuntimeList g_runtimeList;
static std::unique_ptr<zdl::SNPE::SNPE> g_snpe;

static zdl::DlSystem::UserBufferMap g_inputMap, g_outputMap;
static std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> g_userInBufs, g_userOutBufs;
static std::unordered_map<std::string, std::vector<float>> g_appIn, g_appOut;

static std::string g_inputName, g_outputName;
static const bool g_useUB = true;
static const bool g_isTfN = false; // float32 input; model handles internal quant
static const int  g_bitWidth = 32;

// Keep a global chosen-runtime name
static std::string g_activeRuntimeName = "unknown";

static const char* rtToStr(zdl::DlSystem::Runtime_t r) {
    switch (r) {
        case zdl::DlSystem::Runtime_t::CPU: return "CPU";
        case zdl::DlSystem::Runtime_t::GPU: return "GPU";
//        case zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID: return "GPU_HYBRID";
        case zdl::DlSystem::Runtime_t::DSP: return "DSP/HTP";
        default: return "UNSET";
    }
}
// Accessor defined in header
const std::string& getActiveRuntimeName() {
    return g_activeRuntimeName;
}

std::string build_network_BB(const uint8_t * dlc_buffer, size_t dlc_size, char runtime_arg)
{
    auto container = loadContainerFromBuffer(dlc_buffer, dlc_size);
    if (!container) return "Failed opening DLC container";

    g_runtimeList.clear();
    zdl::DlSystem::Runtime_t want = zdl::DlSystem::Runtime_t::CPU;
    if (runtime_arg == 'D') want = zdl::DlSystem::Runtime_t::DSP;
    else if (runtime_arg == 'G') want = zdl::DlSystem::Runtime_t::GPU;
    else if (runtime_arg == 'C') want = zdl::DlSystem::Runtime_t::CPU;

    // Check availability and fall back if needed
    auto chosen = checkRuntime(want);
    LOGI("SNPE: requested runtime=%s, after availability check=%s", rtToStr(want), rtToStr(chosen));
    g_activeRuntimeName = rtToStr(chosen);

    if (!g_runtimeList.add(checkRuntime(want))) return "Cannot set runtime";

    std::lock_guard<std::mutex> lk(g_mtx);
    g_snpe = setBuilderOptions(container, want, g_runtimeList, g_useUB, /*useCaching*/false);
    if (!g_snpe) return "SNPE build failed";

    createInputBufferMap(g_inputMap, g_appIn, g_userInBufs, g_snpe, g_isTfN, g_bitWidth);
    createOutputBufferMap(g_outputMap, g_appOut, g_userOutBufs, g_snpe, g_isTfN, g_bitWidth);

    if (auto ins  = g_snpe->getInputTensorNames())   g_inputName  = (*ins).at(0);
    if (auto outs = g_snpe->getOutputTensorNames())  g_outputName = (*outs).at(0);

    LOGI("SNPE ready. Input: %s  Output: %s", g_inputName.c_str(), g_outputName.c_str());
    return "Model Network Prepare success !!!";
}

bool executeDLC(const float* latent, size_t latentBytes,
                float* outImage, size_t outBytes)
{
    std::lock_guard<std::mutex> lk(g_mtx);
    if (!g_snpe) { LOGE("SNPE not initialized"); return false; }

    auto inIt  = g_appIn.find(g_inputName);
    auto outIt = g_appOut.find(g_outputName);
    if (inIt == g_appIn.end() || outIt == g_appOut.end()) {
        LOGE("Input/Output buffers not found"); return false;
    }

    auto& inVec  = inIt->second;
    auto& outVec = outIt->second;

    size_t inBytesNeeded  = inVec.size()  * sizeof(float);
    size_t outBytesNeeded = outVec.size() * sizeof(float);

    if (latentBytes != inBytesNeeded) {
        LOGE("Latent size mismatch: got %zu, expected %zu", latentBytes, inBytesNeeded);
        return false;
    }
    if (outBytes != outBytesNeeded) {
        LOGE("Output size mismatch: got %zu, expected %zu", outBytes, outBytesNeeded);
        return false;
    }

    std::memcpy(inVec.data(), latent, inBytesNeeded);

    bool ok = g_snpe->execute(g_inputMap, g_outputMap);
    if (!ok) { LOGE("SNPE execute failed"); return false; }
    else {LOGI("SNPE execute succeeded!");}

    std::memcpy(outImage, outVec.data(), outBytesNeeded);
    return true;
}
