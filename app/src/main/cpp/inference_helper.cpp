//
// Created by Chiheb Boussema on 8/9/25.
//
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <android/log.h>
#include <cstdlib>

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"
#include "zdl/DlSystem/DlEnums.hpp"
#include "zdl/DlSystem/String.hpp"
#include "zdl/DlSystem/PlatformConfig.hpp"
#include "zdl/DlContainer/IDlContainer.hpp"
#include "zdl/SNPE/SNPEBuilder.hpp"
#include "DlSystem/ITensorFactory.hpp"

#include "hpp/CreateUserBuffer.hpp"
#include "hpp/inference.h"

bool SetAdspLibraryPath(std::string nativeLibPath) {
    nativeLibPath += ";/data/local/tmp/mv_dlc;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
    __android_log_print(ANDROID_LOG_INFO, "SNPE_VAE", "ADSP Lib Path = %s", nativeLibPath.c_str());
    return setenv("ADSP_LIBRARY_PATH", nativeLibPath.c_str(), 1) == 0;
}

std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromBuffer(const uint8_t * buffer, size_t size) {
    return zdl::DlContainer::IDlContainer::open(buffer, size);
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   bool useCaching)
{
    if (runtimeList.empty()) runtimeList.add(runtime);

    std::string opt = "useAdaptivePD:ON";
    zdl::DlSystem::PlatformConfig platformConfig;
    platformConfig.setPlatformOptions(opt);

    // Log what we’re about to request
    zdl::DlSystem::StringList names = runtimeList.getRuntimeListNames();
    for (const char* n : names) {
        LOGI("SNPE: requested runtime in order: %s", n);
    }

    // Decide profile based on first runtime in the list
    zdl::DlSystem::PerformanceProfile_t profile =
            (runtime == zdl::DlSystem::Runtime_t::CPU)
            ? zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE
            : zdl::DlSystem::PerformanceProfile_t::BURST;

    auto snpe = zdl::SNPE::SNPEBuilder(container.get())
            .setOutputLayers({})
            .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::BURST)
            .setExecutionPriorityHint(zdl::DlSystem::ExecutionPriorityHint_t::HIGH)
            .setRuntimeProcessorOrder(runtimeList)
            .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
            .setPlatformConfig(platformConfig)
            .setInitCacheMode(useCaching)
            .setUnconsumedTensorsAsOutputs(true)
            .build();

    if (!snpe) {
        LOGE("SNPE: build() returned null");
        return nullptr;
    }
    LOGI("SNPE: build() success");
    LOGI("SNPE: PerformanceProfile=%s",
         profile==zdl::DlSystem::PerformanceProfile_t::BURST ? "BURST" : "HIGH_PERFORMANCE");

    return snpe;
}

// ---------- UserBuffer helpers ----------
void createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                      std::unordered_map<std::string, std::vector<float>>& applicationBuffers,
                      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                      std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                      const char * name,
                      bool isTfNBuffer,
                      int bitWidth)
{
    auto bufAttrOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufAttrOpt) throw std::runtime_error(std::string("No attributes for tensor ") + name);

    const auto& shape = (*bufAttrOpt)->getDims();

    size_t elemSize = isTfNBuffer ? (bitWidth/8) : sizeof(float);
    int rank = shape.rank();
    std::vector<size_t> strides(rank);
    strides[rank-1] = elemSize;
    size_t stride = elemSize;
    for (int i = rank - 1; i > 0; --i) { stride *= shape[i]; strides[i-1] = stride; }

    size_t bufBytes = elemSize;
    for (int i = 0; i < rank; ++i) bufBytes *= shape[i];

//    std::unique_ptr<zdl::DlSystem::UserBufferEncoding> enc =
//            isTfNBuffer
//            ? std::unique_ptr<zdl::DlSystem::UserBufferEncodingTfN>(new zdl::DlSystem::UserBufferEncodingTfN(0, 1.0, bitWidth))
//            : std::unique_ptr<zdl::DlSystem::UserBufferEncodingFloat>(new zdl::DlSystem::UserBufferEncodingFloat());
    std::unique_ptr<zdl::DlSystem::UserBufferEncoding> enc;
    if (isTfNBuffer) {
        enc = std::make_unique<zdl::DlSystem::UserBufferEncodingTfN>(0, 1.0, bitWidth);
    } else {
        enc = std::make_unique<zdl::DlSystem::UserBufferEncodingFloat>();
    }

    applicationBuffers.emplace(name, std::vector<float>(bufBytes/sizeof(float)));

    zdl::DlSystem::IUserBufferFactory &ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
    snpeUserBackedBuffers.push_back(
            ubFactory.createUserBuffer(applicationBuffers.at(name).data(),
                                       bufBytes,
                                       strides,
                                       enc.get()));
    userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

void createOutputBufferMap(zdl::DlSystem::UserBufferMap& outputMap,
                           std::unordered_map<std::string, std::vector<float>>& applicationBuffers,
                           std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                           std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                           bool isTfNBuffer,
                           int bitWidth)
{
    const auto& namesOpt = snpe->getOutputTensorNames();
    for (const char* name : *namesOpt) {
        createUserBuffer(outputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name, isTfNBuffer, bitWidth);
    }
}

void createInputBufferMap(zdl::DlSystem::UserBufferMap& inputMap,
                          std::unordered_map<std::string, std::vector<float>>& applicationBuffers,
                          std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                          std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                          bool isTfNBuffer,
                          int bitWidth)
{
    const auto& namesOpt = snpe->getInputTensorNames();
    for (const char* name : *namesOpt) {
        createUserBuffer(inputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name, isTfNBuffer, bitWidth);
    }
}
