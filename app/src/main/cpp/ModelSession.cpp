//
// Created by Chiheb Boussema on 16/9/25.
//
#include "hpp/ModelSession.hpp"
#include "hpp/TensorTypes.hpp"
#include "hpp/CheckRuntime.hpp"

#include "zdl/SNPE/SNPEFactory.hpp"
#include "zdl/SNPE/SNPEBuilder.hpp"
#include "zdl/DlContainer/IDlContainer.hpp"
#include "zdl/DlSystem/PlatformConfig.hpp"
#include "zdl/DlSystem/IUserBufferFactory.hpp"

#include <android/log.h>
#include <sys/time.h>
#include <cstring>
#include <cassert>

#define  LOG_TAG_MS  "SNPE_MS"
#define  LOGI_MS(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG_MS,__VA_ARGS__)
#define  LOGE_MS(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG_MS,__VA_ARGS__)

static const char* rtToStr(zdl::DlSystem::Runtime_t r) {
    using zdl::DlSystem::Runtime_t;
    switch (r) {
        case Runtime_t::CPU: return "CPU";
        case Runtime_t::GPU: return "GPU";
        case Runtime_t::DSP: return "DSP";
        case Runtime_t::AIP_FIXED_TF: return "AIP_FIXED_TF";
        default: return "UNSET";
    }
}

void ModelSession::reCreate(std::string* buildLog= nullptr) {
    using clock = std::chrono::steady_clock;

    LOGI_MS("REBUILDING SESSION");
    zdl::DlSystem::RuntimeList order = opt_.runtimeOrder;
    if (order.empty()) {
        LOGI_MS("Order is empty!");
//        order.add(
//                checkRuntime(zdl::DlSystem::Runtime_t::DSP)
//                )
        if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP, zdl::DlSystem::RuntimeCheckOption_t::UNSIGNEDPD_CHECK) || zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP)) {
            order.add(zdl::DlSystem::Runtime_t::DSP);
        } else {
            order.add(zdl::DlSystem::Runtime_t::CPU);
        }
    }
    // Platform options (HTP PD / adaptive, etc.)
    zdl::DlSystem::PlatformConfig platformConfig;
    platformConfig.setPlatformOptions("useAdaptivePD:ON");

    LOGI_MS("REbuilding SNPE");
    // Build SNPE
    auto t_builder0 = clock::now();
    zdl::SNPE::SNPEBuilder builder(this->container_.get());
    LOGI_MS("Got container");
    auto newSnpe = builder
            .setOutputLayers({})
            .setPerformanceProfile(opt_.perf)
            .setExecutionPriorityHint(zdl::DlSystem::ExecutionPriorityHint_t::HIGH)
            .setRuntimeProcessorOrder(order)
            .setUseUserSuppliedBuffers(opt_.useUserSuppliedBuffers)
            .setPlatformConfig(platformConfig)
            .setInitCacheMode(opt_.initCache)
//            .setCPUFallbackMode(true)
            .setUnconsumedTensorsAsOutputs(true)
            .build();
    auto t_builder1 = clock::now();
    LOGI_MS("SNPE re-builder time: %lld", std::chrono::duration_cast<std::chrono::milliseconds>(t_builder1 - t_builder0).count());

    if (!newSnpe) {
        if (buildLog) *buildLog += "SNPE re-build failed\n";
        LOGE_MS("SNPE re-build failed");
        return;
    }

    // Swap in new graph (old one is freed)
    snpe_.swap(newSnpe);
}

std::unique_ptr<ModelSession> ModelSession::Create(const uint8_t* dlc, size_t bytes,
                     std::shared_ptr<void> dlcOwner,
                     const Options& opt, std::string* buildLog) {
    std::unique_ptr<ModelSession> self(new ModelSession());
    using clock = std::chrono::steady_clock;

    self->dlcBacking_ = std::shared_ptr<const uint8_t>(
        dlc,                   // raw pointer
        [](const uint8_t*){}   // no-op deleter
    );
    self->dlcOwner_ = std::move(dlcOwner);

    // Container
    auto t1 = clock::now();
    auto container = zdl::DlContainer::IDlContainer::open(dlc, bytes);
    auto t2 = clock::now();
    LOGI_MS("DLContainer open time: %lld", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    if (!container) {
        if (buildLog) *buildLog += "Failed to open DLC container\n";
        LOGE_MS("DLC open failed");
        return nullptr;
    }
    self->container_ = std::move(container);
    self->opt_ = opt;

    zdl::DlSystem::RuntimeList order = opt.runtimeOrder;
    if (order.empty()) {
//        order.add(
//                checkRuntime(zdl::DlSystem::Runtime_t::DSP)
//                )
        if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP, zdl::DlSystem::RuntimeCheckOption_t::UNSIGNEDPD_CHECK) || zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP)) {
            order.add(zdl::DlSystem::Runtime_t::DSP);
        } else {
            order.add(zdl::DlSystem::Runtime_t::CPU);
        }
    }// order.add(chosen);

    // Choose runtime actually available (respect given order)
//    zdl::DlSystem::Runtime_t chosen = pickFirstAvailable(opt.runtimeOrder);
    zdl::DlSystem::Runtime_t chosen = checkRuntime(order[0]);
    self->runtimeName_ = rtToStr(chosen);
    LOGI_MS("Selected runtime=%s", self->runtimeName_.c_str());

    // Platform options (HTP PD / adaptive, etc.)
    zdl::DlSystem::PlatformConfig platformConfig;
    platformConfig.setPlatformOptions("useAdaptivePD:ON");

    // (Optional) log what’s requested
    if (buildLog) {
        auto names = order.getRuntimeListNames();
        std::string s = "Runtime order: ";
        for (const char* n : names) { s += n; s += " "; }
        s += "\n";
        *buildLog += s;
    }

    // Build SNPE
    auto t_builder0 = clock::now();
//    zdl::SNPE::SNPEBuilder builder(container.get());
    zdl::SNPE::SNPEBuilder builder(self->container_.get());
    self->snpe_ = builder
            .setOutputLayers({})
            .setPerformanceProfile(opt.perf)
            .setExecutionPriorityHint(zdl::DlSystem::ExecutionPriorityHint_t::HIGH)
            .setRuntimeProcessorOrder(order)
            .setUseUserSuppliedBuffers(opt.useUserSuppliedBuffers)
            .setPlatformConfig(platformConfig)
            .setInitCacheMode(opt.initCache)
//            .setCPUFallbackMode(true)
            .setUnconsumedTensorsAsOutputs(true)
            .build();
    auto t_builder1 = clock::now();
    LOGI_MS("SNPE builder time: %lld", std::chrono::duration_cast<std::chrono::milliseconds>(t_builder1 - t_builder0).count());

    if (!self->snpe_) {
        if (buildLog) *buildLog += "SNPE build failed\n";
        LOGE_MS("SNPE build failed");
        return nullptr;
    }

//    self->builder_ = std::move(builder);
//    self->opt_ = opt;

    // IO metadata
    self->captureIO_();
    if (buildLog) {
        *buildLog += "SNPE build success. Inputs:";
        for (auto& t : self->inputs_) *buildLog += " " + t.name;
        *buildLog += "  Outputs:";
        for (auto& t : self->outputs_) *buildLog += " " + t.name;
        *buildLog += "\n";
    }
    return self;
}

void ModelSession::captureIO_() {
    // Inputs
    auto inNamesOpt = snpe_->getInputTensorNames();
    if (inNamesOpt) {
        const auto& names = *inNamesOpt;
        for (const char* n : names) {
            auto attr = snpe_->getInputOutputBufferAttributes(n);
            if (!attr) continue;
            const auto& shape = (*attr)->getDims();
            TensorInfo t;
            t.name = n;
            t.elementBytes = 4; // float32 for strict boundary
//            t.dims.assign(shape.getDimensions(), shape.getDimensions() + shape.rank());
            t.dims.clear();
            for (size_t i = 0; i < shape.rank(); ++i) t.dims.push_back(shape[i]);
            inputs_.push_back(std::move(t));
        }
    }
    // Outputs
    auto outNamesOpt = snpe_->getOutputTensorNames();
    if (outNamesOpt) {
        const auto& names = *outNamesOpt;
        for (const char* n : names) {
            auto attr = snpe_->getInputOutputBufferAttributes(n);
            if (!attr) continue;
            const auto& shape = (*attr)->getDims();
            TensorInfo t;
            t.name = n;
            t.elementBytes = 4; // float32
//            t.dims.assign(shape.getDimensions(), shape.getDimensions() + shape.rank());
            t.dims.clear();
            for (size_t i = 0; i < shape.rank(); ++i) t.dims.push_back(shape[i]);
            outputs_.push_back(std::move(t));
        }
    }
    // NOTE: If your DLC exposes TfN on IO, you could probe encoding here
    // and log loudly. For strict float32, we keep elementBytes=4 and
    // fail at execute-time if enc != float.
}

void ModelSession::reset() {
    LOGI_MS("[Model Session] Inside reset().");
    snpe_.reset();
//    inputs_.clear();
//    outputs_.clear();
//    runtimeName_.clear();
    if (!snpe_) LOGI_MS("[Model Session] RESET SNPE EMPTY");
}

bool ModelSession::execute(const std::unordered_map<std::string, const void*>& inputPtrs,
                           const std::unordered_map<std::string, void*>& outputPtrs,
                           int64_t* elapsedMs) const {
    using namespace zdl::DlSystem;

    UserBufferMap inMap, outMap;
    std::vector<std::unique_ptr<IUserBuffer>> ubKeepAlive;
    std::vector<std::unique_ptr<UserBufferEncoding>> encKeepAlive;

    auto addOne = [&](const TensorInfo& t, const void* ptr, bool isInput) -> bool {
        if (!ptr) {
            LOGE_MS("Null pointer for %s '%s'", isInput ? "input" : "output", t.name.c_str());
            return false;
        }
        // encoding: float32 strict
//        std::unique_ptr<UserBufferEncoding> enc(new UserBufferEncodingFloat());
        encKeepAlive.emplace_back(new UserBufferEncodingFloat());
        auto* enc = encKeepAlive.back().get();

        auto strides = computePackedStridesBytes(t.dims, t.elementBytes);
        size_t bytes = t.bytes();

        auto& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
        auto ub = ubFactory.createUserBuffer(
                const_cast<void*>(ptr),
                bytes, strides,
                enc
                );
        if (!ub) {
            LOGE_MS("Failed to create UserBuffer for %s", t.name.c_str());
            return false;
        }
        if (isInput) inMap.add(t.name.c_str(), ub.get());
        else         outMap.add(t.name.c_str(), ub.get());

        ubKeepAlive.push_back(std::move(ub));
        return true;
    };

    // Bind inputs
    for (auto& t : inputs_) {
        auto it = inputPtrs.find(t.name);
        if (it == inputPtrs.end()) { LOGE_MS("Missing input: %s", t.name.c_str()); return false; }
        if (!addOne(t, it->second, true)) return false;
    }
    // Bind outputs
    for (auto& t : outputs_) {
        auto it = outputPtrs.find(t.name);
        if (it == outputPtrs.end()) { LOGE_MS("Missing output: %s", t.name.c_str()); return false; }
        if (!addOne(t, it->second, false)) return false;
    }

    // run
    timeval t0{}, t1{};
    gettimeofday(&t0, nullptr);
    bool ok = snpe_->execute(inMap, outMap);
    gettimeofday(&t1, nullptr);

    if (!ok) {
        LOGE_MS("SNPE execute failed");
        return false;
    }
    if (elapsedMs) {
        *elapsedMs = (t1.tv_sec - t0.tv_sec)*1000LL + (t1.tv_usec - t0.tv_usec)/1000LL;
    }
    return true;
}
