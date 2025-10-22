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
                     const Options& opt, std::string* buildLog,
                     std::unordered_map<std::string, std::string> inputEncodings) {

//    LOGI_MS("[ModelSession] 1");
    std::unordered_map<std::string, DType> inputEncodings_map;
    if (!inputEncodings.empty()) {
        for (const auto& it : inputEncodings) {
            LOGI_MS("%s : %s", it.first.c_str(), it.second.c_str());
            inputEncodings_map.insert({it.first, StringToDType(it.second)});
        }
    }
//    LOGI_MS("[ModelSession] 2");

//    if (!inputEncodings_map.empty()) {
//        std::unique_ptr<ModelSession> self(new ModelSession(inputEncodings_map));
//    }
    std::unique_ptr<ModelSession> self(new ModelSession(inputEncodings_map));
//    LOGI_MS("[ModelSession] 3");
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
//    LOGI_MS("DLContainer open time: %lld", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
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
    if (self->inputTypeHints_.empty()) {
        for (const auto& it : self->inputs_) {
            self->inputTypeHints_.insert({it.name, DType::F32});
        }
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
//            LOGI_MS("SNPE_MS %zu", (*attr)->getElementSize());
            if (!attr) continue;
            const auto& shape = (*attr)->getDims();
            TensorInfo t;
            t.name = n;
            t.elementBytes = 4; // float32 for strict boundary
//            t.elementBytes = (*attr)->getElementSize(); // stays 4
//            t.dims.assign(shape.getDimensions(), shape.getDimensions() + shape.rank());
            t.dims.clear();
            for (size_t i = 0; i < shape.rank(); ++i) t.dims.push_back(shape[i]);

            // Default to float32, then override from hints
            t.dtype = DType::F32;
            if (!inputTypeHints_.empty()) {
                auto it = inputTypeHints_.find(t.name);
                if (it != inputTypeHints_.end()) {
                    t.dtype = it->second; // e.g. I32 for "input_ids", maybe "timestep" if exported as int
                }
            }

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

//            t.elementBytes = (*attr)->getElementSize(); // often 4
            t.dtype = DType::F32; // typical UNet/VAE outputs

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



// Helper: convert std::vector<size_t> to TensorShape
static inline zdl::DlSystem::TensorShape toShape(const std::vector<size_t>& dims) {
    std::vector<size_t> tmp = dims; // TensorShape ctor takes pointer+rank
    return zdl::DlSystem::TensorShape(tmp.data(), tmp.size());
}

bool ModelSession::execute(const std::unordered_map<std::string, const void*>& inputPtrs,
                           const std::unordered_map<std::string, void*>& outputPtrs,
                           int64_t* elapsedMs) const {
    using namespace zdl::DlSystem;
    using namespace zdl::SNPE;

    // If ANY input is non-float, use ITensor path for everything.
    bool needITensor = false;
    for (const auto &t: inputs_) {
        if (t.dtype != DType::F32) {
//            needITensor = true;
            break;
        }
    }

    if (needITensor) {
        ITensorFactory factory;
        TensorMap inT, outT;
        std::vector<std::unique_ptr<ITensor>> outKeepAlive;

        // Bind inputs (copies user bytes into internal tensor)
        for (const auto &t: inputs_) {
            auto it = inputPtrs.find(t.name);
            if (it == inputPtrs.end() || !it->second) {
                LOGE_MS("Missing/NULL input: %s", t.name.c_str());
                return false;
            }
            auto shape = toShape(t.dims);
            auto tens = factory.createTensor(
                    shape,
                    reinterpret_cast<const unsigned char *>(it->second),
                    t.bytes()
            );
            if (!tens) {
                LOGE_MS("createTensor failed for input %s", t.name.c_str());
                return false;
            }
            inT.add(t.name.c_str(), tens.release());
        }

        // Prepare output tensors we own (float outputs)
        for (const auto &t: outputs_) {
            auto it = outputPtrs.find(t.name);
            if (it == outputPtrs.end() || !it->second) {
                LOGE_MS("Missing/NULL output: %s", t.name.c_str());
                return false;
            }
            auto shape = toShape(t.dims);
            auto tens = factory.createTensor(shape); // uninitialized, SNPE will fill
            if (!tens) {
                LOGE_MS("createTensor failed for output %s", t.name.c_str());
                return false;
            }
            outT.add(t.name.c_str(), tens.get());
            outKeepAlive.push_back(std::move(tens));
        }

        timeval t0{}, t1{};
        gettimeofday(&t0, nullptr);
        bool ok = snpe_->execute(inT, outT);
        gettimeofday(&t1, nullptr);
        if (!ok) {
            LOGE_MS("SNPE execute (ITensor) failed");
            return false;
        }
        if (elapsedMs)
            *elapsedMs = (t1.tv_sec - t0.tv_sec) * 1000LL + (t1.tv_usec - t0.tv_usec) / 1000LL;

        // Copy outputs back into the workspace buffers the caller provided
        for (size_t i = 0; i < outputs_.size(); ++i) {
            const auto &t = outputs_[i];
            auto tensor = outT.getTensor(t.name.c_str());
//            float *src = const_cast<float*>(outT.getTensor(t.name.c_str())->begin(); // float*
            float *src = const_cast<float*>(&(*tensor->begin()));
            auto *dst = static_cast<float *>(outputPtrs.at(t.name));
            std::memcpy(dst, src, t.bytes());
        }
        return true;
    } else {

        UserBufferMap inMap, outMap;
        std::vector<std::unique_ptr<IUserBuffer>> ubKeepAlive;
        std::vector<std::unique_ptr<UserBufferEncoding>> encKeepAlive;

        auto addUB = [&](const TensorInfo &t, void *ptr, bool isInput) -> bool {
            if (!ptr) {
                LOGE_MS("Null pointer for %s '%s'", isInput ? "input" : "output", t.name.c_str());
                return false;
            }

            auto strides = computePackedStridesBytes(t.dims, t.elementBytes);
            size_t bytes = t.bytes();

            auto &ubFactory = SNPEFactory::getUserBufferFactory();

            // Choose encoding:
            UserBufferEncoding *encRaw = nullptr;
            if (t.dtype == DType::F32) {
                encKeepAlive.emplace_back(new UserBufferEncodingFloatN());
                encRaw = encKeepAlive.back().get();
            } else if (t.dtype == DType::I32) {
                encKeepAlive.emplace_back(new UserBufferEncodingIntN());
//                encRaw = nullptr; // int32: pass no encoding (raw)
                encRaw = encKeepAlive.back().get();
            } else {
                LOGE_MS("Unsupported dtype for %s", t.name.c_str());
                return false;
            }

            auto ub = ubFactory.createUserBuffer(ptr, bytes, strides, encRaw);
            if (!ub) {
                LOGE_MS("Failed to create UserBuffer for %s", t.name.c_str());
                return false;
            }

            if (isInput) inMap.add(t.name.c_str(), ub.get());
            else outMap.add(t.name.c_str(), ub.get());

            ubKeepAlive.push_back(std::move(ub));
            return true;
        };

        // Bind inputs
        for (auto &t: inputs_) {
            auto it = inputPtrs.find(t.name);
            if (it == inputPtrs.end()) {
                LOGE_MS("Missing input: %s", t.name.c_str());
                return false;
            }
            if (!addUB(t, const_cast<void *>(it->second), /*isInput*/true)) return false;
        }
        // Bind outputs (float)
        for (auto &t: outputs_) {
            auto it = outputPtrs.find(t.name);
            if (it == outputPtrs.end()) {
                LOGE_MS("Missing output: %s", t.name.c_str());
                return false;
            }
            if (!addUB(t, it->second, /*isInput*/false)) return false;
        }

        // Run
        timeval t0{}, t1{};
        gettimeofday(&t0, nullptr);
        bool ok = snpe_->execute(inMap, outMap);
        gettimeofday(&t1, nullptr);

        if (!ok) {
            LOGE_MS("SNPE execute failed");
            return false;
        }
        if (elapsedMs)
            *elapsedMs = (t1.tv_sec - t0.tv_sec) * 1000LL + (t1.tv_usec - t0.tv_usec) / 1000LL;
        return true;
    }
}


//bool ModelSession::execute(const std::unordered_map<std::string, const void*>& inputPtrs,
//                           const std::unordered_map<std::string, void*>& outputPtrs,
//                           int64_t* elapsedMs) const
//{
//    using namespace zdl::DlSystem;
//    using namespace zdl::SNPE;
//
//    UserBufferMap inMap, outMap;
//    std::vector<std::unique_ptr<IUserBuffer>> ubKeepAlive;
//    std::vector<std::unique_ptr<UserBufferEncoding>> encKeepAlive;
//
//    auto addUB = [&](const TensorInfo& t, void* ptr, bool isInput) -> bool {
//        if (!ptr) { LOGE_MS("Null pointer for %s '%s'", isInput?"input":"output", t.name.c_str()); return false; }
//
//        auto strides = computePackedStridesBytes(t.dims, t.elementBytes);
//        size_t bytes = t.bytes();
//
//        auto& ubFactory = SNPEFactory::getUserBufferFactory();
//
//        // Choose encoding:
//        UserBufferEncoding* encRaw = nullptr;
//        if (t.dtype == DType::F32) {
//            encKeepAlive.emplace_back(new UserBufferEncodingFloat());
//            encRaw = encKeepAlive.back().get();
//        } else if (t.dtype == DType::I32) {
//            encRaw = nullptr; // int32: pass no encoding (raw)
//        } else {
//            LOGE_MS("Unsupported dtype for %s", t.name.c_str());
//            return false;
//        }
//
//        auto ub = ubFactory.createUserBuffer(ptr, bytes, strides, encRaw);
//        if (!ub) { LOGE_MS("Failed to create UserBuffer for %s", t.name.c_str()); return false; }
//
//        if (isInput) inMap.add(t.name.c_str(), ub.get());
//        else         outMap.add(t.name.c_str(), ub.get());
//
//        ubKeepAlive.push_back(std::move(ub));
//        return true;
//    };
//
//    // Bind inputs
//    for (auto& t : inputs_) {
//        auto it = inputPtrs.find(t.name);
//        if (it == inputPtrs.end()) { LOGE_MS("Missing input: %s", t.name.c_str()); return false; }
//        if (!addUB(t, const_cast<void*>(it->second), /*isInput*/true)) return false;
//    }
//    // Bind outputs (float)
//    for (auto& t : outputs_) {
//        auto it = outputPtrs.find(t.name);
//        if (it == outputPtrs.end()) { LOGE_MS("Missing output: %s", t.name.c_str()); return false; }
//        if (!addUB(t, it->second, /*isInput*/false)) return false;
//    }
//
//    // Run
//    timeval t0{}, t1{};
//    gettimeofday(&t0, nullptr);
//    bool ok = snpe_->execute(inMap, outMap);
//    gettimeofday(&t1, nullptr);
//
//    if (!ok) { LOGE_MS("SNPE execute failed"); return false; }
//    if (elapsedMs) *elapsedMs = (t1.tv_sec - t0.tv_sec)*1000LL + (t1.tv_usec - t0.tv_usec)/1000LL;
//    return true;
//}


//bool execute_old(const std::unordered_map<std::string, const void*>& inputPtrs,
//                           const std::unordered_map<std::string, void*>& outputPtrs,
//                           int64_t* elapsedMs
//                           /*bool inputsAreInt32=false*/) const {
//    using namespace zdl::DlSystem;
//
//    UserBufferMap inMap, outMap;
//    std::vector<std::unique_ptr<IUserBuffer>> ubKeepAlive;
//    std::vector<std::unique_ptr<UserBufferEncoding>> encKeepAlive;
//
//    auto addOne = [&](const TensorInfo& t, const void* ptr, bool isInput) -> bool {
//        if (!ptr) {
//            LOGE_MS("Null pointer for %s '%s'", isInput ? "input" : "output", t.name.c_str());
//            return false;
//        }
//
////        const bool thisIsInt32Input = (isInput && inputsAreInt32);
//
////        UserBufferEncoding* enc = nullptr;
////        if (!thisIsInt32Input) {
//        // encoding: float32 strict
//        //        std::unique_ptr<UserBufferEncoding> enc(new UserBufferEncodingFloat());
//        encKeepAlive.emplace_back(new UserBufferEncodingFloat());
//        auto *enc = encKeepAlive.back().get();
////        enc = encKeepAlive.back().get();
////        }
//
//        auto strides = computePackedStridesBytes(t.dims, t.elementBytes);
//        size_t bytes = t.bytes();
//
//        auto& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
////        std::unique_ptr<IUserBuffer> ub;
////        if (thisIsInt32Input) {
////            // Try overload without encoding if present, else fall back to float-encoding (works in practice).
////            ub.reset(ubFactory.createUserBuffer(const_cast<void*>(ptr), bytes, strides, nullptr));
////        }
//        auto ub = ubFactory.createUserBuffer(
//                const_cast<void*>(ptr),
//                bytes, strides,
//                enc
//                );
//        if (!ub) {
//            LOGE_MS("Failed to create UserBuffer for %s", t.name.c_str());
//            return false;
//        }
//        if (isInput) inMap.add(t.name.c_str(), ub.get());
//        else         outMap.add(t.name.c_str(), ub.get());
//
//        ubKeepAlive.push_back(std::move(ub));
//        return true;
//    };
//
//    // Bind inputs
//    for (auto& t : inputs_) {
//        auto it = inputPtrs.find(t.name);
//        if (it == inputPtrs.end()) { LOGE_MS("Missing input: %s", t.name.c_str()); return false; }
//        if (!addOne(t, it->second, true)) return false;
//    }
//    // Bind outputs
//    for (auto& t : outputs_) {
//        auto it = outputPtrs.find(t.name);
//        if (it == outputPtrs.end()) { LOGE_MS("Missing output: %s", t.name.c_str()); return false; }
//        if (!addOne(t, it->second, false)) return false;
//    }
//
//    // run
//    timeval t0{}, t1{};
//    gettimeofday(&t0, nullptr);
//    bool ok = snpe_->execute(inMap, outMap);
//    gettimeofday(&t1, nullptr);
//
//    if (!ok) {
//        LOGE_MS("SNPE execute failed");
//        return false;
//    }
//    if (elapsedMs) {
//        *elapsedMs = (t1.tv_sec - t0.tv_sec)*1000LL + (t1.tv_usec - t0.tv_usec)/1000LL;
//    }
//    return true;
//}
