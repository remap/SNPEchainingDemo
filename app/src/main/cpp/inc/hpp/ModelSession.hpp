//
// Created by Chiheb Boussema.
//

#ifndef SNPECHAININGDEMO_MODELSESSION_H
#define SNPECHAININGDEMO_MODELSESSION_H

#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPEBuilder.hpp"

#include "TensorTypes.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorShape.hpp"

class ModelSession {
public:
    struct Options {
        zdl::DlSystem::RuntimeList runtimeOrder;
        zdl::DlSystem::PerformanceProfile_t perf =
                zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE;
        bool useUserSuppliedBuffers = true;
        bool initCache = true;
    };

//    struct DlcBacking {
//        std::shared_ptr<MMapFile> file;
//        std::shared_ptr<MMapAsset> asset;
//    };

    ModelSession(const std::unordered_map<std::string, DType>& inputHints = {}): inputTypeHints_(inputHints) {}

    // Factory: takes a DLC buffer
    static std::unique_ptr<ModelSession> Create(const uint8_t* dlc, size_t bytes,
                                                std::shared_ptr<void> dlcOwner,
                                                const Options& opt,
                                                std::string* buildLog /*optional*/,
                                                std::unordered_map<std::string, std::string> inputEncodings,
                                                std::string dlc_name);

    void reCreate(std::string* buildLog);

    // Introspection
    const std::vector<TensorInfo>& inputs()  const { return inputs_;  }
    const std::vector<TensorInfo>& outputs() const { return outputs_; }
    const std::string& selectedRuntimeName() const { return runtimeName_; }
    const zdl::SNPE::SNPE* getSnpe() const { return snpe_.get(); }

    // reset
    void reset();
    void resetEverything() {
        snpe_.reset();
        container_.reset();
        dlcBacking_.reset();
        dlcOwner_.reset();
    }

    // One-shot execution. Pointers must be valid during the call.
    // Returns elapsed ms in *elapsedMs if not null.
    bool execute(const std::unordered_map<std::string, const void*>& inputPtrs,
                 const std::unordered_map<std::string, void*>& outputPtrs,
                 int64_t* elapsedMs
                 /*bool inputsAreInt32*/) const;

private:
    ModelSession() = default;

    // SNPE objects
    std::unique_ptr<zdl::SNPE::SNPE> snpe_;
    std::string runtimeName_;
    Options opt_;
//    std::unique_ptr<zdl::SNPE::SNPEBuilder> builder_;
    std::unique_ptr<zdl::DlContainer::IDlContainer> container_;
    std::shared_ptr<const uint8_t> dlcBacking_;
    std::shared_ptr<void> dlcOwner_;

    // IO metadata (float32 assumed at boundaries)
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;

    std::unordered_map<std::string, DType> inputTypeHints_;

    // Helper to probe IO and fill inputs_/outputs_
    void captureIO_();
};

#endif // SNPECHAININGDEMO_MODELSESSION_H