//
// Created by Chiheb Boussema on 1/15/26.
//

#ifndef SNPECHAININGDEMO_GENERICPIPELINE_H
#define SNPECHAININGDEMO_GENERICPIPELINE_H

#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <android/asset_manager.h>
#include <type_traits>
#include <unistd.h>
#include <fstream>         // std::ifstream  (fixes “undefined template basic_ifstream”)
#include <algorithm> // for std::min
#include <cstring>   // for std::memcpy
#include <iostream>  // for std::cerr or logs

#include "TensorWorkspace.hpp"   // workspace helper
#include "newInferenceHelper.hpp"

#define LOG_TAG "GENERIC_PIPELINE"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)



class GenericPipeline {
public:

    GenericPipeline(AAssetManager* mgr,
                    std::string g_modelDir,
                    std::string models_config_filename,
                    TensorWorkspace& ws) : mgr_(mgr),
                    g_modelDir_(std::move(g_modelDir)),
                    models_config_filename_(std::move(models_config_filename)),
                    ws_(ws) {};

    virtual ~GenericPipeline() = default;

    virtual bool init_networks(char runtime_hint='D') = 0;

protected:
    // Helpers
    static bool readFileToBuffer(const std::string& path, void* dst, size_t bytes, AAssetManager* mgr = nullptr) {
        if (mgr) {
            AAsset* asset = AAssetManager_open(mgr, path.c_str(), AASSET_MODE_BUFFER);
            if (asset) {
                size_t assetLen = AAsset_getLength(asset);
                if (assetLen != bytes) {
                    LOGE("Asset size mismatch: expected %zu, got %zu", bytes, assetLen);
                    AAsset_close(asset);
                    return false;
                }
                int read = AAsset_read(asset, dst, bytes);
                AAsset_close(asset);
                return (read > 0 && static_cast<size_t>(read) == bytes);
            }
        }
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        ifs.read(reinterpret_cast<char*>(dst), bytes);
        return static_cast<size_t>(ifs.gcount()) == bytes;
    };

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

    static void nhwc_to_nchw(float* dst, const float* src, int N, int H, int W, int C) {
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
    };

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
    };

    static void transpose_inplace_nhwc_nchw(float* data, int N, int C, int H, int W, bool nhwc_to_nchw=true) {
        const size_t total = static_cast<size_t>(N) * C * H * W;
        // assuming N = 1 -- may fix later
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
    };

    PipelineCfg findPipe(const MultiPipelinesCfg& mpcfg, const std::string& n) const {
        PipelineCfg out;
        for (const auto& pipe: mpcfg.pipes) {
            if (pipe.name == n) return pipe;
        }
        LOGE("Could not find pipeline %s", n.c_str());
        return out;
    }

    template<class T>
    void logNumbers(const T* p,
                    const TensorInfo& t,
                    const std::string& wsName,
                    const size_t& n=8,
                    const char logLevel='i',
                    const char* initLog="") const {
        static_assert(std::is_arithmetic_v<T>, "logNumbers only supports numeric types (int, float, etc.)");
        std::string vals;
        size_t count = std::min<size_t>(n, t.numel());
        LOGW("TENSOR COUNT %zu", t.numel());
        for (size_t i = 0; i < count; ++i) {
            vals += std::to_string(p[i]);
            if (i + 1 < count) vals += ", ";
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
    };

//    void load_tensors(std::string target_net);

    AAssetManager* mgr_;
    std::string g_modelDir_;
    std::string models_config_filename_;
    TensorWorkspace& ws_;

};



#endif //SNPECHAININGDEMO_GENERICPIPELINE_H
