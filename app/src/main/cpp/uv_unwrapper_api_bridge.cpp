//
// Created by Chiheb Boussema on 11/12/25.
//
// uv_unwrapper_api_bridge.cpp
#include "uv_unwrapper_api_bridge.h"
#include "uv_unwrapper_api.h"

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <string>
#include <iostream>

#include <android/log.h>
#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "UVUnwrapperBridge"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)


namespace UVUnwrapperBridge {

std::vector<int64_t> UVUnwrapper_assign_faces_uv_to_atlas_index_raw(
    const std::vector<float>& vertices,
    const std::vector<int>& indices,
    const std::vector<float>& face_uv_flat,
    const std::vector<int64_t>& face_index)
{
    // Basic argument checks
    const size_t num_vertices = vertices.size() / 3;
    if (vertices.size() % 3 != 0) /*throw std::invalid_argument*/LOGE("[bridge] vertices length must be multiple of 3");
    const size_t num_faces = indices.size() / 3;
    if (indices.size() % 3 != 0) /*throw std::invalid_argument*/LOGE("[bridge] indices length must be multiple of 3");
    if (face_uv_flat.size() != num_faces * 3 * 2) /*throw std::invalid_argument*/LOGE("face_uv_flat size mismatch");
    if (face_index.size() != num_faces) /*throw std::invalid_argument*/LOGE("[bridge] face_index size mismatch");

    LOGI("[bridge] moving indices from int to int64");
    std::vector<int64_t> indices64;
    indices64.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) indices64.push_back(static_cast<int64_t>(indices[i]));

    int64_t* out_assign = nullptr;

    // Call the C API. Let it choose threading (num_threads = 0).
    UVUnwrapperStatus status = assign_faces_uv_to_atlas_index_raw(
        vertices.empty() ? nullptr : vertices.data(),
        num_vertices,
//        indices.empty() ? nullptr : reinterpret_cast<const int64_t*>(indices.data()),
        indices64.data(),
        num_faces,
        face_uv_flat.empty() ? nullptr : face_uv_flat.data(),
        face_index.empty() ? nullptr : face_index.data(),
        &out_assign,
        1 // default threads
    );

    if (status != UV_OK) {
        // If function failed, ensure we don't leak memory (API contract: out_assign only set on success)
//        throw std::runtime_error("assign_faces_uv_to_atlas_index_raw failed with status " + std::to_string((int)status));
        LOGE("[bridge] assign_faces_uv_to_atlas_index_raw failed with status %i", (int)status);
    }

    // Convert to std::vector<int64_t>
    std::vector<int64_t> result;
    result.reserve(num_faces);
    if (out_assign) {
        for (size_t i = 0; i < num_faces; ++i) result.push_back(out_assign[i]);
        // Free memory allocated by the C API
        uv_free(out_assign);
    } else {
        // nothing returned
        result.assign(num_faces, -1);
    }

    return result;
}

} // namespace UVUnwrapperBridge
