//
// Created by Chiheb Boussema.
//

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class DType { F32, I32 };
inline std::unordered_map<std::string, DType> DType_mapping{
        {"float32", DType::F32},
        {"int32",   DType::I32}
};
inline DType StringToDType(const std::string type) {
//    return DType_mapping.at(type);
    auto it = DType_mapping.find(type);
    if (it == DType_mapping.end()) {
        throw std::out_of_range("Unknown DType string: " + type + ". Supported types are 'float32' and 'int32'.");
    }
    return it->second;
}

struct TensorInfo {
    std::string name;
    std::vector<size_t> dims;   // NCHW style (or as exported)
    DType dtype = DType::F32;
    size_t elementBytes = 4;    // float32 default
    // Convenience: total bytes
    size_t bytes() const {
        size_t n = elementBytes;
        for (auto d : dims) n *= d;
        return n;
    }

    size_t numel() const {
        size_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
};

// Simple helper to compute tightly packed strides (in bytes)
inline std::vector<size_t> computePackedStridesBytes(const std::vector<size_t>& dims,
                                                     size_t elemBytes) {
    if (dims.empty()) return {};
    std::vector<size_t> strides(dims.size());
    strides.back() = elemBytes;
    for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i)
        strides[i] = strides[i+1] * dims[i+1];
    return strides;
}
