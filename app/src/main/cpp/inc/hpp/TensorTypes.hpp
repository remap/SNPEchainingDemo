#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct TensorInfo {
    std::string name;
    std::vector<size_t> dims;   // NCHW style (or as exported)
    size_t elementBytes = 4;    // float32 default
    // Convenience: total bytes
    size_t bytes() const {
        size_t n = elementBytes;
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
