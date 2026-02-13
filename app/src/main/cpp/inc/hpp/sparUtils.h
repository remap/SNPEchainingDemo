//
// Created by Chiheb Boussema on 23/12/25.
//

#ifndef SNPECHAININGDEMO_SPARUTILS_H
#define SNPECHAININGDEMO_SPARUTILS_H

// normalize_pc_bbox.hpp
#pragma once
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <limits>
#include <cmath>

// Optional OpenMP: compile with -fopenmp to enable parallelism
#ifdef _OPENMP
  #include <omp.h>
#endif

// Layout expectation:
//  - pc buffer layout: batch-major contiguous: for b in [0..B-1], for i in [0..N-1], for c in [0..C-1]:
//      index = (b * N + i) * C + c
//  - C must be 3, 6, or 9 (function asserts this)

inline void normalize_pc_bbox_inplace(float* pc, size_t B, size_t N, size_t C, float user_scale = 1.0f) {
    assert(pc != nullptr);
    assert(C == 3 || C == 6 || C == 9);
    if (B == 0 || N == 0) return;

    // Process each batch independently -- parallelize across batch dimension
    #pragma omp parallel for if(B > 1)
    for (ptrdiff_t b = 0; b < static_cast<ptrdiff_t>(B); ++b) {
        // compute bounds for x,y,z
        const size_t batch_offset = b * N * C;

        float min_x = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        float min_z = std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();

        // single pass to compute min/max
        for (size_t i = 0; i < N; ++i) {
            const size_t base = batch_offset + i * C;
            const float x = pc[base + 0];
            const float y = pc[base + 1];
            const float z = pc[base + 2];

            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
            if (z < min_z) min_z = z;
            if (z > max_z) max_z = z;
        }

        // compute center and scale (match python logic: scale = max(range_x, range_y, range_z))
        const float center_x = (max_x + min_x) * 0.5f;
        const float center_y = (max_y + min_y) * 0.5f;
        const float center_z = (max_z + min_z) * 0.5f;

        float range_x = max_x - min_x;
        float range_y = max_y - min_y;
        float range_z = max_z - min_z;

        float bbox_scale = std::max(range_x, std::max(range_y, range_z));
        // If user provided user_scale param, the function signature had scale=1.0 default
        // Python code overwrote scale variable; we keep identical behavior:
        // Use the bbox_scale (if zero fallback), ignore user_scale if you prefer original semantics.
        // To honor the user's requested parameter name, we'll multiply by user_scale.
        if (bbox_scale <= 0.0f || !std::isfinite(bbox_scale)) bbox_scale = 1.0f;
//        bbox_scale *= user_scale;

        const float inv_scale = 1.0f / bbox_scale;

        // second pass: normalize in-place
        for (size_t i = 0; i < N; ++i) {
            const size_t base = batch_offset + i * C;
            // normalize xyz
            pc[base + 0] = (pc[base + 0] - center_x) * inv_scale;
            pc[base + 1] = (pc[base + 1] - center_y) * inv_scale;
            pc[base + 2] = (pc[base + 2] - center_z) * inv_scale;
            // extra channels (3..C-1) remain unchanged
        }
    } // end batch loop
}


// Out-of-place version: src is const, dst gets written.
// Both src_pc and dst_pc must be of size at least B * N * C.
// If dst_pc == src_pc, this still works (it copies normalized values into dst).
inline void normalize_pc_bbox(const float* src_pc, float* dst_pc, size_t B, size_t N, size_t C, float user_scale = 1.0f) {
    assert(src_pc != nullptr && dst_pc != nullptr);
    assert(C == 3 || C == 6 || C == 9);
    if (B == 0 || N == 0) return;

    #pragma omp parallel for if(B > 1)
    for (ptrdiff_t b = 0; b < static_cast<ptrdiff_t>(B); ++b) {
        const size_t batch_offset = b * N * C;

        float min_x = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        float min_z = std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();

        // compute min/max from source
        for (size_t i = 0; i < N; ++i) {
            const size_t base = batch_offset + i * C;
            const float x = src_pc[base + 0];
            const float y = src_pc[base + 1];
            const float z = src_pc[base + 2];

            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
            if (z < min_z) min_z = z;
            if (z > max_z) max_z = z;
        }

        const float center_x = (max_x + min_x) * 0.5f;
        const float center_y = (max_y + min_y) * 0.5f;
        const float center_z = (max_z + min_z) * 0.5f;

        float range_x = max_x - min_x;
        float range_y = max_y - min_y;
        float range_z = max_z - min_z;

        float bbox_scale = std::max(range_x, std::max(range_y, range_z));
        if (bbox_scale <= 0.0f || !std::isfinite(bbox_scale)) bbox_scale = 1.0f;
//        bbox_scale *= user_scale;

        const float inv_scale = 1.0f / bbox_scale;

        // write normalized points to destination
        for (size_t i = 0; i < N; ++i) {
            const size_t base = batch_offset + i * C;
            const float x = src_pc[base + 0];
            const float y = src_pc[base + 1];
            const float z = src_pc[base + 2];

            dst_pc[base + 0] = (x - center_x) * inv_scale;
            dst_pc[base + 1] = (y - center_y) * inv_scale;
            dst_pc[base + 2] = (z - center_z) * inv_scale;

            // copy extra channels if any
            for (size_t ch = 3; ch < C; ++ch) {
                dst_pc[base + ch] = src_pc[base + ch];
            }
        }
    } // end batch loop
}


#endif //SNPECHAININGDEMO_SPARUTILS_H
