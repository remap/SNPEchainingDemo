//
// Created by Chiheb Boussema on 23/12/25.
//

#ifndef SNPECHAININGDEMO_BAKER_CPU_H
#define SNPECHAININGDEMO_BAKER_CPU_H

#pragma once

// baker_cpu.h - CPU-only definitions for texture baker
// Clean, portable header intended for Android NDK (ARM64) or desktop CPU use.

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "MeshCPP.h"

namespace texture_baker_cpp {

// Basic 2D/3D float/vector types
    union alignas(8) tb_float2 {
        struct {
            float x, y;
        };
        float data[2];

        float &operator[](size_t idx) {
            if (idx > 1) throw std::runtime_error("tb_float2: bad index");
            return data[idx];
        }

        const float &operator[](size_t idx) const {
            if (idx > 1) throw std::runtime_error("tb_float2: bad index");
            return data[idx];
        }

        bool operator==(const tb_float2 &rhs) const { return x == rhs.x && y == rhs.y; }
    };

    union alignas(4) tb_float3 {
        struct {
            float x, y, z;
        };
        float data[3];

        float &operator[](size_t idx) {
            if (idx > 2) throw std::runtime_error("tb_float3: bad index");
            return data[idx];
        }

        const float &operator[](size_t idx) const {
            if (idx > 2) throw std::runtime_error("tb_float3: bad index");
            return data[idx];
        }
    };

    union alignas(16) tb_float4 {
        struct {
            float x, y, z, w;
        };
        float data[4];

        float &operator[](size_t idx) {
            if (idx > 3) throw std::runtime_error("tb_float4: bad index");
            return data[idx];
        }

        const float &operator[](size_t idx) const {
            if (idx > 3) throw std::runtime_error("tb_float4: bad index");
            return data[idx];
        }
    };

    union alignas(4) tb_int3 {
        struct {
            int x, y, z;
        };
        int data[3];

        int &operator[](size_t idx) {
            if (idx > 2) throw std::runtime_error("tb_int3: bad index");
            return data[idx];
        }

        const int &operator[](size_t idx) const {
            if (idx > 2) throw std::runtime_error("tb_int3: bad index");
            return data[idx];
        }
    };

// Axis-aligned bounding box (2D)
    struct alignas(16) AABB {
        tb_float2 min = {FLT_MAX, FLT_MAX};
        tb_float2 max = {-FLT_MAX, -FLT_MAX};

        // grow the AABB to include a point
        void grow(const tb_float2 &p) {
            min.x = std::min(min.x, p.x);
            min.y = std::min(min.y, p.y);
            max.x = std::max(max.x, p.x);
            max.y = std::max(max.y, p.y);
        }

        void grow(const AABB &b) {
            if (b.min.x != FLT_MAX) {
                grow(b.min);
                grow(b.max);
            }
        }

        bool overlaps(const AABB &other) const {
            return min.x <= other.max.x && max.x >= other.min.x &&
                   min.y <= other.max.y && max.y >= other.min.y;
        }

        bool overlaps(const tb_float2 &point) const {
            return point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y;
        }

        void invalidate() {
            min = {FLT_MAX, FLT_MAX};
            max = {-FLT_MAX, -FLT_MAX};
        }

        float area() const {
            tb_float2 extent = {max.x - min.x, max.y - min.y};
            return extent.x * extent.y;
        }
    };

    struct BVHNode {
        AABB bbox;
        int start = 0, end = 0;
        int left = -1, right = -1;

        int num_triangles() const { return end - start; }

        bool is_leaf() const { return left == -1 && right == -1; }

        float calculate_node_cost() const {
            return num_triangles() * bbox.area();
        }
    };

    struct Triangle {
        tb_float2 v0, v1, v2;
        int index = -1;         // original triangle index
        tb_float2 centroid;     // precomputed centroid
    };

// CPU BVH implementation (kept simple and portable)
    struct BVH {
        std::vector<BVHNode> nodes;
        std::vector<Triangle> triangles;
        std::vector<int> triangle_indices;
        int root = -1;

        // Build BVH from vertex and index arrays
        // vertices: pointer to N x 2 float (tb_float2) array
        // indices: pointer to M x 3 int (tb_int3) array
        // num_indices: number of triangles (M)
        void build(const tb_float2 *vertices, const tb_int3 *indices, const int64_t &num_indices);

        // Intersect a 2D point against triangles stored in the BVH; returns true if hit
        bool intersect(const tb_float2 &point, float &u, float &v, float &w, int &index) const;

        // internal helpers (definitions in .cpp)
        void update_node_bounds(BVHNode &node, AABB &centroidBounds);

        float find_best_split_plane(const BVHNode &node, int &best_axis, int &best_pos,
                                    AABB &centroidBounds);
    };

inline void hide_garbage() {
// CPU rasterize / interpolate helpers (plain host APIs)
//
// NOTE: the functions below return std::vector<float> with the following layouts:
// - rasterize_cpu_host(...)
//     returns a vector of size (bake_resolution * bake_resolution * 4)
//     layout: row-major [y, x, 4] where index = (y * width + x) * 4
//     each pixel stores: [u, v, w, triangle_index_as_float] (triangle_index = -1.0f if no hit)
//
// - interpolate_cpu_host(...)
//     returns a vector of size (height * width * 3)
//     layout: row-major [y, x, 3] where index = (y * width + x) * 3
//     each pixel stores the interpolated 3D attribute (x,y,z)
//
// These are drop-in replacements of the original tensor-based functions but using raw pointers.
//
// vertices: tb_float2* (N x 2)
// indices:  tb_int3*  (M x 3)   (same format used to build BVH)
// num_indices: number of triangles (M)

// Pointer-based rasterize / interpolate helpers (zero-copy):
//
// 1) Pointer-based rasterize (preferred for zero-copy):
//    - rast_result: preallocated float array with size bake_resolution*bake_resolution*4
//                   layout: row-major (y * width + x) * 4
//    - vertices: tb_float2* (num_vertices elements)
//    - indices: tb_int3* (num_triangles elements)
//    - num_triangles: number of triangles (M)
//    - bake_resolution: width == height == bake_resolution
//
//    void rasterize_cpu_host(float* rast_result,
//                            const tb_float2* vertices,
//                            const tb_int3* indices,
//                            const int64_t num_triangles,
//                            const int bake_resolution);
//
// 2) Convenience overload taking flattened host vectors and writing into rast_result:
//    void rasterize_cpu_host(float* rast_result,
//                            const std::vector<float>& vertices_flat,
//                            const std::vector<int>& indices_flat,
//                            const int64_t num_triangles,
//                            const int bake_resolution);
//
// 3) Pointer-based interpolate (writes into caller output buffer):
//    - output_ptr: preallocated float array size width*height*3
//    - attr_ptr: per-vertex attributes (num_vertices*3 floats)
//    - indices_ptr: flat triangle indices (num_triangles*3 ints)
//    - num_triangles: number of triangles
//    - rast_ptr: raster buffer produced by rasterize (width*height*4 floats)
//    - width, height: raster dims
//
//    void interpolate_cpu_host(float* output_ptr,
//                              const float* attr_ptr,
//                              const int* indices_ptr,
//                              const int num_triangles,
//                              const float* rast_ptr,
//                              const int width,
//                              const int height);
//
// 4) Convenience vector-based wrapper:
//    void interpolate_cpu_host(std::vector<float>& output_flat,
//                              const std::vector<float>& attr_flat,
//                              const std::vector<int>& indices_flat,
//                              const std::vector<float>& rast_flat,
//                              const int width,
//                              const int height);


//std::vector<float> rasterize_cpu_host(const tb_float2* vertices, const tb_int3* indices, const int64_t num_indices, int bake_resolution);
//
//// attr: attribute buffer (num_vertices * 3 floats), row-major per-vertex vec3
//// indices: same triangle index layout as above (M x 3)
//// rast: raster buffer (width * height * 4 floats) — typically produced by rasterize_cpu_host
//std::vector<float> interpolate_cpu_host(const float* attr, const int* indices_flat, int num_triangles, const float* rast, int width, int height);
};

void rasterize_cpu_host(float* rast_result,
                        const tb_float2* vertices,
                        const tb_int3* indices,
                        const int64_t num_triangles,
                        const int bake_resolution);

void rasterize_cpu_host(float* rast_result,
                        const std::vector<float>& vertices_flat,
                        const std::vector<int>& indices_flat,
                        const int64_t num_triangles,
                        const int bake_resolution);

void rasterize_cpu_host_triangleTile(float* rast_result,
                        const tb_float2* vertices,
                        const tb_int3* indices,
                        const int64_t num_triangles,
                        const int bake_resolution);

void interpolate_cpu_host(float* output_ptr,
                          const float* attr_ptr,
                          const int* indices_ptr,
                          const int num_triangles,
                          const float* rast_ptr,
                          const int width,
                          const int height);

void interpolate_cpu_host(std::vector<float>& output_flat,
                          const std::vector<float>& attr_flat,
                          const std::vector<int>& indices_flat,
                          const std::vector<float>& rast_flat,
                          const int width,
                          const int height);

struct BuildTexturesInspect {
    std::vector<float> basecolor;        // H*W*3 (float [0,1])
    std::vector<float> bump;             // H*W*3 (float [0,1])
    std::vector<float> bump_filled;             // H*W*3 (float [0,1])
    std::vector<float> albedo;             // H*W*3 (float [0,1])
    // std::vector<float> albedo_filled;             // H*W*3 (float [0,1])

    std::vector<float> gb_nrm;           // mask_count * 3
    std::vector<float> gb_tng;           // mask_count * 3
    std::vector<float> gb_btng;          // mask_count * 3
    std::vector<float> normal_tangent;   // mask_count * 3

    std::vector<int> mask_indices;       // mask indices (flat index)
    int H = 0;
    int W = 0;
    float roughness = 0.0f;
    float metallic = 0.0f;

    std::string basecolor_jpeg_bytes;    // binary bytes (not null-terminated text)
    std::string bump_jpeg_bytes;       // binary bytes for bump/normal texture
    std::string basecolor_path;
    std::string bump_path;
};

//float* dilate_fill_cv(
//    const float* img_in,
//    const uint8_t* mask_in,
//    const int H, const int W,
//    const int iterations = 6
//);
//cv::Mat dilate_fill_cv(
//    const float* img_in,
//    const uint8_t* mask_in,
//    const int H, const int W,
//    const int iterations = 6
//);
cv::Mat dilate_fill_cv(
    const cv::Mat img_in,
    const cv::Mat mask_in,
    const int H, const int W,
    const int iterations = 6
);

BuildTexturesInspect BuildTextures_SaveImages(
    // geometry / attributes (zero-copy pointers)
    const float* v_nrm, size_t v_nrm_count,
    const float* v_tng, size_t v_tng_count,
    const float* v_pos, size_t v_pos_count,
    const float* v_tex, size_t v_tex_count,
    const int* t_pos_idx, size_t t_pos_idx_count, // length = num_triangles * 3
    // rasterization results
    const float* rast_ptr, // size = width*height*4
    const uint8_t* bake_mask_ptr, // size = width*height (0 or 1)
    int width, int height,
    // decoded / per-mask arrays
    const float* features_masked, // mask_count * 3
    const float* perturb_normal_masked, // mask_count * 3
    // numeric parameters
    float roughness,
    float metallic,
    // outputs (file paths)
    const std::string &basecolor_outpath,
    const std::string &bump_outpath,
    bool save_as_jpeg = true
);

bool ExportGLBFromInspect(
    const BuildTexturesInspect& inspect,
    const MeshCPP& mesh,
    const std::string& output_glb_path
);

} // namespace texture_baker_cpp


#endif //SNPECHAININGDEMO_BAKER_CPU_H
