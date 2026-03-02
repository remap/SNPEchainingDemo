//
// Created by Chiheb Boussema on 10/12/25.
//
// uv_unwrapper_api.cpp
#include "uv_unwrapper_api.h"
#include "bvh.h"
#include "intersect.h"
#include "common.h"

#include <vector>
#include <set>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::size_t;

#include "unwrapper_decl.h"
using namespace UVUnwrapper;

#include <android/log.h>
#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "UVUnwrapperAPI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

/**
 * Helper wrappers: create_bvhs(...) and perform_intersection_check(...) are defined
 * in the original unwrapper.cpp (you provided). Make sure those functions are declared
 * in a header (e.g., in bvh.h or intersect.h). If they live in a .cpp without a header,
 * copy their declarations into a header and include it here.
 *
 * This file expects those functions to be linkable from the same library (they are in
 * your bvh / intersect units).
 */

//extern void create_bvhs(BVH* bvhs, Triangle* triangles,
//                 std::vector<std::set<int>> &triangle_per_face, int num_faces,
//                 int start, int end);
//
//extern void perform_intersection_check(BVH *bvhs, int num_bvhs, Triangle *triangles,
//                                uv_float3 *vertex_tri_centroids,
//                                int64_t *assign_indices_ptr,
//                                ssize_t num_indices, int offset,
//                                std::vector<std::set<int>> &triangle_per_face);

/**
 * Core C API implementation
 */
//UVUnwrapperStatus assign_faces_uv_to_atlas_index_raw(
//    const float* vertices, size_t num_vertices,
//    const int64_t* indices, size_t num_faces,
//    const float* face_uv, const int64_t* face_index,
//    int64_t** out_assign_indices,
//    int num_threads)
//{
//    if (!vertices || !indices || !face_uv || !face_index || !out_assign_indices) {
//        return UV_BAD_ARG;
//    }
//    *out_assign_indices = nullptr;
//
//    // Optionally set number of threads for OpenMP
//#ifdef _OPENMP
//    if (num_threads > 0) {
//        omp_set_num_threads(num_threads);
//    }
//#endif
//
//    try {
//        // allocate result buffer (caller must free with uv_free)
//        int64_t* assign_indices = (int64_t*)std::malloc(sizeof(int64_t) * num_faces);
//        if (!assign_indices) return UV_OUT_OF_MEMORY;
//        std::memcpy(assign_indices, face_index, sizeof(int64_t) * num_faces);
//
//        // allocate helpers
//        uv_float3* vertex_tri_centroids = new uv_float3[num_faces];
//        Triangle* triangles = new Triangle[num_faces];
//
//        // per-face triangle index sets (13 buckets as original)
//        std::vector<std::set<int>> triangle_per_face;
//        triangle_per_face.resize(13);
//
//        // Build triangle geometry and initial triangle_per_face
//        // face_uv layout: for face i: [u0.x, u0.y, u1.x, u1.y, u2.x, u2.y] at offset i*6
//#pragma omp parallel for
//        for (int i = 0; i < static_cast<int>(num_faces); ++i) {
//            size_t foff = static_cast<size_t>(i) * 6;
//            triangles[i].v0.x = face_uv[foff + 0];
//            triangles[i].v0.y = face_uv[foff + 1];
//            triangles[i].v1.x = face_uv[foff + 2];
//            triangles[i].v1.y = face_uv[foff + 3];
//            triangles[i].v2.x = face_uv[foff + 4];
//            triangles[i].v2.y = face_uv[foff + 5];
//            triangles[i].centroid = triangle_centroid(triangles[i].v0, triangles[i].v1, triangles[i].v2);
//
//            // centroid
//            uv_float3 cent;
//            cent.x = (triangles[i].v0.x + triangles[i].v1.x + triangles[i].v2.x) / 3.0f;
//            cent.y = (triangles[i].v0.y + triangles[i].v1.y + triangles[i].v2.y) / 3.0f;
//            cent.z = 0.0f;
//            vertex_tri_centroids[i] = cent;
//
//#pragma omp critical
//            {
//                int idx = static_cast<int>(face_index[i]);
//                if (idx < 0) idx = 0;
//                if (idx >= static_cast<int>(triangle_per_face.size())) idx = static_cast<int>(triangle_per_face.size()) - 1;
//                triangle_per_face[idx].insert(i);
//            }
//        }
//
//        // Build BVHs and do intersection checks (two passes as in Python/C++ implementation)
//        BVH* bvhs = new BVH[6];
//        create_bvhs(bvhs, triangles, triangle_per_face, static_cast<int>(num_faces), 0, 6);
//        perform_intersection_check(bvhs, 6, triangles, vertex_tri_centroids, assign_indices, static_cast<ssize_t>(num_faces), 0, triangle_per_face);
//
//        BVH* new_bvhs = new BVH[6];
//        create_bvhs(new_bvhs, triangles, triangle_per_face, static_cast<int>(num_faces), 6, 12);
//        perform_intersection_check(new_bvhs, 6, triangles, vertex_tri_centroids, assign_indices, static_cast<ssize_t>(num_faces), 6, triangle_per_face);
//
//        // cleanup
//        delete[] vertex_tri_centroids;
//        delete[] triangles;
//        delete[] bvhs;
//        delete[] new_bvhs;
//
//        *out_assign_indices = assign_indices;
//        return UV_OK;
//    } catch (...) {
//        return UV_INTERNAL_ERROR;
//    }
//}

//std::vector<int64_t> assign_faces_uv_to_atlas_index_raw(
//    const std::vector<float>& vertices,        // Nv*3  (x,y,z) flattened
//    const std::vector<int>& indices,           // Nf*3  (triangle vertex indices) flattened
//    const std::vector<float>& face_uv_flat,    // Nf*6  (u0,v0,u1,v1,u2,v2) flattened
//    const std::vector<int64_t>& face_index     // Nf    (initial face index per triangle)
//) {
//    if (!vertices || !indices || !face_uv || !face_index || !out_assign_indices) {
//        return UV_BAD_ARG;
//    }
//    *out_assign_indices = nullptr;
//
//    // Optionally set number of threads for OpenMP
//#ifdef _OPENMP
//    if (num_threads > 0) {
//        omp_set_num_threads(num_threads);
//    }
//#endif
//
//    // Number of faces (triangles)
//    const size_t num_faces = indices.size() / 3;
//    // Defensive check: face_uv_flat should be 6 * num_faces
//    if (face_uv_flat.size() < num_faces * 6) {
//        throw std::runtime_error("face_uv_flat too small");
//    }
//    if (vertices.size() % 3 != 0) {
//        throw std::runtime_error("vertices length not divisible by 3");
//    }
//    if (indices.size() % 3 != 0) {
//        throw std::runtime_error("indices length not divisible by 3");
//    }
//    if (face_index.size() != num_faces) {
//        throw std::runtime_error("face_index size mismatch");
//    }
//
//    // Output: assign indices (copy of face_index)
//    std::vector<int64_t> assign_indices = face_index; // copy initial assignments
//
//    // Reserve vectors for triangles and 3D centroids (RAII, exception-safe)
//    std::vector<UVUnwrapper::Triangle> triangles;
//    triangles.resize(num_faces);
//
//    std::vector<UVUnwrapper::uv_float3> vertex_tri_centroids;
//    vertex_tri_centroids.resize(num_faces);
//
//    // Use std::set for triangle_per_face (size 13 buckets to match original code)
//    std::vector<std::set<int>> triangle_per_face;
//    triangle_per_face.resize(13);
//
//    // First pass: fill triangles (UV), compute UV centroid, compute 3D centroid,
//    // and register triangle into triangle_per_face based on initial face_index.
//    //
//    // This loop parallels the original code's #pragma omp parallel for.
//    // We keep the same behavior of doing a critical insertion into std::set
//    // for thread-safety when using OpenMP.
//#pragma omp parallel for
//    for (int i = 0; i < static_cast<int>(num_faces); ++i) {
//        // Read face UVs (flattened as 6 floats per face)
//        const size_t foff = static_cast<size_t>(i) * 6;
//        triangles[i].v0.x = face_uv_flat[foff + 0];
//        triangles[i].v0.y = face_uv_flat[foff + 1];
//        triangles[i].v1.x = face_uv_flat[foff + 2];
//        triangles[i].v1.y = face_uv_flat[foff + 3];
//        triangles[i].v2.x = face_uv_flat[foff + 4];
//        triangles[i].v2.y = face_uv_flat[foff + 5];
//
//        // Compute UV centroid (triangle_centroid defined in UVUnwrapper)
//        triangles[i].centroid = UVUnwrapper::triangle_centroid(
//            triangles[i].v0, triangles[i].v1, triangles[i].v2);
//
//        // Read 3D vertex positions via indices (flattened vertices array)
//        const int idx0 = indices[static_cast<size_t>(i) * 3 + 0];
//        const int idx1 = indices[static_cast<size_t>(i) * 3 + 1];
//        const int idx2 = indices[static_cast<size_t>(i) * 3 + 2];
//
//        // Defensive: ensure indices are in-range; on invalid index we clamp to 0 to avoid UB.
//#ifndef NDEBUG
//        const size_t Nv = vertices.size() / 3;
//        assert(idx0 >= 0 && static_cast<size_t>(idx0) < Nv);
//        assert(idx1 >= 0 && static_cast<size_t>(idx1) < Nv);
//        assert(idx2 >= 0 && static_cast<size_t>(idx2) < Nv);
//#endif
//
//        UVUnwrapper::uv_float3 v0, v1, v2;
//        v0.x = vertices[static_cast<size_t>(idx0) * 3 + 0];
//        v0.y = vertices[static_cast<size_t>(idx0) * 3 + 1];
//        v0.z = vertices[static_cast<size_t>(idx0) * 3 + 2];
//
//        v1.x = vertices[static_cast<size_t>(idx1) * 3 + 0];
//        v1.y = vertices[static_cast<size_t>(idx1) * 3 + 1];
//        v1.z = vertices[static_cast<size_t>(idx1) * 3 + 2];
//
//        v2.x = vertices[static_cast<size_t>(idx2) * 3 + 0];
//        v2.y = vertices[static_cast<size_t>(idx2) * 3 + 1];
//        v2.z = vertices[static_cast<size_t>(idx2) * 3 + 2];
//
//        // 3D centroid for spatial tests
//        vertex_tri_centroids[static_cast<size_t>(i)] =
//            UVUnwrapper::triangle_centroid(v0, v1, v2);
//
//        // Assign this triangle index to the appropriate face bucket.
//        // The original code does this inside a critical section to protect
//        // concurrent insertion into std::set.
//        int face_bucket = static_cast<int>(face_index[static_cast<size_t>(i)]);
//        // Be conservative: clamp to valid bucket range if out-of-range (prevent UB).
//        if (face_bucket < 0) face_bucket = 0;
//        if (face_bucket >= static_cast<int>(triangle_per_face.size()))
//            face_bucket = static_cast<int>(triangle_per_face.size()) - 1;
//
//#pragma omp critical
//        {
//            triangle_per_face[face_bucket].insert(i);
//        }
//    } // end for faces
//
//    // --- Build BVHs for first pass (0..5) and run intersection check ---
//    // Allocate 6 BVHs
//    UVUnwrapper::BVH* bvhs = new UVUnwrapper::BVH[6];
//    // create_bvhs expects: (BVH* bvhs, Triangle* triangles, vector<set<int>>& triangle_per_face,
//    //                      size_t num_faces, int start_index, int end_index)
//    UVUnwrapper::create_bvhs(bvhs, triangles.data(), triangle_per_face, num_faces, 0, 6);
//
//    // Run intersection check (this will mutate assign_indices data)
//    // perform_intersection_check(BVH*, int n_bvhs, Triangle*, uv_float3*, int64_t*, long, int start_index, vector<set<int>>&)
//    UVUnwrapper::perform_intersection_check(
//        bvhs, 6, triangles.data(), vertex_tri_centroids.data(),
//        assign_indices.data(), static_cast<long>(num_faces), 0, triangle_per_face);
//
//    // --- Second pass: build BVHs for indices 6..11 and run intersection check ---
//    UVUnwrapper::BVH* new_bvhs = new UVUnwrapper::BVH[6];
//    UVUnwrapper::create_bvhs(new_bvhs, triangles.data(), triangle_per_face, num_faces, 6, 12);
//
//    UVUnwrapper::perform_intersection_check(
//        new_bvhs, 6, triangles.data(), vertex_tri_centroids.data(),
//        assign_indices.data(), static_cast<long>(num_faces), 6, triangle_per_face);
//
//    // Cleanup BVH arrays (triangles and centroids are RAII-managed)
//    delete[] bvhs;
//    delete[] new_bvhs;
//
//    // Return the assign indices
//    return assign_indices;
//}


UVUnwrapperStatus assign_faces_uv_to_atlas_index_raw(
    const float* vertices, size_t num_vertices,         // vertices: Nv * 3
    const int64_t* indices, size_t num_faces,           // indices: Nf * 3 (int64_t)
    const float* face_uv,                               // face_uv: Nf * 6 (u0,v0,u1,v1,u2,v2)
    const int64_t* face_index,                          // face_index: Nf
    int64_t** out_assign_indices,                       // out: pointer to allocated int64_t array of length Nf
    int num_threads)
{
    // Basic argument validation
    if (!vertices || !indices || !face_uv || !face_index || !out_assign_indices) {
        return UV_BAD_ARG;
    }

    // Defensive check: sizes should be consistent (num_vertices should be divisible by 3, indices length implicitly 3*num_faces)
    if (num_vertices == 0 || num_faces == 0) {
        // return empty assignment array, but still allocate zero bytes (set to nullptr)
        *out_assign_indices = nullptr;
        return UV_OK;
    }

    // Allocate output assignment array (caller is responsible for free()).
    int64_t* assign_indices = (int64_t*) std::malloc(num_faces * sizeof(int64_t));
    if (!assign_indices) {
        *out_assign_indices = nullptr;
        return UV_OUT_OF_MEMORY;
    }

    // Copy initial face_index values into assign_indices (mirrors original behavior).
    std::memcpy(assign_indices, face_index, num_faces * sizeof(int64_t));

    // Prepare arrays for triangles and centroids (RAII via std::vector)
    std::vector<UVUnwrapper::Triangle> triangles;
    try {
        triangles.resize(num_faces);
    } catch (const std::bad_alloc&) {
        std::cerr << "COULD NOT RESIZE TRIANGLES!\n";
        LOGE("COULD NOT RESIZE TRIANGLES!");
        std::free(assign_indices);
        *out_assign_indices = nullptr;
        return UV_OUT_OF_MEMORY;
    }

    std::vector<uv_float3> vertex_tri_centroids;
    try {
        vertex_tri_centroids.resize(num_faces);
    } catch (const std::bad_alloc&) {
        std::cerr << "COULD NOT RESIZE CENTROIDS!\n";
        LOGE("COULD NOT RESIZE CENTROIDS!");
        std::free(assign_indices);
        *out_assign_indices = nullptr;
        return UV_OUT_OF_MEMORY;
    }

    // triangle_per_face: vector of std::set<int> with 13 buckets as in original
    std::vector<std::set<int>> triangle_per_face;
    triangle_per_face.resize(13);


    // quick defensive scan BEFORE heavy work:
    std::cerr << "CHECKING INDICES...\n";
    LOGI("[uv_unwrapper_api:] CHECKING INDICES...");
    {
        const size_t Nv = num_vertices;
        bool bad = false;
        int64_t bad_face = -1;
        int64_t bad_idx_val = -1;
        for (size_t i = 0; i < num_faces; ++i) {
            int64_t idx0 = indices[i*3 + 0];
            int64_t idx1 = indices[i*3 + 1];
            int64_t idx2 = indices[i*3 + 2];
            if (idx0 < 0 || static_cast<size_t>(idx0) >= Nv) { bad = true; bad_face = (int64_t)i; bad_idx_val = idx0; break; }
            if (idx1 < 0 || static_cast<size_t>(idx1) >= Nv) { bad = true; bad_face = (int64_t)i; bad_idx_val = idx1; break; }
            if (idx2 < 0 || static_cast<size_t>(idx2) >= Nv) { bad = true; bad_face = (int64_t)i; bad_idx_val = idx2; break; }
        }
        if (bad) {
            std::fprintf(stderr, "assign_faces_uv_to_atlas_index_raw: BAD INDEX: face %lld has idx %lld (num_vertices=%zu,num_faces=%zu)\n",
                         (long long)bad_face, (long long)bad_idx_val, Nv, num_faces);
            LOGE("[uv_unwrapper_api:] assign_faces_uv_to_atlas_index_raw: BAD INDEX: face %lld has idx %lld (num_vertices=%zu,num_faces=%zu)",
                 (long long)bad_face, (long long)bad_idx_val, Nv, num_faces);
            // Return an error status so Python sees this instead of hanging/crashing deeper.
            *out_assign_indices = nullptr;
            return UV_BAD_ARG;
        }
    }
    std::cerr << "CHECKING INDICES OK!\n";
    LOGI("[uv_unwrapper_api:] CHECKING INDICES OK!");


    // Configure OpenMP threads if available and requested
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#endif

    // First pass: fill triangle UVs, compute UV centroids, compute 3D centroids,
    // and insert triangle index into triangle_per_face face bucket.
    auto t_start = std::chrono::high_resolution_clock::now();
    std::cerr << "[UV unwrap] entering for loop\n";
    LOGI("[uv_unwrapper_api:] entering for loop");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < static_cast<int>(num_faces); ++i) {
        // Read flattened face_uv (6 floats per face)
        size_t foff = static_cast<size_t>(i) * 6;
        triangles[i].v0.x = face_uv[foff + 0];
        triangles[i].v0.y = face_uv[foff + 1];
        triangles[i].v1.x = face_uv[foff + 2];
        triangles[i].v1.y = face_uv[foff + 3];
        triangles[i].v2.x = face_uv[foff + 4];
        triangles[i].v2.y = face_uv[foff + 5];

        // Compute UV centroid using project-provided helper
        triangles[i].centroid = triangle_centroid(triangles[i].v0, triangles[i].v1, triangles[i].v2);

        // Read 3D vertex coordinates for the triangle using indices array
        int64_t idx0 = indices[static_cast<size_t>(i) * 3 + 0];
        int64_t idx1 = indices[static_cast<size_t>(i) * 3 + 1];
        int64_t idx2 = indices[static_cast<size_t>(i) * 3 + 2];

        // Defensive bounds checking (only in debug builds).
//#ifndef NDEBUG
//        const size_t Nv = num_vertices;
//        assert(idx0 >= 0 && static_cast<size_t>(idx0) < Nv);
//        assert(idx1 >= 0 && static_cast<size_t>(idx1) < Nv);
//        assert(idx2 >= 0 && static_cast<size_t>(idx2) < Nv);
//#endif

        // vertices is flattened Nv*3 (x,y,z)
        uv_float3 v0, v1, v2;
        v0.x = vertices[static_cast<size_t>(idx0) * 3 + 0];
        v0.y = vertices[static_cast<size_t>(idx0) * 3 + 1];
        v0.z = vertices[static_cast<size_t>(idx0) * 3 + 2];

        v1.x = vertices[static_cast<size_t>(idx1) * 3 + 0];
        v1.y = vertices[static_cast<size_t>(idx1) * 3 + 1];
        v1.z = vertices[static_cast<size_t>(idx1) * 3 + 2];

        v2.x = vertices[static_cast<size_t>(idx2) * 3 + 0];
        v2.y = vertices[static_cast<size_t>(idx2) * 3 + 1];
        v2.z = vertices[static_cast<size_t>(idx2) * 3 + 2];

        vertex_tri_centroids[static_cast<size_t>(i)] = triangle_centroid(v0, v1, v2);

        // Insert triangle index into appropriate face bucket (thread-safe via critical)
        int face_bucket = static_cast<int>(face_index[static_cast<size_t>(i)]);
        if (face_bucket < 0) face_bucket = 0;
        if (face_bucket >= static_cast<int>(triangle_per_face.size()))
            face_bucket = static_cast<int>(triangle_per_face.size()) - 1;
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            triangle_per_face[face_bucket].insert(i);
        }
    }
    auto t_after_first_pass = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sec1 = t_after_first_pass - t_start;
    std::cout << "first pass time: " << sec1.count() << "s\n";
    LOGI("[uv_unwrapper_api:] first pass time: %f s",sec1.count());

    // Build first BVH set (0..5) and run intersection check
    UVUnwrapper::BVH* bvhs = nullptr;
    try {
        bvhs = new UVUnwrapper::BVH[6];
    } catch (const std::bad_alloc&) {
        std::free(assign_indices);
        *out_assign_indices = nullptr;
        return UV_OUT_OF_MEMORY;
    }

    // create_bvhs signature in your codebase may vary slightly; call accordingly.
    // We cast num_faces to int because original signature used int in some places.
    std::cerr << "Creating bvhs 0-6...\n";
    LOGI("Creating bvhs 0-6...");
    create_bvhs(bvhs, triangles.data(), triangle_per_face, static_cast<int>(num_faces), 0, 6);
    std::cerr << "Done with bvhs, performing intersection checks...\n";
    LOGI("Done with bvhs, performing intersection checks...");
    auto t_after_bvh = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sec2 = t_after_bvh - t_after_first_pass;
    std::cout << "create_bvhs time: " << sec2.count() << "s\n";
    LOGI("create_bvhs time: %f s", sec2.count());
    // perform_intersection_check will modify assign_indices in-place
    perform_intersection_check(
        bvhs,
        6,
        triangles.data(),
        vertex_tri_centroids.data(),
        assign_indices,
        static_cast<long>(num_faces),
        0,
        triangle_per_face
    );
    std::cerr << "Done with intersection checks\n";
    LOGI("Done with intersection checks");
    auto t_after_inter1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sec3 = t_after_inter1 - t_after_bvh;
    std::cout << "intersect1 time: " << sec3.count() << "s\n";
    LOGI("intersect1 time: %f s", sec3.count());

    // Second BVH set (6..11)
    UVUnwrapper::BVH* new_bvhs = nullptr;
    try {
        new_bvhs = new UVUnwrapper::BVH[6];
    } catch (const std::bad_alloc&) {
        delete[] bvhs;
        std::free(assign_indices);
        *out_assign_indices = nullptr;
        return UV_OUT_OF_MEMORY;
    }

    create_bvhs(new_bvhs, triangles.data(), triangle_per_face, static_cast<int>(num_faces), 6, 12);

    perform_intersection_check(
        new_bvhs,
        6,
        triangles.data(),
        vertex_tri_centroids.data(),
        assign_indices,
        static_cast<long>(num_faces),
        6,
        triangle_per_face
    );

    // Cleanup BVHs (triangles & centroids are RAII vectors)
    delete[] bvhs;
    delete[] new_bvhs;

    // Set out pointer for caller; caller is responsible for free()
    *out_assign_indices = assign_indices;

    return UV_OK;
}

void uv_free(void* p) {
    if (p) std::free(p);
}
