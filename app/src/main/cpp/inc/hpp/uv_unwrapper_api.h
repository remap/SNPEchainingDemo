//
// Created by Chiheb Boussema on 10/12/25.
//

#ifndef SNPECHAININGDEMO_UV_UNWRAPPER_API_H
#define SNPECHAININGDEMO_UV_UNWRAPPER_API_H

// uv_unwrapper_api.h
#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Return codes
typedef enum {
    UV_OK = 0,
    UV_BAD_ARG = 1,
    UV_OUT_OF_MEMORY = 2,
    UV_INTERNAL_ERROR = 3
} UVUnwrapperStatus;

/**
 * assign_faces_uv_to_atlas_index_raw
 *
 * Inputs:
 *  - vertices: float array length num_vertices * 3 (row-major: x,y,z)
 *  - num_vertices: number of vertices
 *  - indices: int64 array length num_faces * 3 (row-major triangle indices referencing vertices)
 *  - num_faces: number of faces (triangles)
 *  - face_uv: float array length num_faces * 3 * 2 (for each face: uv0.x,uv0.y, uv1.x,uv1.y, uv2.x,uv2.y)
 *  - face_index: int64 array length num_faces (initial atlas index per face)
 *
 * Output:
 *  - out_assign_indices: pointer to newly allocated int64_t[num_faces], set on success.
 *                         Caller must free with uv_free().
 *
 *  - num_threads: optional, set 0 to use default (OpenMP default) or >0 to set number of threads.
 *
 * Returns UV_OK on success.
 */
UVUnwrapperStatus assign_faces_uv_to_atlas_index_raw(
    const float* vertices, size_t num_vertices,
    const int64_t* indices, size_t num_faces,
    const float* face_uv, const int64_t* face_index,
    int64_t** out_assign_indices,
    int num_threads);

/** Free memory returned by assign_faces_uv_to_atlas_index_raw */
void uv_free(void* p);

#ifdef __cplusplus
} // extern "C"
#endif


#endif //SNPECHAININGDEMO_UV_UNWRAPPER_API_H
