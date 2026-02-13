//
// Created by Chiheb Boussema on 10/12/25.
//

#ifndef SNPECHAININGDEMO_UNWRAPPERCLASS_H
#define SNPECHAININGDEMO_UNWRAPPERCLASS_H

#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <array>
#include <utility>
#include <algorithm>

// Small result struct: unique uv coords + vtex_idx (Nf x 3)
struct MeshUVResult {
    // unique_uv: flattened Utex x 2
    // stored as [u0, v0, u1, v1, ...]
    std::vector<float> unique_uv;

    // vtex_idx: flattened Nf x 3 (index into unique_uv by vertex)
    std::vector<int> vtex_idx;

    // number of triangles (Nf)
    size_t num_faces() const { return vtex_idx.size() / 3; }
    // number of unique uv vertices
    size_t num_unique_uv() const { return unique_uv.size() / 2; }
};

namespace UVUnwrapperBridge {
    // Bridge function (must be implemented / linked from your BVH-based file).
    // Returns assign_indices length = Nf
    std::vector<int64_t> UVUnwrapper_assign_faces_uv_to_atlas_index_raw(
        const std::vector<float>& vertices,   // Nv*3
        const std::vector<int>& indices,      // Nf*3
        const std::vector<float>& face_uv_flat, // Nf*3*2 flattened
        const std::vector<int64_t>& face_index  // Nf
    );
}

class Unwrapper {
public:
    Unwrapper() = default;

    // Main entrypoint: emulate Python Unwrapper.forward(...)
    // Inputs:
    //   vertex_positions: Nv*3 flattened (x,y,z)
    //   vertex_normals:   Nv*3 flattened normals
    //   triangle_idxs:    Nf*3 flattened triangle vertex indices (ints)
    //   island_padding:   padding float
    //
    // Returns MeshUVResult with unique_uv and vtex_idx (Nf*3 ints).
    MeshUVResult forward(
        const std::vector<float>& vertex_positions,
        const std::vector<float>& vertex_normals,
        const std::vector<int>& triangle_idxs,
        float island_padding = 0.02f
    );

private:
    // helpers translated from python methods
    void align_mesh_with_main_axis(
        std::vector<float>& vertex_positions, // in-out
        std::vector<float>& vertex_normals    // in-out
    );

    // box assign: returns face_uv (flattened Nf*3*2) and face_index (Nf)
    void box_assign_vertex_to_cube_face(
        const std::vector<float>& vertex_positions,
        const std::vector<float>& vertex_normals,
        const std::vector<int>& triangle_idxs,
        const std::array<float,6>& bbox, // [minx,miny,minz,maxx,maxy,maxz]
        std::vector<float>& out_face_uv_flat, // out: Nf*3*2
        std::vector<int64_t>& out_face_index    // out: Nf
    );

    // rotate uv slices: modifies face_uv_flat in-place (Nf*3*2)
    void rotate_uv_slices_consistent_space(
        const std::vector<float>& vertex_positions,
        const std::vector<float>& vertex_normals,
        const std::vector<int>& triangle_idxs,
        std::vector<float>& face_uv_flat,
        const std::vector<int64_t>& face_index
    );

    // find offsets and divisors similar to python _find_slice_offset_and_scale
    void find_slice_offset_and_scale(
        const std::vector<int64_t>& index,
        std::vector<float>& offset_x,
        std::vector<float>& offset_y,
        std::vector<float>& div_x,
        std::vector<float>& div_y
    );

    // handle slices and remaining uvs
    void handle_slice_uvs(
        std::vector<float>& face_uv_flat, // Nf*3*2
        const std::vector<int64_t>& index,
        float island_padding,
        int max_index = 12
    );

    void handle_remaining_uvs(
        std::vector<float>& face_uv_flat,
        const std::vector<int64_t>& index,
        float island_padding
    );

    // Distribute face_uv into atlas with offsets/divs
    std::vector<float> distribute_individual_uvs_in_atlas(
        const std::vector<float>& face_uv_flat, // Nf*3*2
        const std::vector<int64_t>& assigned_faces,
        const std::vector<float>& offset_x,
        const std::vector<float>& offset_y,
        const std::vector<float>& div_x,
        const std::vector<float>& div_y,
        float island_padding
    );

    // unique face uv -> unique_uv + vtex_idx
    MeshUVResult get_unique_face_uv(const std::vector<float>& uv_flat, size_t Nf);
};


#endif //SNPECHAININGDEMO_UNWRAPPERCLASS_H
