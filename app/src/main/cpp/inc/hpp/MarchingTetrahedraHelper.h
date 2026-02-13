//
// Created by Chiheb Boussema on 9/12/25.
//

#ifndef SNPECHAININGDEMO_MARCHINGTETRAHEDRAHELPER_H
#define SNPECHAININGDEMO_MARCHINGTETRAHEDRAHELPER_H

// MarchingTetrahedraTwoPass.h
// Two-pass Marching Tetrahedra implementation optimized for large meshes.
// - Uses packed uint64_t keys for edges: key = (uint64_t(min_idx) << 32) | uint64_t(max_idx)
// - Sort + unique on keys, then single interpolation pass.
// - Matches Python semantics: occupancy = (sdf > 0), deformation applied as grid + grid_scale * tanh(deformation).
//
// Usage:
//   MarchingTetrahedraHelper helper(
//       resolution, grid_vertices_vec, indices_vec, /*points_min=*/0.0f, /*points_max=*/1.0f
//   );
//   Mesh mesh = helper.forward(level_ptr, deformation_ptr_or_null);
//
// Notes:
//   - Designed for high-end devices; peak memory ~200-350MiB for your sizes (~536k verts, ~3M tets).
//   - Places to add parallelism are clearly annotated (first pass edge collection, interpolation over unique edges, triangle assembly).
//
#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <cassert>

namespace mtd2 {

// Simple Vec3
struct Vec3 {
    float x, y, z;
    Vec3() : x(0.f), y(0.f), z(0.f) {}
    Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};

// Output mesh (flattened arrays)
struct Mesh {
    std::vector<float> v_pos;        // flattened Nx3
    std::vector<uint32_t> t_pos_idx; // flattened Mx3
};

// Helper: pack two uint32 into uint64 (min << 32) | max (ensures ordering)
inline uint64_t pack_edge_key(uint32_t a, uint32_t b) {
    uint32_t vmin = (a <= b) ? a : b;
    uint32_t vmax = (a <= b) ? b : a;
    return (static_cast<uint64_t>(vmin) << 32) | static_cast<uint64_t>(vmax);
}
inline void unpack_edge_key(uint64_t key, uint32_t &out_min, uint32_t &out_max) {
    out_min = static_cast<uint32_t>(key >> 32);
    out_max = static_cast<uint32_t>(key & 0xffffffffu);
}

// Local mapping for tet edges (edge index -> (local vertex a, local vertex b))
static constexpr uint8_t EDGE_TO_LOCAL_VERTS[6][2] = {
    {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
};

// Exact triangle table taken from your Python source (16 x up to 6 entries).
static constexpr int TRI_TABLE[16][6] = {
    {-1, -1, -1, -1, -1, -1},
    { 1,  0,  2, -1, -1, -1},
    { 4,  0,  3, -1, -1, -1},
    { 1,  4,  2,  1,  3,  4},
    { 3,  1,  5, -1, -1, -1},
    { 2,  3,  0,  2,  5,  3},
    { 1,  4,  0,  1,  5,  4},
    { 4,  2,  5, -1, -1, -1},
    { 4,  5,  2, -1, -1, -1},
    { 4,  1,  0,  4,  5,  1},
    { 3,  2,  0,  3,  5,  2},
    { 1,  3,  5, -1, -1, -1},
    { 4,  1,  2,  4,  3,  1},
    { 3,  0,  4, -1, -1, -1},
    { 2,  0,  1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1}
};

class MarchingTetrahedraHelper {
public:
    // Constructor: store resolution, grid vertices, tet indices, and points_range
    // grid_vertices: Nv Vec3
    // indices: num_tets * 4 ints (global vertex indices)
    MarchingTetrahedraHelper(
        const std::vector<Vec3>& grid_vertices,
        const std::vector<int>& indices,
        int resolution = 160,
        float points_min = 0.0f,
        float points_max = 1.0f
    )
    : resolution_(resolution),
      grid_vertices_(grid_vertices),
      indices_(indices),
      points_min_(points_min),
      points_max_(points_max)
    {
        assert(resolution_ > 0);
        assert(!grid_vertices_.empty());
        assert(indices_.size() % 4 == 0);
    }

    // Forward: level_ptr is length Nv (SDF). deformation_ptr is length Nv*3 or nullptr.
    // Returns Mesh with interpolated vertices and triangles.
    Mesh forward(const float* level_ptr, const float* deformation_ptr, const std::vector<mtd2::Vec3>& bbox = {}) const {
        const size_t Nv = grid_vertices_.size();
        const size_t num_tets = indices_.size() / 4;

        // Compute grid_scale = (points_max - points_min) / resolution
        const float grid_scale = (points_max_ - points_min_) / static_cast<float>(resolution_);

        // Step 0: build deformed grid (grid + grid_scale * tanh(deformation)) if provided
        std::vector<Vec3> grid; grid.reserve(Nv);
        if (deformation_ptr) {
            for (size_t i = 0; i < Nv; ++i) {
                const float dx = deformation_ptr[i*3 + 0];
                const float dy = deformation_ptr[i*3 + 1];
                const float dz = deformation_ptr[i*3 + 2];
                Vec3 dv(std::tanh(dx) * grid_scale, std::tanh(dy) * grid_scale, std::tanh(dz) * grid_scale);
                grid.push_back( grid_vertices_[i] + dv );
            }
        } else {
            grid = grid_vertices_; // copy
        }

        // --- FIRST PASS: collect edge keys from valid tets ---
        // valid tet = a tet that has some vertices inside and some outside w.r.t sdf>0
        std::vector<uint64_t> edge_keys;
        edge_keys.reserve(num_tets * 2); // heuristic; will grow if needed

        // Keep list of valid tet indices and their case (so we don't recompute later)
        std::vector<uint32_t> valid_tet_indices;
        valid_tet_indices.reserve(num_tets / 4);

        std::vector<uint8_t> valid_tet_case; // case_idx per valid tet
        valid_tet_case.reserve(num_tets / 4);

        // You can parallelize this loop easily; it's a read-only pass on level_ptr and indices_.
        for (size_t t = 0; t < num_tets; ++t) {
            // load local indices and sdf values
            int iv[4];
            float vf[4];
            for (int j = 0; j < 4; ++j) {
                iv[j] = indices_[t*4 + j];
                // sanity: clamp invalid indices (shouldn't happen)
                if (iv[j] < 0 || static_cast<size_t>(iv[j]) >= Nv) {
                    vf[j] = 0.0f;
                } else {
                    vf[j] = level_ptr[ static_cast<size_t>(iv[j]) ];
                }
            }

            // occupancy bits: sdf > 0 (match Python)
            int case_idx = 0;
            if (vf[0] > 0.0f) case_idx |= 1;
            if (vf[1] > 0.0f) case_idx |= 2;
            if (vf[2] > 0.0f) case_idx |= 4;
            if (vf[3] > 0.0f) case_idx |= 8;

            // count inside vertices:
            int inside_count = ((case_idx & 1) != 0) + ((case_idx & 2) != 0) + ((case_idx & 4) != 0) + ((case_idx & 8) != 0);
            if (inside_count == 0 || inside_count == 4) {
                // fully outside or fully inside -> no surface intersects this tet
                continue;
            }

            // valid tet: store index and case
            valid_tet_indices.push_back(static_cast<uint32_t>(t));
            valid_tet_case.push_back(static_cast<uint8_t>(case_idx));

            // collect its 6 edges (pack as uint64_t)
            // we reserve edges in pack-order with min/max enforced by pack function
            for (int e = 0; e < 6; ++e) {
                uint32_t a_local = EDGE_TO_LOCAL_VERTS[e][0];
                uint32_t b_local = EDGE_TO_LOCAL_VERTS[e][1];
                uint32_t a_global = static_cast<uint32_t>(iv[a_local]);
                uint32_t b_global = static_cast<uint32_t>(iv[b_local]);
                edge_keys.push_back( pack_edge_key(a_global, b_global) );
            }
        } // end first pass

        if (edge_keys.empty()) {
            // nothing intersected, return empty mesh
            LOGW("[MarchingTets] Nothing intersected, returning empty mesh!");
            return Mesh();
        }

        // --- DEDUP: sort + unique ---
        std::sort(edge_keys.begin(), edge_keys.end());
        auto it_end = std::unique(edge_keys.begin(), edge_keys.end());
        edge_keys.erase(it_end, edge_keys.end());
        const size_t unique_edges = edge_keys.size();

        // Build a map: edge_key -> index in edge_keys vector (0..unique_edges-1)
        // We build this map once; it's smaller and faster than doing binary_search repeatedly during triangle assembly.
        std::unordered_map<uint64_t, uint32_t> edgekey_to_idx;
        edgekey_to_idx.reserve(unique_edges * 1.3f);
        for (uint32_t i = 0; i < static_cast<uint32_t>(unique_edges); ++i) {
            edgekey_to_idx.emplace(edge_keys[i], i);
        }

        // --- SECOND PASS: interpolate only crossing edges and assign mesh vertex IDs ---
        // edge_vertex_id[i] = mesh vid for edge_keys[i], or UINT32_MAX if edge doesn't cross (both sides same sign)
        std::vector<uint32_t> edge_vertex_id(unique_edges, std::numeric_limits<uint32_t>::max());
        Mesh mesh;
        mesh.v_pos.reserve(unique_edges * 1); // rough heuristic (many edges will not cross)

        // tensor scaling
        mtd2::Vec3 tgt_min(0.0f, 0.0f, 0.0f);
        mtd2::Vec3 tgt_max(1.0f, 1.0f, 1.0f);
        if (bbox.size() >= 2) {
            tgt_min = bbox[0];
            tgt_max = bbox[1];
        }
        float points_min = 0.0f;
        float points_max = 1.0f;
        float inp_scale = points_max - points_min;
        // input scale is (0,1) so neglected here
        mtd2::Vec3 v_pos_scaling((tgt_max.x - tgt_min.x)*(1.0f/inp_scale),
                                 (tgt_max.y - tgt_min.y)*(1.0f/inp_scale),
                                 (tgt_max.y - tgt_min.z)*(1.0f/inp_scale)
                                 );

        // Iterate unique edges; can parallelize: each edge's interpolation is independent.
        // If parallelizing, careful to atomically append vertices or precompute counts and allocate.
        for (uint32_t ei = 0; ei < static_cast<uint32_t>(unique_edges); ++ei) {
            uint32_t a_global, b_global;
            unpack_edge_key(edge_keys[ei], a_global, b_global);

            // Safeguard indices
            if (a_global >= Nv || b_global >= Nv) continue;

            float valA = level_ptr[a_global];
            float valB = level_ptr[b_global];
            // Edge crosses the surface if exactly one endpoint has sdf > 0 (matching python's mask_edges)
            const bool insideA = (valA > 0.0f);
            const bool insideB = (valB > 0.0f);
            if (insideA == insideB) {
                // either both inside or both outside => this edge doesn't cross the 0-level
                continue;
            }

            // safe interpolation
            float denom = valB - valA;
            float t;
            if (std::fabs(denom) < 1e-12f) {
                t = 0.5f;
            } else {
                t = (-valA) / denom; // p = A + t*(B-A)
            }
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;

            const Vec3& P1 = grid[a_global];
            const Vec3& P2 = grid[b_global];
            Vec3 P = P1 + (P2 - P1) * t;

            // scale position tensor
            P.x = (P.x - points_min) * v_pos_scaling.x + tgt_min.x;
            P.y = (P.y - points_min) * v_pos_scaling.y + tgt_min.y;
            P.z = (P.z - points_min) * v_pos_scaling.z + tgt_min.z;

            // append vertex
            mesh.v_pos.push_back(P.x);
            mesh.v_pos.push_back(P.y);
            mesh.v_pos.push_back(P.z);
            uint32_t new_vid = static_cast<uint32_t>(mesh.v_pos.size()/3 - 1);
            edge_vertex_id[ei] = new_vid;
        }

        // --- THIRD PASS: assemble triangles using valid_tet_indices and the TRI_TABLE ---
        // Reserve triangle space heuristically
        mesh.t_pos_idx.reserve(valid_tet_indices.size() * 2 * 3);

        for (size_t vi = 0; vi < valid_tet_indices.size(); ++vi) {
            uint32_t t = valid_tet_indices[vi];
            uint8_t case_idx = valid_tet_case[vi];

            const int* trirow = TRI_TABLE[case_idx];

            // For each triangle in row (up to 2 triangles, 3 entries each)
            for (int tri_e = 0; tri_e < 6; tri_e += 3) {
                if (trirow[tri_e] == -1) break;

                uint32_t tri_vids[3];
                bool tri_valid = true;

                for (int k = 0; k < 3; ++k) {
                    int edge_local = trirow[tri_e + k];
                    uint32_t a_local = EDGE_TO_LOCAL_VERTS[edge_local][0];
                    uint32_t b_local = EDGE_TO_LOCAL_VERTS[edge_local][1];

                    // global vertex indices for the tet
                    uint32_t a_global = static_cast<uint32_t>( indices_[t*4 + a_local] );
                    uint32_t b_global = static_cast<uint32_t>( indices_[t*4 + b_local] );

                    uint64_t ekey = pack_edge_key(a_global, b_global);

                    auto it = edgekey_to_idx.find(ekey);
                    if (it == edgekey_to_idx.end()) {
                        tri_valid = false;
                        break;
                    }
                    uint32_t edge_idx = it->second;
                    uint32_t vid = edge_vertex_id[edge_idx];
                    if (vid == std::numeric_limits<uint32_t>::max()) {
                        // edge did not cross surface (shouldn't happen if tri table consistent), skip triangle
                        tri_valid = false;
                        break;
                    }
                    tri_vids[k] = vid;
                }

                if (tri_valid) {
                    // Append triangle indices (winding as per table)
                    mesh.t_pos_idx.push_back(tri_vids[0]);
                    mesh.t_pos_idx.push_back(tri_vids[1]);
                    mesh.t_pos_idx.push_back(tri_vids[2]);
                }
            }
        }

        // Done
        return mesh;
    }

private:
    int resolution_;
    std::vector<Vec3> grid_vertices_;
    std::vector<int> indices_;
    float points_min_, points_max_;
};

} // namespace mtd2



#endif //SNPECHAININGDEMO_MARCHINGTETRAHEDRAHELPER_H
