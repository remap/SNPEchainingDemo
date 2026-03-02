//
// Created by Chiheb Boussema on 11/12/25.
//
#include "MeshCPP.h"
#include <iostream>
#include <unordered_set>
#include <set>

#include <android/log.h>
#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "MESH_CPP"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)


// ------------------------ Constructor ------------------------
MeshCPP::MeshCPP(const std::vector<float>& v_pos_flat, const std::vector<int>& t_pos_idx_flat)
    : v_pos_(v_pos_flat), t_pos_idx_(t_pos_idx_flat)
{
    assert(v_pos_.size() % 3 == 0);
    assert(t_pos_idx_.size() % 3 == 0);
    invalidate_cached();
}

void MeshCPP::invalidate_cached() {
    v_nrm_.clear(); v_tng_.clear(); v_tex_.clear(); edges_.clear();
    has_v_nrm_ = has_v_tng_ = has_v_tex_ = has_edges_ = false;
}

// ------------------------ Normals ------------------------
const std::vector<float>& MeshCPP::v_nrm() {
    if (!has_v_nrm_) compute_vertex_normal_impl();
    return v_nrm_;
}

void MeshCPP::compute_vertex_normal_impl() {
    const size_t Nv = num_vertices_from_flat(v_pos_);
    const size_t Nf = num_faces_from_flat(t_pos_idx_);

    v_nrm_.assign(Nv * 3, 0.0f);

    // accumulate face normals
    for (size_t fi = 0; fi < Nf; ++fi) {
        int i0 = t_pos_idx_[fi*3 + 0];
        int i1 = t_pos_idx_[fi*3 + 1];
        int i2 = t_pos_idx_[fi*3 + 2];
        // bounds check minimally (avoid crash)
        if ((size_t)i0 >= Nv || (size_t)i1 >= Nv || (size_t)i2 >= Nv) continue;

        // read positions
        Vec3 v0(v_pos_[i0*3+0], v_pos_[i0*3+1], v_pos_[i0*3+2]);
        Vec3 v1(v_pos_[i1*3+0], v_pos_[i1*3+1], v_pos_[i1*3+2]);
        Vec3 v2(v_pos_[i2*3+0], v_pos_[i2*3+1], v_pos_[i2*3+2]);

        // face normal = cross(v1-v0, v2-v0)
        Vec3 fn;
        Vec3 a = v1 - v0;
        Vec3 b = v2 - v0;
        fn.x = a.y * b.z - a.z * b.y;
        fn.y = a.z * b.x - a.x * b.z;
        fn.z = a.x * b.y - a.y * b.x;

        // add to vertex normals
        v_nrm_[i0*3 + 0] += fn.x; v_nrm_[i0*3 + 1] += fn.y; v_nrm_[i0*3 + 2] += fn.z;
        v_nrm_[i1*3 + 0] += fn.x; v_nrm_[i1*3 + 1] += fn.y; v_nrm_[i1*3 + 2] += fn.z;
        v_nrm_[i2*3 + 0] += fn.x; v_nrm_[i2*3 + 1] += fn.y; v_nrm_[i2*3 + 2] += fn.z;
    }

    // normalize and replace zero normals with default (0,0,1)
    for (size_t vi = 0; vi < Nv; ++vi) {
        float nx = v_nrm_[vi*3 + 0];
        float ny = v_nrm_[vi*3 + 1];
        float nz = v_nrm_[vi*3 + 2];
        float norm2 = nx*nx + ny*ny + nz*nz;
        if (norm2 <= 1e-20f) {
            v_nrm_[vi*3 + 0] = 0.0f;
            v_nrm_[vi*3 + 1] = 0.0f;
            v_nrm_[vi*3 + 2] = 1.0f;
        } else {
            float inv = 1.0f / std::sqrt(norm2);
            v_nrm_[vi*3 + 0] = nx * inv;
            v_nrm_[vi*3 + 1] = ny * inv;
            v_nrm_[vi*3 + 2] = nz * inv;
        }
    }

    has_v_nrm_ = true;
}

// ------------------------ Tangents ------------------------
const std::vector<float>& MeshCPP::v_tng() {
    if (!has_v_tng_) compute_vertex_tangent_impl();
    return v_tng_;
}

void MeshCPP::compute_vertex_tangent_impl() {
    // This expects that v_tex_ is present (per-vertex uv) and v_nrm_ is present
    if (!has_v_tex_) {
        // The python behavior: v_tng is computed after unwrap_uv, which sets v_tex.
        // If v_tex missing, attempt to compute normals first and then bail (no tangent).
        if (!has_v_nrm_) compute_vertex_normal_impl();
        // produce zero tangents
        const size_t Nv = num_vertices_from_flat(v_pos_);
        v_tng_.assign(Nv*3, 0.0f);
        has_v_tng_ = true;
        return;
    }
    if (!has_v_nrm_) compute_vertex_normal_impl();

    const size_t Nv = num_vertices_from_flat(v_pos_);
    const size_t Nf = num_faces_from_flat(t_pos_idx_);

    v_tng_.assign(Nv*3, 0.0f);
    std::vector<int> tansum(Nv, 0);

    // For each triangle
    for (size_t fi = 0; fi < Nf; ++fi) {
        int i0 = t_pos_idx_[fi*3 + 0];
        int i1 = t_pos_idx_[fi*3 + 1];
        int i2 = t_pos_idx_[fi*3 + 2];
        if ((size_t)i0 >= Nv || (size_t)i1 >= Nv || (size_t)i2 >= Nv) continue;

        Vec3 p0(v_pos_[i0*3+0], v_pos_[i0*3+1], v_pos_[i0*3+2]);
        Vec3 p1(v_pos_[i1*3+0], v_pos_[i1*3+1], v_pos_[i1*3+2]);
        Vec3 p2(v_pos_[i2*3+0], v_pos_[i2*3+1], v_pos_[i2*3+2]);

        // tex coords (per vertex) in v_tex_ flattened Nx2
        float u0 = v_tex_[i0*2 + 0], v0 = v_tex_[i0*2 + 1];
        float u1 = v_tex_[i1*2 + 0], v1 = v_tex_[i1*2 + 1];
        float u2 = v_tex_[i2*2 + 0], v2 = v_tex_[i2*2 + 1];

        // duv and dpos
        float duv1x = u1 - u0, duv1y = v1 - v0;
        float duv2x = u2 - u0, duv2y = v2 - v0;

        Vec3 dpos1 = p1 - p0;
        Vec3 dpos2 = p2 - p0;

        // tng_nom = dpos1 * duv2.y - dpos2 * duv1.y  (vector scaled by scalar)
        Vec3 tng_nom = Vec3(
            dpos1.x * duv2y - dpos2.x * duv1y,
            dpos1.y * duv2y - dpos2.y * duv1y,
            dpos1.z * duv2y - dpos2.z * duv1y
        );

        float denom = duv1x * duv2y - duv1y * duv2x;
//        float denom_safe = (std::fabs(denom) < 1e-6f) ? (denom < 0 ? -1e-6f : 1e-6f) : denom; // this seems to make more sense but not the original python implementation
        float denom_safe = std::max(denom, 1e-6f); // this is the original python implementation
        Vec3 tang = tng_nom * (1.0f / denom_safe);

        // add to all three vertices
        v_tng_[i0*3 + 0] += tang.x; v_tng_[i0*3 + 1] += tang.y; v_tng_[i0*3 + 2] += tang.z; tansum[i0] += 1;
        v_tng_[i1*3 + 0] += tang.x; v_tng_[i1*3 + 1] += tang.y; v_tng_[i1*3 + 2] += tang.z; tansum[i1] += 1;
        v_tng_[i2*3 + 0] += tang.x; v_tng_[i2*3 + 1] += tang.y; v_tng_[i2*3 + 2] += tang.z; tansum[i2] += 1;
    }

    // average and orthogonalize with normals
    for (size_t vi = 0; vi < Nv; ++vi) {
        if (tansum[vi] > 0) {
            v_tng_[vi*3 + 0] /= (float)tansum[vi];
            v_tng_[vi*3 + 1] /= (float)tansum[vi];
            v_tng_[vi*3 + 2] /= (float)tansum[vi];
        }
        // normalize tangent
        float tx = v_tng_[vi*3 + 0], ty = v_tng_[vi*3 + 1], tz = v_tng_[vi*3 + 2];
        float tnorm = std::sqrt(tx*tx + ty*ty + tz*tz);
        if (tnorm < 1e-12f) {
            // fallback tangent
            tx = 1.0f; ty = 0.0f; tz = 0.0f;
            tnorm = 1.0f;
        }
        tx /= tnorm; ty /= tnorm; tz /= tnorm;

        // make tangent perpendicular to normal: t = normalize(t - dot(t,n) * n)
        float nx = v_nrm_[vi*3+0], ny = v_nrm_[vi*3+1], nz = v_nrm_[vi*3+2];
        float dottn = tx*nx + ty*ny + tz*nz;
        tx = tx - dottn * nx;
        ty = ty - dottn * ny;
        tz = tz - dottn * nz;
        float newnorm = std::sqrt(tx*tx + ty*ty + tz*tz);
        if (newnorm < 1e-12f) {
            // fallback
            tx = 1.0f; ty = 0.0f; tz = 0.0f;
        } else {
            tx /= newnorm; ty /= newnorm; tz /= newnorm;
        }
        v_tng_[vi*3 + 0] = tx; v_tng_[vi*3 + 1] = ty; v_tng_[vi*3 + 2] = tz;
    }

    has_v_tng_ = true;
}

// ------------------------ Edges ------------------------
const std::vector<int>& MeshCPP::edges() {
    if (!has_edges_) compute_edges_impl();
    return edges_;
}

void MeshCPP::compute_edges_impl() {
    // Build edges from faces, sort (min,max) per edge, unique
    const size_t Nf = num_faces_from_flat(t_pos_idx_);
    std::vector<std::pair<int,int>> edges_local;
    edges_local.reserve(Nf * 3);

    for (size_t fi = 0; fi < Nf; ++fi) {
        int a = t_pos_idx_[fi*3 + 0];
        int b = t_pos_idx_[fi*3 + 1];
        int c = t_pos_idx_[fi*3 + 2];
        auto add_edge = [&](int u, int v) {
            if (u > v) std::swap(u,v);
            edges_local.emplace_back(u,v);
        };
        add_edge(a,b); add_edge(b,c); add_edge(c,a);
    }

    std::sort(edges_local.begin(), edges_local.end());
    edges_local.erase(std::unique(edges_local.begin(), edges_local.end()), edges_local.end());

    edges_.clear();
    edges_.reserve(edges_local.size()*2);
    for (auto &e : edges_local) { edges_.push_back(e.first); edges_.push_back(e.second); }

    has_edges_ = true;
}

// ------------------------ Unwrap (core) ------------------------
void MeshCPP::unwrap_uv(float island_padding) {
    if (v_pos_.empty()) { LOGE("[Mesh_CPP] v_pos_ empty"); return; }
    if (t_pos_idx_.empty()) { LOGE("[Mesh_CPP] t_pos_idx_ empty"); return; }
    if (v_pos_.size() % 3 != 0) { LOGE("[Mesh_CPP] v_pos_ len not multiple of 3"); return; }
    if (t_pos_idx_.size() % 3 != 0) { LOGE("[Mesh_CPP] t_pos_idx_ len not multiple of 3"); return; }
    // verify indices
    size_t Nv = v_pos_.size() / 3;
    for (size_t i=0;i<t_pos_idx_.size();++i) {
        int id = t_pos_idx_[i];
        if (id < 0 || static_cast<size_t>(id) >= Nv) {
            LOGE("[Mesh_CPP] invalid index in t_pos_idx_[%zu] = %d  (Nv=%zu)", i, id, Nv);
            return;
        }
    }
    // Ensure normals present for Unwrapper
    if (!has_v_nrm_) compute_vertex_normal_impl();
    LOGI("[Mesh_CPP:] Here 1");
    // Call Unwrapper (your C++ class) which returns MeshUVResult (unique_uv + vtex_idx)
    Unwrapper uw;
    // Unwrapper::forward expects arguments:
    // (const std::vector<float>& vertex_positions, const std::vector<float>& vertex_normals,
    //  const std::vector<int>& triangle_idxs, float island_padding)
    LOGI("CALLING uw.forward");
    MeshUVResult result = uw.forward(v_pos_, v_nrm_, t_pos_idx_, island_padding);
    LOGI("DONE with uw.forward");
    if (result.unique_uv.empty() || result.vtex_idx.empty()) {
        LOGE("[Mesh_CPP] uw.forward returned empty result; aborting unwrap_uv");
        return;
    }
    LOGW("[MESH_CPP:] UV: %lu, INDICES: %lu", result.num_unique_uv(), result.num_faces());
    {
        std::string inds_log = "";
        for (size_t k = 0; k < 10; ++k) inds_log += std::to_string(result.vtex_idx[k]) + ", ";
        LOGW("[MESH_CPP:] indices: %s", inds_log.c_str());
//        std::minmax_element(result.vtex_idx.begin(), result.vtex_idx.end());
        LOGW("max indices: %s", std::to_string(std::max_element( result.vtex_idx.begin(), result.vtex_idx.end())[0]).c_str());
    }
    // result.unique_uv : flattened [u0,v0, u1,v1, ...]
    // result.vtex_idx : flattened Nf*3 indices into unique_uv

    const size_t Nf = num_faces_from_flat(t_pos_idx_);
    assert(result.vtex_idx.size() == Nf * 3);

    // Build uv_flat per-corner: uv_flat[i] = unique_uv[ vtex_idx[i] ]
    // Then flatten it to per-vertex after duplication
    // Step: create individual_vertices = v_pos_[ t_pos_idx_.view(-1) ]
    std::vector<float> new_vpos;
    new_vpos.reserve(Nf * 3 * 3); // tripled triangles -> 3 coords each

    std::vector<int> new_faces;
    new_faces.reserve(Nf * 3);

    std::vector<float> new_vtex; // Nx2 (same length as new_vpos /3)
    new_vtex.reserve(Nf * 3 * 2);

    for (size_t fi = 0; fi < Nf; ++fi) {
        for (int corner = 0; corner < 3; ++corner) {
            int vid = t_pos_idx_[fi*3 + corner];
//            int vid = result.vtex_idx[fi*3 + corner];
            // copy pos
            new_vpos.push_back(v_pos_[vid*3 + 0]);
            new_vpos.push_back(v_pos_[vid*3 + 1]);
            new_vpos.push_back(v_pos_[vid*3 + 2]);

            // add face index (sequential)
            int new_vid = static_cast<int>( (new_vpos.size() / 3) - 1 );
            new_faces.push_back(new_vid);

            // uv index from result
            int uv_idx = static_cast<int>(result.vtex_idx[fi*3 + corner]); // index into unique_uv
            // bounds-check
            size_t uniq_uv_count = result.unique_uv.size() / 2;
            if (uv_idx < 0 || static_cast<size_t>(uv_idx) >= uniq_uv_count) {
                // fallback 0,0
                std::cerr << "FALLBACK UV IDX " << uv_idx << " (should be 0-" << uniq_uv_count-1 << ")\n";
                LOGE("FALLBACK UV IDX %lu (should be 0-%lu)", uv_idx, uniq_uv_count-1);
                new_vtex.push_back(0.0f); new_vtex.push_back(0.0f);
            } else {
                float u = result.unique_uv[uv_idx*2 + 0];
                float v = result.unique_uv[uv_idx*2 + 1];
                assert(u >= .0f && u <= 1.0f);
                assert(u >= .0f && u <= 1.0f);
                assert(v >= .0f && v <= 1.0f);
                assert(v >= .0f && v <= 1.0f);
                new_vtex.push_back(u);
                new_vtex.push_back(v);
            }
        }
    }

    {
        LOGW("[MESH_CPP:] NEW_V_POS: %lu", new_vpos.size()/3);
        std::string new_faces_log = "";
        for (size_t k = 0; k < 10; ++k) new_faces_log += std::to_string(new_faces[k]) + ", ";
        LOGW("new t_pos_idx: %s", new_faces_log.c_str());
        LOGW("max: %s", std::to_string(std::max_element(new_faces.begin(), new_faces.end())[0]).c_str());
    }

    // Replace internal arrays with duplicated mesh
    v_pos_.swap(new_vpos);
    t_pos_idx_.swap(new_faces);
    v_tex_.swap(new_vtex);

    // mark caches invalid except normals/tangents recompute
    has_v_tex_ = true;
    // recompute normals & tangents & edges
    compute_vertex_normal_impl();
    has_v_tng_ = false; v_tng_.clear();
    compute_vertex_tangent_impl();
    compute_edges_impl();
}


const std::vector<float>& MeshCPP::v_tex() {
    if (this->v_tex_.empty()) {
        // call the unwrap computation that fills v_tex_
        // make sure MeshCPP::unwrap_uv() populates v_tex_
        this->unwrap_uv();
    }
    return this->v_tex_;
}