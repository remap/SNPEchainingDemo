//
// Created by Chiheb Boussema on 11/12/25.
//

#ifndef SNPECHAININGDEMO_MESHCPP_H
#define SNPECHAININGDEMO_MESHCPP_H

#pragma once
#include <vector>
#include <cstdint>
#include <optional>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>
#include <iostream>
#include "UnwrapperClass.h" // your provided header that defines Unwrapper and MeshUVResult
#include <optional>
#include <functional> // Required for std::reference_wrapper

// Simple 3D vector helper
struct Vec3 {
    float x, y, z;
    Vec3(): x(0.f), y(0.f), z(0.f) {}
    Vec3(float _x, float _y, float _z): x(_x), y(_y), z(_z) {}
    Vec3 operator+(const Vec3 &o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
    Vec3 operator-(const Vec3 &o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    Vec3 operator*(float s) const { return Vec3(x*s, y*s, z*s); }
};

class MeshCPP {
public:
    // Constructor: supply flattened v_pos (Nv*3) and flattened t_pos_idx (Nf*3)
    MeshCPP(const std::vector<float>& v_pos_flat, const std::vector<int>& t_pos_idx_flat);

    MeshCPP(const std::vector<float>& v_pos_flat,
            const std::vector<int>& t_pos_idx_flat,
            const std::vector<float>& v_nrm,
            const std::vector<float>& v_tng,
            const std::vector<float>& v_tex)
            : v_pos_(v_pos_flat),
            t_pos_idx_(t_pos_idx_flat),
            v_nrm_(v_nrm),
            v_tng_(v_tng),
            v_tex_(v_tex)
            {
                has_v_nrm_ = true;
                has_v_tex_ = true;
                has_v_tng_ = true;
            }

    // Accessors
    const std::vector<float>& v_pos() const { return v_pos_; }          // flattened Nx3
    const std::vector<int>& t_pos_idx() const { return t_pos_idx_; }    // flattened Mx3
    const std::vector<float>& v_nrm();  // compute lazily
    const std::vector<float>& v_tng();  // compute lazily
    const std::vector<float>& v_tex();  // after unwrap_uv, per-vertex uv flattened Nx2
    const std::vector<int>& edges();    // flattened Ne x 2

    void set_v_pos(std::vector<float> vpos) { v_pos_ = vpos; }
    void set_v_norm(std::vector<float> vnrm) { v_nrm_ = vnrm; has_v_nrm_ = true; }
    void set_v_tng(std::vector<float> vtng) { v_tng_ = vtng; has_v_tng_ = true; }
    void set_v_tex(std::vector<float> vtex) { v_tex_ = vtex; has_v_tex_ = true; }
    void set_t_pos_idx(std::vector<int> tposidx) { t_pos_idx_ = tposidx; }

    std::optional<std::reference_wrapper<const std::vector<float>>> get_v_nrm() const {
//        static const std::vector<float> empty;
        if (!has_v_nrm_) {
//            std::cerr << "Warning: Normals not computed." << std::endl;
            std::clog << "Warning: v_nrm_ requested but not initialized." << std::endl;
//            return empty;
            return std::nullopt;
        }
        return std::cref(v_nrm_);
    }
    std::optional<std::reference_wrapper<const std::vector<float>>> get_v_tng() const {
        if (!has_v_tng_) {
            std::clog << "Warning: v_tng_ requested but not initialized." << std::endl;
            return std::nullopt;
        }
        return std::cref(v_tng_);
    }
    std::optional<std::reference_wrapper<const std::vector<float>>> get_v_tex() const {
        if (!has_v_tex_) {
            std::clog << "Warning: v_tex_ requested but not initialized." << std::endl;
            return std::nullopt;
        }
        return std::cref(v_tex_);
    }
//    std::optional<std::reference_wrapper<const std::vector<float>>> get_v_pos() const {
//        if (!has_v_tng_) {
//            std::clog << "Warning: v_tng_ requested but not initialized." << std::endl;
//            return std::nullopt;
//        }
//        return std::cref(v_tng_);
//    }
//    std::optional<std::reference_wrapper<const std::vector<float>>> get_t_pos_idx() const {
//        if (!has_v_tng_) {
//            std::clog << "Warning: v_tng_ requested but not initialized." << std::endl;
//            return std::nullopt;
//        }
//        return std::cref(v_tng_);
//    }

    // Forces recomputation (useful if you mutate positions externally)
    void invalidate_cached();

    // The core method: runs Unwrapper and updates the mesh (duplicates vertices at seams)
    // island_padding default 0.02 to match python
    void unwrap_uv(float island_padding = 0.02f);

private:
    // helpers that mirror python names
    void compute_vertex_normal_impl();
    void compute_vertex_tangent_impl();
    void compute_edges_impl();

    // internal data (flattened arrays)
    std::vector<float> v_pos_;       // Nx3
    std::vector<int>   t_pos_idx_;   // Nf*3

    // cached (empty if not computed)
    std::vector<float> v_nrm_;       // Nx3
    std::vector<float> v_tng_;       // Nx3
    std::vector<float> v_tex_;       // Nx2 (per-vertex, created by unwrap)
    std::vector<int>   edges_;       // Ne*2

    // flags
    bool has_v_nrm_ = false;
    bool has_v_tng_ = false;
    bool has_v_tex_ = false;
    bool has_edges_ = false;

    // small helpers
    static inline size_t num_vertices_from_flat(const std::vector<float>& flat) { return flat.size() / 3; }
    static inline size_t num_faces_from_flat(const std::vector<int>& flat) { return flat.size() / 3; }

    // For future: you may pass a pointer to a Unwrapper instance to avoid re-allocations.
};

#endif //SNPECHAININGDEMO_MESHCPP_H
