//
// Created by Chiheb Boussema on 10/12/25.
//

#include "UnwrapperClass.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <cassert>
#include <unordered_map>
#include <cstring>
#include <iostream>
#include <set>
// Bridge header (declares UVUnwrapper_assign_faces_uv_to_atlas_index_raw)
#include "uv_unwrapper_api_bridge.h"

#include <android/log.h>
#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "UNWRAPPER_CLASS"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)


static inline float clampf(float x, float a, float b) {
    return std::min(std::max(x, a), b);
}

static inline size_t div_round_up(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static inline void vec_minmax3(const std::vector<float>& v, std::array<float,6>& out_bbox) {
    assert(v.size() % 3 == 0);
    size_t Nv = v.size()/3;
    float minx = std::numeric_limits<float>::infinity();
    float miny = std::numeric_limits<float>::infinity();
    float minz = std::numeric_limits<float>::infinity();
    float maxx = -std::numeric_limits<float>::infinity();
    float maxy = -std::numeric_limits<float>::infinity();
    float maxz = -std::numeric_limits<float>::infinity();
    for (size_t i=0;i<Nv;i++){
        float x = v[3*i+0];
        float y = v[3*i+1];
        float z = v[3*i+2];
        if (x<minx) minx=x;
        if (y<miny) miny=y;
        if (z<minz) minz=z;
        if (x>maxx) maxx=x;
        if (y>maxy) maxy=y;
        if (z>maxz) maxz=z;
    }
    out_bbox[0]=minx; out_bbox[1]=miny; out_bbox[2]=minz;
    out_bbox[3]=maxx; out_bbox[4]=maxy; out_bbox[5]=maxz;
}

static inline void normalize_inplace3(std::vector<float>& v) {
    assert(v.size()%3==0);
    size_t N = v.size()/3;
    for (size_t i=0;i<N;i++){
        float x=v[3*i+0], y=v[3*i+1], z=v[3*i+2];
        float n = std::sqrt(x*x + y*y + z*z) + 1e-12f;
        v[3*i+0] = x / n;
        v[3*i+1] = y / n;
        v[3*i+2] = z / n;
    }
}

// simple cross product of two 3-vectors
static inline std::array<float,3> cross3(const std::array<float,3>& a, const std::array<float,3>& b) {
    return { a[1]*b[2] - a[2]*b[1],
             a[2]*b[0] - a[0]*b[2],
             a[0]*b[1] - a[1]*b[0] };
}

// dot product
static inline float dot3(const std::array<float,3>& a, const std::array<float,3>& b){
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// helper: dot of 3-vectors
static inline float dot3f(const std::array<float,3>& a, const std::array<float,3>& b){
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static inline void normalize3(std::array<float,3>& v) {
    float n = std::sqrt(dot3f(v,v));
    if (n < 1e-12f) return;
    v[0] /= n; v[1] /= n; v[2] /= n;
}
static inline std::array<float,3> sub3(const std::array<float,3>& a, const std::array<float,3>& b){
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}
static inline std::array<float,3> mul3(const std::array<float,3>& a, float s){
    return {a[0]*s, a[1]*s, a[2]*s};
}

// compute top eigenvector of symmetric 3x3 matrix M using power iteration
static std::array<float,3> power_top_eigenvector(const std::array<std::array<double,3>,3>& M, int iters=64) {
    std::array<float,3> x = {1.0f, 0.0f, 0.0f};
    normalize3(x);
    for (int k=0;k<iters;k++) {
        // y = M * x
        std::array<double,3> y = {
            M[0][0]*x[0] + M[0][1]*x[1] + M[0][2]*x[2],
            M[1][0]*x[0] + M[1][1]*x[1] + M[1][2]*x[2],
            M[2][0]*x[0] + M[2][1]*x[1] + M[2][2]*x[2]
        };
        float ynorm = std::sqrt((float)(y[0]*y[0] + y[1]*y[1] + y[2]*y[2])) + 1e-20f;
        x[0] = (float)(y[0]/ynorm);
        x[1] = (float)(y[1]/ynorm);
        x[2] = (float)(y[2]/ynorm);
    }
    normalize3(x);
    return x;
}

void Unwrapper::align_mesh_with_main_axis(std::vector<float>& vertex_positions,
                                         std::vector<float>& vertex_normals) {
    assert(vertex_positions.size() % 3 == 0);
    size_t Nv = vertex_positions.size()/3;
    if (Nv == 0) return;

    // compute mean
    double mx=0,my=0,mz=0;
    for (size_t i=0;i<Nv;i++){
        mx += vertex_positions[3*i+0];
        my += vertex_positions[3*i+1];
        mz += vertex_positions[3*i+2];
    }
    mx /= (double)Nv; my /= (double)Nv; mz /= (double)Nv;

    // centered positions & covariance matrix (3x3)
    std::array<std::array<double,3>,3> C; // row-major
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) C[r][c]=0.0;
    for (size_t i=0;i<Nv;i++){
        double x = vertex_positions[3*i+0] - mx;
        double y = vertex_positions[3*i+1] - my;
        double z = vertex_positions[3*i+2] - mz;
        C[0][0] += x*x; C[0][1] += x*y; C[0][2] += x*z;
        C[1][0] += y*x; C[1][1] += y*y; C[1][2] += y*z;
        C[2][0] += z*x; C[2][1] += z*y; C[2][2] += z*z;
    }
    double invN = 1.0 / (double)Nv;
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) C[r][c] *= invN;

    // top eigenvector
    std::array<float,3> main_axis = power_top_eigenvector(C, 80);

    // deflate: compute M2 = C - lambda * (main_axis ⊗ main_axis)
    // approximate lambda = main_axis^T * C * main_axis
    std::array<double,3> Cx = {
        C[0][0]*main_axis[0] + C[0][1]*main_axis[1] + C[0][2]*main_axis[2],
        C[1][0]*main_axis[0] + C[1][1]*main_axis[1] + C[1][2]*main_axis[2],
        C[2][0]*main_axis[0] + C[2][1]*main_axis[1] + C[2][2]*main_axis[2]
    };
    double lambda = main_axis[0]*Cx[0] + main_axis[1]*Cx[1] + main_axis[2]*Cx[2];
    std::array<std::array<double,3>,3> M2;
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) M2[r][c] = C[r][c] - lambda * (double)main_axis[r] * (double)main_axis[c];

    // second eigenvector of M2
    std::array<float,3> second_axis = power_top_eigenvector(M2, 80);

    // orthogonalize (numerical safety)
    float proj = dot3f(second_axis, main_axis);
    std::array<float,3> tmp = sub3(second_axis, mul3(main_axis, proj));
    second_axis = tmp;
    normalize3(second_axis);

    // third axis via cross
    std::array<float,3> third_axis = cross3(main_axis, second_axis);
    normalize3(third_axis);

    // Now align axes to canonical axes like python: pick argmax abs and
    // permute them so that axes[idx] = vector. This resolves sign/ordering collisions.
    int main_axis_max_idx = 0; float mval = std::fabs(main_axis[0]);
    for (int i=1;i<3;i++) if (std::fabs(main_axis[i]) > mval) { mval = std::fabs(main_axis[i]); main_axis_max_idx = i; }
    int second_axis_max_idx = 0; float sval = std::fabs(second_axis[0]);
    for (int i=1;i<3;i++) if (std::fabs(second_axis[i]) > sval) { sval = std::fabs(second_axis[i]); second_axis_max_idx = i; }
    int third_axis_max_idx = 0; float tval = std::fabs(third_axis[0]);
    for (int i=1;i<3;i++) if (std::fabs(third_axis[i]) > tval) { tval = std::fabs(third_axis[i]); third_axis_max_idx = i; }

    // resolve duplicates (same logic as Python)
    std::set<int> all_axes = {0,1,2};
    int cur_index = 1;
    while ( (size_t) ( (int)std::set<int>({main_axis_max_idx, second_axis_max_idx, third_axis_max_idx}).size() ) != 3) {
        std::set<int> present = {main_axis_max_idx, second_axis_max_idx, third_axis_max_idx};
        int missing = -1;
        for (int v: all_axes) if (!present.count(v)) missing = v;
        if (cur_index == 1) third_axis_max_idx = missing;
        else if (cur_index == 2) second_axis_max_idx = missing;
        else break;
        cur_index++;
    }

    // place axes into array so axes[idx] = vector
    std::array<std::array<float,3>,3> axes;
    axes[0] = {0.0f,0.0f,0.0f};
    axes[1] = {0.0f,0.0f,0.0f};
    axes[2] = {0.0f,0.0f,0.0f};
    axes[main_axis_max_idx] = main_axis;
    axes[second_axis_max_idx] = second_axis;
    axes[third_axis_max_idx] = third_axis;

    // Build rotation matrix such that rot_rows[i] = axes[i] (so rot_mat rows are axes)
    // Apply rotation like Python: new_coord_i = dot(rot_row_i, pos)
    for (size_t i=0;i<Nv;i++){
        std::array<float,3> p = { vertex_positions[3*i+0] - (float)mx,
                                  vertex_positions[3*i+1] - (float)my,
                                  vertex_positions[3*i+2] - (float)mz };
        std::array<float,3> q = {
            dot3f(axes[0], p),
            dot3f(axes[1], p),
            dot3f(axes[2], p)
        };
        vertex_positions[3*i+0] = q[0];
        vertex_positions[3*i+1] = q[1];
        vertex_positions[3*i+2] = q[2];
    }

    // Rotate normals use same rows multiplication (but normals should NOT be shifted by mean)
    size_t Nn = vertex_normals.size()/3;
    for (size_t i=0;i<Nn;i++){
        std::array<float,3> p = { vertex_normals[3*i+0], vertex_normals[3*i+1], vertex_normals[3*i+2] };
        std::array<float,3> q = {
            dot3f(axes[0], p),
            dot3f(axes[1], p),
            dot3f(axes[2], p)
        };
        vertex_normals[3*i+0] = q[0];
        vertex_normals[3*i+1] = q[1];
        vertex_normals[3*i+2] = q[2];
    }
    normalize_inplace3(vertex_normals);
}


void align_mesh_with_main_axis_old(std::vector<float>& vertex_positions,
                                         std::vector<float>& vertex_normals) {
    // Quick deterministic heuristic: use covariance matrix diagonal (variances) to pick axes.
    // This is simpler than full PCA but aligns big axis to largest variance.
    assert(vertex_positions.size() % 3 == 0);
    size_t Nv = vertex_positions.size()/3;

    // Compute mean
    double mx=0, my=0, mz=0;
    for (size_t i=0;i<Nv;i++){
        mx += vertex_positions[3*i+0];
        my += vertex_positions[3*i+1];
        mz += vertex_positions[3*i+2];
    }
    mx /= (double)Nv; my /= (double)Nv; mz /= (double)Nv;

    // Compute variances (diagonal of covariance)
    double vx=0, vy=0, vz=0;
    for (size_t i=0;i<Nv;i++){
        double dx = vertex_positions[3*i+0] - mx;
        double dy = vertex_positions[3*i+1] - my;
        double dz = vertex_positions[3*i+2] - mz;
        vx += dx*dx; vy += dy*dy; vz += dz*dz;
    }
    vx /= (double)Nv; vy /= (double)Nv; vz /= (double)Nv;

    // sort axes by variance
    // axes indices 0=x,1=y,2=z
    std::array<int,3> idx = {0,1,2};
    std::array<double,3> var = {vx,vy,vz};
    // bubble sort small
    for (int a=0;a<3;a++){
        for (int b=a+1;b<3;b++){
            if (var[b] > var[a]) {
                std::swap(var[a], var[b]);
                std::swap(idx[a], idx[b]);
            }
        }
    }
    // Build rotation matrix whose columns are the selected axes (unit vectors)
    // For the heuristic we form axes aligned to canonical axes but permuted.
    // main_axis is axis with largest variance, secondary second, third derived by cross.
    std::array<float,3> main_axis = {0,0,0};
    std::array<float,3> second_axis = {0,0,0};
    std::array<float,3> third_axis = {0,0,0};
    main_axis[idx[0]] = 1.0f;
    second_axis[idx[1]] = 1.0f;
    // third is cross(main, second)
    third_axis = cross3(main_axis, second_axis);
    // if cross produced zero vector (rare if idx[0]==idx[1]) then try different orientation
    float third_norm = std::sqrt(dot3(third_axis, third_axis));
    if (third_norm < 1e-6f) {
        // fallback: choose remaining axis
        for (int i=0;i<3;i++) if (i!=idx[0] && i!=idx[1]) { third_axis = {0,0,0}; third_axis[i]=1.0f; break; }
    }

    // rotation matrix(rot) such that new_pos = rot * old_pos (3x3)
    // columns are axes in original coordinates: [main_axis, second_axis, third_axis]
    std::array<std::array<float,3>,3> rot;
    rot[0] = main_axis;
    rot[1] = second_axis;
    rot[2] = third_axis;

    // Apply rotation: new[n] = rot_T * old[n] (matching python einsum "ij,nj->ni")
    for (size_t i=0;i<Nv;i++){
        std::array<float,3> p = { vertex_positions[3*i+0], vertex_positions[3*i+1], vertex_positions[3*i+2] };
        std::array<float,3> q = {
//            rot[0][0]*p[0] + rot[1][0]*p[1] + rot[2][0]*p[2],
//            rot[0][1]*p[0] + rot[1][1]*p[1] + rot[2][1]*p[2],
//            rot[0][2]*p[0] + rot[1][2]*p[1] + rot[2][2]*p[2]
            rot[0][0]*p[0] + rot[0][1]*p[1] + rot[0][2]*p[2],
            rot[1][0]*p[0] + rot[1][1]*p[1] + rot[1][2]*p[2],
            rot[2][0]*p[0] + rot[2][1]*p[1] + rot[2][2]*p[2]
        };
        vertex_positions[3*i+0] = q[0];
        vertex_positions[3*i+1] = q[1];
        vertex_positions[3*i+2] = q[2];
    }

    // Normals rotate the same way
    size_t Nn = vertex_normals.size()/3;
    for (size_t i=0;i<Nn;i++){
        std::array<float,3> p = { vertex_normals[3*i+0], vertex_normals[3*i+1], vertex_normals[3*i+2] };
        std::array<float,3> q = {
//            rot[0][0]*p[0] + rot[1][0]*p[1] + rot[2][0]*p[2],
//            rot[0][1]*p[0] + rot[1][1]*p[1] + rot[2][1]*p[2],
//            rot[0][2]*p[0] + rot[1][2]*p[1] + rot[2][2]*p[2]
            rot[0][0]*p[0] + rot[0][1]*p[1] + rot[0][2]*p[2],
            rot[1][0]*p[0] + rot[1][1]*p[1] + rot[1][2]*p[2],
            rot[2][0]*p[0] + rot[2][1]*p[1] + rot[2][2]*p[2]
        };
        vertex_normals[3*i+0] = q[0];
        vertex_normals[3*i+1] = q[1];
        vertex_normals[3*i+2] = q[2];
    }
    // normalize normals
    normalize_inplace3(vertex_normals);
}

void Unwrapper::box_assign_vertex_to_cube_face(
    const std::vector<float>& vertex_positions,
    const std::vector<float>& vertex_normals,
    const std::vector<int>& triangle_idxs,
    const std::array<float,6>& bbox,
    std::vector<float>& out_face_uv_flat,
    std::vector<int64_t>& out_face_index)
{
    // Inputs validation
    assert(vertex_positions.size()%3==0);
    assert(vertex_normals.size()%3==0);
    assert(triangle_idxs.size()%3==0);
    size_t Nv = vertex_positions.size()/3;
    size_t Nf = triangle_idxs.size()/3;

    out_face_uv_flat.assign(Nf*3*2, 0.0f);
    out_face_index.assign(Nf, 0);

    // Precompute normalized coords v_pos_normalized (Nv x 3)
    std::vector<std::array<float,3>> vposn(Nv);
    {
        float minx = bbox[0], miny = bbox[1], minz = bbox[2];
        float maxx = bbox[3], maxy = bbox[4], maxz = bbox[5];
        float dx = maxx - minx; if (dx==0) dx=1e-6f;
        float dy = maxy - miny; if (dy==0) dy=1e-6f;
        float dz = maxz - minz; if (dz==0) dz=1e-6f;
        for (size_t i=0;i<Nv;i++){
            float x = vertex_positions[3*i+0];
            float y = vertex_positions[3*i+1];
            float z = vertex_positions[3*i+2];
            // normalized 0..1 then to -1..1
            float nx = 2.0f * ((x - minx) / dx) - 1.0f;
            float ny = 2.0f * ((y - miny) / dy) - 1.0f;
            float nz = 2.0f * ((z - minz) / dz) - 1.0f;
            vposn[i] = {nx, ny, nz};
        }
    }

    // For each triangle calc tri_stack (Nf x 3 x 3) and tri_stack_abs
    for (size_t f=0; f<Nf; ++f) {
        int i0 = triangle_idxs[3*f + 0];
        int i1 = triangle_idxs[3*f + 1];
        int i2 = triangle_idxs[3*f + 2];

        auto v0 = vposn[i0];
        auto v1 = vposn[i1];
        auto v2 = vposn[i2];

        // tri_stack_nrm: average normals per face
        std::array<float,3> vn0 = { vertex_normals[3*i0+0], vertex_normals[3*i0+1], vertex_normals[3*i0+2] };
        std::array<float,3> vn1 = { vertex_normals[3*i1+0], vertex_normals[3*i1+1], vertex_normals[3*i1+2] };
        std::array<float,3> vn2 = { vertex_normals[3*i2+0], vertex_normals[3*i2+1], vertex_normals[3*i2+2] };
        std::array<float,3> face_nrm = { vn0[0]+vn1[0]+vn2[0], vn0[1]+vn1[1]+vn2[1], vn0[2]+vn1[2]+vn2[2] };
        float norm = std::sqrt(face_nrm[0]*face_nrm[0] + face_nrm[1]*face_nrm[1] + face_nrm[2]*face_nrm[2]) + 1e-9f;
        face_nrm[0] /= norm; face_nrm[1] /= norm; face_nrm[2] /= norm;

        // Determine index among 6 axes by dot product with axis set
        // axis: 0:[1,0,0],1:[-1,0,0],2:[0,1,0],3:[0,-1,0],4:[0,0,1],5:[0,0,-1]
        std::array<std::array<float,3>,6> axis = {
            std::array<float,3>{1,0,0},
            std::array<float,3>{-1,0,0},
            std::array<float,3>{0,1,0},
            std::array<float,3>{0,-1,0},
            std::array<float,3>{0,0,1},
            std::array<float,3>{0,0,-1}
        };
        float best = -1e9f; int best_idx = 0;
        for (int a=0;a<6;++a){
            float val = face_nrm[0]*axis[a][0] + face_nrm[1]*axis[a][1] + face_nrm[2]*axis[a][2];
            if (val > best) { best = val; best_idx = a; }
        }
        out_face_index[f] = best_idx;

        // compute abs_x, abs_y, abs_z across the three verts: we want per-triangle vectors
        float abs_x = std::max({ std::fabs(v0[0]), std::fabs(v1[0]), std::fabs(v2[0]) });
        float abs_y = std::max({ std::fabs(v0[1]), std::fabs(v1[1]), std::fabs(v2[1]) });
        float abs_z = std::max({ std::fabs(v0[2]), std::fabs(v1[2]), std::fabs(v2[2]) });

        // max_axis (single float) = corresponding max of abs_x/abs_y/abs_z
        float max_axis = 1.0f;
        float uc0=0, uc1=0, uc2=0;
        float vc0=0, vc1=0, vc2=0;
        switch (best_idx) {
            case 0: // +x
            case 1: // -x
                max_axis = abs_x;
                uc0 = v0[1]; vc0 = -v0[2];
                uc1 = v1[1]; vc1 = -v1[2];
                uc2 = v2[1]; vc2 = -v2[2];
                break;
            case 2: // +y
            case 3: // -y
                max_axis = abs_y;
                uc0 = v0[0]; vc0 = -v0[2];
                uc1 = v1[0]; vc1 = -v1[2];
                uc2 = v2[0]; vc2 = -v2[2];
                break;
            case 4: // +z
                max_axis = abs_z;
                uc0 = v0[0]; vc0 = v0[1];
                uc1 = v1[0]; vc1 = v1[1];
                uc2 = v2[0]; vc2 = v2[1];
                break;
            case 5: // -z
                max_axis = abs_z;
                uc0 = v0[0]; vc0 = -v0[1];
                uc1 = v1[0]; vc1 = -v1[1];
                uc2 = v2[0]; vc2 = -v2[1];
                break;
        }
        // avoid divide by zero
        float max_dim_div = max_axis;
        if (max_dim_div == 0.0f) max_dim_div = 1.0f;
        max_dim_div = 1.0f; // the way max_dim_div in original python was computed is just nonsensical and only works because it ends up being 1

        // uc from [-1,1] to [0,1]: ((uc / max_dim_div + 1) * 0.5)
        float u0 = clampf((uc0 / max_dim_div + 1.0f) * 0.5f, 0.0f, 1.0f);
        float v0c = clampf((vc0 / max_dim_div + 1.0f) * 0.5f, 0.0f, 1.0f);
        float u1 = clampf((uc1 / max_dim_div + 1.0f) * 0.5f, 0.0f, 1.0f);
        float v1c = clampf((vc1 / max_dim_div + 1.0f) * 0.5f, 0.0f, 1.0f);
        float u2 = clampf((uc2 / max_dim_div + 1.0f) * 0.5f, 0.0f, 1.0f);
        float v2c = clampf((vc2 / max_dim_div + 1.0f) * 0.5f, 0.0f, 1.0f);

        // write into out_face_uv_flat: for triangle f, vertex 0..2, (u,v)
        out_face_uv_flat[(f*3 + 0)*2 + 0] = u0;
        out_face_uv_flat[(f*3 + 0)*2 + 1] = v0c;
        out_face_uv_flat[(f*3 + 1)*2 + 0] = u1;
        out_face_uv_flat[(f*3 + 1)*2 + 1] = v1c;
        out_face_uv_flat[(f*3 + 2)*2 + 0] = u2;
        out_face_uv_flat[(f*3 + 2)*2 + 1] = v2c;
    }
}

// rotate_uv_slices_consistent_space - simplified: compute triangle tangents and rotate each atlas slice
void Unwrapper::rotate_uv_slices_consistent_space(
    const std::vector<float>& vertex_positions,
    const std::vector<float>& vertex_normals,
    const std::vector<int>& triangle_idxs,
    std::vector<float>& face_uv_flat,
    const std::vector<int64_t>& face_index)
{
    // For each atlas index (0..5 and possible overlaps), compute mean actual tangent and expected tangent and rotate the UVs.
    // We'll compute per-triangle tangents similar to python: derive per-triangle tangent from dpos/duv
    size_t Nf = triangle_idxs.size()/3;
    // compute per-triangle tangents (3 components) and per-triangle expected tangents (3 components)
    std::vector<std::array<float,3>> tangents(Nf);
    std::vector<std::array<float,3>> expected_tangents(Nf);

    for (size_t f=0; f<Nf; ++f) {
        int i0 = triangle_idxs[3*f + 0];
        int i1 = triangle_idxs[3*f + 1];
        int i2 = triangle_idxs[3*f + 2];

        // positions
        std::array<float,3> p0 = { vertex_positions[3*i0+0], vertex_positions[3*i0+1], vertex_positions[3*i0+2] };
        std::array<float,3> p1 = { vertex_positions[3*i1+0], vertex_positions[3*i1+1], vertex_positions[3*i1+2] };
        std::array<float,3> p2 = { vertex_positions[3*i2+0], vertex_positions[3*i2+1], vertex_positions[3*i2+2] };

        // uv coordinates (for this triangle)
        float u0 = face_uv_flat[(f*3 + 0)*2 + 0];
        float v0 = face_uv_flat[(f*3 + 0)*2 + 1];
        float u1 = face_uv_flat[(f*3 + 1)*2 + 0];
        float v1 = face_uv_flat[(f*3 + 1)*2 + 1];
        float u2 = face_uv_flat[(f*3 + 2)*2 + 0];
        float v2 = face_uv_flat[(f*3 + 2)*2 + 1];

        // duv and dpos
        std::array<float,2> duv1 = { u1 - u0, v1 - v0 };
        std::array<float,2> duv2 = { u2 - u0, v2 - v0 };
        std::array<float,3> dpos1 = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
        std::array<float,3> dpos2 = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };

        // tng_nom = dpos1 * duv2.y - dpos2 * duv1.y  (component-wise)
        std::array<float,3> tng_nom = {
            dpos1[0]*duv2[1] - dpos2[0]*duv1[1],
            dpos1[1]*duv2[1] - dpos2[1]*duv1[1],
            dpos1[2]*duv2[1] - dpos2[2]*duv1[1]
        };
        float denom = duv1[0]*duv2[1] - duv1[1]*duv2[0];
        float denom_safe = (std::fabs(denom) < 1e-6f) ? (denom < 0 ? -1e-6f : 1e-6f) : denom;
        std::array<float,3> tang = { tng_nom[0]/denom_safe, tng_nom[1]/denom_safe, tng_nom[2]/denom_safe };
        // store tang normalized
        float tn = std::sqrt(tang[0]*tang[0] + tang[1]*tang[1] + tang[2]*tang[2]) + 1e-9f;
        tangents[f] = { tang[0]/tn, tang[1]/tn, tang[2]/tn };

        // expected tangent: use the Python pos_stack trick: [-y, x, 0] then projected
        std::array<float,3> pos_stack0 = {-p0[1], p0[0], 0.0f};
        std::array<float,3> vn0 = { vertex_normals[3*i0+0], vertex_normals[3*i0+1], vertex_normals[3*i0+2] };
        std::array<float,3> cross1 = cross3(pos_stack0, vn0);
        std::array<float,3> expected = cross3(vn0, cross1);
        float en = std::sqrt(expected[0]*expected[0] + expected[1]*expected[1] + expected[2]*expected[2]) + 1e-9f;
        expected_tangents[f] = { expected[0]/en, expected[1]/en, expected[2]/en };
    }

    // For each atlas slice index (0..5) compute mean actual and expected tangents and find angle to rotate uvs
    int max_index = 6; // we'll only rotate first 6 by default (python did index %6)
    for (int slice = 0; slice < 6; ++slice) {
        // collect triangles belonging to this slice (face_index % 6 == slice)
        std::vector<int> tri_idx;
        for (size_t f=0; f<Nf; ++f) {
            if (static_cast<int>(face_index[f] % 6) == slice) tri_idx.push_back((int)f);
        }
        if (tri_idx.empty()) continue;

        // compute mean actual and expected tangents (2D by taking XY only)
        std::array<float,2> actual_mean = {0.0f, 0.0f};
        std::array<float,2> expected_mean = {0.0f, 0.0f};
        for (int f : tri_idx) {
            actual_mean[0] += tangents[f][0];
            actual_mean[1] += tangents[f][1];
            expected_mean[0] += expected_tangents[f][0];
            expected_mean[1] += expected_tangents[f][1];
        }
        actual_mean[0] /= (float)tri_idx.size(); actual_mean[1] /= (float)tri_idx.size();
        expected_mean[0] /= (float)tri_idx.size(); expected_mean[1] /= (float)tri_idx.size();

        // compute 2D rotation angle between actual_mean and expected_mean using atan2(cross, dot)
        float dot = actual_mean[0]*expected_mean[0] + actual_mean[1]*expected_mean[1];
        float cross2 = actual_mean[0]*expected_mean[1] - actual_mean[1]*expected_mean[0];
        float angle = std::atan2(cross2, dot);
        float c = std::cos(angle), s = std::sin(angle);

        // rotate UVs of triangles in tri_idx
        // center uv to -1..1, rotate, rescale to 0..1
        float uv_min_u = std::numeric_limits<float>::infinity(), uv_max_u = -uv_min_u;
        float uv_min_v = uv_min_u, uv_max_v = -uv_min_u;
        // gather all u/v values for those triangles first
        for (int f : tri_idx) {
            for (int vi=0; vi<3; ++vi) {
                float u = face_uv_flat[(f*3 + vi)*2 + 0];
                float v = face_uv_flat[(f*3 + vi)*2 + 1];
                if (u < uv_min_u) uv_min_u = u;
                if (u > uv_max_u) uv_max_u = u;
                if (v < uv_min_v) uv_min_v = v;
                if (v > uv_max_v) uv_max_v = v;
            }
        }
        // rotate and renormalize
        for (int f : tri_idx) {
            for (int vi=0; vi<3; ++vi) {
                float u = face_uv_flat[(f*3 + vi)*2 + 0];
                float v = face_uv_flat[(f*3 + vi)*2 + 1];
                // center to -1..1
                float uc = (u - uv_min_u) / (uv_max_u - uv_min_u + 1e-9f) * 2.0f - 1.0f;
                float vc = (v - uv_min_v) / (uv_max_v - uv_min_v + 1e-9f) * 2.0f - 1.0f;
                // rotate
                float ru = c*uc - s*vc;
                float rv = s*uc + c*vc;
                // rescale to 0..1
                // We'll normalize by the current rotated mins/max to fill the same range:
                // For simplicity and stability, map back directly using tanh-like limit
                float newu = (ru + 1.0f) * 0.5f;
                float newv = (rv + 1.0f) * 0.5f;
                // clamp
                newu = clampf(newu, 0.0f, 1.0f);
                newv = clampf(newv, 0.0f, 1.0f);
                face_uv_flat[(f*3 + vi)*2 + 0] = newu;
                face_uv_flat[(f*3 + vi)*2 + 1] = newv;
            }
        }
    }
}

void Unwrapper::find_slice_offset_and_scale(
    const std::vector<int64_t>& index,
    std::vector<float>& offset_x,
    std::vector<float>& offset_y,
    std::vector<float>& div_x,
    std::vector<float>& div_y)
{
    // index length = Nf
    size_t Nf = index.size();
    offset_x.assign(Nf, 0.0f);
    offset_y.assign(Nf, 0.0f);
    div_x.assign(Nf, 3.0f); // default 3//2 in python, but set to float 3 for safety
    div_y.assign(Nf, 3.0f);

    const float off = 1.0f/3.0f;
    const float dupl_off = 1.0f/6.0f;
    const int offset_x_vals[6] = {0,1,2,0,1,2};
    const int offset_y_vals[6] = {0,0,0,1,1,1};

    // compute max index
    int max_index = 0;
    for (auto v : index) if (v > max_index) max_index = (int)v;

    for (int i=0;i<=max_index;++i) {
        // find mask
        bool any=false;
        for (size_t f=0; f<Nf; ++f) {
            if (index[f] == i) { any = true; break; }
        }
        if (!any) continue;

        int base = i / 6;
        int ix = offset_x_vals[i % 6];
        int iy = offset_y_vals[i % 6];
        auto x_offset_calc = [&](int x, int ii)->float {
            int offset_calc = ii / 6;
            if (offset_calc == 0) return off * x;
            else return dupl_off * x + std::min(offset_calc - 1, 1) * 0.5f;
        };
        auto y_offset_calc = [&](int x, int ii)->float {
            int offset_calc = ii / 6;
            if (offset_calc == 0) return off * x;
            else return dupl_off * x + off * 2.0f;
        };

        for (size_t f=0; f<Nf; ++f) {
            if (index[f] == i) {
                offset_x[f] = x_offset_calc(ix, i);
                offset_y[f] = y_offset_calc(iy, i);
            }
        }
    }

    // div_x initial: 6//2 => 3. For index >=6 => 6, index >=12 => 2; div_y same but index>=12 => 3
    for (size_t f=0; f<Nf; ++f) {
        int64_t idx = index[f];
        if (idx >= 12) { div_x[f] = 2.0f; div_y[f] = 3.0f; }
        else if (idx >= 6) { div_x[f] = 6.0f; div_y[f] = 6.0f; }
        else { div_x[f] = 3.0f; div_y[f] = 3.0f; }
    }
}

void Unwrapper::handle_slice_uvs(
    std::vector<float>& face_uv_flat,
    const std::vector<int64_t>& index,
    float island_padding,
    int max_index)
{
    // face_uv_flat length = Nf*3*2
    size_t Nf = index.size();
    // get uc, vc arrays
    std::vector<float> uc(Nf*3), vc(Nf*3);
    for (size_t i=0;i<Nf*3;i++){
        uc[i] = face_uv_flat[i*2 + 0];
        vc[i] = face_uv_flat[i*2 + 1];
    }

    // for index in [6, max_index), scale each slice to fill patch (clamped at 0.5 min scale)
    for (int idx = 6; idx < max_index; ++idx) {
        // build list of vertex positions belonging to triangles whose face index == idx
        std::vector<int> vidx; vidx.reserve(128);
        for (size_t f=0; f<Nf; ++f) if (index[f] == idx) {
            size_t base = f*3;
            vidx.push_back((int)(base+0));
            vidx.push_back((int)(base+1));
            vidx.push_back((int)(base+2));
        }
        if (vidx.empty()) continue;

        // for each triangle vertex list, normalize to [0,1] per-triangle group
        float min_u = std::numeric_limits<float>::infinity(), max_u = -min_u;
        float min_v = min_u, max_v = -min_u;
        for (int vi : vidx) {
            if (uc[vi] < min_u) min_u = uc[vi];
            if (uc[vi] > max_u) max_u = uc[vi];
            if (vc[vi] < min_v) min_v = vc[vi];
            if (vc[vi] > max_v) max_v = vc[vi];
        }
//        float range_u = max_u - min_u; if (range_u < 1e-6f) range_u = 1e-6f;
//        float range_v = max_v - min_v; if (range_v < 1e-6f) range_v = 1e-6f;
        // Clip the denominator to 0.5 to match PyTorch's .clip(0.5)
        float range_u = std::max(max_u - min_u, 0.5f);
        float range_v = std::max(max_v - min_v, 0.5f);

        for (int vi : vidx) {
            float nu = (uc[vi] - min_u) / range_u;
//            if (nu < 0.5f) nu = nu; // clip below
//            if (nu > 1.0f) nu = 1.0f;
            float nv = (vc[vi] - min_v) / range_v;
//            if (nv < 0.5f) nv = nv;
//            if (nv > 1.0f) nv = 1.0f;
//            // safety clip to 0.5..1.0 to match clip(0.5) semantics approximation
//            if (nu < 0.5f) nu = 0.5f;
//            if (nv < 0.5f) nv = 0.5f;
            uc[vi] = nu;
            vc[vi] = nv;
        }
    }

    // pad UVs to island_padding
    for (size_t i=0;i<Nf*3;i++){
        uc[i] = clampf( uc[i] * (1.0f - 2.0f*island_padding) + island_padding, 0.0f, 1.0f );
        vc[i] = clampf( vc[i] * (1.0f - 2.0f*island_padding) + island_padding, 0.0f, 1.0f );
    }

    // write back
    for (size_t i=0;i<Nf*3;i++){
        face_uv_flat[i*2 + 0] = uc[i];
        face_uv_flat[i*2 + 1] = vc[i];
    }
}

void Unwrapper::handle_remaining_uvs(
    std::vector<float>& face_uv_flat,
    const std::vector<int64_t>& index,
    float island_padding)
{
    size_t Nf = index.size();
    // find remaining_filter (index >= 12)
    std::vector<int> remaining_faces;
    for (size_t f=0; f<Nf; ++f) if (index[f] >= 12) remaining_faces.push_back((int)f);
    size_t squares_left = remaining_faces.size();
    if (squares_left == 0) return;

    // build uc/vc arrays for those triangles (per triangle)
    // uc_mat: squares_left x 3
    std::vector<std::array<float,3>> ucarr(squares_left), vcarr(squares_left);
    for (size_t k=0;k<squares_left;k++){
        int f = remaining_faces[k];
        for (int vi=0;vi<3;vi++){
            ucarr[k][vi] = face_uv_flat[(f*3 + vi)*2 + 0];
            vcarr[k][vi] = face_uv_flat[(f*3 + vi)*2 + 1];
        }
    }
    // Now normalize each triangle's coords
    // compute grid dims as in python: ratio = 0.5*(1/3)
    float ratio = 0.5f * (1.0f/3.0f);
    float mult = std::sqrt((float)squares_left / ratio);
    int num_square_width = (int)std::ceil(0.5f * mult);
    if (num_square_width < 1) num_square_width = 1;
    int num_square_height = (int)std::ceil((float)squares_left / num_square_width);
    float width = 1.0f / (float)num_square_width;
    float height = 1.0f / (float)num_square_height;

    // normalize per-triangle to 0..1
    for (size_t k=0;k<squares_left;k++){
        float min_u = std::min({ ucarr[k][0], ucarr[k][1], ucarr[k][2] });
        float max_u = std::max({ ucarr[k][0], ucarr[k][1], ucarr[k][2] });
        float min_v = std::min({ vcarr[k][0], vcarr[k][1], vcarr[k][2] });
        float max_v = std::max({ vcarr[k][0], vcarr[k][1], vcarr[k][2] });
        float ru = (max_u - min_u);
        float rv = (max_v - min_v);
        if (ru < 1e-6f) ru = 1e-6f;
        if (rv < 1e-6f) rv = 1e-6f;
        for (int vi=0;vi<3;vi++){
            ucarr[k][vi] = (ucarr[k][vi] - min_u) / ru;
            vcarr[k][vi] = (vcarr[k][vi] - min_v) / rv;
            // add small padding scaling
            ucarr[k][vi] = clampf( ucarr[k][vi] * (1 - island_padding * num_square_width * 0.25f) + island_padding * num_square_width * 0.125f, 0.0f, 1.0f);
            vcarr[k][vi] = clampf( vcarr[k][vi] * (1 - island_padding * num_square_height * 0.25f) + island_padding * num_square_height * 0.125f, 0.0f, 1.0f);
            // scale to cell width/height
            ucarr[k][vi] = ucarr[k][vi] * width;
            vcarr[k][vi] = vcarr[k][vi] * height;
        }
    }

    // Move each triangle to its own spot in the grid and write back to face_uv_flat
    for (size_t k=0;k<squares_left;k++){
        int f = remaining_faces[k];
        int idx = (int)k;
        int x_idx = idx % num_square_width;
        int y_idx = idx / num_square_width;
        for (int vi=0; vi<3; ++vi) {
            float u = ucarr[k][vi] + x_idx * width;
            float v = vcarr[k][vi] + y_idx * height;
            u = clampf( u * (1 - 2.0f * island_padding * 0.5f) + island_padding * 0.5f, 0.0f, 1.0f );
            v = clampf( v * (1 - 2.0f * island_padding * 0.5f) + island_padding * 0.5f, 0.0f, 1.0f );
            face_uv_flat[(f*3 + vi)*2 + 0] = u;
            face_uv_flat[(f*3 + vi)*2 + 1] = v;
        }
    }
}

std::vector<float> Unwrapper::distribute_individual_uvs_in_atlas(
    const std::vector<float>& face_uv_flat,
    const std::vector<int64_t>& assigned_faces,
    const std::vector<float>& offset_x,
    const std::vector<float>& offset_y,
    const std::vector<float>& div_x,
    const std::vector<float>& div_y,
    float island_padding)
{
    // face_uv_flat: Nf*3*2, assigned_faces length Nf
    size_t Nf = assigned_faces.size();
    // Make a copy to mutate via slice & remaining handlers
    std::vector<float> placed = face_uv_flat;

    // First handle slices (modify in place)
    handle_slice_uvs(placed, assigned_faces, island_padding, 12);
    // Then handle remaining
    handle_remaining_uvs(placed, assigned_faces, island_padding);

    // Then offset / divide
    // uc = uc / div_x[:, None] + offset_x[:, None]
    // vc = vc / div_y[:, None] + offset_y[:, None]
    std::vector<float> out_uv_flat = placed; // copy
    for (size_t f=0; f<Nf; ++f) {
        float ox = offset_x[f];
        float oy = offset_y[f];
        float dx = div_x[f];
        float dy = div_y[f];
        for (int vi=0; vi<3; ++vi) {
            size_t ptr = (f*3 + vi)*2;
            float uc = placed[ptr + 0] / dx + ox;
            float vc = placed[ptr + 1] / dy + oy;
            out_uv_flat[ptr + 0] = clampf(uc, 0.0f, 1.0f);
            out_uv_flat[ptr + 1] = clampf(vc, 0.0f, 1.0f);
        }
    }

    return out_uv_flat;
}

MeshUVResult Unwrapper::get_unique_face_uv(const std::vector<float>& uv_flat, size_t Nf) {
    // uv_flat length = Nf*3*2
    // Create a map from uv pair to unique index
    MeshUVResult out;
    out.vtex_idx.resize(Nf*3);

//    struct PairHash {
//        size_t operator()(const std::pair<float,float>& p) const noexcept {
//            // bitwise hash (not perfect), but okay for float keys as we use exact float equality
//            uint64_t a; std::memcpy(&a, &p.first, sizeof(float));
//            uint64_t b; std::memcpy(&b, &p.second, sizeof(float));
//            return std::hash<uint64_t>()(a ^ (b<<1));
//        }
//    };
    struct PairHash {
        size_t operator()(const std::pair<float,float>& p) const noexcept {
            uint32_t a_bits = 0u, b_bits = 0u;
            static_assert(sizeof(uint32_t) == sizeof(float), "float must be 32-bit");
            std::memcpy(&a_bits, &p.first, sizeof(float));
            std::memcpy(&b_bits, &p.second, sizeof(float));
            // combine into 64-bit to avoid collisions
            uint64_t key = (static_cast<uint64_t>(a_bits) << 32) | static_cast<uint64_t>(b_bits);
            return std::hash<uint64_t>()(key);
        }
    };
    struct PairEq {
        bool operator()(const std::pair<float,float>& a, const std::pair<float,float>& b) const noexcept {
            return a.first==b.first && a.second==b.second;
        }
    };

    std::unordered_map<std::pair<float,float>, int, PairHash, PairEq> mapUV;
    mapUV.reserve(Nf*3 * 2);

    int uid = 0;
    for (size_t f=0; f<Nf; ++f) {
        for (int vi=0; vi<3; ++vi) {
            size_t ptr = (f*3 + vi)*2;
            std::pair<float,float> p = { uv_flat[ptr+0], uv_flat[ptr+1] };
            auto it = mapUV.find(p);
            if (it == mapUV.end()) {
                mapUV.emplace(p, uid);
                out.unique_uv.push_back(p.first);
                out.unique_uv.push_back(p.second);
                out.vtex_idx[f*3 + vi] = uid;
                uid++;
            } else {
                out.vtex_idx[f*3 + vi] = it->second;
            }
        }
    }
    return out;
}

MeshUVResult Unwrapper::forward(
    const std::vector<float>& vertex_positions_in,
    const std::vector<float>& vertex_normals_in,
    const std::vector<int>& triangle_idxs,
    float island_padding)
{
    // Copy inputs to local mutable buffers
    std::vector<float> vpos = vertex_positions_in;
    std::vector<float> vnrm = vertex_normals_in;

    // Basic sanity checks
    if (vpos.empty()) {
        LOGE("Unwrapper::forward: vpos is empty!");
        return MeshUVResult{};
    }
    if (vpos.size() % 3 != 0) {
        LOGE("Unwrapper::forward: vpos size not multiple of 3: %zu", vpos.size());
        return MeshUVResult{};
    }
    if (!vnrm.empty() && vnrm.size() != vpos.size()) {
        LOGE("Unwrapper::forward: vnrm size (%zu) != vpos size (%zu)", vnrm.size(), vpos.size());
        return MeshUVResult{};
    }
    if (triangle_idxs.empty()) {
        LOGE("Unwrapper::forward: triangle_idxs is empty!");
        return MeshUVResult{};
    }
    if (triangle_idxs.size() % 3 != 0) {
        LOGE("Unwrapper::forward: triangle_idxs size not multiple of 3: %zu", triangle_idxs.size());
        return MeshUVResult{};
    }

    // Validate triangle indices are in-range
    const size_t Nv = vpos.size() / 3;
    for (size_t i = 0; i < triangle_idxs.size(); ++i) {
        int idx = triangle_idxs[i];
        if (idx < 0 || static_cast<size_t>(idx) >= Nv) {
            LOGE("Unwrapper::forward: triangle idx out of range at pos %zu -> %d (Nv=%zu)", i, idx, Nv);
            return MeshUVResult{};
        }
    }

    bool nan_in_pos = false;
    for (float v : vpos) {
        if (std::isnan(v)) {
            nan_in_pos = true;
            break;
        }
    }
    if (nan_in_pos) LOGE("[1] Found NaN in v_pos!");

    // Align mesh
    LOGI("aligning mesh with main axis");
    align_mesh_with_main_axis(vpos, vnrm);
    nan_in_pos = false;
    for (float v : vpos) {
        if (std::isnan(v)) {
            nan_in_pos = true;
            break;
        }
    }
    if (nan_in_pos) LOGE("[2] Found NaN in v_pos!");

    // Compute bbox
    std::array<float,6> bbox;
    LOGI("computing minmax vpos bbox");
    vec_minmax3(vpos, bbox);
    nan_in_pos = false;
    for (float v : vpos) {
        if (std::isnan(v)) {
            nan_in_pos = true;
            break;
        }
    }
    if (nan_in_pos) LOGE("[3] Found NaN in v_pos!");
//    std::cerr << "bbox: ";
//    for (const auto& el : bbox) std::cerr << el << ", ";
//    std::cerr << "\n";

    // Step 1: box assign -> face_uv_flat (Nf*3*2) and face_index (Nf)
    std::vector<float> face_uv_flat;
    std::vector<int64_t> face_index;
    LOGI("box_assign_vertex_to_cube_face");
    box_assign_vertex_to_cube_face(vpos, vnrm, triangle_idxs, bbox, face_uv_flat, face_index);
    std::string log1 = "face_uv_flat | face_index: \n";
    for (size_t i = 0; i < 5; i++) {
        log1 += "ind: " + std::to_string(i) + "\n";
        log1 += "ind: " + std::to_string(face_uv_flat[i*6 + 0]) + " | " + std::to_string(face_index[i])+"\n";
        log1 += "ind: " + std::to_string(face_uv_flat[i*6 + 1]) + " | " + std::to_string(face_index[i])+"\n";
        log1 += "ind: " + std::to_string(face_uv_flat[i*6 + 2]) + " | " + std::to_string(face_index[i])+"\n";
        log1 += "ind: " + std::to_string(face_uv_flat[i*6 + 3]) + " | " + std::to_string(face_index[i])+"\n";
        log1 += "ind: " + std::to_string(face_uv_flat[i*6 + 4]) + " | " + std::to_string(face_index[i])+"\n";
        log1 += "ind: " + std::to_string(face_uv_flat[i*6 + 5]) + " | " + std::to_string(face_index[i])+"\n";
    }
    LOGI("%s", log1.c_str());

    // Check outputs of box_assign
    const size_t Nf = triangle_idxs.size()/3;
    if (face_uv_flat.size() != Nf * 6) {
        LOGE("Unwrapper::forward: unexpected face_uv_flat size %zu, expected %zu (Nf=%zu)", face_uv_flat.size(), (size_t)Nf*6, Nf);
        return MeshUVResult{};
    }
    if (face_index.size() != Nf) {
        LOGE("Unwrapper::forward: unexpected face_index size %zu, expected %zu", face_index.size(), Nf);
        return MeshUVResult{};
    }
    // Step 2: rotate uv slices consistent
    LOGI("rotate_uv_slices_consistent_space");
    rotate_uv_slices_consistent_space(vpos, vnrm, triangle_idxs, face_uv_flat, face_index);
//    std::cerr << "face_uv_flat: \n";
//    for (size_t i = 0; i < 5; i++) {
//        std::cerr << "ind: " << i << "\n";
//        std::cerr << face_uv_flat[i*6 + 0] << "\n";
//        std::cerr << face_uv_flat[i*6 + 1] << "\n";
//        std::cerr << face_uv_flat[i*6 + 2] << "\n";
//        std::cerr << face_uv_flat[i*6 + 3] << "\n";
//        std::cerr << face_uv_flat[i*6 + 4] << "\n";
//        std::cerr << face_uv_flat[i*6 + 5] << "\n";
//    }
//    std::cerr << "\n";

    // Step 3: ask the BVH-based C++ function to assign faces to atlas indices
    // We need to call external bridge: it expects vertices (Nv*3), indices (Nf*3),
    // face_uv flattened as Nf*3 x 2 (i.e. same as face_uv_flat), and face_index
    LOGI("UVUnwrapper_assign_faces_uv_to_atlas_index_raw...");
    LOGI("triangle_idx size: %zu  | /3--> %zu",triangle_idxs.size(), triangle_idxs.size()/3);
    std::vector<int64_t> assigned_atlas_index = UVUnwrapperBridge::UVUnwrapper_assign_faces_uv_to_atlas_index_raw(
        vpos,
        triangle_idxs,
        face_uv_flat,
        face_index
    );
//    std::cerr << "assigned_atlas_index:\n";
//    for (size_t i = 0; i < 20; i++) {
//        std::cerr << i << ": " << assigned_atlas_index[i] << "\n";
//    }
//    std::cerr << "\n";
    if (assigned_atlas_index.size() != Nf) {
        LOGE("Unwrapper::forward: assigned_atlas_index size %zu != Nf %zu", assigned_atlas_index.size(), Nf);
        return MeshUVResult{};
    }

    // Step 4: find slice offsets and scale
    std::vector<float> offset_x, offset_y, div_x, div_y;
    LOGI("find_slice_offset_and_scale");
    find_slice_offset_and_scale(assigned_atlas_index, offset_x, offset_y, div_x, div_y);

    // Step 5: distribute into atlas
    LOGI("distribute_individual_uvs_in_atlas");
    std::vector<float> placed_uv = distribute_individual_uvs_in_atlas(face_uv_flat, assigned_atlas_index, offset_x, offset_y, div_x, div_y, island_padding);
    if (placed_uv.empty()) {
        LOGE("Unwrapper::forward: placed_uv is empty after distribution");
        return MeshUVResult{};
    }

    // Step 6: get unique uv and mapping
    LOGI("get_unique_face_uv");
    MeshUVResult res = get_unique_face_uv(placed_uv, triangle_idxs.size()/3);
    // final check:
    if (res.unique_uv.empty() || res.vtex_idx.empty()) {
        LOGE("Unwrapper::forward: result empty (unique_uv=%zu, vtex_idx=%zu)", res.unique_uv.size(), res.vtex_idx.size());
    }

    return res;
}
