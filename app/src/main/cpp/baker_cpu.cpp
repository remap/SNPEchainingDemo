//
// Created by Chiheb Boussema on 23/12/25.
//

// baker_cpu.cpp - CPU-only rasterize/interpolate implementations
// Adapted from SPAR3D original file. Two APIs provided:
//  - Pointer-based: low-level, no allocations, suitable for JNI
//  - Vector wrappers: convenience, copies into typed arrays then calls pointer API

#include "baker_cpu.h"
#include "tiny_gltf.h"

#include <chrono>
#include <cmath>
#include <queue>
#include <numeric>
#include <iostream>
#include <omp.h>
#include <algorithm> // for min/max
#include <atomic>

#include <vector>
#include <string>
#include <random>
#include <cassert>



//#ifndef __ARM_ARCH_ISA_A64
//  #include <immintrin.h>
//#endif
//
//#if defined(__aarch64__) || defined(__ARM_NEON)
//  #include <arm_neon.h>
//#endif
//
//#ifndef _MSC_VER
//  #include <immintrin.h> // SSE/AVX for x86
//#endif

// Architecture-specific SIMD headers:
// - include x86 intrinsics only for x86/x64 builds
// - include ARM NEON only for ARM builds
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  // x86/x64 (SSE/AVX)
  #include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
  // ARM NEON
  #include <arm_neon.h>
#else
  // No platform intrinsics available — code should use scalar fallbacks
#endif


// #define TIMING
#define BINS 8


#include <android/log.h>
#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "BAKER_CPU"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)


namespace texture_baker_cpp {

// ------------------------- helpers from original -------------------------

// Calculate centroid of triangle
tb_float2 triangle_centroid(const tb_float2 &v0, const tb_float2 &v1,
                            const tb_float2 &v2) {
  return {(v0.x + v1.x + v2.x) * 0.3333f,
          (v0.y + v1.y + v2.y) * 0.3333f};
}

float BVH::find_best_split_plane(const BVHNode &node, int &best_axis,
                                 int &best_pos, AABB &centroidBounds) {
  float best_cost = std::numeric_limits<float>::max();

  for (int axis = 0; axis < 2; ++axis) {
    float boundsMin = centroidBounds.min[axis];
    float boundsMax = centroidBounds.max[axis];
    if (boundsMin == boundsMax) continue;

    float scale = BINS / (boundsMax - boundsMin);
    float leftCountArea[BINS - 1], rightCountArea[BINS - 1];
    int leftSum = 0, rightSum = 0;

#ifndef __ARM_ARCH_ISA_A64
#ifndef _MSC_VER
    if (__builtin_cpu_supports("sse"))
#elif (defined(_M_AMD64) || defined(_M_X64))
    if constexpr (true)
#endif
    {
//#ifndef _MSC_VER
      __m128 min4[BINS], max4[BINS];
      unsigned int count[BINS];
      for (unsigned int i = 0; i < BINS; ++i) {
        min4[i] = _mm_set_ps1(1e30f);
        max4[i] = _mm_set_ps1(-1e30f);
        count[i] = 0;
      }
      for (int i = node.start; i < node.end; ++i) {
        int tri_idx = triangle_indices[i];
        const Triangle &triangle = triangles[tri_idx];

        int binIdx = std::min(BINS - 1,
            (int)((triangle.centroid[axis] - boundsMin) * scale));
        count[binIdx]++;

        __m128 v0 = _mm_set_ps(triangle.v0.x, triangle.v0.y, 0.0f, 0.0f);
        __m128 v1 = _mm_set_ps(triangle.v1.x, triangle.v1.y, 0.0f, 0.0f);
        __m128 v2 = _mm_set_ps(triangle.v2.x, triangle.v2.y, 0.0f, 0.0f);

        min4[binIdx] = _mm_min_ps(min4[binIdx], v0);
        max4[binIdx] = _mm_max_ps(max4[binIdx], v0);
        min4[binIdx] = _mm_min_ps(min4[binIdx], v1);
        max4[binIdx] = _mm_max_ps(max4[binIdx], v1);
        min4[binIdx] = _mm_min_ps(min4[binIdx], v2);
        max4[binIdx] = _mm_max_ps(max4[binIdx], v2);
      }

      __m128 leftMin4 = _mm_set_ps1(1e30f), rightMin4 = leftMin4;
      __m128 leftMax4 = _mm_set_ps1(-1e30f), rightMax4 = leftMax4;

      for (int i = 0; i < BINS - 1; ++i) {
        leftSum += count[i];
        rightSum += count[BINS - 1 - i];
        leftMin4 = _mm_min_ps(leftMin4, min4[i]);
        rightMin4 = _mm_min_ps(rightMin4, min4[BINS - 2 - i]);
        leftMax4 = _mm_max_ps(leftMax4, max4[i]);
        rightMax4 = _mm_max_ps(rightMax4, max4[BINS - 2 - i]);

        float le[4], re[4];
        _mm_store_ps(le, _mm_sub_ps(leftMax4, leftMin4));
        _mm_store_ps(re, _mm_sub_ps(rightMax4, rightMin4));
        leftCountArea[i] = leftSum * (le[2] * le[3]);
        rightCountArea[BINS - 2 - i] = rightSum * (re[2] * re[3]);
      }
    }
#else
      // MSVC path could be added if needed
      if constexpr (false) {
      }
#endif
//    } else
//#endif
    {
      struct Bin {
        AABB bounds;
        int triCount = 0;
      } bins[BINS];

      for (int i = node.start; i < node.end; ++i) {
        int tri_idx = triangle_indices[i];
        const Triangle &triangle = triangles[tri_idx];

        int binIdx = std::min(BINS - 1,
            (int)((triangle.centroid[axis] - boundsMin) * scale));
        bins[binIdx].triCount++;
        bins[binIdx].bounds.grow(triangle.v0);
        bins[binIdx].bounds.grow(triangle.v1);
        bins[binIdx].bounds.grow(triangle.v2);
      }

      AABB leftBox, rightBox;
      for (int i = 0; i < BINS - 1; ++i) {
        leftSum += bins[i].triCount;
        leftBox.grow(bins[i].bounds);
        leftCountArea[i] = leftSum * leftBox.area();

        rightSum += bins[BINS - 1 - i].triCount;
        rightBox.grow(bins[BINS - 1 - i].bounds);
        rightCountArea[BINS - 2 - i] = rightSum * rightBox.area();
      }
    }

    scale = (boundsMax - boundsMin) / BINS;
    for (int i = 0; i < BINS - 1; ++i) {
      float planeCost = leftCountArea[i] + rightCountArea[i];
      if (planeCost < best_cost) {
        best_axis = axis;
        best_pos = i + 1;
        best_cost = planeCost;
      }
    }
  }

  return best_cost;
}

void BVH::update_node_bounds(BVHNode &node, AABB &centroidBounds) {
#ifndef __ARM_ARCH_ISA_A64
#ifndef _MSC_VER
  if (__builtin_cpu_supports("sse"))
#elif (defined(_M_AMD64) || defined(_M_X64))
  if constexpr (true)
#endif
  {
//#ifndef _MSC_VER
    __m128 min4 = _mm_set_ps1(1e30f), max4 = _mm_set_ps1(-1e30f);
    __m128 cmin4 = _mm_set_ps1(1e30f), cmax4 = _mm_set_ps1(-1e30f);

    for (int i = node.start; i < node.end; i += 2) {
      int tri_idx1 = triangle_indices[i];
      const Triangle &leafTri1 = triangles[tri_idx1];

      __m128 v0, v1, v2, centroid;
      if (i + 1 < node.end) {
        int tri_idx2 = triangle_indices[i + 1];
        const Triangle &leafTri2 = triangles[tri_idx2];

        v0 = _mm_set_ps(leafTri1.v0.x, leafTri1.v0.y, leafTri2.v0.x,
                        leafTri2.v0.y);
        v1 = _mm_set_ps(leafTri1.v1.x, leafTri1.v1.y, leafTri2.v1.x,
                        leafTri2.v1.y);
        v2 = _mm_set_ps(leafTri1.v2.x, leafTri1.v2.y, leafTri2.v2.x,
                        leafTri2.v2.y);
        centroid = _mm_set_ps(leafTri1.centroid.x, leafTri1.centroid.y,
                              leafTri2.centroid.x, leafTri2.centroid.y);
      } else {
        v0 = _mm_set_ps(leafTri1.v0.x, leafTri1.v0.y, leafTri1.v0.x,
                        leafTri1.v0.y);
        v1 = _mm_set_ps(leafTri1.v1.x, leafTri1.v1.y, leafTri1.v1.x,
                        leafTri1.v1.y);
        v2 = _mm_set_ps(leafTri1.v2.x, leafTri1.v2.y, leafTri1.v2.x,
                        leafTri1.v2.y);
        centroid = _mm_set_ps(leafTri1.centroid.x, leafTri1.centroid.y,
                              leafTri1.centroid.x, leafTri1.centroid.y);
      }

      min4 = _mm_min_ps(min4, v0);
      max4 = _mm_max_ps(max4, v0);
      min4 = _mm_min_ps(min4, v1);
      max4 = _mm_max_ps(max4, v1);
      min4 = _mm_min_ps(min4, v2);
      max4 = _mm_max_ps(max4, v2);
      cmin4 = _mm_min_ps(cmin4, centroid);
      cmax4 = _mm_max_ps(cmax4, centroid);
    }

    float min_values[4], max_values[4], cmin_values[4], cmax_values[4];
    _mm_store_ps(min_values, min4);
    _mm_store_ps(max_values, max4);
    _mm_store_ps(cmin_values, cmin4);
    _mm_store_ps(cmax_values, cmax4);

    node.bbox.min.x = std::min(min_values[3], min_values[1]);
    node.bbox.min.y = std::min(min_values[2], min_values[0]);
    node.bbox.max.x = std::max(max_values[3], max_values[1]);
    node.bbox.max.y = std::max(max_values[2], max_values[0]);

    centroidBounds.min.x = std::min(cmin_values[3], cmin_values[1]);
    centroidBounds.min.y = std::min(cmin_values[2], cmin_values[0]);
    centroidBounds.max.x = std::max(cmax_values[3], cmax_values[1]);
    centroidBounds.max.y = std::max(cmax_values[2], cmax_values[0]);
  }
#else
    // MSVC SSE path could go here
    if constexpr (false) {
//#endif
  }
#endif
  // Scalar fallback / initialization region
  {
    node.bbox.invalidate();
    centroidBounds.invalidate();

    for (int i = node.start; i < node.end; ++i) {
      int tri_idx = triangle_indices[i];
      const Triangle &tri = triangles[tri_idx];
      node.bbox.grow(tri.v0);
      node.bbox.grow(tri.v1);
      node.bbox.grow(tri.v2);
      centroidBounds.grow(tri.centroid);
    }
  }
}

void BVH::build(const tb_float2 *vertices, const tb_int3 *indices,
                const int64_t &num_indices) {
#ifdef TIMING
  auto start = std::chrono::high_resolution_clock::now();
#endif

  triangles.clear();
  triangle_indices.clear();
  nodes.clear();

//  for (size_t i = 0; i < static_cast<size_t>(num_indices); ++i) {
  for (size_t i = 0; i < num_indices; ++i) {
    tb_int3 idx = indices[i];
    triangles.push_back({
      vertices[idx.x],
      vertices[idx.y],
      vertices[idx.z],
      static_cast<int>(i),
      triangle_centroid(vertices[idx.x], vertices[idx.y], vertices[idx.z])
    });
  }

  triangle_indices.resize(triangles.size());
  std::iota(triangle_indices.begin(), triangle_indices.end(), 0);

  nodes.reserve(triangles.size() * 2 + 1);
  nodes.push_back({});
  root = 0;

  struct QueueEntry { int node_idx, start, end; };
  std::queue<QueueEntry> node_queue;
//  node_queue.push({root, 0, static_cast<int>(triangles.size())});
  node_queue.push({root, 0, (int)triangles.size()});

  while (!node_queue.empty()) {
    QueueEntry current = node_queue.front();
    node_queue.pop();

    int node_idx = current.node_idx;
    int start = current.start;
    int end = current.end;

    BVHNode &node = nodes[node_idx];
    node.start = start;
    node.end = end;

    AABB centroidBounds;
    update_node_bounds(node, centroidBounds);

//    int best_axis = 0, best_pos = 0;
    int best_axis, best_pos;
    float splitCost = find_best_split_plane(node, best_axis, best_pos, centroidBounds);
    float nosplitCost = node.calculate_node_cost();

    if (splitCost >= nosplitCost) {
      node.left = node.right = -1;
      continue;
    }

    float scale = BINS / (centroidBounds.max[best_axis] - centroidBounds.min[best_axis]);
    int i = node.start;
    int j = node.end - 1;

    while (i <= j) {
      int tri_idx = triangle_indices[i];
      tb_float2 tcentr = triangles[tri_idx].centroid;
      int binIdx = std::min(BINS - 1,
        (int)((tcentr[best_axis] - centroidBounds.min[best_axis]) * scale));
      if (binIdx < best_pos) i++;
      else std::swap(triangle_indices[i], triangle_indices[j--]);
    }

    int leftCount = i - node.start;
    if (leftCount == 0 || leftCount == node.num_triangles()) {
      node.left = node.right = -1;
      continue;
    }

    int mid = i;
//    node.left = static_cast<int>(nodes.size());
    node.left = nodes.size();
    nodes.push_back({});
    node_queue.push({node.left, start, mid});

    // refresh node alias as original code did
    // node reference might be invalidated by push_back on vector; re-acquire:
//    BVHNode &node_ref = nodes[node_idx];
//    node_ref.right = static_cast<int>(nodes.size());
//    node_ref.right = nodes.size();
    node = nodes[node_idx];
    node.right = nodes.size();
    nodes.push_back({});
//    node_queue.push({node_ref.right, mid, end});
    node_queue.push({node.right, mid, end});
  }

#ifdef TIMING
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "BVH build time: " << elapsed.count() << "s" << std::endl;
#endif
}

// clamp & barycentric (unchanged)
static inline float clamp_f(float val, float minVal, float maxVal) {
  return std::min(std::max(val, minVal), maxVal);
}

bool barycentric_coordinates(tb_float2 xy, tb_float2 v1, tb_float2 v2,
                             tb_float2 v3, float &u, float &v, float &w) {
  tb_float2 v1v2 = {v2.x - v1.x, v2.y - v1.y};
  tb_float2 v1v3 = {v3.x - v1.x, v3.y - v1.y};
  tb_float2 xyv1 = {xy.x - v1.x, xy.y - v1.y};

  float d00 = v1v2.x * v1v2.x + v1v2.y * v1v2.y;
  float d01 = v1v2.x * v1v3.x + v1v2.y * v1v3.y;
  float d11 = v1v3.x * v1v3.x + v1v3.y * v1v3.y;
  float d20 = xyv1.x * v1v2.x + xyv1.y * v1v2.y;
  float d21 = xyv1.x * v1v3.x + xyv1.y * v1v3.y;

  float denom = d00 * d11 - d01 * d01;
//  if (denom == 0.0f) {
//    u = v = w = 0.0f;
//    return false;
//  }
  v = (d11 * d20 - d01 * d21) / denom;
  w = (d00 * d21 - d01 * d20) / denom;
  u = 1.0f - v - w;

  return (v >= 0.0f) && (w >= 0.0f) && (v + w <= 1.0f);
}

bool BVH::intersect(const tb_float2 &point, float &u, float &v, float &w,
                    int &index) const {
  const int max_stack_size = 64;
  int node_stack[max_stack_size];
  int stack_size = 0;

  node_stack[stack_size++] = root;

  while (stack_size > 0) {
    int node_idx = node_stack[--stack_size];
    const BVHNode &node = nodes[node_idx];

    if (node.is_leaf()) {
      for (int i = node.start; i < node.end; ++i) {
        const Triangle &tri = triangles[triangle_indices[i]];
        if (barycentric_coordinates(point, tri.v0, tri.v1, tri.v2, u, v, w)) {
          index = tri.index;
          return true;
        }
      }
    } else {
      if (nodes[node.right].bbox.overlaps(point)) {
        if (stack_size < max_stack_size) node_stack[stack_size++] = node.right;
        else throw std::runtime_error("Node stack overflow");
      }
      if (nodes[node.left].bbox.overlaps(point)) {
        if (stack_size < max_stack_size) node_stack[stack_size++] = node.left;
        else throw std::runtime_error("Node stack overflow");
      }
    }
  }

  return false;
}

// ---------------------- New pointer-based APIs ---------------------------

// Pointer-based rasterize:
// rast_result: preallocated float array of size (bake_resolution * bake_resolution * 4)
// vertices: pointer to tb_float2 array (num_vertices elements) - used only for BVH construction
// indices: pointer to tb_int3 array (num_triangles elements)
// num_triangles: number of triangles (M)
// bake_resolution: width==height==bake_resolution
void rasterize_cpu_host(float* rast_result,
                        const tb_float2* vertices,
                        const tb_int3* indices,
                        const int64_t num_triangles,
                        const int bake_resolution) {
  if (!rast_result || !vertices || !indices) return;

  const int width = bake_resolution;
  const int height = bake_resolution;
  const int num_pixels = width * height;

  BVH bvh;
  auto t_start = std::chrono::high_resolution_clock::now();
  bvh.build(vertices, indices, num_triangles);
  auto t_after_bvh = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sec = t_after_bvh - t_start;
  std::cout << "create_bvhs time: " << sec.count() << "s\n";
#ifdef TIMING
  auto start = std::chrono::high_resolution_clock::now();
#endif

#pragma omp parallel for
  for (int idx = 0; idx < num_pixels; ++idx) {
    int x = idx / height;
    int y = idx % height;
    int out_idx = idx * 4;

    tb_float2 pixel_coord = { static_cast<float>(y) / height, static_cast<float>(x) / width };
    pixel_coord.x = clamp_f(pixel_coord.x, 0.0f, 1.0f);
    pixel_coord.y = 1.0f - clamp_f(pixel_coord.y, 0.0f, 1.0f);

    float u,v,w;
    int triangle_idx;// = -1;
    if (bvh.intersect(pixel_coord, u, v, w, triangle_idx)) {
      rast_result[out_idx + 0] = u;
      rast_result[out_idx + 1] = v;
      rast_result[out_idx + 2] = w;
      rast_result[out_idx + 3] = static_cast<float>(triangle_idx);
    } else {
      rast_result[out_idx + 0] = 0.0f;
      rast_result[out_idx + 1] = 0.0f;
      rast_result[out_idx + 2] = 0.0f;
      rast_result[out_idx + 3] = -1.0f;
    }
  }
  auto t_after_for = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sec2 = t_after_for - t_after_bvh;
  std::cout << "for loop with intersect time: " << sec2.count() << "s\n";

#ifdef TIMING
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Rasterization time: " << elapsed.count() << "s" << std::endl;
#endif
//  output is rast_result
}

// Vector wrapper for rasterize:
// vertices_flat: layout [vx0, vy0, vx1, vy1, ...] length = num_vertices * 2
// indices_flat: layout [i0, j0, k0, i1, j1, k1, ...] length = num_triangles * 3
void rasterize_cpu_host(float* rast_result,
                        const std::vector<float>& vertices_flat, /* v_tex */
                        const std::vector<int>& indices_flat, /* t_pos_idx */
                        const int64_t num_triangles,
                        const int bake_resolution) {
  // Validate sizes minimally
  if (vertices_flat.empty() || indices_flat.empty() || !rast_result) return;
  if (indices_flat.size() < static_cast<size_t>(num_triangles * 3)) return;

  // Build typed arrays (temporary) from flat inputs
  size_t num_vertices = vertices_flat.size() / 2;
  std::vector<tb_float2> vertices(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    vertices[i].x = vertices_flat[2*i + 0];
    vertices[i].y = vertices_flat[2*i + 1];
  }

  std::vector<tb_int3> tris(num_triangles);
  for (int64_t t = 0; t < num_triangles; ++t) {
    tris[t].x = indices_flat[3*t + 0];
    tris[t].y = indices_flat[3*t + 1];
    tris[t].z = indices_flat[3*t + 2];
  }

  rasterize_cpu_host(rast_result, vertices.data(), tris.data(), num_triangles, bake_resolution);
}

// --- triangle-centric rasterizer with tile-binning ---
// Replace previous rasterize_cpu_host(ptr, vertices, indices, ...) with this.


void rasterize_cpu_host_triangleTile(float* rast_result,
                        const tb_float2* vertices,
                        const tb_int3* indices,
                        const int64_t num_triangles,
                        const int bake_resolution) {
  if (!rast_result || !vertices || !indices) return;
  const int res = bake_resolution;
  const int width = res;
  const int height = res;
  const int num_pixels = width * height;

  // initialize output: (H * W * 4)
  // each pixel: [u, v, w, tri_index_as_float]; default tri index = -1
  // We'll initialize whole buffer here (single-threaded cheap relative to raster).
  {
    float *p = rast_result;
    for (int i = 0; i < num_pixels; ++i) {
      p[4*i + 0] = 0.0f;
      p[4*i + 1] = 0.0f;
      p[4*i + 2] = 0.0f;
      p[4*i + 3] = -1.0f;
    }
  }

  // --- Tile parameters ---
  const int tile_size = 16; // good default; tune to 8..32
  const int tiles_x = (width  + tile_size - 1) / tile_size;
  const int tiles_y = (height + tile_size - 1) / tile_size;
  const int num_tiles = tiles_x * tiles_y;

  // Precompute per-triangle integer pixel-space bounding boxes and transformed verts
  struct TriPacked {
    float x0, y0; // pixel-space floats
    float x1, y1;
    float x2, y2;
    float area2;  // twice area (signed) used as denominator
    int minx, miny, maxx, maxy; // inclusive pixel bbox (clamped to image)
    int tri_index;
  };
  std::vector<TriPacked> tri_data;
  tri_data.reserve(num_triangles);
  auto tri_s = std::chrono::high_resolution_clock::now();

  for (int t = 0; t < static_cast<int>(num_triangles); ++t) {
    const tb_int3 &tri = indices[t];
    const tb_float2 &uv0 = vertices[tri.x];
    const tb_float2 &uv1 = vertices[tri.y];
    const tb_float2 &uv2 = vertices[tri.z];

    // Map UV -> pixel space (consistent with original code)
    float x0 = uv0.x * (float)res;
    float y0 = (1.0f - uv0.y) * (float)res;
    float x1 = uv1.x * (float)res;
    float y1 = (1.0f - uv1.y) * (float)res;
    float x2 = uv2.x * (float)res;
    float y2 = (1.0f - uv2.y) * (float)res;
//    float y0 = uv0.x * (float)res;
//    float x0 = (1.0f - uv0.y) * (float)res;
//    float y1 = uv1.x * (float)res;
//    float x1 = (1.0f - uv1.y) * (float)res;
//    float y2 = uv2.x * (float)res;
//    float x2 = (1.0f - uv2.y) * (float)res;

    // compute bbox in integer pixel coords (clamped)
    float fxmin = std::floor(std::min({x0, x1, x2}));
    float fymin = std::floor(std::min({y0, y1, y2}));
    float fxmax = std::ceil (std::max({x0, x1, x2}));
    float fymax = std::ceil (std::max({y0, y1, y2}));

    int minx = static_cast<int>(std::max(0.0f, fxmin));
    int miny = static_cast<int>(std::max(0.0f, fymin));
    int maxx = static_cast<int>(std::min((float)(width-1), fxmax));
    int maxy = static_cast<int>(std::min((float)(height-1), fymax));

    // compute twice the signed area for barycentric denominator
    float area2 = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0); // = cross(v1-v0, v2-v0)

    // skip degenerate triangles quickly
    if (std::abs(area2) <= 1e-9f || minx > maxx || miny > maxy) {
      // push a degenerate entry with empty bbox so it won't be used
      tri_data.push_back({x0,y0,x1,y1,x2,y2, area2, width, height, -1, -1, t});
      continue;
    }

    TriPacked td;
    td.x0 = x0; td.y0 = y0;
    td.x1 = x1; td.y1 = y1;
    td.x2 = x2; td.y2 = y2;
    td.area2 = area2;
    td.minx = minx; td.miny = miny;
    td.maxx = maxx; td.maxy = maxy;
    td.tri_index = t;
    tri_data.push_back(td);
  }
  auto tri_e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = tri_e - tri_s;
  std::cout << "triangle coordinates into pixel space time: " << elapsed.count() << "s" << std::endl;

  // --- Build tile->triangle lists (simple binning) ---
  // Pre-allocate per-tile vectors to avoid repeated allocations
  std::vector<std::vector<int>> tile_tris(num_tiles);
  // Reserve approximate capacity: average triangles per tile
  int avg_tris_per_tile = std::max(1, static_cast<int>(num_triangles / std::max(1, num_tiles)));
  for (int i = 0; i < num_tiles; ++i) tile_tris[i].reserve(avg_tris_per_tile);

  auto ti_to_t_s = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < static_cast<int>(num_triangles); ++t) {
    const TriPacked &td = tri_data[t];
    if (td.minx > td.maxx || td.miny > td.maxy) continue;
    int tile_x0 = td.minx / tile_size;
    int tile_x1 = td.maxx / tile_size;
    int tile_y0 = td.miny / tile_size;
    int tile_y1 = td.maxy / tile_size;
    tile_x0 = std::max(0, std::min(tile_x0, tiles_x-1));
    tile_x1 = std::max(0, std::min(tile_x1, tiles_x-1));
    tile_y0 = std::max(0, std::min(tile_y0, tiles_y-1));
    tile_y1 = std::max(0, std::min(tile_y1, tiles_y-1));

    for (int ty = tile_y0; ty <= tile_y1; ++ty) {
      for (int tx = tile_x0; tx <= tile_x1; ++tx) {
        int tile_id = ty * tiles_x + tx;
        tile_tris[tile_id].push_back(t);
      }
    }
  }
  auto ti_to_t_e = std::chrono::high_resolution_clock::now();
  elapsed = ti_to_t_e - ti_to_t_s;
  std::cout << "tile->triangle time: " << elapsed.count() << "s" << std::endl;

  // --- Rasterize tiles in parallel (each tile owns its pixels -> no atomics) ---
  // We'll parallelize over tile_id. Use static schedule for uniform distribution.
  auto rast_tiles_s = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for schedule(static)
  for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
    int tx = tile_id % tiles_x;
    int ty = tile_id / tiles_x;
    int px0 = tx * tile_size;
    int py0 = ty * tile_size;
    int px1 = std::min(px0 + tile_size - 1, width - 1);
    int py1 = std::min(py0 + tile_size - 1, height - 1);

    // pointer to rast_result for convenience
    float* out_base = rast_result;

    // For each triangle assigned to this tile, rasterize its intersection with the tile bbox
    const std::vector<int> &tris = tile_tris[tile_id];
    for (int tidx : tris) {
      const TriPacked &tri = tri_data[tidx];

      // quick reject if triangle bbox doesn't intersect tile region
      if (tri.minx > px1 || tri.maxx < px0 || tri.miny > py1 || tri.maxy < py0) continue;

      // compute integer bbox intersection
      int bx0 = std::max(px0, tri.minx);
      int bx1 = std::min(px1, tri.maxx);
      int by0 = std::max(py0, tri.miny);
      int by1 = std::min(py1, tri.maxy);

      // Precompute edge coefficients for E(x,y) = (x - x0)*(y1 - y0) - (y - y0)*(x1 - x0)
      // We'll compute three edges e0,e1,e2 for vertices (x0,y0),(x1,y1),(x2,y2)
      float x0 = tri.x0, y0 = tri.y0;
      float x1 = tri.x1, y1 = tri.y1;
      float x2 = tri.x2, y2 = tri.y2;
      // Edge function coefficients: E0(x,y)=A0*x + B0*y + C0 etc
      float A0 = (y1 - y0);
      float B0 = -(x1 - x0);
      float C0 = x1*y0 - y1*x0; // such that E0(x,y) = A0*x + B0*y + C0

      float A1 = (y2 - y1);
      float B1 = -(x2 - x1);
      float C1 = x2*y1 - y2*x1;

      float A2 = (y0 - y2);
      float B2 = -(x0 - x2);
      float C2 = x0*y2 - y0*x2;

      float area2 = tri.area2; // denom for barycentric


      // Evaluate sample at pixel centers (x + 0.5, y + 0.5).
       for (int y = by0; y <= by1; ++y) {
           float py = float(y) + 0.5f;
           for (int x = bx0; x <= bx1; ++x) {
               float px = float(x) + 0.5f;
               // Compute signed areas (cross products)
               float a0 = (x1 - px) * (y2 - py) - (y1 - py) * (x2 - px); // area of tri (p, v1, v2)
               float a1 = (x2 - px) * (y0 - py) - (y2 - py) * (x0 - px); // area of tri (p, v2, v0)
               float a2 = (x0 - px) * (y1 - py) - (y0 - py) * (x1 - px); // area of tri (p, v0, v1)

               // total (signed) area:
               float area_total = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0);

               // inside test: consistent with area_total sign
               if ((a0 * area_total >= -1e-8f) &&
               (a1 * area_total >= -1e-8f) &&
               (a2 * area_total >= -1e-8f)) {
                   float u = a0 / area_total;
                   float v = a1 / area_total;
                   float w = a2 / area_total;

                   int pix_idx = (y * width + x) * 4;
                   out_base[pix_idx + 0] = u;
                   out_base[pix_idx + 1] = v;
                   out_base[pix_idx + 2] = w;
                   out_base[pix_idx + 3] = float(tri.tri_index);
               }
           }
       }


      // We'll evaluate E at integer pixel coordinates (x,y). Pixel sample = (x, y)
      // To speed up, compute E for start of row and increment by A? Actually delta_x = A, delta_y = B; we use row-major inner loop.
//      for (int y = by0; y <= by1; ++y) {
//        // compute E values at (bx0, y)
//        float ex0 = A0 * float(bx0) + B0 * float(y) + C0;
//        float ex1 = A1 * float(bx0) + B1 * float(y) + C1;
//        float ex2 = A2 * float(bx0) + B2 * float(y) + C2;
//
//        // step per +1 x increments
//        const float step0 = A0;
//        const float step1 = A1;
//        const float step2 = A2;
//
//        // iterate across x in the scanline
//        int out_row_base = y * width;
//        for (int x = bx0; x <= bx1; ++x) {
//          // inside-triangle test (winding: accept >= 0 for a consistent fill rule)
//          bool inside =
//              (ex0 * area2 >= -1e-8f) &&
//              (ex1 * area2 >= -1e-8f) &&
//              (ex2 * area2 >= -1e-8f);
//
//          if (inside) {
//            // compute barycentric coords: using edge values scaled by area2
//            // u = E1 / area2? There are different relations; using standard formula:
//            // u = E1 / area2, v = E2 / area2, w = 1 - u - v
//            float u = ex1 / area2;
//            float v = ex2 / area2;
//            float w = 1.0f - u - v;
//
//            int pix_idx = (out_row_base + x) * 4;
//            out_base[pix_idx + 0] = u;
//            out_base[pix_idx + 1] = v;
//            out_base[pix_idx + 2] = w;
//            out_base[pix_idx + 3] = static_cast<float>(tri.tri_index);
//          }
//
//          // increment edge values for next x
//          ex0 += step0;
//          ex1 += step1;
//          ex2 += step2;
//        } // x
//      } // y
    } // per triangle
  } // tiles loop
  auto rast_tiles_e = std::chrono::high_resolution_clock::now();
  elapsed = rast_tiles_e - rast_tiles_s;
  std::cout << "rasterize tiles time: " << elapsed.count() << "s" << std::endl;

  // done; rast_result filled in place
}


// ---------------------- Interpolate pointer-based APIs -------------------

// Pointer-based interpolate:
// output_ptr: preallocated float array of size (width * height * 3)
// attr_ptr: per-vertex attributes (num_vertices * 3 floats) layout [x0,y0,z0, x1,y1,z1,...]
// indices_ptr: flat triangle indices (num_triangles * 3 ints) layout as above
// num_triangles: number of triangles
// rast_ptr: raster buffer produced by rasterize (width*height*4 floats)
// width, height: raster dims
void interpolate_cpu_host(float* output_ptr,
                          const float* attr_ptr,
                          const int* indices_ptr,
                          const int num_triangles,
                          const float* rast_ptr,
                          const int width, /* equal to bake_resolution */
                          const int height /* equal to bake_resolution */
                          ) {
  if (!output_ptr || !attr_ptr || !indices_ptr || !rast_ptr) return;

  const int num_pixels = width * height;

#pragma omp parallel for
  for (int idx = 0; idx < num_pixels; ++idx) {
    int rast_idx = idx * 4;
    float b0 = rast_ptr[rast_idx + 0];
    float b1 = rast_ptr[rast_idx + 1];
    float b2 = rast_ptr[rast_idx + 2];
    int triangle_idx = static_cast<int>(rast_ptr[rast_idx + 3]);

    if (triangle_idx < 0) {
      output_ptr[idx * 3 + 0] = 0.0f;
      output_ptr[idx * 3 + 1] = 0.0f;
      output_ptr[idx * 3 + 2] = 0.0f;
      continue;
    }

    int tri_base = triangle_idx * 3;
    int ix = indices_ptr[tri_base + 0];
    int iy = indices_ptr[tri_base + 1];
    int iz = indices_ptr[tri_base + 2];

    // read vertex attributes
    float v1x = attr_ptr[3*ix + 0], v1y = attr_ptr[3*ix + 1], v1z = attr_ptr[3*ix + 2];
    float v2x = attr_ptr[3*iy + 0], v2y = attr_ptr[3*iy + 1], v2z = attr_ptr[3*iy + 2];
    float v3x = attr_ptr[3*iz + 0], v3y = attr_ptr[3*iz + 1], v3z = attr_ptr[3*iz + 2];

    output_ptr[idx * 3 + 0] = v1x * b0 + v2x * b1 + v3x * b2;
    output_ptr[idx * 3 + 1] = v1y * b0 + v2y * b1 + v3y * b2;
    output_ptr[idx * 3 + 2] = v1z * b0 + v2z * b1 + v3z * b2;
  }
}

// Vector wrapper for interpolate:
// attr_flat: length = num_vertices * 3
// indices_flat: length = num_triangles * 3
// rast_flat: length = width * height * 4
// output vector will be filled into output_flat (preallocated size width*height*3)
void interpolate_cpu_host(std::vector<float>& output_flat,
                          const std::vector<float>& attr_flat,
                          const std::vector<int>& indices_flat,
                          const std::vector<float>& rast_flat,
                          const int width,
                          const int height) {
  if (output_flat.size() < static_cast<size_t>(width * height * 3)) return;
  if (rast_flat.size() < static_cast<size_t>(width * height * 4)) return;

  // Call pointer version directly
  interpolate_cpu_host(output_flat.data(),
                       attr_flat.data(),
                       indices_flat.data(),
                       static_cast<int>(indices_flat.size() / 3),
                       rast_flat.data(),
                       width,
                       height);
}


void hider() {
//cv::Mat dilate_fill_cv_exact_fixed(const cv::Mat &img_f32, const cv::Mat &mask_u8, int iterations)
//{
//  CV_Assert(img_f32.type() == CV_32FC3);
//  CV_Assert(mask_u8.type() == CV_8UC1);
//  const int H = img_f32.rows;
//  const int W = img_f32.cols;
//  CV_Assert(H >= 1 && W >= 1);
//
//  // out will be mutated each iteration (corresponds to oldImg in Python)
//  cv::Mat out = img_f32.clone();
//
//  // curMask as 0/1 uchar
//  cv::Mat curMask(H, W, CV_8UC1);
//  for (int y = 0; y < H; ++y) {
//      const uchar* src = mask_u8.ptr<uchar>(y);
//      uchar* dst = curMask.ptr<uchar>(y);
//      for (int x = 0; x < W; ++x) dst[x] = src[x] ? 1 : 0;
//  }
//
//  // Pre-allocate some structures reused per iteration
//  // We'll implement unfold manually: for kernel 3x3, stride=1, padding=0
//  const int kH = 3, kW = 3;
//  const int n_patch_y = H - kH + 1;
//  const int n_patch_x = W - kW + 1;
//  if (n_patch_y <= 0 || n_patch_x <= 0) return out; // trivial for tiny images
//
//  // Helper: compute mask_conv = conv2d(newMask, ones(3x3), padding=1)
//  auto compute_mask_conv = [&](const cv::Mat &m)->cv::Mat {
//      cv::Mat conv(H, W, CV_32F, cv::Scalar(0.0f));
//      for (int y = 0; y < H; ++y) {
//          for (int x = 0; x < W; ++x) {
//              int y0 = std::max(0, y-1), y1 = std::min(H-1, y+1);
//              int x0 = std::max(0, x-1), x1 = std::min(W-1, x+1);
//              float s = 0.0f;
//              for (int yy = y0; yy <= y1; ++yy) {
//                  const uchar* row = m.ptr<uchar>(yy);
//                  for (int xx = x0; xx <= x1; ++xx) s += (row[xx] ? 1.0f : 0.0f);
//              }
//              conv.ptr<float>(y)[x] = s;
//          }
//      }
//      return conv;
//  };
//
//  for (int it = 0; it < iterations; ++it) {
//      // 1) compute newMask via dilation with zero padding semantics (same as PyTorch max_pool2d with padding=1)
//      cv::Mat dilated;
//      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
//      cv::dilate(curMask, dilated, kernel, cv::Point(-1,-1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
//
//      // diff = dilated - curMask (uchar 0/1)
//      cv::Mat diff;
//      cv::subtract(dilated, curMask, diff);
//
//      if (cv::countNonZero(diff) == 0) break;
//
//      // Precompute mask_conv for normalization (float)
//      cv::Mat mask_conv = compute_mask_conv(dilated); // H x W float
//
//      // We'll build accum (CV_32FC3) and counts (CV_32F) by explicit fold:
//      cv::Mat accum(H, W, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
//      cv::Mat counts(H, W, CV_32F, cv::Scalar(0.0f));
//
//      // === Unfold loop: iterate over patch top-left (py,px) matching F.unfold with kernel=3x3, stride=1, padding=0 ===
//      // For each patch we compute:
//      //   - masked sum over mask_prev (curMask) inside patch
//      //   - patch_count = number of valid pixels (mask_prev==1)
//      //   - mean_color = sum / max(1, patch_count)
//      //   - for each local offset (dy,dx) in patch: if dilated at absolute (py+dy, px+dx) is 1, add mean_color to accum at that absolute pixel and increment counts at that pixel by 1.
//      //
//      // This exactly mirrors:
//      //   img_unfold = F.unfold(oldImg, (3,3))
//      //   mask_unfold = F.unfold(oldMask, (3,3))
//      //   mean_color = (img_unfold.sum(dim=2) / mask_unfold.sum(dim=2).clip(1)).unsqueeze(2)
//      //   fill_color = (mean_color * new_mask_unfold)
//      //   newImg = F.fold(fill_color, (H,W), (3,3)) / mask_conv.clamp(1)
//
//      for (int py = 0; py <= H - kH; ++py) {
//          for (int px = 0; px <= W - kW; ++px) {
//              // compute masked sum and count for this 3x3 patch (from curMask and out)
//              float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
//              int patch_count = 0;
//              for (int dy = 0; dy < kH; ++dy) {
//                  const uchar* mask_row = curMask.ptr<uchar>(py + dy);
//                  const cv::Vec3f* img_row = out.ptr<cv::Vec3f>(py + dy);
//                  for (int dx = 0; dx < kW; ++dx) {
//                      int ax = px + dx;
//                      if (mask_row[ax]) {
//                          const cv::Vec3f &pix = img_row[ax];
//                          sum0 += pix[0];
//                          sum1 += pix[1];
//                          sum2 += pix[2];
//                          ++patch_count;
//                      }
//                  }
//              }
//              float denom = (patch_count > 0) ? (float)patch_count : 1.0f;
//              float m0 = sum0 / denom;
//              float m1 = sum1 / denom;
//              float m2 = sum2 / denom;
//
//              // now fold: for each local (dy,dx) in the patch, if dilated at that absolute pixel is 1,
//              // add mean_color to accum[ay,ax] and increment counts[ay,ax] by 1.
//              for (int dy = 0; dy < kH; ++dy) {
//                  int ay = py + dy;
//                  const uchar* newmask_row = dilated.ptr<uchar>(ay);
//                  cv::Vec3f* accum_row = accum.ptr<cv::Vec3f>(ay);
//                  float* counts_row = counts.ptr<float>(ay);
//                  for (int dx = 0; dx < kW; ++dx) {
//                      int ax = px + dx;
//                      if (newmask_row[ax]) {
//                          accum_row[ax][0] += m0;
//                          accum_row[ax][1] += m1;
//                          accum_row[ax][2] += m2;
//                          counts_row[ax] += 1.0f;
//                      }
//                  }
//              }
//          }
//      }
//
//      // Now normalize: newImg = accum / mask_conv.clamp(min=1.0)
//      cv::Mat newImg(H, W, CV_32FC3, cv::Scalar(0.0f,0.0f,0.0f));
//      for (int y = 0; y < H; ++y) {
//          const cv::Vec3f* acc_row = accum.ptr<cv::Vec3f>(y);
//          const float* maskc_row = mask_conv.ptr<float>(y);
//          cv::Vec3f* new_row = newImg.ptr<cv::Vec3f>(y);
//          for (int x = 0; x < W; ++x) {
//              float denom_norm = maskc_row[x];
//              if (denom_norm < 1.0f) denom_norm = 1.0f;
//              new_row[x][0] = acc_row[x][0] / denom_norm;
//              new_row[x][1] = acc_row[x][1] / denom_norm;
//              new_row[x][2] = acc_row[x][2] / denom_norm;
//          }
//      }
//
//      // Apply only to newly-dilated pixels: out = lerp(old=out, new=newImg, alpha=diff)
//      for (int y = 0; y < H; ++y) {
//          const uchar* diff_row = diff.ptr<uchar>(y);
//          cv::Vec3f* out_row = out.ptr<cv::Vec3f>(y);
//          const cv::Vec3f* new_row = newImg.ptr<cv::Vec3f>(y);
//          for (int x = 0; x < W; ++x) {
//              if (diff_row[x]) {
//                  // torch.lerp(oldImg, newImg, diffMask) for diffMask==1 => newImg, for 0 => oldImg.
//                  out_row[x] = new_row[x];
//              }
//          }
//      }
//
//      // update mask for next iter
//      curMask = dilated;
//  } // iterations
//
//  return out;
//}
}

void compute_single_unfold_3x3_fast(
    const cv::Mat &mat,                 // CV_32FC3 HxW  OR CV_8UC1 HxW when mask==true
    std::vector<float> &mat_unfold,     // out: size (mask? 9 : 27) * N
    int &out_N,                         // out: number of patches (L)
    const bool mask
) {
    if (mask) {
        CV_Assert(mat.type() == CV_8UC1);
    } else {
        CV_Assert(mat.type() == CV_32FC3);
    }

    const int H = mat.rows;
    const int W = mat.cols;
    const int C = 3;
    const int kH = 3, kW = 3;
    const int n_py = H - kH + 1;
    const int n_px = W - kW + 1;
    if (n_py <= 0 || n_px <= 0) {
        out_N = 0;
        mat_unfold.clear();
        return;
    }
    const int N = n_py * n_px;
    out_N = N;

    const int patch_elems = kH * kW;            // 9
    const int mat_channels = C * patch_elems;   // 27
    const int num_elems = mask ? patch_elems : mat_channels;
    mat_unfold.assign((size_t)num_elems * (size_t)N, 0.0f);

    if (!mask) {
        // Colour case: CV_32FC3
        const int out_stride = N; // number of patches; each block has contiguous stride N
        for (int c = 0; c < C; ++c) {
            for (int ky = 0; ky < kH; ++ky) {
                for (int kx = 0; kx < kW; ++kx) {
                    const int block_idx = (c * kH + ky) * kW + kx;
                    float* dst = &mat_unfold[(size_t)block_idx * (size_t)out_stride];
                    // For each top-left py (0..n_py-1) we will copy n_px values in a row
                    for (int py = 0; py < n_py; ++py) {
                        const cv::Vec3f* src_row = mat.ptr<cv::Vec3f>(py + ky) + kx;
                        const int dst_row_offset = py * n_px;
                        for (int px = 0; px < n_px; ++px) {
                            dst[dst_row_offset + px] = src_row[px][c];
                        }
                    }
                }
            }
        }
    } else {
        // Mask case: CV_8UC1 -> output layout (patch_elems, N)
        const int out_stride = N;
        for (int ky = 0; ky < kH; ++ky) {
            for (int kx = 0; kx < kW; ++kx) {
                const int block_idx = ky * kW + kx;
                float* dst = &mat_unfold[(size_t)block_idx * (size_t)out_stride];
                for (int py = 0; py < n_py; ++py) {
                    const uchar* src_row = mat.ptr<uchar>(py + ky) + kx;
                    const int dst_row_offset = py * n_px;
                    for (int px = 0; px < n_px; ++px) {
                        dst[dst_row_offset + px] = src_row[px] ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }
}

//float* dilate_fill_cv(
//    const float* img_in,
//    const uint8_t* mask_in,
//    const int H, const int W,
//    const int iterations = 6
//) {
//    auto all_start = std::chrono::high_resolution_clock::now();
//    // Validate input array shapes and types
//    if (!img_in) throw std::runtime_error("img_in is null");
//    if (!mask_in) throw std::runtime_error("mask_in is null");
//    if (H <= 0 || W <= 0) throw std::runtime_error("invalid H or W");
//
//    const int C = 3;
//    const int P = 9;
//
//    // Wrap input data in cv::Mat without copying (rows, cols, type, data ptr)
//    // Note: OpenCV expects interleaved HWC float (CV_32FC3)
//    cv::Mat img_mat(H, W, CV_32FC3, const_cast<float*>(img_in));
//    // Mask: expect 0/1 or 0/255 but type must be uint8
//    cv::Mat mask_mat(H, W, CV_8UC1, const_cast<uint8_t*>(mask_in));
//
//    // cv::Mat oldMask(H, W, CV_8UC1, static_cast<void*>(mask_buf.ptr));// = mask_mat.clone();
//    cv::Mat oldMask = cv::Mat::zeros(H, W, CV_8UC1);
//    oldMask = mask_mat.clone();
//    CV_Assert(oldMask.type() == CV_8UC1);
//    // cv::Mat oldImg(H, W, CV_32FC3, static_cast<void*>(img_buf.ptr));// = img_mat.clone();
//    cv::Mat oldImg(H, W, CV_32FC3);
//    oldImg = img_mat.clone();
//
//    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
//    cv::Mat newMask;
//    cv::Mat mask_kernel = cv::Mat::ones(3, 3, CV_32F);
//
//    for (int i = 0; i < iterations; i++) {
//        // 1. max pool 2d
//        cv::dilate(oldMask, newMask, kernel,
//                cv::Point(-1,-1),   // anchor default
//                1,                  // iterations
//                cv::BORDER_CONSTANT,// zero padding semantics
//                cv::Scalar(0));     // pad value = 0
//
//        // 2. unfold
//        cv::Mat img_in_unfold = oldImg.clone();
//        cv::Mat new_mask_in_unfold = newMask.clone();
//        cv::Mat old_mask_in_unfold = oldMask.clone();
//        std::vector<float> img_unfold;   // will become size = 3*9*N
//        std::vector<float> new_mask_unfold;  // will become size = 9*N
//        std::vector<float> old_mask_unfold;  // will become size = 9*N
//        int N = 0;                       // number of patches (L)
//
//        auto start = std::chrono::high_resolution_clock::now();
//        compute_single_unfold_3x3_fast(img_in_unfold, img_unfold, N, false);
//        auto end = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> elapsed = end - start;
//        std::cout << "single_unfold img time: " << elapsed.count() << "s" << std::endl;
//
//        start = std::chrono::high_resolution_clock::now();
//        compute_single_unfold_3x3_fast(old_mask_in_unfold, old_mask_unfold, N, true);
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "single_unfold old_mask_in_unfold time: " << elapsed.count() << "s" << std::endl;
//
//        start = std::chrono::high_resolution_clock::now();
//        compute_single_unfold_3x3_fast(new_mask_in_unfold, new_mask_unfold, N, true);
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "single_unfold new_mask_in_unfold time: " << elapsed.count() << "s" << std::endl;
//
//        // 3. calculate mean color
//        std::vector<float> mean_color(3 * N, 0.0f);
//        start = std::chrono::high_resolution_clock::now();
//        #pragma omp parallel for schedule(static)
//        for (int l = 0; l < N; ++l) {
//            // mask_unfold.sum(dim=2)
//            float mask_sum = 0.0f;
//            for (int p = 0; p < 9; ++p) {
//                mask_sum += old_mask_unfold[p * N + l];
//            }
//            // .clip(1)
//            float denom = (mask_sum > 0.0f) ? mask_sum : 1.0f;
//            // img_unfold.sum(dim=2) / denom
//            for (int c = 0; c < 3; ++c) {
//                float s = 0.0f;
//                for (int p = 0; p < 9; ++p) {
//                    s += img_unfold[(c * 9 + p) * N + l];
//                }
//                mean_color[c * N + l] = s / denom;
//            }
//        }
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "mean_color time: " << elapsed.count() << "s" << std::endl;
//
//        // 4. fill color
//        std::vector<float> fill_color(C * P * N, 0.0f);
//        start = std::chrono::high_resolution_clock::now();
//        #pragma omp parallel for collapse(2) schedule(static)
//        for (int l = 0; l < N; ++l) {
//            for (int c = 0; c < C; ++c) {
//                float mc = mean_color[c * N + l];  // mean_color[0,c,0,l]
//                for (int p = 0; p < P; ++p) {
//                    float nm = new_mask_unfold[p * N + l]; // new_mask_unfold[0,0,p,l]
//                    fill_color[(c * P + p) * N + l] = mc * nm;
//                }
//            }
//        }
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "fill_color time: " << elapsed.count() << "s" << std::endl;
//
//        // 5. conv2d
//        // assume newMask_cv is CV_8UC1 or CV_32F (values 0/1), size HxW
//        cv::Mat newMask_f;
//        if (newMask.type() == CV_8U) {
//            newMask.convertTo(newMask_f, CV_32F); // 0/1 -> float
//        } else {
//            newMask_f = newMask;
//        }
//        // 3x3 ones kernel (float)
//        // cv::Mat mask_kernel = cv::Mat::ones(3, 3, CV_32F);
//        // filter2D with constant (zero) padding -> exactly conv2d with padding=1
//        cv::Mat mask_conv_f; // CV_32F HxW
//        start = std::chrono::high_resolution_clock::now();
//        cv::filter2D(
//            newMask_f,               // src
//            mask_conv_f,             // dst
//            CV_32F,                  // desired depth
//            mask_kernel,                  // kernel (3x3 ones)
//            cv::Point(-1, -1),       // anchor = center
//            0.0,                     // delta
//            cv::BORDER_CONSTANT      // borderType -> zero padding
//        );
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "filter2D time: " << elapsed.count() << "s" << std::endl;
//
//        // 6. fold
//        cv::Mat newImg_accum(H, W, CV_32FC3, cv::Scalar(0,0,0));
//        start = std::chrono::high_resolution_clock::now();
//        const int out_W = W - 2;
//        for (int l = 0; l < N; ++l) {
//            int py = l / out_W;
//            int px = l % out_W;
//            for (int c = 0; c < C; ++c) {
//                for (int p = 0; p < P; ++p) {
//                    int ky = p / 3;
//                    int kx = p % 3;
//                    int ay = py + ky;
//                    int ax = px + kx;
//                    float v = fill_color[(c * P + p) * N + l];
//                    newImg_accum.at<cv::Vec3f>(ay, ax)[c] += v;
//                }
//            }
//        }
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "newImg_accum time: " << elapsed.count() << "s" << std::endl;
//
//        cv::Mat newImg(H, W, CV_32FC3, cv::Scalar(0,0,0));
//        start = std::chrono::high_resolution_clock::now();
//        for (int y = 0; y < H; ++y) {
//            const float* mc = mask_conv_f.ptr<float>(y);
//            cv::Vec3f* dst = newImg.ptr<cv::Vec3f>(y);
//            const cv::Vec3f* src = newImg_accum.ptr<cv::Vec3f>(y);
//            for (int x = 0; x < W; ++x) {
//                float denom = mc[x] > 0.0f ? mc[x] : 1.0f;
//                dst[x][0] = src[x][0] / denom;
//                dst[x][1] = src[x][1] / denom;
//                dst[x][2] = src[x][2] / denom;
//            }
//        }
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "newImg time: " << elapsed.count() << "s" << std::endl;
//
//        // 7. diff and lerp
//        cv::Mat diffMask;
//        cv::subtract(newMask, oldMask, diffMask, cv::noArray(), CV_8U); // CV_8U result
//        // replace oldMask with newMask for next iteration
//        oldMask = newMask.clone(); // or oldMask = newMask; if you want shared ref
//
//        // if no pixels changed, skip the lerp
//        if (cv::countNonZero(diffMask) == 0) {
//            // oldImg unchanged
//        } else {
//            start = std::chrono::high_resolution_clock::now();
//            // convert diffMask to float and expand to 3 channels so it broadcasts over RGB
//            cv::Mat diffMaskF;
//            diffMask.convertTo(diffMaskF, CV_32F); // values 0.0 or 1.0
//            cv::Mat diff3;
//            {
//                cv::Mat ch[] = { diffMaskF, diffMaskF, diffMaskF };
//                cv::merge(ch, 3, diff3); // CV_32F, size HxW, 3 channels
//            }
//            // tmp = (newImg - oldImg)
//            cv::Mat tmp;
//            cv::subtract(newImg, oldImg, tmp); // CV_32FC3
//            // tmp = diff3 * tmp  (elementwise)
//            cv::multiply(tmp, diff3, tmp); // CV_32FC3
//            // oldImg = oldImg + tmp  (in-place update)
//            cv::add(oldImg, tmp, oldImg); // oldImg now = oldImg + diff*(new-old)
//            end = std::chrono::high_resolution_clock::now();
//            elapsed = end - start;
//            std::cout << "merge,substract,multiply,add time: " << elapsed.count() << "s" << std::endl;
//
//        }
//
//        if (i == iterations - 1) {
//            // allocate output buffer: C * H * W floats
////            size_t out_size = static_cast<size_t>(C) * static_cast<size_t>(H) * static_cast<size_t>(W);
////            float* out_ptr = new float[out_size];
////            // fill buffer in channel-major order (CHW) — same indexing as your py return
////            for (int c = 0; c < C; ++c) {
////                for (int y = 0; y < H; ++y) {
////                    const cv::Vec3f* row = oldImg.ptr<cv::Vec3f>(y);
////                    for (int x = 0; x < W; ++x) {
////                        float v = row[x][c];
////                        size_t idx = (static_cast<size_t>(c) * static_cast<size_t>(H) + static_cast<size_t>(y)) * static_cast<size_t>(W) + static_cast<size_t>(x);
////                        out_ptr[idx] = v;
////                    }
////                }
////            }
//            // --- RETURN HWC buffer ---
//            size_t out_size = static_cast<size_t>(H) * static_cast<size_t>(W) * static_cast<size_t>(C);
//            float* out_ptr = new float[out_size];
//            for (int y = 0; y < H; ++y) {
//                const cv::Vec3f* row = oldImg.ptr<cv::Vec3f>(y);
//                for (int x = 0; x < W; ++x) {
//                    // store channels contiguous: (y * W + x) * C + c
//                    size_t base = (static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x)) * static_cast<size_t>(C);
//                    out_ptr[base + 0] = row[x][0];
//                    out_ptr[base + 1] = row[x][1];
//                    out_ptr[base + 2] = row[x][2];
//                }
//            }
//
//            auto all_end = std::chrono::high_resolution_clock::now();
//            std::chrono::duration<double> all_elapsed = all_end - all_start;
//            std::cout << "ALL time: " << all_elapsed.count() << "s" << std::endl;
//
//            return out_ptr;
//        }
//    } // end iterations
//     // if iterations <= 0 (shouldn't usually happen), return a copy of oldImg
////    size_t out_size = static_cast<size_t>(C) * static_cast<size_t>(H) * static_cast<size_t>(W);
////    float* out_ptr = new float[out_size];
////    for (int c = 0; c < C; ++c) {
////        for (int y = 0; y < H; ++y) {
////            const cv::Vec3f* row = oldImg.ptr<cv::Vec3f>(y);
////            for (int x = 0; x < W; ++x) {
////                float v = row[x][c];
////                size_t idx = (static_cast<size_t>(c) * static_cast<size_t>(H) + static_cast<size_t>(y)) * static_cast<size_t>(W) + static_cast<size_t>(x);
////                out_ptr[idx] = v;
////            }
////        }
////    }
//    size_t out_size = static_cast<size_t>(H) * static_cast<size_t>(W) * static_cast<size_t>(C);
//    float* out_ptr = new float[out_size];
//    for (int y = 0; y < H; ++y) {
//        const cv::Vec3f* row = oldImg.ptr<cv::Vec3f>(y);
//        for (int x = 0; x < W; ++x) {
//            size_t base = (static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x)) * static_cast<size_t>(C);
//            out_ptr[base + 0] = row[x][0];
//            out_ptr[base + 1] = row[x][1];
//            out_ptr[base + 2] = row[x][2];
//        }
//    }
//    auto all_end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> all_elapsed = all_end - all_start;
//    std::cout << "ALL time: " << all_elapsed.count() << "s" << std::endl;
//    return out_ptr;
//
//}
//
//cv::Mat dilate_fill_cv(
//    const float* img_in,
//    const uint8_t* mask_in,
//    const int H, const int W,
//    const int iterations = 6
//) {
//    auto all_start = std::chrono::high_resolution_clock::now();
//    // Validate input array shapes and types
//    if (!img_in) throw std::runtime_error("img_in is null");
//    if (!mask_in) throw std::runtime_error("mask_in is null");
//    if (H <= 0 || W <= 0) throw std::runtime_error("invalid H or W");
//
//    const int C = 3;
//    const int P = 9;
//
//    // Wrap input data in cv::Mat without copying (rows, cols, type, data ptr)
//    // Note: OpenCV expects interleaved HWC float (CV_32FC3)
//    cv::Mat img_mat(H, W, CV_32FC3, const_cast<float*>(img_in));
//    // Mask: expect 0/1 or 0/255 but type must be uint8
//    cv::Mat mask_mat(H, W, CV_8UC1, const_cast<uint8_t*>(mask_in));
//
//    // cv::Mat oldMask(H, W, CV_8UC1, static_cast<void*>(mask_buf.ptr));// = mask_mat.clone();
//    cv::Mat oldMask = cv::Mat::zeros(H, W, CV_8UC1);
//    oldMask = mask_mat.clone();
//    CV_Assert(oldMask.type() == CV_8UC1);
//    // cv::Mat oldImg(H, W, CV_32FC3, static_cast<void*>(img_buf.ptr));// = img_mat.clone();
//    cv::Mat oldImg(H, W, CV_32FC3);
//    oldImg = img_mat.clone();
//
//    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
//    cv::Mat newMask;
//    cv::Mat mask_kernel = cv::Mat::ones(3, 3, CV_32F);
//
//    for (int i = 0; i < iterations; i++) {
//        // 1. max pool 2d
//        cv::dilate(oldMask, newMask, kernel,
//                cv::Point(-1,-1),   // anchor default
//                1,                  // iterations
//                cv::BORDER_CONSTANT,// zero padding semantics
//                cv::Scalar(0));     // pad value = 0
//
//        // 2. unfold
//        cv::Mat img_in_unfold = oldImg.clone();
//        cv::Mat new_mask_in_unfold = newMask.clone();
//        cv::Mat old_mask_in_unfold = oldMask.clone();
//        std::vector<float> img_unfold;   // will become size = 3*9*N
//        std::vector<float> new_mask_unfold;  // will become size = 9*N
//        std::vector<float> old_mask_unfold;  // will become size = 9*N
//        int N = 0;                       // number of patches (L)
//
//        auto start = std::chrono::high_resolution_clock::now();
//        compute_single_unfold_3x3_fast(img_in_unfold, img_unfold, N, false);
//        auto end = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> elapsed = end - start;
//        std::cout << "single_unfold img time: " << elapsed.count() << "s" << std::endl;
//
//        start = std::chrono::high_resolution_clock::now();
//        compute_single_unfold_3x3_fast(old_mask_in_unfold, old_mask_unfold, N, true);
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "single_unfold old_mask_in_unfold time: " << elapsed.count() << "s" << std::endl;
//
//        start = std::chrono::high_resolution_clock::now();
//        compute_single_unfold_3x3_fast(new_mask_in_unfold, new_mask_unfold, N, true);
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "single_unfold new_mask_in_unfold time: " << elapsed.count() << "s" << std::endl;
//
//        // 3. calculate mean color
//        std::vector<float> mean_color(3 * N, 0.0f);
//        start = std::chrono::high_resolution_clock::now();
//        #pragma omp parallel for schedule(static)
//        for (int l = 0; l < N; ++l) {
//            // mask_unfold.sum(dim=2)
//            float mask_sum = 0.0f;
//            for (int p = 0; p < 9; ++p) {
//                mask_sum += old_mask_unfold[p * N + l];
//            }
//            // .clip(1)
//            float denom = (mask_sum > 0.0f) ? mask_sum : 1.0f;
//            // img_unfold.sum(dim=2) / denom
//            for (int c = 0; c < 3; ++c) {
//                float s = 0.0f;
//                for (int p = 0; p < 9; ++p) {
//                    s += img_unfold[(c * 9 + p) * N + l];
//                }
//                mean_color[c * N + l] = s / denom;
//            }
//        }
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "mean_color time: " << elapsed.count() << "s" << std::endl;
//
//        // 4. fill color
//        std::vector<float> fill_color(C * P * N, 0.0f);
//        start = std::chrono::high_resolution_clock::now();
//        #pragma omp parallel for collapse(2) schedule(static)
//        for (int l = 0; l < N; ++l) {
//            for (int c = 0; c < C; ++c) {
//                float mc = mean_color[c * N + l];  // mean_color[0,c,0,l]
//                for (int p = 0; p < P; ++p) {
//                    float nm = new_mask_unfold[p * N + l]; // new_mask_unfold[0,0,p,l]
//                    fill_color[(c * P + p) * N + l] = mc * nm;
//                }
//            }
//        }
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "fill_color time: " << elapsed.count() << "s" << std::endl;
//
//        // 5. conv2d
//        // assume newMask_cv is CV_8UC1 or CV_32F (values 0/1), size HxW
//        cv::Mat newMask_f;
//        if (newMask.type() == CV_8U) {
//            newMask.convertTo(newMask_f, CV_32F); // 0/1 -> float
//        } else {
//            newMask_f = newMask;
//        }
//        // 3x3 ones kernel (float)
//        // cv::Mat mask_kernel = cv::Mat::ones(3, 3, CV_32F);
//        // filter2D with constant (zero) padding -> exactly conv2d with padding=1
//        cv::Mat mask_conv_f; // CV_32F HxW
//        start = std::chrono::high_resolution_clock::now();
//        cv::filter2D(
//            newMask_f,               // src
//            mask_conv_f,             // dst
//            CV_32F,                  // desired depth
//            mask_kernel,                  // kernel (3x3 ones)
//            cv::Point(-1, -1),       // anchor = center
//            0.0,                     // delta
//            cv::BORDER_CONSTANT      // borderType -> zero padding
//        );
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "filter2D time: " << elapsed.count() << "s" << std::endl;
//
//        // 6. fold
//        cv::Mat newImg_accum(H, W, CV_32FC3, cv::Scalar(0,0,0));
//        start = std::chrono::high_resolution_clock::now();
//        const int out_W = W - 2;
//        for (int l = 0; l < N; ++l) {
//            int py = l / out_W;
//            int px = l % out_W;
//            for (int c = 0; c < C; ++c) {
//                for (int p = 0; p < P; ++p) {
//                    int ky = p / 3;
//                    int kx = p % 3;
//                    int ay = py + ky;
//                    int ax = px + kx;
//                    float v = fill_color[(c * P + p) * N + l];
//                    newImg_accum.at<cv::Vec3f>(ay, ax)[c] += v;
//                }
//            }
//        }
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "newImg_accum time: " << elapsed.count() << "s" << std::endl;
//
//        cv::Mat newImg(H, W, CV_32FC3, cv::Scalar(0,0,0));
//        start = std::chrono::high_resolution_clock::now();
//        for (int y = 0; y < H; ++y) {
//            const float* mc = mask_conv_f.ptr<float>(y);
//            cv::Vec3f* dst = newImg.ptr<cv::Vec3f>(y);
//            const cv::Vec3f* src = newImg_accum.ptr<cv::Vec3f>(y);
//            for (int x = 0; x < W; ++x) {
//                float denom = mc[x] > 0.0f ? mc[x] : 1.0f;
//                dst[x][0] = src[x][0] / denom;
//                dst[x][1] = src[x][1] / denom;
//                dst[x][2] = src[x][2] / denom;
//            }
//        }
//        end = std::chrono::high_resolution_clock::now();
//        elapsed = end - start;
//        std::cout << "newImg time: " << elapsed.count() << "s" << std::endl;
//
//        // 7. diff and lerp
//        cv::Mat diffMask;
//        cv::subtract(newMask, oldMask, diffMask, cv::noArray(), CV_8U); // CV_8U result
//        // replace oldMask with newMask for next iteration
//        oldMask = newMask.clone(); // or oldMask = newMask; if you want shared ref
//
//        // if no pixels changed, skip the lerp
//        if (cv::countNonZero(diffMask) == 0) {
//            // oldImg unchanged
//        } else {
//            start = std::chrono::high_resolution_clock::now();
//            // convert diffMask to float and expand to 3 channels so it broadcasts over RGB
//            cv::Mat diffMaskF;
//            diffMask.convertTo(diffMaskF, CV_32F); // values 0.0 or 1.0
//            cv::Mat diff3;
//            {
//                cv::Mat ch[] = { diffMaskF, diffMaskF, diffMaskF };
//                cv::merge(ch, 3, diff3); // CV_32F, size HxW, 3 channels
//            }
//            // tmp = (newImg - oldImg)
//            cv::Mat tmp;
//            cv::subtract(newImg, oldImg, tmp); // CV_32FC3
//            // tmp = diff3 * tmp  (elementwise)
//            cv::multiply(tmp, diff3, tmp); // CV_32FC3
//            // oldImg = oldImg + tmp  (in-place update)
//            cv::add(oldImg, tmp, oldImg); // oldImg now = oldImg + diff*(new-old)
//            end = std::chrono::high_resolution_clock::now();
//            elapsed = end - start;
//            std::cout << "merge,substract,multiply,add time: " << elapsed.count() << "s" << std::endl;
//
//        }
//
//    } // end iterations
//
//    auto all_end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> all_elapsed = all_end - all_start;
//    std::cout << "ALL time: " << all_elapsed.count() << "s" << std::endl;
//    return oldImg;
//
//}

cv::Mat float32_to_uint8_cv(const cv::Mat &img_f32, bool dither=false, const cv::Mat &dither_mask = cv::Mat()) {
    CV_Assert(img_f32.type() == CV_32FC3);
    cv::Mat out(img_f32.rows, img_f32.cols, CV_8UC3);
    if (!dither) {
        cv::Mat tmp;
        img_f32.convertTo(tmp, CV_32FC3, 256.0);
        for (int r = 0; r < tmp.rows; ++r) {
            const float* src = tmp.ptr<float>(r);
            uchar* dst = out.ptr<uchar>(r);
            for (int c = 0; c < tmp.cols; ++c) {
                for (int ch = 0; ch < 3; ++ch) {
                    float v = src[c*3 + ch];
                    int vi = static_cast<int>(std::floor(v));
                    if (vi < 0) vi = 0;
                    if (vi > 255) vi = 255;
                    dst[c*3 + ch] = static_cast<uchar>(vi);
                }
            }
        }
        return out;
    } else {
        std::mt19937 rng((unsigned)std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (int r = 0; r < img_f32.rows; ++r) {
            const cv::Vec3f* src = img_f32.ptr<cv::Vec3f>(r);
            uchar* dst = out.ptr<uchar>(r);
            for (int c = 0; c < img_f32.cols; ++c) {
                float noise = dist(rng);
                if (!dither_mask.empty()) {
                    float mask_val = dither_mask.at<float>(r,c);
                    if (mask_val < 0.5f) noise = 0.0f;
                }
                for (int ch = 0; ch < 3; ++ch) {
                    float v = src[c][ch] * 256.0f + noise;
                    int vi = static_cast<int>(std::floor(v));
                    if (vi < 0) vi = 0;
                    if (vi > 255) vi = 255;
                    dst[c*3 + ch] = static_cast<uchar>(vi);
                }
            }
        }
        return out;
    }
}

void normalize_rows_inplace(float* data, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        float x = data[3*i + 0];
        float y = data[3*i + 1];
        float z = data[3*i + 2];
        float len = std::sqrt(x*x + y*y + z*z);
        if (len > 1e-9f) {
            data[3*i + 0] = x / len;
            data[3*i + 1] = y / len;
            data[3*i + 2] = z / len;
        } else {
            data[3*i + 0] = 0.0f;
            data[3*i + 1] = 0.0f;
            data[3*i + 2] = 1.0f;
        }
    }
}

void cross_rows(const float* a, const float* b, float* out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        float ax = a[3*i + 0], ay = a[3*i + 1], az = a[3*i + 2];
        float bx = b[3*i + 0], by = b[3*i + 1], bz = b[3*i + 2];
        out[3*i + 0] = ay * bz - az * by;
        out[3*i + 1] = az * bx - ax * bz;
        out[3*i + 2] = ax * by - ay * bx;
    }
}


cv::Mat dilate_fill_cv(
    const cv::Mat img_in,
    const cv::Mat mask_in,
    const int H, const int W,
    const int iterations
) {
    auto all_start = std::chrono::high_resolution_clock::now();
    // Validate input array shapes and types
//    if (!img_in) throw std::runtime_error("img_in is null");
//    if (!mask_in) throw std::runtime_error("mask_in is null");
//    if (H <= 0 || W <= 0) throw std::runtime_error("invalid H or W");

    const int C = 3;
    const int P = 9;

    // Wrap input data in cv::Mat without copying (rows, cols, type, data ptr)
    // Note: OpenCV expects interleaved HWC float (CV_32FC3)
//    cv::Mat img_mat(H, W, CV_32FC3, const_cast<float*>(img_in));
//    // Mask: expect 0/1 or 0/255 but type must be uint8
//    cv::Mat mask_mat(H, W, CV_8UC1, const_cast<uint8_t*>(mask_in));

    // cv::Mat oldMask(H, W, CV_8UC1, static_cast<void*>(mask_buf.ptr));// = mask_mat.clone();
    cv::Mat oldMask = cv::Mat::zeros(H, W, CV_8UC1);
    oldMask = mask_in.clone();
    CV_Assert(oldMask.type() == CV_8UC1);
    // cv::Mat oldImg(H, W, CV_32FC3, static_cast<void*>(img_buf.ptr));// = img_mat.clone();
    cv::Mat oldImg(H, W, CV_32FC3);
    oldImg = img_in.clone();

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::Mat newMask;
    cv::Mat mask_kernel = cv::Mat::ones(3, 3, CV_32F);

    for (int i = 0; i < iterations; i++) {
        // 1. max pool 2d
        cv::dilate(oldMask, newMask, kernel,
                cv::Point(-1,-1),   // anchor default
                1,                  // iterations
                cv::BORDER_CONSTANT,// zero padding semantics
                cv::Scalar(0));     // pad value = 0

        // 2. unfold
        cv::Mat img_in_unfold = oldImg.clone();
        cv::Mat new_mask_in_unfold = newMask.clone();
        cv::Mat old_mask_in_unfold = oldMask.clone();
        std::vector<float> img_unfold;   // will become size = 3*9*N
        std::vector<float> new_mask_unfold;  // will become size = 9*N
        std::vector<float> old_mask_unfold;  // will become size = 9*N
        int N = 0;                       // number of patches (L)

        auto start = std::chrono::high_resolution_clock::now();
        compute_single_unfold_3x3_fast(img_in_unfold, img_unfold, N, false);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "single_unfold img time: " << elapsed.count() << "s" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        compute_single_unfold_3x3_fast(old_mask_in_unfold, old_mask_unfold, N, true);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "single_unfold old_mask_in_unfold time: " << elapsed.count() << "s" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        compute_single_unfold_3x3_fast(new_mask_in_unfold, new_mask_unfold, N, true);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "single_unfold new_mask_in_unfold time: " << elapsed.count() << "s" << std::endl;

        // 3. calculate mean color
        std::vector<float> mean_color(3 * N, 0.0f);
        start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(static)
        for (int l = 0; l < N; ++l) {
            // mask_unfold.sum(dim=2)
            float mask_sum = 0.0f;
            for (int p = 0; p < 9; ++p) {
                mask_sum += old_mask_unfold[p * N + l];
            }
            // .clip(1)
            float denom = (mask_sum > 0.0f) ? mask_sum : 1.0f;
            // img_unfold.sum(dim=2) / denom
            for (int c = 0; c < 3; ++c) {
                float s = 0.0f;
                for (int p = 0; p < 9; ++p) {
                    s += img_unfold[(c * 9 + p) * N + l];
                }
                mean_color[c * N + l] = s / denom;
            }
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "mean_color time: " << elapsed.count() << "s" << std::endl;

        // 4. fill color
        std::vector<float> fill_color(C * P * N, 0.0f);
        start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int l = 0; l < N; ++l) {
            for (int c = 0; c < C; ++c) {
                float mc = mean_color[c * N + l];  // mean_color[0,c,0,l]
                for (int p = 0; p < P; ++p) {
                    float nm = new_mask_unfold[p * N + l]; // new_mask_unfold[0,0,p,l]
                    fill_color[(c * P + p) * N + l] = mc * nm;
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "fill_color time: " << elapsed.count() << "s" << std::endl;

        // 5. conv2d
        // assume newMask_cv is CV_8UC1 or CV_32F (values 0/1), size HxW
        cv::Mat newMask_f;
        if (newMask.type() == CV_8U) {
            newMask.convertTo(newMask_f, CV_32F); // 0/1 -> float
        } else {
            newMask_f = newMask;
        }
        // 3x3 ones kernel (float)
        // cv::Mat mask_kernel = cv::Mat::ones(3, 3, CV_32F);
        // filter2D with constant (zero) padding -> exactly conv2d with padding=1
        cv::Mat mask_conv_f; // CV_32F HxW
        start = std::chrono::high_resolution_clock::now();
        cv::filter2D(
            newMask_f,               // src
            mask_conv_f,             // dst
            CV_32F,                  // desired depth
            mask_kernel,                  // kernel (3x3 ones)
            cv::Point(-1, -1),       // anchor = center
            0.0,                     // delta
            cv::BORDER_CONSTANT      // borderType -> zero padding
        );
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "filter2D time: " << elapsed.count() << "s" << std::endl;

        // 6. fold
        cv::Mat newImg_accum(H, W, CV_32FC3, cv::Scalar(0,0,0));
        start = std::chrono::high_resolution_clock::now();
        const int out_W = W - 2;
        for (int l = 0; l < N; ++l) {
            int py = l / out_W;
            int px = l % out_W;
            for (int c = 0; c < C; ++c) {
                for (int p = 0; p < P; ++p) {
                    int ky = p / 3;
                    int kx = p % 3;
                    int ay = py + ky;
                    int ax = px + kx;
                    float v = fill_color[(c * P + p) * N + l];
                    newImg_accum.at<cv::Vec3f>(ay, ax)[c] += v;
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "newImg_accum time: " << elapsed.count() << "s" << std::endl;

        cv::Mat newImg(H, W, CV_32FC3, cv::Scalar(0,0,0));
        start = std::chrono::high_resolution_clock::now();
        for (int y = 0; y < H; ++y) {
            const float* mc = mask_conv_f.ptr<float>(y);
            cv::Vec3f* dst = newImg.ptr<cv::Vec3f>(y);
            const cv::Vec3f* src = newImg_accum.ptr<cv::Vec3f>(y);
            for (int x = 0; x < W; ++x) {
                float denom = mc[x] > 0.0f ? mc[x] : 1.0f;
                dst[x][0] = src[x][0] / denom;
                dst[x][1] = src[x][1] / denom;
                dst[x][2] = src[x][2] / denom;
            }
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "newImg time: " << elapsed.count() << "s" << std::endl;

        // 7. diff and lerp
        cv::Mat diffMask;
        cv::subtract(newMask, oldMask, diffMask, cv::noArray(), CV_8U); // CV_8U result
        // replace oldMask with newMask for next iteration
        oldMask = newMask.clone(); // or oldMask = newMask; if you want shared ref

        // if no pixels changed, skip the lerp
        if (cv::countNonZero(diffMask) == 0) {
            // oldImg unchanged
        } else {
            start = std::chrono::high_resolution_clock::now();
            // convert diffMask to float and expand to 3 channels so it broadcasts over RGB
            cv::Mat diffMaskF;
            diffMask.convertTo(diffMaskF, CV_32F); // values 0.0 or 1.0
            cv::Mat diff3;
            {
                cv::Mat ch[] = { diffMaskF, diffMaskF, diffMaskF };
                cv::merge(ch, 3, diff3); // CV_32F, size HxW, 3 channels
            }
            // tmp = (newImg - oldImg)
            cv::Mat tmp;
            cv::subtract(newImg, oldImg, tmp); // CV_32FC3
            // tmp = diff3 * tmp  (elementwise)
            cv::multiply(tmp, diff3, tmp); // CV_32FC3
            // oldImg = oldImg + tmp  (in-place update)
            cv::add(oldImg, tmp, oldImg); // oldImg now = oldImg + diff*(new-old)
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "merge,substract,multiply,add time: " << elapsed.count() << "s" << std::endl;

        }

    } // end iterations

    auto all_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> all_elapsed = all_end - all_start;
    std::cout << "ALL time: " << all_elapsed.count() << "s" << std::endl;
    return oldImg;
}

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
    bool save_as_jpeg
) {
    BuildTexturesInspect inspect;
    inspect.H = height;
    inspect.W = width;
    inspect.roughness = roughness;
    inspect.metallic = metallic;

    if (!rast_ptr || !bake_mask_ptr) {
        LOGE("rast or bake_mask pointers null");
        throw std::runtime_error("rast or bake_mask pointers null");
    }
    if (width <= 0 || height <= 0) {
        LOGE("invalid dims");
        throw std::runtime_error("invalid dims");
    }
    const int W = width, H = height;
    const int num_pixels = W * H;

    cv::Mat rast_mat(H, W, CV_32FC4, const_cast<float*>(rast_ptr));
    cv::Mat mask_u8(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        uchar* row = mask_u8.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            row[x] = (bake_mask_ptr[idx] != 0) ? 1 : 0;
        }
    }

    std::vector<int> mask_indices;
    mask_indices.reserve(num_pixels);
    for (int i = 0; i < num_pixels; ++i) {
        if (bake_mask_ptr[i]) mask_indices.push_back(i);
    }
    size_t mask_count = mask_indices.size();
    inspect.mask_indices = mask_indices;

    // Step 1: interpolate normals and tangents into flat buffers
    std::vector<float> nrm_flat(num_pixels * 3, 0.0f);
    std::vector<float> tng_flat(num_pixels * 3, 0.0f);

    int num_triangles = static_cast<int>(t_pos_idx_count / 3);
    // call your pointer-based interpolate (assumed in texture_baker_cpp)
    if (v_nrm && v_nrm_count >= 3) {
        interpolate_cpu_host(nrm_flat.data(), v_nrm, t_pos_idx, num_triangles, rast_ptr, W, H);
    } else {
        LOGE("v_nrm not provided or invalid");
        throw std::runtime_error("v_nrm not provided or invalid");
    }

    if (v_tng && v_tng_count >= 3) {
        interpolate_cpu_host(tng_flat.data(), v_tng, t_pos_idx, num_triangles, rast_ptr, W, H);
    } else {
        std::fill(tng_flat.begin(), tng_flat.end(), 0.0f);
    }

    // Step 2: gather gb_nrm and normalize
    std::vector<float> gb_nrm(3 * mask_count);
    for (size_t k = 0; k < mask_count; ++k) {
        int idx = mask_indices[k];
        gb_nrm[3*k + 0] = nrm_flat[3*idx + 0];
        gb_nrm[3*k + 1] = nrm_flat[3*idx + 1];
        gb_nrm[3*k + 2] = nrm_flat[3*idx + 2];
    }
    normalize_rows_inplace(gb_nrm.data(), mask_count);
    inspect.gb_nrm = gb_nrm;

    // Step 3: build cv::Mat f_albedo and f_bump and fill
    cv::Mat f_albedo(H, W, CV_32FC3, cv::Scalar(0,0,0));
    cv::Mat f_bump(H, W, CV_32FC3, cv::Scalar(0,0,0));

    if (features_masked) {
        for (size_t k = 0; k < mask_count; ++k) {
            int idx = mask_indices[k];
            int y = idx / W; int x = idx % W;
            cv::Vec3f val(features_masked[3*k + 0], features_masked[3*k + 1], features_masked[3*k + 2]);
            f_albedo.at<cv::Vec3f>(y,x) = val;
        }
    }

    // Normal / bump handling:
    if (perturb_normal_masked) {
        std::vector<float> gb_tng(3 * mask_count);
        for (size_t k = 0; k < mask_count; ++k) {
            int idx = mask_indices[k];
            gb_tng[3*k + 0] = tng_flat[3*idx + 0];
            gb_tng[3*k + 1] = tng_flat[3*idx + 1];
            gb_tng[3*k + 2] = tng_flat[3*idx + 2];
        }
        normalize_rows_inplace(gb_tng.data(), mask_count);
        inspect.gb_tng = gb_tng;

        std::vector<float> gb_btng(3 * mask_count);
        cross_rows(gb_nrm.data(), gb_tng.data(), gb_btng.data(), mask_count);
        normalize_rows_inplace(gb_btng.data(), mask_count);
        inspect.gb_btng = gb_btng;

        std::vector<float> normal_masked(3 * mask_count);
        for (size_t k = 0; k < mask_count; ++k) {
            normal_masked[3*k + 0] = perturb_normal_masked[3*k + 0];
            normal_masked[3*k + 1] = perturb_normal_masked[3*k + 1];
            normal_masked[3*k + 2] = perturb_normal_masked[3*k + 2];
        }
        normalize_rows_inplace(normal_masked.data(), mask_count);

        std::vector<float> normal_tangent(3 * mask_count);
        for (size_t k = 0; k < mask_count; ++k) {
            Eigen::Matrix3f T;
            T.col(0) = Eigen::Vector3f(gb_tng[3*k + 0], gb_tng[3*k + 1], gb_tng[3*k + 2]);
            T.col(1) = Eigen::Vector3f(gb_btng[3*k + 0], gb_btng[3*k + 1], gb_btng[3*k + 2]);
            T.col(2) = Eigen::Vector3f(gb_nrm[3*k + 0], gb_nrm[3*k + 1], gb_nrm[3*k + 2]);

            Eigen::Vector3f n_local(normal_masked[3*k + 0], normal_masked[3*k + 1], normal_masked[3*k + 2]);
            Eigen::Vector3f n_world = T.transpose() * n_local;
            n_world = (n_world * 0.5f + Eigen::Vector3f::Ones() * 0.5f).cwiseMax(0.0f).cwiseMin(1.0f);
            normal_tangent[3*k + 0] = n_world[0];
            normal_tangent[3*k + 1] = n_world[1];
            normal_tangent[3*k + 2] = n_world[2];
            int idx = mask_indices[k]; int y = idx / W; int x = idx % W;
            f_bump.at<cv::Vec3f>(y,x) = cv::Vec3f(n_world[0], n_world[1], n_world[2]);
        }
        inspect.normal_tangent = normal_tangent;
    } else {
        for (size_t k = 0; k < mask_count; ++k) {
            int idx = mask_indices[k];
            int y = idx / W, x = idx % W;
            float nx = nrm_flat[3*idx + 0], ny = nrm_flat[3*idx + 1], nz = nrm_flat[3*idx + 2];
            float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len > 1e-9f) { nx/=len; ny/=len; nz/=len; }
            nx = std::min(std::max(nx * 0.5f + 0.5f, 0.0f), 1.0f);
            ny = std::min(std::max(ny * 0.5f + 0.5f, 0.0f), 1.0f);
            nz = std::min(std::max(nz * 0.5f + 0.5f, 0.0f), 1.0f);
            f_bump.at<cv::Vec3f>(y,x) = cv::Vec3f(nx, ny, nz);
        }
    }

    // Step 4: dilate-fill
    int iterations = std::max(1, W / 150);
//    cv::Mat f_albedo_bgr;
//    cv::cvtColor(f_albedo, f_albedo_bgr, cv::COLOR_RGB2BGR);
    cv::Mat f_albedo_filled = dilate_fill_cv(f_albedo, mask_u8, H, W, iterations);
//    cv::Mat f_albedo_filled;
//    cv::cvtColor(f_albedo_filled_bgr, f_albedo_filled, cv::COLOR_BGR2RGB);
    cv::Mat f_bump_filled = dilate_fill_cv(f_bump, mask_u8, H, W, iterations);
    // create the "bump_up" reference (same shape as f_bump_filled)
    cv::Mat bump_up(f_bump_filled.rows, f_bump_filled.cols, CV_32FC3, cv::Scalar(0.5f, 0.5f, 1.0f));
    // compute dither_mask: true where bump_np equals bump_up (per-channel)
    cv::Mat equal_mask(f_bump_filled.rows, f_bump_filled.cols, CV_8UC1, cv::Scalar(0));
    for (int y=0; y<f_bump_filled.rows; ++y) {
        const cv::Vec3f* brow = f_bump_filled.ptr<cv::Vec3f>(y);
        uchar* mout = equal_mask.ptr<uchar>(y);
        for (int x=0; x<f_bump_filled.cols; ++x) {
            bool all_eq = true;
            for (int c=0; c<3; ++c) {
                // careful with float equality: Python used exact equality after conversion
                // It is okay to test near-equality here; we use a tiny epsilon.
                float a = brow[x][c];
                float target = (c < 2) ? 0.5f : 1.0f;
                if (std::abs(a - target) > 1e-6f) { all_eq = false; break; }
            }
            mout[x] = all_eq ? 1 : 0;
        }
    }
    // convert equal_mask to float dither_mask expected by float32_to_uint8_cv
    cv::Mat dither_mask_float;
    equal_mask.convertTo(dither_mask_float, CV_32F, 1.0/*/255.0*/); // if equal_mask is 0/255
    // If equal_mask is 0/1 already, convertTo will produce 0/1 floats directly.


    // Step 5: save images if requested
    try {
        cv::Mat basecolor_u8 = float32_to_uint8_cv(f_albedo_filled, false);
//        cv::Mat bump_u8 = float32_to_uint8_cv(f_bump_filled, false);
        cv::Mat bump_u8 = float32_to_uint8_cv(f_bump_filled, /*dither=*/true, dither_mask_float);

//        cv::Mat base_bgr, bump_bgr;
//        cv::cvtColor(basecolor_u8, base_bgr, cv::COLOR_RGB2BGR);
//        cv::cvtColor(bump_u8, bump_bgr, cv::COLOR_RGB2BGR);

        if (save_as_jpeg) {
            std::vector<int> jparams = {cv::IMWRITE_JPEG_QUALITY, 95};
            if (!cv::imwrite(basecolor_outpath, basecolor_u8, jparams)) {
                LOGE("Failed to write %s", basecolor_outpath.c_str());
            }
            if (!cv::imwrite(bump_outpath, bump_u8, jparams)) {
                LOGE("Failed to write %s", bump_outpath.c_str());
            }
        } else {
            cv::imwrite(basecolor_outpath, basecolor_u8);
            cv::imwrite(bump_outpath, bump_u8);
        }
        std::vector<uchar> jpg_buf, bsc_buf;
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 };
        cv::imencode(".jpg", bump_u8, jpg_buf, params); // jpg_buf contains bytes
        std::string bump_jpg_bytes(reinterpret_cast<char*>(jpg_buf.data()), jpg_buf.size());
        inspect.bump_jpeg_bytes.assign(reinterpret_cast<char*>(jpg_buf.data()), jpg_buf.size());
//        inspect.bump_path = bump_outpath;

        cv::imencode(".jpg", basecolor_u8, bsc_buf, params); // jpg_buf contains bytes
        std::string bsc_jpg_bytes(reinterpret_cast<char*>(bsc_buf.data()), bsc_buf.size());
        inspect.basecolor_jpeg_bytes.assign(reinterpret_cast<char*>(bsc_buf.data()), bsc_buf.size());
//        inspect.basecolor_path = basecolor_outpath;

    } catch (const std::exception &e) {
        LOGE("Exception while saving images: %s", e.what());
    }

    // Fill inspect basecolor and bump as float32 linear arrays (H*W*3)
    inspect.basecolor.resize(num_pixels * 3);
    inspect.bump.resize(num_pixels * 3);
    inspect.bump_filled.resize(num_pixels * 3);
    inspect.albedo.resize(num_pixels * 3);
    for (int y = 0; y < H; ++y) {
        const cv::Vec3f* rowA = f_albedo_filled.ptr<cv::Vec3f>(y);
        const cv::Vec3f* rowB = f_bump_filled.ptr<cv::Vec3f>(y);
        const cv::Vec3f* rowC = f_albedo.ptr<cv::Vec3f>(y);
        const cv::Vec3f* rowD = f_bump.ptr<cv::Vec3f>(y);
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            inspect.basecolor[3*idx + 0] = rowA[x][0];
            inspect.basecolor[3*idx + 1] = rowA[x][1];
            inspect.basecolor[3*idx + 2] = rowA[x][2];
            inspect.bump_filled[3*idx + 0] = rowB[x][0];
            inspect.bump_filled[3*idx + 1] = rowB[x][1];
            inspect.bump_filled[3*idx + 2] = rowB[x][2];

            inspect.bump[3*idx + 0] = rowD[x][0];
            inspect.bump[3*idx + 1] = rowD[x][1];
            inspect.bump[3*idx + 2] = rowD[x][2];

            inspect.albedo[3*idx + 0] = rowC[x][0];
            inspect.albedo[3*idx + 1] = rowC[x][1];
            inspect.albedo[3*idx + 2] = rowC[x][2];
        }
    }

    return inspect;
}

// Helper: convert float image (H*W*3 floats in [0,1]) -> PNG bytes (vector<unsigned char>)
static bool encode_image_png_from_float3(const std::vector<float>& float_img, int H, int W, std::vector<unsigned char>& out_png, bool use_bgr=false) {
    if ((int)float_img.size() != H * W * 3) {
        std::cerr << "encode_image_png_from_float3: size mismatch\n";
        return false;
    }
    // Build cv::Mat CV_8UC3 (0..255) from floats
    cv::Mat mat(H, W, CV_8UC3);
    for (int y = 0; y < H; ++y) {
        uchar* row = mat.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            int idx = (y * W + x) * 3;
            // clamp & scale
            float r = float_img[idx + 0];
            float g = float_img[idx + 1];
            float b = float_img[idx + 2];
            auto clamp = [](float v)->int {
                int vi = static_cast<int>(std::floor(v * 255.0f + 0.5f));
                if (vi < 0) vi = 0;
                if (vi > 255) vi = 255;
                return vi;
            };
            int ri = clamp(r), gi = clamp(g), bi = clamp(b);
            if (!use_bgr) {
                row[x*3 + 0] = static_cast<uchar>(ri);
                row[x*3 + 1] = static_cast<uchar>(gi);
                row[x*3 + 2] = static_cast<uchar>(bi);
            } else {
                // OpenCV default BGR ordering
                row[x*3 + 0] = static_cast<uchar>(bi);
                row[x*3 + 1] = static_cast<uchar>(gi);
                row[x*3 + 2] = static_cast<uchar>(ri);
            }
        }
    }
    // encode PNG
    std::vector<int> params; // PNG uses default params
    bool ok = cv::imencode(".png", mat, out_png, params);
    if (!ok) {
        std::cerr << "imencode PNG failed\n";
    }
    return ok;
}

// Helper: compute per-vertex normals if mesh doesn't provide them
// v_pos: flattened Nx3, t_pos_idx: flattened Mx3
static std::vector<float> compute_vertex_normals(const std::vector<float>& v_pos, const std::vector<int>& t_pos_idx) {
    size_t Nv = v_pos.size() / 3;
    size_t Nf = t_pos_idx.size() / 3;
    std::vector<float> v_nrm(Nv * 3, 0.0f);

    for (size_t f = 0; f < Nf; ++f) {
        int i0 = t_pos_idx[3*f + 0];
        int i1 = t_pos_idx[3*f + 1];
        int i2 = t_pos_idx[3*f + 2];
        Eigen::Vector3f p0(v_pos[3*i0 + 0], v_pos[3*i0 + 1], v_pos[3*i0 + 2]);
        Eigen::Vector3f p1(v_pos[3*i1 + 0], v_pos[3*i1 + 1], v_pos[3*i1 + 2]);
        Eigen::Vector3f p2(v_pos[3*i2 + 0], v_pos[3*i2 + 1], v_pos[3*i2 + 2]);
        Eigen::Vector3f faceN = (p1 - p0).cross(p2 - p0);
        // accumulate
        v_nrm[3*i0 + 0] += faceN.x(); v_nrm[3*i0 + 1] += faceN.y(); v_nrm[3*i0 + 2] += faceN.z();
        v_nrm[3*i1 + 0] += faceN.x(); v_nrm[3*i1 + 1] += faceN.y(); v_nrm[3*i1 + 2] += faceN.z();
        v_nrm[3*i2 + 0] += faceN.x(); v_nrm[3*i2 + 1] += faceN.y(); v_nrm[3*i2 + 2] += faceN.z();
    }
    // normalize
    for (size_t i = 0; i < Nv; ++i) {
        float x = v_nrm[3*i + 0], y = v_nrm[3*i + 1], z = v_nrm[3*i + 2];
        float len = std::sqrt(x*x + y*y + z*z);
        if (len > 1e-9f) {
            v_nrm[3*i + 0] = x / len;
            v_nrm[3*i + 1] = y / len;
            v_nrm[3*i + 2] = z / len;
        } else {
            v_nrm[3*i + 0] = 0; v_nrm[3*i + 1] = 0; v_nrm[3*i + 2] = 1;
        }
    }
    return v_nrm;
}

// Apply a 4x4 transform (row-major or column-major? we choose row-major here)
// accepts rotation_mat as pointer to 16 floats in row-major order (r00,r01,r02,r03, r10,...)
static void apply_4x4_transform_to_positions_and_normals(
    std::vector<float>& v_pos_flat,                // Nx3 floats, in-out
    std::vector<float>& v_nrm_flat,                // Nx3 floats, in-out (may be empty -> skip)
    std::vector<float>& v_tng_flat,                // Nx3 floats, in-out (may be empty -> skip)
    const float* rotation_mat_4x4)                 // pointer to 16 floats row-major
{
    if (!rotation_mat_4x4) return;
    size_t Nv = v_pos_flat.size() / 3;
    // read 3x3 linear part and translation
    Eigen::Matrix3f M;
    M << rotation_mat_4x4[0], rotation_mat_4x4[1], rotation_mat_4x4[2],
         rotation_mat_4x4[4], rotation_mat_4x4[5], rotation_mat_4x4[6],
         rotation_mat_4x4[8], rotation_mat_4x4[9], rotation_mat_4x4[10];

    Eigen::Vector3f T(rotation_mat_4x4[3], rotation_mat_4x4[7], rotation_mat_4x4[11]);

    // For orthonormal rotation M_invT = M (for pure rotation). If you want robust,
    // compute inverse-transpose for normals (here we compute it anyway).
    Eigen::Matrix3f normal_mat = (M.inverse()).transpose();

    for (size_t i = 0; i < Nv; ++i) {
        Eigen::Vector3f p(v_pos_flat[3*i + 0], v_pos_flat[3*i + 1], v_pos_flat[3*i + 2]);
        Eigen::Vector3f p2 = M * p + T;
        v_pos_flat[3*i + 0] = p2.x();
        v_pos_flat[3*i + 1] = p2.y();
        v_pos_flat[3*i + 2] = p2.z();
        if (v_nrm_flat.size() == v_pos_flat.size()) {
            Eigen::Vector3f n(v_nrm_flat[3*i + 0], v_nrm_flat[3*i + 1], v_nrm_flat[3*i + 2]);
            Eigen::Vector3f n2 = normal_mat * n;
            float len = n2.norm();
            if (len > 1e-9f) n2 /= len;
            v_nrm_flat[3*i + 0] = n2.x();
            v_nrm_flat[3*i + 1] = n2.y();
            v_nrm_flat[3*i + 2] = n2.z();
        }
        if ( v_tng_flat.size() == v_pos_flat.size() ) {
            Eigen::Vector3f t(v_tng_flat[3 * i + 0], v_tng_flat[3 * i + 1], v_tng_flat[3 * i + 2]);
            Eigen::Vector3f t_rot = normal_mat * t;
            // re-normalize just in case
            float t_norm = t_rot.norm();
            if (t_norm > 1e-9f) t_rot /= t_norm;
            v_tng_flat[3 * i + 0] = t_rot.x();
            v_tng_flat[3 * i + 1] = t_rot.y();
            v_tng_flat[3 * i + 2] = t_rot.z();
        }
    }
}

// Invert mesh: flip triangle winding and optionally flip vertex normals in place.
static void invert_mesh_winding_and_normals(
    std::vector<int>& t_pos_idx_flat,   // M*3 ints, in-out
    std::vector<float>* v_nrm_flat_ptr  // optional pointer to Nx3 normals; if provided, we negate normals
) {
    size_t M3 = t_pos_idx_flat.size();
    if (M3 % 3 != 0) return;
    for (size_t f = 0; f < M3; f += 3) {
        // f: indices [f, f+1, f+2] -> swap last two to change winding
        std::swap(t_pos_idx_flat[f + 1], t_pos_idx_flat[f + 2]);
    }
//    if (v_nrm_flat_ptr && v_nrm_flat_ptr->size() % 3 == 0) {
//        auto &n = *v_nrm_flat_ptr;
//        for (size_t i = 0; i < n.size(); ++i) n[i] = -n[i]; // flip normals
//    }
}


// Main exporter
// - inspect: BuildTexturesInspect (contains float images in [0,1])
// - mesh: MeshCPP (must contain v_pos, t_pos_idx, v_tex; v_nrm optional)
// - output_glb_path: where to write .glb
// Returns true on success.
bool ExportGLBFromInspectOld(
    const BuildTexturesInspect& inspect,
    const MeshCPP& mesh,
    const std::string& output_glb_path
) {
    using namespace tinygltf;

    // Validate
    if (inspect.H <= 0 || inspect.W <= 0) {
        std::cerr << "Invalid image dims\n";
        return false;
    }

    const int H = inspect.H;
    const int W = inspect.W;

    // Encode images to PNG buffers
    std::vector<unsigned char> basecolor_png, normal_png;
    if (!encode_image_png_from_float3(inspect.basecolor, H, W, basecolor_png, /*use_bgr=*/true)) {
        std::cerr << "Failed to encode basecolor PNG\n";
        return false;
    }
    if (!encode_image_png_from_float3(inspect.bump_filled, H, W, normal_png, /*use_bgr=*/true)) {
        std::cerr << "Failed to encode normal PNG\n";
        return false;
    }

    // Get or build JPEG bytes for basecolor and bump
    std::vector<unsigned char> base_jpg_bytes;
    std::vector<unsigned char> bump_jpg_bytes;
    if (!inspect.basecolor_jpeg_bytes.empty()) {
        std::cerr << "extracting basecolor jpeg encoding\n";
        base_jpg_bytes.assign(inspect.basecolor_jpeg_bytes.begin(), inspect.basecolor_jpeg_bytes.end());
    } else {
        std::cerr << "encoding basecolor\n";
        // encode from float basecolor
        cv::Mat base_f(H, W, CV_32FC3);
        // fill base_f from inspect.basecolor
        for (int y = 0; y < H; ++y) {
            cv::Vec3f* row = base_f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < W; ++x) {
                int idx = (y * W + x) * 3;
                row[x][0] = inspect.basecolor[idx + 0];
                row[x][1] = inspect.basecolor[idx + 1];
                row[x][2] = inspect.basecolor[idx + 2];
            }
        }
        cv::Mat base_u8 = float32_to_uint8_cv(base_f, false);
        cv::Mat base_bgr;
        cv::cvtColor(base_u8, base_bgr, cv::COLOR_RGB2BGR);
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 };
        if (!cv::imencode(".jpg", base_bgr, base_jpg_bytes, params)) {
            std::cerr << "ExportGLBFromInspect: failed to encode basecolor\n";
            return false;
        }
    }
    if (!inspect.bump_jpeg_bytes.empty()) {
        std::cerr << "extracting bump jpeg encoding\n";
        bump_jpg_bytes.assign(inspect.bump_jpeg_bytes.begin(), inspect.bump_jpeg_bytes.end());
    } else {
        std::cerr << "encoding bump\n";
        cv::Mat norm_f(H, W, CV_32FC3);
        for (int y = 0; y < H; ++y) {
            cv::Vec3f* row = norm_f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < W; ++x) {
                int idx = (y * W + x) * 3;
                row[x][0] = inspect.bump_filled[idx + 0];
                row[x][1] = inspect.bump_filled[idx + 1];
                row[x][2] = inspect.bump_filled[idx + 2];
            }
        }
        cv::Mat bump_up(norm_f.rows, norm_f.cols, CV_32FC3, cv::Scalar(0.5f, 0.5f, 1.0f));
        // compute dither_mask: true where bump_np equals bump_up (per-channel)
        cv::Mat equal_mask(norm_f.rows, norm_f.cols, CV_8UC1, cv::Scalar(0));
        for (int y=0; y<norm_f.rows; ++y) {
            const cv::Vec3f* brow = norm_f.ptr<cv::Vec3f>(y);
            uchar* mout = equal_mask.ptr<uchar>(y);
            for (int x=0; x<norm_f.cols; ++x) {
                bool all_eq = true;
                for (int c=0; c<3; ++c) {
                    // careful with float equality: Python used exact equality after conversion
                    // It is okay to test near-equality here; we use a tiny epsilon.
                    float a = brow[x][c];
                    float target = (c < 2) ? 0.5f : 1.0f; // the values in bump_up
                    if (std::abs(a - target) > 1e-6f) { all_eq = false; break; }
                }
                mout[x] = all_eq ? 1 : 0;
            }
        }
        // convert equal_mask to float dither_mask expected by float32_to_uint8_cv
        cv::Mat dither_mask_float;
        equal_mask.convertTo(dither_mask_float, CV_32F, 1.0/255.0); // if equal_mask is 0/255
        // dithered normal? If you had an already dithered bump_u8, use that.
        cv::Mat bump_u8 = float32_to_uint8_cv(norm_f, true /*dither*/, equal_mask);
        cv::Mat bump_bgr; cv::cvtColor(bump_u8, bump_bgr, cv::COLOR_RGB2BGR);
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 };
        if (!cv::imencode(".jpg", bump_bgr, bump_jpg_bytes, params)) {
            std::cerr << "ExportGLBFromInspect: failed to encode normal\n";
            return false;
        }
    }

    // Prepare geometry arrays
    std::vector<float> vpos = mesh.v_pos();
    std::vector<int> tidx = mesh.t_pos_idx();
    std::vector<float> vnormal;
    bool have_vertex_normals = false;
    try {
//        const std::vector<float>& maybe = mesh.get_v_nrm();
        auto maybe = mesh.get_v_nrm();
        if (maybe) {
//            if (maybe.size() == vpos.size()) {
            vnormal = maybe.value().get();
            if (vnormal.size() == vpos.size()) have_vertex_normals = true;
        }
    } catch (...) {
        // fall back
    }
    if (!have_vertex_normals) {
        vnormal = compute_vertex_normals(vpos, tidx);
    }
    std::vector<float> vtex;// = mesh.get_v_tex();
    auto vtex_maybe = mesh.get_v_tex();
    if (vtex_maybe) vtex = vtex_maybe.value().get();
    if (vtex.size() / 2 != vpos.size() / 3) {
        std::cerr << "Warning: v_tex size doesn't match vertex count; proceeding but UV may be wrong\n";
    }

    // apply transform and invert
    std::array<float,16> transform_mat{};
    transform_mat[0] = 0; transform_mat[1] = -1; transform_mat[2] = 0; transform_mat[3] = 0;
    transform_mat[4] = 0; transform_mat[5] = 0; transform_mat[6] = 1; transform_mat[7] = 0;
    transform_mat[8] = -1; transform_mat[9] = 0; transform_mat[10] = 0; transform_mat[11] = 0;
    transform_mat[12] = 0; transform_mat[13] = 0; transform_mat[14] = 0; transform_mat[15] = 1;
//    apply_4x4_transform_to_positions_and_normals(vpos, vnormal, transform_mat.data());
    invert_mesh_winding_and_normals(tidx, &vnormal);

    // Build binary buffer (GLB binary chunk) by concatenating all buffer data in a single vector<unsigned char>
    std::vector<unsigned char> bin; bin.reserve(1 << 20);

    Model model;
    model.asset.version = "2.0";
    model.asset.generator = "custom_baker_cpp";

    // Helper to append raw bytes and return bufferView index
    auto append_binary_data = [&](const void* data_ptr, size_t byte_length, int target /* 34962 = ARRAY_BUFFER, 34963 = ELEMENT_ARRAY_BUFFER */)->int {
        // alignment to 4 bytes for glTF
        size_t offset = bin.size();
        size_t pad = (4 - (offset % 4)) % 4;
        for (size_t p = 0; p < pad; ++p) bin.push_back(0);
        int viewOffset = static_cast<int>(bin.size());
        const unsigned char* src = reinterpret_cast<const unsigned char*>(data_ptr);
        bin.insert(bin.end(), src, src + byte_length);

        // create bufferView
        BufferView bv;
        bv.buffer = 0; // single buffer
        bv.byteOffset = viewOffset;
        bv.byteLength = static_cast<int>(byte_length);
        bv.target = target; // glTF target hint
        model.bufferViews.push_back(bv);
        return static_cast<int>(model.bufferViews.size() - 1);
    };

    // 1) indices (uint32)
    std::vector<uint32_t> indices_u32(tidx.size());
    for (size_t i = 0; i < tidx.size(); ++i) indices_u32[i] = static_cast<uint32_t>(tidx[i]);
    int bv_indices = append_binary_data(indices_u32.data(), indices_u32.size() * sizeof(uint32_t), TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER);

    // create accessor for indices
    {
        Accessor acc;
        acc.bufferView = bv_indices;
        acc.byteOffset = 0;
        acc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
        acc.count = static_cast<int>(indices_u32.size());
        acc.type = TINYGLTF_TYPE_SCALAR;
        acc.maxValues = { (double)*std::max_element(indices_u32.begin(), indices_u32.end()) };
        acc.minValues = { (double)*std::min_element(indices_u32.begin(), indices_u32.end()) };
        model.accessors.push_back(acc);
    }
    int accessor_indices = static_cast<int>(model.accessors.size() - 1);

    // 2) positions (FLOAT, VEC3)
    int bv_positions = append_binary_data(vpos.data(), vpos.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);
    {
        Accessor acc;
        acc.bufferView = bv_positions;
        acc.byteOffset = 0;
        acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        acc.count = static_cast<int>(vpos.size() / 3);
        acc.type = TINYGLTF_TYPE_VEC3;
        // compute min/max
        Eigen::Vector3f minv(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        Eigen::Vector3f maxv(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < vpos.size() / 3; ++i) {
            float x = vpos[3*i + 0], y = vpos[3*i + 1], z = vpos[3*i + 2];
            minv.x() = std::min(minv.x(), x); minv.y() = std::min(minv.y(), y); minv.z() = std::min(minv.z(), z);
            maxv.x() = std::max(maxv.x(), x); maxv.y() = std::max(maxv.y(), y); maxv.z() = std::max(maxv.z(), z);
        }
        acc.minValues = { minv.x(), minv.y(), minv.z() };
        acc.maxValues = { maxv.x(), maxv.y(), maxv.z() };
        model.accessors.push_back(acc);
    }
    int accessor_positions = static_cast<int>(model.accessors.size() - 1);

    // 3) normals (FLOAT, VEC3)
    int bv_normals = append_binary_data(vnormal.data(), vnormal.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);
    {
        Accessor acc;
        acc.bufferView = bv_normals;
        acc.byteOffset = 0;
        acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        acc.count = static_cast<int>(vnormal.size() / 3);
        acc.type = TINYGLTF_TYPE_VEC3;
        acc.minValues = { -1.0, -1.0, -1.0 };
        acc.maxValues = { 1.0, 1.0, 1.0 };
        model.accessors.push_back(acc);
    }
    int accessor_normals = static_cast<int>(model.accessors.size() - 1);

    // 4) texcoords (FLOAT, VEC2)
    int bv_uv = append_binary_data(vtex.data(), vtex.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);
    {
        Accessor acc;
        acc.bufferView = bv_uv;
        acc.byteOffset = 0;
        acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        acc.count = static_cast<int>(vtex.size() / 2);
        acc.type = TINYGLTF_TYPE_VEC2;
//        acc.minValues = { 0.0, 0.0 };
//        acc.maxValues = { 1.0, 1.0 };

        Eigen::Vector2f minv(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        Eigen::Vector2f maxv(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < vtex.size() / 2; ++i) {
            float x = vpos[2*i + 0], y = vpos[2*i + 1];
            minv.x() = std::min(minv.x(), x); minv.y() = std::min(minv.y(), y);
            maxv.x() = std::max(maxv.x(), x); maxv.y() = std::max(maxv.y(), y);
        }
        acc.minValues = { minv.x(), minv.y() };
        acc.maxValues = { maxv.x(), maxv.y() };

        model.accessors.push_back(acc);
    }
    int accessor_uv = static_cast<int>(model.accessors.size() - 1);

    // 5) add the binary images as bufferViews and tinygltf::Image entries
//    int bv_basecolor = append_binary_data(basecolor_png.data(), basecolor_png.size(), 0); // target=0 for images
    int bv_basecolor = append_binary_data(base_jpg_bytes.data(), base_jpg_bytes.size(), 0); // target=0 for images
    int idx_image_basecolor = static_cast<int>(model.bufferViews.size() - 1);

    Image img_basecolor;
    img_basecolor.bufferView = idx_image_basecolor;
    img_basecolor.mimeType = "image/png";
    img_basecolor.name = "basecolor";
    model.images.push_back(img_basecolor);
    int image_basecolor_index = static_cast<int>(model.images.size() - 1);

    int bv_normal = append_binary_data(normal_png.data(), normal_png.size(), 0);
    Image img_normal;
    img_normal.bufferView = static_cast<int>(model.bufferViews.size() - 1);
    img_normal.mimeType = "image/png";
    img_normal.name = "normal";
    model.images.push_back(img_normal);
    int image_normal_index = static_cast<int>(model.images.size() - 1);

    // 6) textures (image -> sampler)
    Texture tex_basecolor; tex_basecolor.source = image_basecolor_index;
    model.textures.push_back(tex_basecolor);
    int tex_basecolor_index = static_cast<int>(model.textures.size() - 1);

    Texture tex_normal; tex_normal.source = image_normal_index;
    model.textures.push_back(tex_normal);
    int tex_normal_index = static_cast<int>(model.textures.size() - 1);

    // 7) material (PBR)
    Material mat;
    mat.name = "baked_material";
    // pbrMetallicRoughness
    PbrMetallicRoughness pbr;
    pbr.baseColorTexture.index = tex_basecolor_index;
    pbr.metallicFactor = inspect.metallic;
    pbr.roughnessFactor = inspect.roughness;
//    mat.values["pbrMetallicRoughness"] = ParameterValue(); // not required but keep structure
    mat.pbrMetallicRoughness = pbr;
    mat.doubleSided = true;

    // normal texture
    NormalTextureInfo normalInfo;
    normalInfo.index = tex_normal_index;
    mat.normalTexture = normalInfo;

    model.materials.push_back(mat);
    int material_index = static_cast<int>(model.materials.size() - 1);

    // 8) mesh primitive
    Mesh mesh_out;
    mesh_out.name = "baked_mesh";
    Primitive prim;
    prim.indices = accessor_indices;
    prim.attributes["POSITION"] = accessor_positions;
    prim.attributes["NORMAL"] = accessor_normals;
    prim.attributes["TEXCOORD_0"] = accessor_uv;
    prim.material = material_index;
    mesh_out.primitives.push_back(prim);
    model.meshes.push_back(mesh_out);
    int mesh_index = static_cast<int>(model.meshes.size() - 1);

    // 9) node + scene
    Node node;
    node.mesh = mesh_index;
    model.nodes.push_back(node);
    model.scenes.push_back(Scene());
    model.scenes[0].nodes.push_back(0);
    model.defaultScene = 0;

    // 10) buffer (single buffer containing bin)
    Buffer mainBuffer;
//    mainBuffer.byteLength = static_cast<int>(bin.size());
    mainBuffer.data = bin;
    model.buffers.push_back(mainBuffer);
//    model.buffers[0].data = bin;

    // FINALLY, write .glb
    TinyGLTF writer;
//    writer.WriteImageData = false; // images are in bufferViews (we set them)
//    bool write_ok = writer.WriteGltfSceneToFile(&model, output_glb_path, true, true, true, true);
    bool write_ok = writer.WriteGltfSceneToFile(&model, output_glb_path, /* embedImages = */ true, /* embedBuffers = */ true, /* pretty_print = */ false, /* write_binary = */ true);
    if (!write_ok) {
        std::cerr << "tinygltf failed to write " << output_glb_path << std::endl;
        return false;
    }
    std::cout << "Wrote GLB: " << output_glb_path << std::endl;
    return true;
}

bool ExportGLBFromInspect(
    const BuildTexturesInspect& inspect,
    const MeshCPP& mesh,
    const std::string& output_glb_path
) {
    LOGI("[Export:] Entered export function...");

    using namespace tinygltf;

    if (inspect.H <= 0 || inspect.W <= 0) {
        LOGE("[Export:] Invalid image dims");
        return false;
    }
    const int H = inspect.H;
    const int W = inspect.W;

    // Encode images to PNG buffers (you already have helpers)
//    std::vector<unsigned char> basecolor_png, normal_png;
//    if (!encode_image_png_from_float3(inspect.basecolor, H, W, basecolor_png, /*use_bgr=*/true)) {
//        std::cerr << "Failed to encode basecolor PNG\n";
//        return false;
//    }
//    if (!encode_image_png_from_float3(inspect.bump_filled, H, W, normal_png, /*use_bgr=*/true)) {
//        std::cerr << "Failed to encode normal PNG\n";
//        return false;
//    }

    // Get or build JPEG bytes for basecolor and bump
    LOGI("[Export:] building jpeg bytes for basecolor and bump...");
    std::vector<unsigned char> base_jpg_bytes;
    std::vector<unsigned char> bump_jpg_bytes;
    if (false) { // (!inspect.basecolor_jpeg_bytes.empty()) {
        LOGI("[Export:] extracting basecolor jpeg encoding");
        base_jpg_bytes.assign(inspect.basecolor_jpeg_bytes.begin(), inspect.basecolor_jpeg_bytes.end());
    } else {
        LOGI("[Export:] encoding basecolor");
        // encode from float basecolor
        cv::Mat base_f(H, W, CV_32FC3);
        // fill base_f from inspect.basecolor
        LOGI("[Export:] fill base_f from inspect.basecolor...");
        for (int y = 0; y < H; ++y) {
            cv::Vec3f* row = base_f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < W; ++x) {
                int idx = (y * W + x) * 3;
                row[x][0] = inspect.basecolor[idx + 0];
                row[x][1] = inspect.basecolor[idx + 1];
                row[x][2] = inspect.basecolor[idx + 2];
            }
        }
        cv::Mat base_u8 = float32_to_uint8_cv(base_f, false);
        cv::Mat base_bgr;
        cv::cvtColor(base_u8, base_bgr, cv::COLOR_RGB2BGR);
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 };
        if (!cv::imencode(".jpg", base_bgr, base_jpg_bytes, params)) {
            LOGE("[Export:] ExportGLBFromInspect: failed to encode basecolor");
            return false;
        }
        LOGI("[Export:] encoded basecolor!");
    }
    if (false) { // (!inspect.bump_jpeg_bytes.empty()) {
        LOGI("[Export:] extracting bump jpeg encoding");
        bump_jpg_bytes.assign(inspect.bump_jpeg_bytes.begin(), inspect.bump_jpeg_bytes.end());
    } else {
        LOGI("[Export:] encoding bump");
        cv::Mat norm_f(H, W, CV_32FC3);
        for (int y = 0; y < H; ++y) {
            cv::Vec3f* row = norm_f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < W; ++x) {
                int idx = (y * W + x) * 3;
                row[x][0] = inspect.bump_filled[idx + 0];
                row[x][1] = inspect.bump_filled[idx + 1];
                row[x][2] = inspect.bump_filled[idx + 2];
            }
        }
        cv::Mat bump_up(norm_f.rows, norm_f.cols, CV_32FC3, cv::Scalar(0.5f, 0.5f, 1.0f));
        // compute dither_mask: true where bump_np equals bump_up (per-channel)
        LOGI("[Export:] computing dither mask...");
        cv::Mat equal_mask(norm_f.rows, norm_f.cols, CV_8UC1, cv::Scalar(0));
        for (int y=0; y<norm_f.rows; ++y) {
            const cv::Vec3f* brow = norm_f.ptr<cv::Vec3f>(y);
            uchar* mout = equal_mask.ptr<uchar>(y);
            for (int x=0; x<norm_f.cols; ++x) {
                bool all_eq = true;
                for (int c=0; c<3; ++c) {
                    // careful with float equality: Python used exact equality after conversion
                    // It is okay to test near-equality here; we use a tiny epsilon.
                    float a = brow[x][c];
                    float target = (c < 2) ? 0.5f : 1.0f; // the values in bump_up
                    if (std::abs(a - target) > 1e-6f) { all_eq = false; break; }
                }
                mout[x] = all_eq ? 1 : 0;
            }
        }
        // convert equal_mask to float dither_mask expected by float32_to_uint8_cv
        cv::Mat dither_mask_float;
        equal_mask.convertTo(dither_mask_float, CV_32F, 1.0/*/255.0*/); // if equal_mask is 0/255
        // dithered normal? If you had an already dithered bump_u8, use that.
        cv::Mat bump_u8 = float32_to_uint8_cv(norm_f, true /*dither*/, dither_mask_float);//equal_mask);
        cv::Mat bump_bgr; cv::cvtColor(bump_u8, bump_bgr, cv::COLOR_RGB2BGR);
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 };
        if (!cv::imencode(".jpg", bump_bgr, bump_jpg_bytes, params)) {
            LOGE("[Export:] ExportGLBFromInspect: failed to encode normal");
            return false;
        }
        LOGI("[Export:] encoded normals!");
    }

    // Geometry
    LOGI("[Export:] geometry...");
    std::vector<float> vpos = mesh.v_pos();       // Nx3
    std::vector<int> tidx = mesh.t_pos_idx();     // Mx3 -> flattened indices
    std::vector<float> vnormal;
    // If MeshCPP provides normals via get_v_nrm() returning optional-like, use it; otherwise compute.
    auto maybe_nrm = mesh.get_v_nrm(); // adapt to your MeshCPP API
    if (maybe_nrm && maybe_nrm.value().get().size() == vpos.size()) {
        vnormal = *maybe_nrm;
    } else {
        LOGI("[Export:] computing vertex normals...");
        vnormal = compute_vertex_normals(vpos, tidx); // implement if not present
    }

    std::vector<float> vtex;
    auto maybe_vtex = mesh.get_v_tex();
    if (maybe_vtex) vtex = *maybe_vtex;

    std::vector<float> vtng;
    auto maybe_vtng = mesh.get_v_tng();
    if (maybe_vtng) vtng = *maybe_vtng;

    // Apply transform/invert operations (your matrix code)
    LOGI("[Export:] make and apply transform...");
    std::array<float,16> transform_mat{};
    transform_mat[0] = 0; transform_mat[1] = -1; transform_mat[2] = 0; transform_mat[3] = 0;
    transform_mat[4] = 0; transform_mat[5] = 0; transform_mat[6] = 1; transform_mat[7] = 0;
    transform_mat[8] = -1; transform_mat[9] = 0; transform_mat[10] = 0; transform_mat[11] = 0;
    transform_mat[12] = 0; transform_mat[13] = 0; transform_mat[14] = 0; transform_mat[15] = 1;
//    apply_4x4_transform_to_positions_and_normals(vpos, vnormal, transform_mat.data());
    apply_4x4_transform_to_positions_and_normals(vpos, vnormal, vtng, transform_mat.data());
    LOGI("[Export:] invert mesh winding");
    invert_mesh_winding_and_normals(const_cast<std::vector<int>&>(tidx), &vnormal);

    // Build binary buffer
    std::vector<unsigned char> bin; bin.reserve(1 << 20);

    Model model;
    model.asset.version = "2.0";
    model.asset.generator = "custom_baker_cpp";

    // append helper returns bufferView index
    auto append_binary_data = [&](const void* data_ptr, size_t byte_length, int target)->int {
        size_t pad = (4 - (bin.size() % 4)) % 4;
        for (size_t p = 0; p < pad; ++p) bin.push_back(0);
        int viewOffset = static_cast<int>(bin.size());
        if (byte_length > 0) {
            const unsigned char* src = reinterpret_cast<const unsigned char*>(data_ptr);
            bin.insert(bin.end(), src, src + byte_length);
        }
        BufferView bv;
        bv.buffer = 0;
        bv.byteOffset = viewOffset;
        bv.byteLength = static_cast<int>(byte_length);
        bv.target = target;
        model.bufferViews.push_back(bv);
        return static_cast<int>(model.bufferViews.size() - 1);
    };

    // 1) indices (uint32)
    LOGI("[Export:] build indices...");
    std::vector<uint32_t> indices_u32(tidx.size());
    for (size_t i = 0; i < tidx.size(); ++i) indices_u32[i] = static_cast<uint32_t>(tidx[i]);
    int bv_indices = append_binary_data(indices_u32.data(), indices_u32.size() * sizeof(uint32_t), TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER);

    // accessor indices
    {
        Accessor acc;
        acc.bufferView = bv_indices;
        acc.byteOffset = 0;
        acc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
        acc.count = static_cast<int>(indices_u32.size()); // number of indices
        acc.type = TINYGLTF_TYPE_SCALAR;
        acc.maxValues = { static_cast<double>(*std::max_element(indices_u32.begin(), indices_u32.end())) };
        acc.minValues = { static_cast<double>(*std::min_element(indices_u32.begin(), indices_u32.end())) };
        model.accessors.push_back(acc);
    }
    int accessor_indices = static_cast<int>(model.accessors.size() - 1);

    // 2) positions
    LOGI("[Export:] build positions...");
    int bv_positions = append_binary_data(vpos.data(), vpos.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);
    {
        Accessor acc;
        acc.bufferView = bv_positions;
        acc.byteOffset = 0;
        acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        acc.count = static_cast<int>(vpos.size() / 3);
        acc.type = TINYGLTF_TYPE_VEC3;
        // compute min/max
        Eigen::Vector3f minv( std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity());
        Eigen::Vector3f maxv(-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < vpos.size()/3; ++i) {
            float x = vpos[3*i + 0], y = vpos[3*i + 1], z = vpos[3*i + 2];
            minv.x() = std::min(minv.x(), x); minv.y() = std::min(minv.y(), y); minv.z() = std::min(minv.z(), z);
            maxv.x() = std::max(maxv.x(), x); maxv.y() = std::max(maxv.y(), y); maxv.z() = std::max(maxv.z(), z);
        }
        acc.minValues = { minv.x(), minv.y(), minv.z() };
        acc.maxValues = { maxv.x(), maxv.y(), maxv.z() };
        model.accessors.push_back(acc);
    }
    int accessor_positions = static_cast<int>(model.accessors.size() - 1);

    // 3) normals (only if present)
    LOGI("[Export:] build normals...");
    int accessor_normals = -1;
    if (!vnormal.empty()) {
        int bv_normals = append_binary_data(vnormal.data(), vnormal.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);
        Accessor acc;
        acc.bufferView = bv_normals;
        acc.byteOffset = 0;
        acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        acc.count = static_cast<int>(vnormal.size() / 3);
        acc.type = TINYGLTF_TYPE_VEC3;
        acc.minValues = { -1.0, -1.0, -1.0 };
        acc.maxValues = { 1.0, 1.0, 1.0 };
        model.accessors.push_back(acc);
        accessor_normals = static_cast<int>(model.accessors.size() - 1);
    }

    // 4) texcoords (only if present)
    LOGI("[Export:] build textcoords...");
    int accessor_uv = -1;
    if (!vtex.empty()) {
        std::vector<float> vtex_flipped = vtex;
//        acc.minValues = { 0.0, 0.0 };
//        acc.maxValues = { 1.0, 1.0 };
        Eigen::Vector2f minv_tex(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        Eigen::Vector2f maxv_tex(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < vtex.size() / 2; ++i) {
            float x = vtex[2*i + 0], y = vtex[2*i + 1];
            minv_tex.x() = std::min(minv_tex.x(), x); minv_tex.y() = std::min(minv_tex.y(), y);
            maxv_tex.x() = std::max(maxv_tex.x(), x); maxv_tex.y() = std::max(maxv_tex.y(), y);
            vtex_flipped[2*i + 1] = 1.0f - y;
        }
        int bv_uv = append_binary_data(vtex_flipped.data(), vtex_flipped.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);
        Accessor acc;
        acc.bufferView = bv_uv;
        acc.byteOffset = 0;
        acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        acc.count = static_cast<int>(vtex.size() / 2);
        acc.type = TINYGLTF_TYPE_VEC2;
        acc.minValues = { minv_tex.x(), minv_tex.y() };
        acc.maxValues = { maxv_tex.x(), maxv_tex.y() };
        model.accessors.push_back(acc);
        accessor_uv = static_cast<int>(model.accessors.size() - 1);
    }

    // add winding to tangents
    LOGI("[Export:] add winding to tangents");
    std::vector<float> vtangent;
    if (!vtng.empty()) {
        if (vtng.size() == vpos.size() ) {
            vtangent.reserve((vpos.size() / 3) * 4);
            for (size_t i = 0; i < vtng.size(); i += 3) {
                vtangent.push_back(vtng[i + 0]);
                vtangent.push_back(vtng[i + 1]);
                vtangent.push_back(vtng[i + 2]);
                vtangent.push_back(1.0f); // Default sign
            }
        }
    }
    int accessor_tangent = -1;
    if (!vtangent.empty()) {
        int bv_tangent = append_binary_data(vtangent.data(), vtangent.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);

        tinygltf::Accessor acc;
        acc.bufferView = bv_tangent;
        acc.byteOffset = 0;
        acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        acc.count = static_cast<int>(vtangent.size() / 4);
        acc.type = TINYGLTF_TYPE_VEC4; // MUST be VEC4
        Eigen::Vector3f minvtng( std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity());
        Eigen::Vector3f maxvtng(-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < vtng.size()/3; ++i) {
            float x = vtng[3*i + 0], y = vtng[3*i + 1], z = vtng[3*i + 2];
            minvtng.x() = std::min(minvtng.x(), x); minvtng.y() = std::min(minvtng.y(), y); minvtng.z() = std::min(minvtng.z(), z);
            maxvtng.x() = std::max(maxvtng.x(), x); maxvtng.y() = std::max(maxvtng.y(), y); maxvtng.z() = std::max(maxvtng.z(), z);
        }
        acc.minValues = { minvtng.x(), minvtng.y(), minvtng.z(), 1.0f }; // -1.0f
        acc.maxValues = { maxvtng.x(), maxvtng.y(), maxvtng.z(), 1.0f };

        model.accessors.push_back(acc);
        accessor_tangent = static_cast<int>(model.accessors.size() - 1);
    }

    // 5) images -> add bufferViews and images (use returned indices)
    LOGI("[Export:] add basecolor to bufferview");
//    int bv_basecolor = append_binary_data(basecolor_png.data(), basecolor_png.size(), /*target=*/0);
    int bv_basecolor = append_binary_data(base_jpg_bytes.data(), base_jpg_bytes.size(), /*target=*/0);
    Image img_basecolor;
    img_basecolor.bufferView = bv_basecolor;
//    img_basecolor.mimeType = "image/png";
    img_basecolor.mimeType = "image/jpeg";
    img_basecolor.name = "basecolor";
    model.images.push_back(img_basecolor);
    int image_basecolor_index = static_cast<int>(model.images.size()-1);
//    std::cerr << "image basecolor index = " + std::to_string(image_basecolor_index);

//    int bv_normal = append_binary_data(normal_png.data(), normal_png.size(), /*target=*/0);
//    Image img_normal;
//    img_normal.bufferView = bv_normal;
//    img_normal.mimeType = "image/png";
//    img_normal.name = "normal";
//    model.images.push_back(img_normal);
//    int image_normal_index = static_cast<int>(model.images.size()-1);

    LOGI("[Export:] add bump_tex to bufferview");
    int bv_bump_tex = append_binary_data(bump_jpg_bytes.data(), bump_jpg_bytes.size(), /*target=*/0);
    Image img_bump;
    img_bump.bufferView = bv_bump_tex;
    img_bump.mimeType = "image/jpeg";
    img_bump.name = "normal";
    model.images.push_back(img_bump);
    int image_bump_index = static_cast<int>(model.images.size()-1);
//    std::cerr << "image bump index = " + std::to_string(image_bump_index);

    // 6) textures
    LOGI("[Export:] push texture indeces");
    Texture tex_basecolor;
    tex_basecolor.source = image_basecolor_index;
    model.textures.push_back(tex_basecolor);
    int tex_basecolor_index = static_cast<int>(model.textures.size()-1);
//    std::cerr << "tex_basecolor_index = " + std::to_string(tex_basecolor_index);

//    Texture tex_normal; tex_normal.source = image_normal_index;
//    model.textures.push_back(tex_normal);
//    int tex_normal_index = static_cast<int>(model.textures.size()-1);

    Texture tex_bump; tex_bump.source = image_bump_index;
    model.textures.push_back(tex_bump);
    int tex_bump_index = static_cast<int>(model.textures.size()-1);
//    std::cerr << "tex_bump_index = " + std::to_string(tex_bump_index);

    // 7) material
    LOGI("[Export:] build baked material...");
    Material mat;
    mat.name = "baked_material";
    PbrMetallicRoughness pbr;
    pbr.baseColorTexture.index = tex_basecolor_index;
    pbr.metallicFactor = inspect.metallic;
    pbr.roughnessFactor = inspect.roughness;
    mat.pbrMetallicRoughness = pbr;
    NormalTextureInfo normalInfo;
    normalInfo.index = tex_bump_index; //tex_normal_index;
    mat.normalTexture = normalInfo;
    mat.doubleSided = true;
    model.materials.push_back(mat);
    int material_index = static_cast<int>(model.materials.size()-1);

    // 8) mesh & primitive
    LOGI("[Export:] build mesh and primitives...");
    Mesh mesh_out;
    mesh_out.name = "baked_mesh";
    Primitive prim;
    prim.indices = accessor_indices;
    prim.attributes["POSITION"] = accessor_positions;
    if (accessor_normals >= 0) prim.attributes["NORMAL"] = accessor_normals;
    if (accessor_uv >= 0) prim.attributes["TEXCOORD_0"] = accessor_uv;
    if (accessor_tangent >= 0) prim.attributes["TANGENT"] = accessor_tangent;
    prim.material = material_index;
    prim.mode = TINYGLTF_MODE_TRIANGLES; // IMPORTANT
    mesh_out.primitives.push_back(prim);
    model.meshes.push_back(mesh_out);

    // 9) node / scene
    LOGI("[Export:] build scenes/nodes...");
    Node node;
    node.mesh = static_cast<int>(model.meshes.size()-1);
    model.nodes.push_back(node);
    Scene scene;
    scene.nodes.push_back(static_cast<int>(model.nodes.size()-1));
    model.scenes.push_back(scene);
    model.defaultScene = 0;

    // 10) buffer
    LOGI("[Export:] push buffer");
    Buffer mainBuffer;
//    mainBuffer.byteLength = static_cast<int>(bin.size());
    mainBuffer.data = bin;
    model.buffers.push_back(mainBuffer);

    // Write GLB (embed images & buffers, writeBinary = true)
    LOGI("[Export:] writing GLB...");
    TinyGLTF writer;
    bool write_ok = writer.WriteGltfSceneToFile(&model, output_glb_path, /*embedImages=*/true, /*embedBuffers=*/true, /*prettyPrint=*/false, /*writeBinary=*/true);
    if (!write_ok) {
        LOGE("[Export:] tinygltf failed to write %s", output_glb_path.c_str());
        return false;
    }
    LOGI("[Export:] Wrote GLB: %s", output_glb_path.c_str());
    return true;
}


} // namespace texture_baker_cpp
