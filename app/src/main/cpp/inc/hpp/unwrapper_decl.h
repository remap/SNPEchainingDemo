//
// Created by Chiheb Boussema on 10/12/25.
//

#ifndef SNPECHAININGDEMO_UNWRAPPER_DECL_H
#define SNPECHAININGDEMO_UNWRAPPER_DECL_H

// unwrapper_decl.h
#pragma once
#include "bvh.h"
#include <vector>
#include <set>

namespace UVUnwrapper {
  void create_bvhs(BVH* bvhs, Triangle* triangles,
                   std::vector<std::set<int>>& triangle_per_face, int num_faces,
                   int start, int end);

  void perform_intersection_check(BVH* bvhs, int num_bvhs, Triangle* triangles,
                                  uv_float3* vertex_tri_centroids,
                                  int64_t* assign_indices_ptr,
                                  ssize_t num_indices, int offset,
                                  std::vector<std::set<int>>& triangle_per_face);
}


#endif //SNPECHAININGDEMO_UNWRAPPER_DECL_H
