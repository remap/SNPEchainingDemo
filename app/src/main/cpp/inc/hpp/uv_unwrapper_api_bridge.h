//
// Created by Chiheb Boussema on 10/12/25.
//

#ifndef SNPECHAININGDEMO_UV_UNWRAPPER_API_BRIDGE_H
#define SNPECHAININGDEMO_UV_UNWRAPPER_API_BRIDGE_H

#pragma once
#include <vector>
#include <cstdint>

// in uv_unwrapper_api_bridge.h
namespace UVUnwrapperBridge {
  // The function name below must be implemented in your existing codebase.
  // It returns assign_indices (size Nf) by value.
  std::vector<int64_t> UVUnwrapper_assign_faces_uv_to_atlas_index_raw(
      const std::vector<float>& vertices,   // Nv*3
      const std::vector<int>& indices,      // Nf*3
      const std::vector<float>& face_uv_flat, // Nf*3*2 flattened (v0.u, v0.v, v1.u, v1.v, v2.u, v2.v repeat)
      const std::vector<int64_t>& face_index  // Nf
  );
}


#endif //SNPECHAININGDEMO_UV_UNWRAPPER_API_BRIDGE_H
