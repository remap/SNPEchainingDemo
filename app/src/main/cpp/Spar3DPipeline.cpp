//
// Created by Chiheb Boussema on 1/16/26.
//
#include "Spar3DPipeline.h"
#include "initTensorsHelper.h"
#include "baker_cpu.h"
#include <cassert>
#include "omp.h"
#include <limits>    // for std::numeric_limits
#include <cmath>     // for std::isfinite
#include <algorithm> // for std::max
#include <cstddef>   // for ptrdiff_t, size_t
#include <vector>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include <unistd.h>

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "SPAR3D_PIPELINE"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)


size_t readProcessRssKb() {
    // Read /proc/self/statm: fields are (pages): size resident share text lib data dt
    std::ifstream ifs("/proc/self/statm");
    if (!ifs) return 0;
    long size_pages = 0, rss_pages = 0;
    ifs >> size_pages >> rss_pages;
    ifs.close();
    long page_size_kb = sysconf(_SC_PAGESIZE) / 1024;
    if (page_size_kb <= 0) page_size_kb = 4; // fallback
    return static_cast<size_t>(rss_pages) * static_cast<size_t>(page_size_kb);
}


inline void normalize_pc_bbox_inplace(float* pc, size_t B, size_t N, size_t C, float user_scale = 1.0f) {
    assert(pc != nullptr);
    assert(C == 3 || C == 6 || C == 9);
    if (B == 0 || N == 0) return;

    // Process each batch independently -- parallelize across batch dimension
    #pragma omp parallel for if(B > 1)
    for (ptrdiff_t b = 0; b < static_cast<ptrdiff_t>(B); ++b) {
        // compute bounds for x,y,z
        const size_t batch_offset = b * N * C;

        float min_x = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        float min_z = std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();

        // single pass to compute min/max
        for (size_t i = 0; i < N; ++i) {
            const size_t base = batch_offset + i * C;
            const float x = pc[base + 0];
            const float y = pc[base + 1];
            const float z = pc[base + 2];

            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
            if (z < min_z) min_z = z;
            if (z > max_z) max_z = z;
        }

        // compute center and scale (match python logic: scale = max(range_x, range_y, range_z))
        const float center_x = (max_x + min_x) * 0.5f;
        const float center_y = (max_y + min_y) * 0.5f;
        const float center_z = (max_z + min_z) * 0.5f;

        float range_x = max_x - min_x;
        float range_y = max_y - min_y;
        float range_z = max_z - min_z;

        float bbox_scale = std::max(range_x, std::max(range_y, range_z));
        // If user provided user_scale param, the function signature had scale=1.0 default
        // Python code overwrote scale variable; we keep identical behavior:
        // Use the bbox_scale (if zero fallback), ignore user_scale if you prefer original semantics.
        // To honor the user's requested parameter name, we'll multiply by user_scale.
        if (bbox_scale <= 0.0f || !std::isfinite(bbox_scale)) bbox_scale = 1.0f;
        bbox_scale *= user_scale;

        const float inv_scale = 1.0f / bbox_scale;

        // second pass: normalize in-place
        for (size_t i = 0; i < N; ++i) {
            const size_t base = batch_offset + i * C;
            // normalize xyz
            pc[base + 0] = (pc[base + 0] - center_x) * inv_scale;
            pc[base + 1] = (pc[base + 1] - center_y) * inv_scale;
            pc[base + 2] = (pc[base + 2] - center_z) * inv_scale;
            // extra channels (3..C-1) remain unchanged
        }
    } // end batch loop
}

// Out-of-place version: src is const, dst gets written.
// Both src_pc and dst_pc must be of size at least B * N * C.
// If dst_pc == src_pc, this still works (it copies normalized values into dst).
inline void normalize_pc_bbox(const float* src_pc, float* dst_pc, size_t B, size_t N, size_t C, float user_scale = 1.0f) {
    assert(src_pc != nullptr && dst_pc != nullptr);
    assert(C == 3 || C == 6 || C == 9);
    if (B == 0 || N == 0) return;

    #pragma omp parallel for if(B > 1)
    for (ptrdiff_t b = 0; b < static_cast<ptrdiff_t>(B); ++b) {
        const size_t batch_offset = b * N * C;

        float min_x = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        float min_z = std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();

        // compute min/max from source
        for (size_t i = 0; i < N; ++i) {
            const size_t base = batch_offset + i * C;
            const float x = src_pc[base + 0];
            const float y = src_pc[base + 1];
            const float z = src_pc[base + 2];

            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
            if (z < min_z) min_z = z;
            if (z > max_z) max_z = z;
        }

        const float center_x = (max_x + min_x) * 0.5f;
        const float center_y = (max_y + min_y) * 0.5f;
        const float center_z = (max_z + min_z) * 0.5f;

        float range_x = max_x - min_x;
        float range_y = max_y - min_y;
        float range_z = max_z - min_z;

        float bbox_scale = std::max(range_x, std::max(range_y, range_z));
        if (bbox_scale <= 0.0f || !std::isfinite(bbox_scale)) bbox_scale = 1.0f;
        bbox_scale *= user_scale;

        const float inv_scale = 1.0f / bbox_scale;

        // write normalized points to destination
        for (size_t i = 0; i < N; ++i) {
            const size_t base = batch_offset + i * C;
            const float x = src_pc[base + 0];
            const float y = src_pc[base + 1];
            const float z = src_pc[base + 2];

            dst_pc[base + 0] = (x - center_x) * inv_scale;
            dst_pc[base + 1] = (y - center_y) * inv_scale;
            dst_pc[base + 2] = (z - center_z) * inv_scale;

            // copy extra channels if any
            for (size_t ch = 3; ch < C; ++ch) {
                dst_pc[base + ch] = src_pc[base + ch];
            }
        }
    } // end batch loop
}

// Helper to load a vector directly from an asset
template <typename T>
inline bool loadVectorFromAsset(AAssetManager* mgr, const char* filename, std::vector<T>& out_vec) {
    // 1. Open the asset
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    if (!asset) {
        // LOGE("Could not open asset: %s", filename);
        return false;
    }

    // 2. Get exact size in bytes
    off_t length = AAsset_getLength(asset);

    // Check if valid size for type T
    if (length % sizeof(T) != 0) {
        // LOGE("Asset size mismatch for type T");
        AAsset_close(asset);
        return false;
    }

    // 3. Resize the vector to hold exactly this many elements
    size_t num_elements = static_cast<size_t>(length) / sizeof(T);
    out_vec.resize(num_elements);

    // 4. Read directly into the vector's memory
    // valid because std::vector guarantees contiguous memory
    int read_bytes = AAsset_read(asset, out_vec.data(), static_cast<size_t>(length));

    AAsset_close(asset);

    return (read_bytes == length);
}

static std::vector<uint8_t> get_mask(const std::vector<float>& rast, int resolution) {
    // Total number of pixels
    const int num_pixels = resolution * resolution;

    // Output mask: Use uint8_t (0 or 1) instead of bool for thread safety and speed
    std::vector<uint8_t> mask(num_pixels);

    // Parallel loop
    #pragma omp parallel for
    for (int i = 0; i < num_pixels; ++i) {
        // We want the 4th channel (index 3) of every pixel.
        // Memory layout: [R, G, B, A, R, G, B, A ...]
        // Index mapping: i * 4 + 3

        // This comparison returns true (1) or false (0), which casts efficiently to uint8_t
        mask[i] = (rast[i * 4 + 3] >= 0.0f);
    }

    return mask;
}

/**
 * Efficiently performs: dest = src[mask]
 * Compacts sparse 3D data into a dense buffer in parallel.
 * * @param dest       Pointer to destination buffer (must be large enough! e.g., H*W*3)
 * @param src        Pointer to source data (size H*W*3)
 * @param mask       Pointer to mask data (size H*W, 0 or 1)
 * @param num_pixels Total pixels (H*W)
 * @return           The number of valid points written
 */
template <typename T>
static int compact_masked_parallel(T* dest, const float* src, const uint8_t* mask, int num_pixels, TensorWorkspace* ws= nullptr) {
//static int compact_masked_parallel(float* dest, const float* src, const uint8_t* mask, int num_pixels) {
    LOGI("CMP 1");
    LOGI("Type of dest: %s", typeid(dest).name());
    int total_valid_points = 0;
    float* effective_dst = nullptr;

    std::vector<int> thread_offsets;
    std::vector<int> thread_counts;

    // We can't use a simple 'parallel for' because we don't know the write index
    // for each thread without checking previous threads.
    // Algorithm: 2-Pass Parallel Compaction
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n_threads = omp_get_num_threads();

        // 1. Determine this thread's chunk of the image
        int items_per_thread = (num_pixels + n_threads - 1) / n_threads;
        int start_idx = std::min(tid * items_per_thread, num_pixels);
        int end_idx = std::min((tid + 1) * items_per_thread, num_pixels);

        // 2. PASS 1: Count valid items in this chunk
        int local_count = 0;
        for (int i = start_idx; i < end_idx; ++i) {
            if (mask[i]) local_count++;
        }

        // 3. Share counts and compute offsets (Prefix Sum)
        // We use a static array or shared vector for inter-thread communication
//        /*static*/ std::vector<int> thread_offsets;
//        /*static*/ std::vector<int> thread_counts;

        #pragma omp single
        {
            thread_offsets.resize(n_threads);
            thread_counts.resize(n_threads);
        }
        // barrier implied
        thread_counts[tid] = local_count;
        #pragma omp barrier // Wait for all threads to write counts
        #pragma omp single
        {
            LOGI("CMP 2");
            int current_offset = 0;
            for (int t = 0; t < n_threads; ++t) {
                thread_offsets[t] = current_offset;
                current_offset += thread_counts[t];
            }
            total_valid_points = current_offset;
            LOGW("total valid points: %zu", total_valid_points);
            if constexpr (std::is_same_v<std::remove_cv_t<T>, std::string>) {
                LOGI("CMP 3");
                assert(ws != nullptr);
                TensorInfo dest_tinfo;
                dest_tinfo.name = *dest;
                dest_tinfo.dims = {(size_t)total_valid_points, 3};
                {
                    std::string emsg;
                    LOGW("dest numel: %zu, bytes: %zu", dest_tinfo.numel(), dest_tinfo.bytes());
                    if (!ensureWorkspaceBuffer(*ws, dest_tinfo.name, dest_tinfo, &emsg)) {
                        LOGE("COULD NOT CREATE TENSOR %s; %s", dest_tinfo.name.c_str(), emsg.c_str());
                    }
                }
                effective_dst = static_cast<float*>(ws->data(dest_tinfo.name));
            } else {
                LOGI("CMP 4");
                effective_dst = (float*)dest;
            }
        }
        // barrier implied
        // 4. PASS 2: Copy data to calculated global position
        // All threads write to 'effective_dst'
        LOGI("CMP 5");
        if (effective_dst != nullptr) {
            LOGI("CMP 6");
            int write_ptr = thread_offsets[tid];
            for (int i = start_idx; i < end_idx; ++i) {
                if (mask[i]) {
                    // Copy 3 floats (x, y, z)
                    int src_offset = i * 3;
                    int dst_offset = write_ptr * 3;

                    effective_dst[dst_offset + 0] = src[src_offset + 0];
                    effective_dst[dst_offset + 1] = src[src_offset + 1];
                    effective_dst[dst_offset + 2] = src[src_offset + 2];

                    write_ptr++;
                }
            }
        }
    }
    LOGI("CMP 7");
    return total_valid_points;
}

// Optimized Grid Sample for Triplanes
// - Matches 'scale_tensor' (Linear mapping, no pre-clamp)
// - Matches 'grid_sample' with padding_mode='zeros' (Out of bounds = 0)
// - Zero Copy
static void query_triplane_optimized(
    const float* positions,     // [N, 3]
    const float* triplanes,     // [3, C, H, W]
    float* output_buffer,       // [N, 3*C]
    const int N,                // Number of points
    const int C,                // Channels (e.g., 40)
    const int H,                // Height (e.g., 384)
    const int W,                // Width (e.g., 384)
    const float radius          // e.g., 0.87
) {
    const int plane_stride = C * H * W;
    const int channel_stride = H * W;

    // --- 1. Precompute Scaling Constants ---
    // scale_tensor maps [-R, R] -> [-1, 1] linearly.
    // grid_sample (align_corners=True) maps [-1, 1] -> [0, W-1].
    //
    // Combined Transformation (World -> Pixel):
    // pixel = ( (world / radius) + 1 ) * 0.5 * (W - 1)
    // pixel = (world + radius) * (W - 1) / (2 * radius)

    const float scale_common = (W - 1.0f) / (2.0f * radius);

    // We can compute pixel = world * scale + offset
    // offset = radius * scale_common = radius * (W-1)/(2R) = (W-1)/2
    const float map_scale = scale_common;
    const float map_offset = (W - 1.0f) * 0.5f;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {

        // Input Positions
        const float px = positions[i * 3 + 0];
        const float py = positions[i * 3 + 1];
        const float pz = positions[i * 3 + 2];

        // Coordinate sets for 3 planes: XY, XZ, YZ
        const float p_coords_u[3] = {px, px, py};
        const float p_coords_v[3] = {py, pz, pz};

        float* point_out_ptr = output_buffer + i * (3 * C);

        for (int p = 0; p < 3; ++p) {

            // --- 2. Map to Pixel Coordinates (No Clamp) ---
            float u = p_coords_u[p] * map_scale + map_offset;
            float v = p_coords_v[p] * map_scale + map_offset;

            // --- 3. Bilinear Interpolation w/ Zero Padding ---

            // Floor to find top-left integer coordinate
            int u0 = static_cast<int>(std::floor(u));
            int v0 = static_cast<int>(std::floor(v));
            int u1 = u0 + 1;
            int v1 = v0 + 1;

            // Compute weights based on distance from floor
            float w_u = u - u0;
            float w_v = v - v0;

            float w00 = (1.0f - w_u) * (1.0f - w_v);
            float w10 = w_u * (1.0f - w_v);
            float w01 = (1.0f - w_u) * w_v;
            float w11 = w_u * w_v;

            // Check bounds for Zero Padding
            // If a coordinate is outside [0, W-1], that corner is invalid.
            bool valid_u0 = (u0 >= 0 && u0 < W);
            bool valid_u1 = (u1 >= 0 && u1 < W);
            bool valid_v0 = (v0 >= 0 && v0 < H);
            bool valid_v1 = (v1 >= 0 && v1 < H);

            // Calculate memory offsets for the 4 corners
            // If invalid, we set offset to 0 (safe read) but mask the weight to 0.0
            int idx00 = (valid_v0 && valid_u0) ? (v0 * W + u0) : 0;
            int idx10 = (valid_v0 && valid_u1) ? (v0 * W + u1) : 0;
            int idx01 = (valid_v1 && valid_u0) ? (v1 * W + u0) : 0;
            int idx11 = (valid_v1 && valid_u1) ? (v1 * W + u1) : 0;

            // Mask weights for invalid corners (Implementing 'padding_mode="zeros"')
            float mw00 = (valid_v0 && valid_u0) ? w00 : 0.0f;
            float mw10 = (valid_v0 && valid_u1) ? w10 : 0.0f;
            float mw01 = (valid_v1 && valid_u0) ? w01 : 0.0f;
            float mw11 = (valid_v1 && valid_u1) ? w11 : 0.0f;

            const float* plane_ptr = triplanes + p * plane_stride;

            // --- 4. Channel Gather ---
            // Unrolling hint might help depending on compiler,
            // but simple loop is usually auto-vectorized well by NEON
            for (int c = 0; c < C; ++c) {
                const float* ch_ptr = plane_ptr + c * channel_stride;

                float val = mw00 * ch_ptr[idx00] +
                            mw10 * ch_ptr[idx10] +
                            mw01 * ch_ptr[idx01] +
                            mw11 * ch_ptr[idx11];

                point_out_ptr[p * C + c] = val;
            }
        }
    }
}

static void scale_tensor_inplace(float* __restrict__ data, const size_t num_points,
                                 const float inp_scale_min, const float inp_scale_max,
                                 const float tgt_scale_min, const float tgt_scale_max) {
    // dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    // dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    // we precompute scale and offset
    const float s = (tgt_scale_max - tgt_scale_min) / (inp_scale_max - inp_scale_min);
    const float o = -inp_scale_min * s + tgt_scale_min;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_points; ++i) {
        size_t idx = i * 3;
        // The compiler will auto-vectorize these 3 lines into SIMD instructions
        data[idx + 0] = data[idx + 0] * s + o; // X
        data[idx + 1] = data[idx + 1] * s + o; // Y
        data[idx + 2] = data[idx + 2] * s + o; // Z
    }
}

bool Spar3DPipeline::init_networks(char runtime_hint) {
    std::string log;
    bool reset_sessions = true; //true;

    // read config file
    std::string cfgText;
    std::string emsg;
    if (!readAssetToString(mgr_, models_config_filename_.c_str(), cfgText, &emsg)) {
        LOGE("Config read failed: %s", emsg.c_str());
        return false;
    }
    // parse config file
    MultiPipelinesCfg mp_cfg;
    {
        std::string emsg;
        if (!ParseMultiConfig(cfgText, mp_cfg, &emsg)) {
            LOGE("Config parse failed: %s", emsg.c_str());
            return false;
        }
        if (mp_cfg.pipes.empty()) {
            LOGE("Config has no pipelines");
            return false;
        }
        if (!mp_cfg.baseDir.empty()) {
            LOGI("Found pipelines base directory: %s", mp_cfg.baseDir.c_str());
            g_modelDir_ = mp_cfg.baseDir;
            if (g_modelDir_.at(g_modelDir_.length()-1) != '/') g_modelDir_ += '/';
        }
    }
    // build networks
//    std::vector<std::string> pipeline_names = {
//            "preparer",
//            "pdiff_cond",
//            "scene_codes1",
////            "scene_codes2",
////            "image_estimator",
//            "triplanesToProtoMesh",
//            "padded_decoder",
//            "one_step_denoiser"
//    };
    std::map<std::string, GraphRunner *> pipelines;
    pipelines = {
            {"preparer",             gr_runners_.img_preparer.get()},
            {"pdiff_cond",           gr_runners_.pdiff_cond.get()},
            {"scene_codes1",         gr_runners_.scene_codes1.get()},
            {"scene_codes2",         gr_runners_.scene_codes2.get()},
            {"image_estimator",      gr_runners_.image_estimator.get()},
            {"triplanesToProtoMesh", gr_runners_.triplanesToProtoMesh.get()},
            {"padded_decoder",         gr_runners_.padded_decoder.get()},
            {"one_step_denoiser",    gr_runners_.one_step_denoiser.get()},
            {"scene_codes2_3",    gr_runners_.scene_codes2_3.get()},
//            {"tester_allvision",    gr_runners_.tester_allvision.get()}
    };
    for (auto& it : pipelines) {
        std::string pipe_name = it.first;
        LOGI("Building %s", pipe_name.c_str());
        auto pipe = this->findPipe(mp_cfg, pipe_name);
        if (pipe.models.empty()) { LOGE("Missing pipeline %s", pipe_name.c_str()); return false; }
        log += buildArbitraryChainFromConfig(mgr_,
                                           g_modelDir_,
                                           pipe,
                                           ws_,
                                           *it.second,
                                           runtime_hint,
                                           reset_sessions);
    }
    LOGI("Building networks complete!");
    return true;
}

// Helper to keep logic clean
struct CropTransform {
    float scale;
    float tx, ty;
};

// Returns the processed CHW or HWC data (depending on what your model needs)
std::vector<float> Spar3DPipeline::preprocessImage(uint8_t* pixelData, int width, int height, const int tgt_size, const bool _01, const bool HWC) {

    size_t shape = tgt_size * tgt_size * 4;
    std::vector<float> result(shape);
    preprocessImage(pixelData, width, height, result.data(), shape, tgt_size, _01, HWC);
    return result;

    // 1. Wrap the Android pixel data in a cv::Mat (Zero copy if possible)
    // Assumes Android Bitmap is ARGB_8888 (which is actually RGBA in memory)
//    cv::Mat original(height, width, CV_8UC4, pixelData);
//
//    // 2. Extract Alpha Channel to find the object
//    cv::Mat alpha;
//    cv::extractChannel(original, alpha, 3);
//
//    // 3. Find Bounding Box (Equivalent to np.flatnonzero)
//    // Threshold alpha to be sure we ignore semi-transparent noise
//    cv::Mat mask;
//    cv::threshold(alpha, mask, 0.5, 255, cv::THRESH_BINARY);
////    cv::Rect bbox = cv::boundingRect(mask);
////
////    // Handle empty image case
////    if (bbox.empty()) {
////        // Return blank black tensor or handle error
////        return std::vector<float>(tgt_size * tgt_size * 4, 0.0f);
////    }
//
//    // 4. Calculate Centering Geometry (Python: foreground_crop)
//    // Reduce to X-axis (collapse rows) -> checking columns
//    cv::Mat col_sum;
//    cv::reduce(mask, col_sum, 0, cv::REDUCE_MAX, CV_8U); // Max is safer than Sum for binary 0/255
//    // Reduce to Y-axis (collapse cols) -> checking rows
//    cv::Mat row_sum;
//    cv::reduce(mask, row_sum, 1, cv::REDUCE_MAX, CV_8U);
//    int min_x = -1, max_x = -1;
//    int min_y = -1, max_y = -1;
//    // Find X bounds (scan the collapsed row)
//    uint8_t* pCols = col_sum.data;
//    for (int x = 0; x < width; ++x) {
//        if (pCols[x] > 0) {
//            if (min_x == -1) min_x = x;
//            max_x = x;
//        }
//    }
//    // Find Y bounds (scan the collapsed col)
//    uint8_t* pRows = row_sum.data;
//    for (int y = 0; y < height; ++y) {
//        if (pRows[y] > 0) {
//            if (min_y == -1) min_y = y;
//            max_y = y;
//        }
//    }
//
//    LOGW("max x: %zu, max y: %zu, min x: %zu, min y: %zu", max_x, max_y, min_x, min_y);
//
//    float h = max_y - min_y;
//    float w = max_x - min_x;
//    float yc = (max_y + min_y) / 2.0f;
//    float xc = (max_x + min_x) / 2.0f;
//
//    float crop_ratio = 1.3f;
//    float scale_source = std::max(h, w) * crop_ratio;
////    LOGW("[preprocessing:] h: %zu, w %zu, y: %zu, yc: %zu, x: %zu, xc: %zu, scale: %zu",
////         h, w, bbox.y, yc, bbox.x, xc, scale_source);
//    LOGW("[preprocessing:] h: %f, w %f, yc: %f, xc: %f, scale: %f",
//         h, w, yc, xc, scale_source);
//
//    // We want to map the square region centered at (xc, yc) with size 'scale_source'
//    // into a 512x512 destination image.
//    float target_size = (float)tgt_size; // 512.0f;
//    float s = 1.0f;// target_size / scale_source;
//
//    // Affine Matrix: [ s  0  tx ]
//    //                [ 0  s  ty ]
//    float tx = xc - (scale_source / 2.0f) ;
//    float ty = yc - (scale_source / 2.0f) ;
//
//    cv::Mat M = (cv::Mat_<double>(2, 3) << s, 0, tx, 0, s, ty);
//
//    // 5. Warp (Crop + Center in one step)
//    cv::Mat warped;// = original.clone();
//    cv::warpAffine(original, warped, M,  cv::Size((int)scale_source, (int)scale_source),
//                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0)
//                   );
//
//    // 6. Resize image
//    cv::Mat resizedImg;
//    cv::resize(warped, resizedImg, cv::Size(tgt_size, tgt_size), 1.0f, 1.0f,cv::INTER_LANCZOS4);
//
//    // Convert to Float
//    cv::Mat float_img;
////    warped.convertTo(float_img, CV_32FC4); // 0..255 range
//    resizedImg.convertTo(float_img, CV_32FC4); // 0..255 range
//
//    // 7. Pack into std::vector (HWC -> CHW if needed)
//    int num_pixels = tgt_size * tgt_size; //height * width; //
//    size_t output_size = num_pixels*4;//tgt_size * tgt_size * 4
//    std::vector<float> tensor_data(output_size);
//    // Efficient HWC to CHW loop
//    // OpenCV stores as BGR or RGB depending on how you loaded, usually BGR by default but
//    // Android Bitmap is RGBA, so cvtColor RGBA2RGB gives RGB.
//    float* ptr = reinterpret_cast<float*>(float_img.data);
//    for (int i = 0; i < num_pixels; ++i) {
//        tensor_data[i] = ptr[i * 4 + 0];                // R
//        tensor_data[num_pixels + i] = ptr[i * 4 + 1];   // G
//        tensor_data[2 * num_pixels + i] = ptr[i * 4 + 2]; // B
//        tensor_data[3 * num_pixels + i] = ptr[i * 4 + 3]; // A
//    }
//    return tensor_data;
}

// Returns the processed CHW or HWC data (depending on what your model needs)
void Spar3DPipeline::preprocessImage(uint8_t* pixelData, int width, int height, float* output_buffer, const size_t output_elements, const int tgt_size, const bool _01, const bool HWC) {

    // 1. Wrap the Android pixel data in a cv::Mat (Zero copy if possible)
    // Assumes Android Bitmap is ARGB_8888 (which is actually RGBA in memory)
    cv::Mat original(height, width, CV_8UC4, pixelData);

    // 2. Extract Alpha Channel to find the object
    cv::Mat alpha;
    cv::extractChannel(original, alpha, 3);

    // 3. Find Bounding Box (Equivalent to np.flatnonzero)
    // Threshold alpha to be sure we ignore semi-transparent noise
    cv::Mat mask;
    cv::threshold(alpha, mask, 0.5, 255, cv::THRESH_BINARY);
//    cv::Rect bbox = cv::boundingRect(mask);

    // 4. Calculate Centering Geometry (Python: foreground_crop)
    // Reduce to X-axis (collapse rows) -> checking columns
    cv::Mat col_sum;
    cv::reduce(mask, col_sum, 0, cv::REDUCE_MAX, CV_8U); // Max is safer than Sum for binary 0/255
    // Reduce to Y-axis (collapse cols) -> checking rows
    cv::Mat row_sum;
    cv::reduce(mask, row_sum, 1, cv::REDUCE_MAX, CV_8U);
    int min_x = -1, max_x = -1;
    int min_y = -1, max_y = -1;
    // Find X bounds (scan the collapsed row)
    uint8_t* pCols = col_sum.data;
    for (int x = 0; x < width; ++x) {
        if (pCols[x] > 0) {
            if (min_x == -1) min_x = x;
            max_x = x;
        }
    }
    // Find Y bounds (scan the collapsed col)
    uint8_t* pRows = row_sum.data;
    for (int y = 0; y < height; ++y) {
        if (pRows[y] > 0) {
            if (min_y == -1) min_y = y;
            max_y = y;
        }
    }
    // handle empty case
    if (min_x == -1 || min_y == -1) {
        min_x = 0; max_x = width - 1;
        min_y = 0; max_y = height - 1;
    }
//    LOGW("max x: %zu, max y: %zu, min x: %zu, min y: %zu", max_x, max_y, min_x, min_y);
    float h = max_y - min_y;
    float w = max_x - min_x;
    float yc = (max_y + min_y) * 0.5f;
    float xc = (max_x + min_x) * 0.5f;

    float crop_ratio = 1.3f;
    float scale_source = std::max(h, w) * crop_ratio;
    LOGW("[preprocessing:] h: %f, w %f, yc: %f, xc: %f, scale: %f",
         h, w, yc, xc, scale_source);

    // We want to map the square region centered at (xc, yc) with size 'scale_source'
    // into a 512x512 destination image.
    float target_size = (float)tgt_size; // 512.0f;
    float s = 1.0f;// target_size / scale_source;

    // Affine Matrix: [ s  0  tx ]
    //                [ 0  s  ty ]
    float tx = xc - (scale_source * 0.5f) ;
    float ty = yc - (scale_source * 0.5f) ;

    cv::Mat M = (cv::Mat_<double>(2, 3) << s, 0, tx, 0, s, ty);

    // 5. Warp (Crop + Center in one step)
    cv::Mat warped; // = original.clone();
    cv::warpAffine(original, warped, M,  cv::Size((int)scale_source, (int)scale_source),
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0)
                   );

    // 6. Resize to target size
    cv::Mat resizedImg;
    cv::resize(warped, resizedImg, cv::Size(tgt_size, tgt_size), 1.0, 1.0,cv::INTER_LANCZOS4);

    // Convert to Float
    const double cv_scale = _01 ? (1.0/255.0) : 1.0;
    cv::Mat float_img;
    resizedImg.convertTo(float_img, CV_32FC4, cv_scale); // 0..255 or 0..1 range
    double minVal, maxVal;
    cv::minMaxLoc(float_img, &minVal, &maxVal);
    LOGW("Pixel Range Check: Min = %f, Max = %f", minVal, maxVal);
    if (!float_img.isContinuous()) {
        float_img = float_img.clone(); // cheap for small images like 512^2; else iterate rows
    }

    // 7. Pack into std::vector (HWC -> CHW if needed)
//    std::vector<float> tensor_data(tgt_size * tgt_size * 4);
    // Efficient HWC to CHW loop
    // OpenCV stores as BGR or RGB depending on how you loaded, usually BGR by default but
    // Android Bitmap is RGBA, so cvtColor RGBA2RGB gives RGB.
    int num_pixels = tgt_size * tgt_size;
    float* ptr = float_img.ptr<float>(0); // reinterpret_cast<float*>(float_img.data);
    float* dst = output_buffer;
    assert(output_elements >= num_pixels*4);
//    const float m = _01 ? (1.0f/255.0f) : 1.0f;
    if (HWC) {
        for (int i = 0; i < num_pixels; ++i) {
            dst[i * 4 + 0] = ptr[i * 4 + 0];
            dst[i * 4 + 1] = ptr[i * 4 + 1];
            dst[i * 4 + 2] = ptr[i * 4 + 2];
            dst[i * 4 + 3] = ptr[i * 4 + 3];
        }
    } else {
        float *r_plane = dst;
        float *g_plane = dst + num_pixels;
        float *b_plane = dst + (2 * num_pixels);
        float *a_plane = dst + (3 * num_pixels);
        // Unroll the loop slightly or let compiler optimize
        for (int i = 0; i < num_pixels; ++i) {
            r_plane[i] = ptr[i * 4 + 0]; // R
            g_plane[i] = ptr[i * 4 + 1]; // G
            b_plane[i] = ptr[i * 4 + 2]; // B
            a_plane[i] = ptr[i * 4 + 3]; // A
        }
    }
}


void Spar3DPipeline::test_spill(uint8_t* img, int ori_width, int ori_height) {
    std::string execution_summary;
    const bool reset_sessions = true;

    // first need to convert to RGBA, remove background and resize to 512 then get array in [0,1]
    // input image is likely CHW, so make sure to transpose if needed
    LOGI("[Pipeline:] preprocessing input image...");
    const auto input_img_wsName = gr_runners_.img_preparer->last().inputBinding.at("image_array");
    auto* input_img_array = static_cast<float*>(ws_.data(input_img_wsName)); // should be 512x512x4
    const auto& input_img_tinfo = ws_.tinfoOf(input_img_wsName);
    LOGI("input image tensor dimensions:");
    for (const auto& d : input_img_tinfo->dims) {
        LOGI("%zu", d);
    }
    assert(input_img_tinfo->numel() == inf_config_.img_height * inf_config_.img_width * 4);
    preprocessImage(img, ori_width, ori_height,
                    input_img_array, input_img_tinfo->numel(),
                    inf_config_.img_width, true, true);

    LOGI("[Pipeline:] Network-preparing image...");
    execution_summary += runGraph(*gr_runners_.img_preparer, reset_sessions);

    LOGI("[Pipeline:] running pdiff_cond...");
    execution_summary += runGraph(*gr_runners_.pdiff_cond, reset_sessions);

    auto* rgb_cond_unsq = static_cast<float*>(ws_.data("rgb_cond_unsq"));
    auto* mask_cond_unsq = static_cast<float*>(ws_.data("mask_cond_unsq"));
    auto* allvision_tester_tensor = static_cast<float*>(ws_.data("allvision_tester_tensor"));
    LOGI("initializing allvision tensor...");
    int num_pixels = 512 * 512;
    #pragma omp parallel for
    for (int i = 0; i < num_pixels; ++i) {
        // Calculate input offsets
        int rgb_idx = i * 3;  // Stride of 3 for RGB
        int mask_idx = i;     // Stride of 1 for Mask
        // Calculate output offset
        int out_idx = i * 4;  // Stride of 4 for RGBA
        // Copy R, G, B
        allvision_tester_tensor[out_idx + 0] = rgb_cond_unsq[rgb_idx + 0];
        allvision_tester_tensor[out_idx + 1] = rgb_cond_unsq[rgb_idx + 1];
        allvision_tester_tensor[out_idx + 2] = rgb_cond_unsq[rgb_idx + 2];
        // Copy Alpha (Mask)
        allvision_tester_tensor[out_idx + 3] = mask_cond_unsq[mask_idx];
    }

    LOGI("RUNNING ALL VISION TESTER...");
    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < 19; ++i) {
//        execution_summary += runGraph(*gr_runners_.tester_allvision, false);
//    }
    execution_summary += runGraph(*gr_runners_.tester_allvision, reset_sessions);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    LOGW("allvision runtime: %f s", elapsed.count());
    LOGI("DONE WITH ALL VISION TESTER!");

}

void Spar3DPipeline::test_imest(uint8_t* img, int ori_width, int ori_height) {
    std::string execution_summary;
    const bool reset_sessions = false;

    // first need to convert to RGBA, remove background and resize to 512 then get array in [0,1]
    // input image is likely CHW, so make sure to transpose if needed
    LOGI("[Pipeline:] preprocessing input image...");
    const auto input_img_wsName = gr_runners_.img_preparer->last().inputBinding.at("image_array");
    auto* input_img_array = static_cast<float*>(ws_.data(input_img_wsName)); // should be 512x512x4
    const auto& input_img_tinfo = ws_.tinfoOf(input_img_wsName);
    LOGI("input image tensor dimensions:");
    for (const auto& d : input_img_tinfo->dims) {
        LOGI("%zu", d);
    }
    assert(input_img_tinfo->numel() == inf_config_.img_height * inf_config_.img_width * 4);
    preprocessImage(img, ori_width, ori_height,
                    input_img_array, input_img_tinfo->numel(),
                    inf_config_.img_width, true, true);

    LOGI("[Pipeline:] Network-preparing image...");
    execution_summary += runGraph(*gr_runners_.img_preparer, reset_sessions);

    LOGI("[Pipeline:] running pdiff_cond...");
    execution_summary += runGraph(*gr_runners_.pdiff_cond, reset_sessions);

    auto* rgb_cond_unsq = static_cast<float*>(ws_.data("rgb_cond_unsq"));
    auto* mask_cond_unsq = static_cast<float*>(ws_.data("mask_cond_unsq"));

    LOGI("[Pipeline:] initializing rgb_mask_image...");
    const auto rgb_mask_image_wsName = gr_runners_.image_estimator->last().inputBinding.at("rgb_mask_image");
    const auto& rgb_mask_image_tinfo = ws_.tinfoOf(rgb_mask_image_wsName);
    auto* rgb_mask_image = static_cast<float*>(ws_.data(rgb_mask_image_wsName));
    std::string rgb_mask_log = "rgb_mask_image dimensions: ";
    for (const auto& d : rgb_mask_image_tinfo->dims) rgb_mask_log += std::to_string(d) + " ";
    LOGI("%s", rgb_mask_log.c_str());
    int num_pixels = 512 * 512;
    #pragma omp parallel for
    for (int i = 0; i < num_pixels; ++i) {
        // Calculate input offsets
        int rgb_idx = i * 3;  // Stride of 3 for RGB
        int mask_idx = i;     // Stride of 1 for Mask
        // Calculate output offset
        int out_idx = i * 4;  // Stride of 4 for RGBA
        // Copy R, G, B
        rgb_mask_image[out_idx + 0] = rgb_cond_unsq[rgb_idx + 0];
        rgb_mask_image[out_idx + 1] = rgb_cond_unsq[rgb_idx + 1];
        rgb_mask_image[out_idx + 2] = rgb_cond_unsq[rgb_idx + 2];
        // Copy Alpha (Mask)
        rgb_mask_image[out_idx + 3] = mask_cond_unsq[mask_idx];
    }

    LOGI("RUNNING IMAGE ESTIMATOR...");
    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < 19; ++i) {
//        execution_summary += runGraph(*gr_runners_.tester_allvision, false);
//    }
    execution_summary += runGraph(*gr_runners_.image_estimator, reset_sessions);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    LOGW("Image estimator runtime: %f s", elapsed.count());

    auto* r0_t = static_cast<float*>(ws_.data("roughness_0"));
    auto r0 = r0_t[0];
    auto* r1_t = static_cast<float*>(ws_.data("roughness_1"));
    auto r1 = r1_t[0];
    auto* m1_t = static_cast<float*>(ws_.data("metallic_1"));
    auto m1 = m1_t[0];
    auto* m0_t = static_cast<float*>(ws_.data("metallic_0"));
    auto m0 = m0_t[0];
    LOGI("roughness 0: %f", r0);
    float mc0 = distributionUtils::softplus(m1 + 1.0f);
    float mc1 = distributionUtils::softplus(m0 + 1.0f);
    LOGW("metallic concentration1: %f, concentration0: %f", mc1, mc0);

    distributionUtils::Beta roughness_dist(r0, r1, 1.0f, 1.0f);
    distributionUtils::Beta metallic_dist(m0, m1, 1.0f, 1.0f);
    float roughness = roughness_dist.distribution_eval(inf_config_.roughness_distribution_eval);
    float metallic = metallic_dist.distribution_eval(inf_config_.metallic_distribution_eval);
    LOGW("roughness: %f, metallic: %f", roughness, metallic);
    LOGI("DONE WITH Image estimator test!");
}

static void releaseGRtensors(GraphRunner& gr, TensorWorkspace& ws, bool inputs_only=false) {
    for (auto& n : gr.getNodes()) {
        for (auto& input : n.session.get()->inputs()) {
            const auto wsname = n.inputBinding.at(input.name);
                LOGI("Releasing %s / %s", input.name.c_str(), wsname.c_str());
                if (ws.has(wsname)) ws.release(wsname);
        }
        if (!inputs_only) {
            for (auto& output : n.session.get()->outputs()) {
                const auto wsname = n.outputBinding.at(output.name);
                LOGI("Releasing %s / %s", output.name.c_str(), wsname.c_str());
                if (ws.has(wsname)) ws.release(wsname);
            }
        }
    }
}

void Spar3DPipeline::unscale_channels(float* tensor, const TensorInfo& tinfo) {
    size_t B = tinfo.dims[0];
    size_t N = tinfo.dims[1];
    size_t C = tinfo.dims[2];
    if (N != 6) {LOGW("CANNOT UNSCALE TENSOR AS N != %zu", N); return;}
    std::vector<float> inv_scales(N);
    std::vector<float> biases(N);
    for(size_t i = 0; i < N; ++i) {
        biases[i] = inf_config_.sampler_channel_biases[i];
        inv_scales[i] = 1.0f / inf_config_.sampler_channel_scales[i];
    }
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            float *dst = tensor + (b * N * C) + (n * C);
            float bias = biases[n];
            float scale_inv = inv_scales[n];
            // contiguous memory access
            for (size_t c = 0; c < C; ++c) {
                dst[c] = (dst[c] - bias) * scale_inv;
            }
        }
    }
}

void Spar3DPipeline::prepareToRunSC2() {
    // initialize tokens and triplane_tokens from .bin files
    const auto tokens_wsName = gr_runners_.scene_codes2_3->last().inputBinding.at("tokens");
    auto* tokens = static_cast<float*>(ws_.data(tokens_wsName));
    const auto& tokens_tinfo = ws_.tinfoOf(tokens_wsName);

    const auto triplane_tokens_wsName = "triplane_tokens";
    auto* triplane_tokens = static_cast<float*>(ws_.data(triplane_tokens_wsName));
    const auto& triplane_tokens_tinfo = ws_.tinfoOf(triplane_tokens_wsName);

    std::string triplane_tokens_path = "/sdcard/Android/data/com.example.snpechainingdemo/files/spar3d/triplane_tokens.bin";
    std::string tokens_path = "/sdcard/Android/data/com.example.snpechainingdemo/files/spar3d/tokens.bin";

    if (!readFileToBuffer(triplane_tokens_path, triplane_tokens, triplane_tokens_tinfo->bytes())) {
        LOGE("COULD NOT INITIALIZE TRIPLANE TOKENS FROM FILE %s", triplane_tokens_path.c_str());
        return;
    }
    LOGI("Read triplane tokens from file!");
    logNumbers(triplane_tokens, *triplane_tokens_tinfo, triplane_tokens_wsName);
    if (!readFileToBuffer(tokens_path, tokens, tokens_tinfo->bytes())) {
        LOGE("COULD NOT INITIALIZE TOKENS FROM FILE %s", tokens_path.c_str());
        return;
    }
    LOGI("Read tokens from file!");
    logNumbers(tokens, *tokens_tinfo, tokens_wsName);

    // initialize init_latents from .bin file
    const auto& sc2_nodes = gr_runners_.scene_codes2->getNodes();
    const auto init_latents_wsName = gr_runners_.scene_codes2->getNode(sc2_nodes[0].name).inputBinding.at("init_latents");
    auto* init_latents = static_cast<float*>(ws_.data(init_latents_wsName));
    const auto& init_latents_tinfo = ws_.tinfoOf(init_latents_wsName);

    std::string init_latents_path = "/sdcard/Android/data/com.example.snpechainingdemo/files/spar3d/init_latents.bin";
    if (!readFileToBuffer(init_latents_path, init_latents, init_latents_tinfo->bytes())) {
        LOGE("COULD NOT INITIALIZE INIT LATENTS FROM FILE %s", triplane_tokens_path.c_str());
        return;
    }
    LOGI("Read init_latents from file!");

    // initialize cross_mask and latent_mask from .bin files
    const auto latent_mask_wsName = gr_runners_.scene_codes2->getNode(sc2_nodes[0].name).inputBinding.at("latent_mask");
    auto* latent_mask = static_cast<float*>(ws_.data(latent_mask_wsName));
    const auto& latent_mask_tinfo = ws_.tinfoOf(latent_mask_wsName);
    std::string latent_mask_asset = "latent_mask.bin";
    if (!readAssetToBuffer(mgr_, latent_mask_asset.c_str(), latent_mask, latent_mask_tinfo->bytes())) {
        LOGE("COULD NOT INITIALIZE LATENT MASK FROM ASSET %s", latent_mask_asset.c_str());
        return;
    }

    const auto cross_mask_wsName = gr_runners_.scene_codes2->getNode(sc2_nodes[0].name).inputBinding.at("cross_mask");
    auto* cross_mask = static_cast<float*>(ws_.data(cross_mask_wsName));
    const auto& cross_mask_tinfo = ws_.tinfoOf(cross_mask_wsName);
    std::string cross_mask_asset = "cross_mask.bin";
    if (!readAssetToBuffer(mgr_, cross_mask_asset.c_str(), cross_mask, cross_mask_tinfo->bytes())) {
        LOGE("COULD NOT INITIALIZE CROSS MASK FROM ASSET %s", cross_mask_asset.c_str());
        return;
    }

    // concatenate outputs of scene_codes1
    const auto input_image_tokens_wsName = gr_runners_.scene_codes1->last().outputBinding.at("output_0");
    auto* input_image_tokens = static_cast<float*>(ws_.data(input_image_tokens_wsName));
    const auto& input_image_tokens_tinfo = ws_.tinfoOf(input_image_tokens_wsName);

    const auto pc_embeds_wsName = gr_runners_.scene_codes1->last().outputBinding.at("output_1");
    auto* pc_embeds = static_cast<float*>(ws_.data(pc_embeds_wsName));
    const auto& pc_embeds_tinfo = ws_.tinfoOf(pc_embeds_wsName);

    const auto cross_tokens_wsName = gr_runners_.scene_codes2->getNode(sc2_nodes[0].name).inputBinding.at("cross_tokens");
    auto* cross_tokens = static_cast<float*>(ws_.data(cross_tokens_wsName));
    const auto& cross_tokens_tinfo = ws_.tinfoOf(cross_tokens_wsName);
    // cross_tokens = torch.cat([padded_input_image_tokens, pc_emebds], dim=1)
    // input_image_tokens is to be padded by zeros from dimension 1297 to 1312 so that it is divisible by 32
    // in practice no need to pad because cross_tokens is already zero-initialized, so we only need to copy the first 1297
    // then offset by the number of elements between 1297 and 1312 before further concatenating pc_embeds
    {
        assert(input_image_tokens_tinfo->dims[0] == 1 &&
               "Batch size must be 1 for linear concatenation!");
        assert(pc_embeds_tinfo->dims[0] == 1 && "Batch size must be 1 for pc_embeds!");
        assert(input_image_tokens_tinfo->dims[2] == pc_embeds_tinfo->dims[2] &&
               "Feature dimensions mismatch!");
    }
    std::memcpy(cross_tokens, input_image_tokens, input_image_tokens_tinfo->bytes());
    size_t offset_elements = input_image_tokens_tinfo->numel() + (15 * input_image_tokens_tinfo->dims[2]);
    auto* ct_dst = cross_tokens + offset_elements;
    {
        size_t required_space = offset_elements + pc_embeds_tinfo->numel();
        assert(required_space <= cross_tokens_tinfo->numel() && "Buffer overflow detected!");
    }
    std::memcpy(ct_dst, pc_embeds, pc_embeds_tinfo->bytes());

    LOGI("[Pipeline:] Done preparing tensors for scene_codes2!");
}

static void dump_tensor_to_file(const char* filepath, const float* data, size_t num_elements) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        LOGE("Failed to open file for writing: %s", filepath);
        return;
    }
    // Write the raw bytes of the float array
    file.write(reinterpret_cast<const char*>(data), num_elements * sizeof(float));
    file.close();
    LOGW("Successfully dumped %zu elements to %s", num_elements, filepath);
}

std::vector<float> Spar3DPipeline::runImageEstimator(std::string& execution_summary, const bool reset_sessions) {
    LOGI("Getting rgb_cond_unsq and mask_cond_unsq");
    auto* rgb_cond_unsq = static_cast<float*>(ws_.data("rgb_cond_unsq"));
    auto* mask_cond_unsq = static_cast<float*>(ws_.data("mask_cond_unsq"));
    if (!rgb_cond_unsq) {
        LOGE("rgb_cond_unsq empty!");
        return {0.0f,0.0f};
    }
    if (!mask_cond_unsq) {
        LOGE("mask_cond_unsq empty!");
        return {0.0f,0.0f};
    }

    LOGI("[Pipeline:] initializing rgb_mask_image...");
    const auto rgb_mask_image_wsName = gr_runners_.image_estimator->last().inputBinding.at("rgb_mask_image");
    const auto& rgb_mask_image_tinfo = ws_.tinfoOf(rgb_mask_image_wsName);
    auto* rgb_mask_image = static_cast<float*>(ws_.data(rgb_mask_image_wsName));
    std::string rgb_mask_log = "rgb_mask_image dimensions: ";
    for (const auto& d : rgb_mask_image_tinfo->dims) rgb_mask_log += std::to_string(d) + " ";
    LOGI("%s", rgb_mask_log.c_str());
    assert(inf_config_.img_width == rgb_mask_image_tinfo->dims[3]);
    int num_pixels = inf_config_.img_width * inf_config_.img_height; // 512 * 512;
    #pragma omp parallel for
    for (int i = 0; i < num_pixels; ++i) {
        // Calculate input offsets
        int rgb_idx = i * 3;  // Stride of 3 for RGB
        int mask_idx = i;     // Stride of 1 for Mask
        // Calculate output offset
        int out_idx = i * 4;  // Stride of 4 for RGBA
        // Copy R, G, B
        rgb_mask_image[out_idx + 0] = rgb_cond_unsq[rgb_idx + 0];
        rgb_mask_image[out_idx + 1] = rgb_cond_unsq[rgb_idx + 1];
        rgb_mask_image[out_idx + 2] = rgb_cond_unsq[rgb_idx + 2];
        // Copy Alpha (Mask)
        rgb_mask_image[out_idx + 3] = mask_cond_unsq[mask_idx];
    }

    LOGI("RUNNING IMAGE ESTIMATOR...");
    auto start = std::chrono::high_resolution_clock::now();
    execution_summary += runGraph(*gr_runners_.image_estimator, reset_sessions);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    LOGW("Image estimator runtime: %f s", elapsed.count());

    // make a beta distribution ouf of the outputs
    auto* r0_t = static_cast<float*>(ws_.data("roughness_0"));
    auto r0 = r0_t[0];
    auto* r1_t = static_cast<float*>(ws_.data("roughness_1"));
    auto r1 = r1_t[0];
    auto* m1_t = static_cast<float*>(ws_.data("metallic_1"));
    auto m1 = m1_t[0];
    auto* m0_t = static_cast<float*>(ws_.data("metallic_0"));
    auto m0 = m0_t[0];
    LOGI("roughness concentration1: %f, concentration0: %f", r1, r0);
    float mc0 = distributionUtils::softplus(m1 + 1.0f);
    float mc1 = distributionUtils::softplus(m0 + 1.0f);
    LOGI("metallic concentration1: %f, concentration0: %f", mc1, mc0);

    distributionUtils::Beta roughness_dist(r0, r1, 1.0f, 1.0f);
    distributionUtils::Beta metallic_dist(m0, m1, 1.0f, 1.0f);
    float roughness = roughness_dist.distribution_eval(inf_config_.roughness_distribution_eval);
    float metallic = metallic_dist.distribution_eval(inf_config_.metallic_distribution_eval);
    LOGW("roughness: %f, metallic: %f", roughness, metallic);
    LOGI("DONE WITH Image estimator!");
    std::vector<float> out = {roughness, metallic};
    return out;
}

void Spar3DPipeline::overall_pipeline(uint8_t* img, int ori_width, int ori_height, const std::string& glb_output_path) {
    std::string execution_summary;
    const bool reset_sessions = false;

    // first need to convert to RGBA, remove background and resize to 512 then get array in [0,1]
    // input image is likely CHW, so make sure to transpose if needed
    LOGI("[Pipeline:] preprocessing input image...");
    const auto input_img_wsName = gr_runners_.img_preparer->last().inputBinding.at("image_array");
    auto* input_img_array = static_cast<float*>(ws_.data(input_img_wsName)); // should be 512x512x4
    const auto& input_img_tinfo = ws_.tinfoOf(input_img_wsName);
    LOGI("input image tensor dimensions:");
    for (const auto& d : input_img_tinfo->dims) {
        LOGI("%zu", d);
    }
    assert(input_img_tinfo->numel() == inf_config_.img_height * inf_config_.img_width * 4);
    preprocessImage(img, ori_width, ori_height,
                    input_img_array, input_img_tinfo->numel(),
                    inf_config_.img_width, true, true);

    LOGI("[Pipeline:] Running image-preparing network...");
    execution_summary += runGraph(*gr_runners_.img_preparer, reset_sessions);
    LOGI("[Pipeline:] completely releasing img_preparer");
    gr_runners_.img_preparer->clear_everything_in_session(gr_runners_.img_preparer->last());

    LOGI("[Pipeline:] running pdiff_cond...");
    execution_summary += runGraph(*gr_runners_.pdiff_cond, reset_sessions);
    LOGI("[Pipeline:] completely releasing pdiff_cond");
//    gr_runners_.pdiff_cond->clear_everything_in_session(gr_runners_.pdiff_cond->last());
    gr_runners_.pdiff_cond->clear_everything_all_sessions();

    LOGI("[Pipeline:] preparing noise and conditioning...");
    // guidance_scale != 0 and != 1.0
    // condition should already have been initialized by zeros and should be [2,1297,1024]
    // copy cond_tokens in the first dimension
    const auto cond_tokens_wsName = gr_runners_.pdiff_cond->last().outputBinding.at("output_0");
    auto* cond_tokens = static_cast<float*>(ws_.data(cond_tokens_wsName));
    const auto& cond_tokens_tinfo = ws_.tinfoOf(cond_tokens_wsName);

    const auto denoiser_condition_wsName = gr_runners_.one_step_denoiser->last().inputBinding.at("condition");
    auto* denoiser_condition = static_cast<float*>(ws_.data(denoiser_condition_wsName));
    const auto& denoiser_condition_tinfo = ws_.tinfoOf(denoiser_condition_wsName);
    logNumbers(denoiser_condition, *denoiser_condition_tinfo, denoiser_condition_wsName);

    std::memcpy(denoiser_condition, cond_tokens, cond_tokens_tinfo->bytes());
    logNumbers(denoiser_condition, *denoiser_condition_tinfo, denoiser_condition_wsName);

    // fill noise tensor with randn noise
    const auto noise_wsName = gr_runners_.one_step_denoiser->last().inputBinding.at("noise");
    const auto& noise_tinfo = ws_.tinfoOf(noise_wsName);
    auto* noise = static_cast<float*>(ws_.data(noise_wsName));
//    uint32_t seed = 42;
//    size_t bytes_to_fill = noise_tinfo->bytes() / 2;
//    fillRandom(noise, bytes_to_fill, 0.0, 1.0, seed);
//    // replicate
//    std::memcpy(noise + noise_tinfo->numel()/2, noise, bytes_to_fill);
    // read noise from file
    if (!readAssetToBuffer(mgr_, "noise42.bin", noise, noise_tinfo->bytes())) {
        LOGW("[Pipeline] COULD NOT READ NOISE FROM ASSET!");
    }
    logNumbers(noise, *noise_tinfo, noise_wsName);

    LOGI("[Pipeline:] running denoiser loop...");
    const auto timestep_wsName = gr_runners_.one_step_denoiser->last().inputBinding.at("t");
    const auto& timestep_tinfo = ws_.tinfoOf(timestep_wsName);
    auto* timestep = static_cast<int32_t*>(ws_.data(timestep_wsName));

    const auto denoising_loop_sample_wsName = gr_runners_.one_step_denoiser->last().outputBinding.at("output_0");
    LOGI("[Pipeline:] Found tensor workspace for denoising loop sample: WS name: %s", denoising_loop_sample_wsName.c_str());
    auto* denoising_loop_sample = static_cast<float*>(ws_.data(denoising_loop_sample_wsName));

    for (int i = inf_config_.num_timesteps - 1; i >= 0; --i) {
        LOGI("[denoiser loop:] ind: %d", i);
        // prepare timestep tensor
        timestep[0] = static_cast<int32_t>(i);
        timestep[1] = static_cast<int32_t>(i);
        logNumbers(timestep, *timestep_tinfo, timestep_wsName);
        // everything is ready to call the one-step denoiser
        execution_summary += runGraph(*gr_runners_.one_step_denoiser, false);
        // update the noisy image for denoising
        std::memcpy(noise, denoising_loop_sample, noise_tinfo->bytes());
//        break; // TODO: remove this
    }
    // get sample
    const auto denoised_sample_wsName = gr_runners_.one_step_denoiser->last().outputBinding.at("output_1");
    LOGI("[Pipeline:] Found tensor workspace for denoiser sample: WS name: %s", denoised_sample_wsName.c_str());
//    gr_runners_.one_step_denoiser->last().session.get()->reset(); // free up memory
    auto* denoised_sample = static_cast<float*>(ws_.data(denoised_sample_wsName));
    const auto& denoised_sample_tinfo = ws_.tinfoOf(denoised_sample_wsName);
    int d_m = denoised_sample_tinfo->dims[1];
    assert(d_m == 6);
    int d_n = denoised_sample_tinfo->dims[2];
    assert(d_n == 512);
    unscale_channels(denoised_sample, *denoised_sample_tinfo);
    logNumbers(denoised_sample, *denoised_sample_tinfo, denoised_sample_wsName);

    LOGI("[Pipeline:] completely releasing one_step_denoiser");
    gr_runners_.one_step_denoiser->clear_everything_in_session(gr_runners_.one_step_denoiser->last());

    const auto pc_cond_wsName = gr_runners_.scene_codes1->last().inputBinding.at("pc_cond");
    auto* pc_cond = static_cast<float*>(ws_.data(pc_cond_wsName));
    const auto& pc_cond_tinfo = ws_.tinfoOf(pc_cond_wsName);
    LOGI("pc_cond dims:");
    for (auto d : pc_cond_tinfo->dims) {
        LOGI("%lu", d);
    }

    // transpose denoised_sample and fill pc_cond
    // we can do a linear walkthrough here because the number of rows is small = 6
    // Source Pointers: Create 6 pointers, one for the start of each row
    const float* src0 = denoised_sample;           // Start of Row 0
    const float* src1 = src0 + d_n;                // Start of Row 1
    const float* src2 = src1 + d_n;                // ...
    const float* src3 = src2 + d_n;
    const float* src4 = src3 + d_n;
    const float* src5 = src4 + d_n;

    float* dst = pc_cond; // Destination walker

    // Loop 512 times (once per column)
    for (int i = 0; i < d_n; ++i) {
        // 1. Read 6 values (one from each row cursor)
        float v0 = src0[i];
        float v1 = src1[i];
        float v2 = src2[i];
        float v3 = src3[i];
        float v4 = src4[i];
        float v5 = src5[i];
        // 2. Write them contiguously into the destination (which acts as a row in the 512x6 matrix)
        dst[0] = v0;
        dst[1] = v1;
        dst[2] = v2;
        dst[3] = v3;
        dst[4] = v4;
        dst[5] = v5;
        // 3. Advance destination pointer by 6
        dst += d_m;
    }
    normalize_pc_bbox_inplace(pc_cond, pc_cond_tinfo->dims[0], pc_cond_tinfo->dims[1], pc_cond_tinfo->dims[2]);
    logNumbers(pc_cond, *pc_cond_tinfo, pc_cond_wsName);

    // call scene encoders
    LOGI("[Pipeline:] Running scene_codes1...");
    execution_summary += runGraph(*gr_runners_.scene_codes1, reset_sessions);
    LOGI("[Pipeline:] completely releasing scene_codes1");
    gr_runners_.scene_codes1->clear_everything_in_session(gr_runners_.scene_codes1->last());

    {
        const auto out_scene1_0_wsName = gr_runners_.scene_codes1->last().outputBinding.at("output_0");
        auto* out_scene1_0 = static_cast<float*>(ws_.data(out_scene1_0_wsName));
        const auto& out_scene1_0_tinfo = ws_.tinfoOf(out_scene1_0_wsName);
        const auto out_scene1_1_wsName = gr_runners_.scene_codes1->last().outputBinding.at("output_1");
        auto* out_scene1_1 = static_cast<float*>(ws_.data(out_scene1_1_wsName));
        const auto& out_scene1_1_tinfo = ws_.tinfoOf(out_scene1_1_wsName);
        logNumbers(out_scene1_0, *out_scene1_0_tinfo, out_scene1_0_wsName);
        logNumbers(out_scene1_1, *out_scene1_1_tinfo, out_scene1_1_wsName);
    }

    LOGI("[Pipeline:] Preparing to run scene_codes2...");
    prepareToRunSC2();
    LOGI("[Pipeline:] Running scene_codes_2...");
    execution_summary += runGraph(*gr_runners_.scene_codes2, reset_sessions);

    auto* tt0 = static_cast<float*>(ws_.data("triplane_tokens_2"));
    auto* lt0 = static_cast<float*>(ws_.data("latent_tokens_2"));
    auto* tt3 = static_cast<float*>(ws_.data("triplane_tokens_3"));
    auto* lt3 = static_cast<float*>(ws_.data("latent_tokens_3"));
    const auto& tt3_tinfo = ws_.tinfoOf("triplane_tokens_3");
    const auto& lt3_tinfo = ws_.tinfoOf("latent_tokens_3");
    std::memcpy(tt3, tt0, tt3_tinfo->bytes());
    std::memcpy(lt3, lt0, lt3_tinfo->bytes());

    LOGI("Releasing tensors from memory and completely resetting scene_codes2!");
    LOGI("RSS before release and reset: %zu KB", readProcessRssKb());
    ws_.release("triplane_tokens_2");
    ws_.release("latent_tokens_2");
    for (auto& n : gr_runners_.scene_codes2->getNodes()) {
        for (auto &input: n.session.get()->inputs()) {
//            if (input.name != "pc_embeds" && input.name != "input_image_tokens") {
            if (input.name != "cross_tokens" && input.name != "latent_mask" && input.name != "cross_mask") {
                const auto wsname = n.inputBinding.at(input.name);
                LOGI("Releasing %s / %s", input.name.c_str(), wsname.c_str());
                if (ws_.has(wsname)) ws_.release(wsname);
            }
        }
        n.session.get()->resetEverything();
    }
    usleep(200*1000);
    LOGI("RSS after reset and sleep: %zu KB", readProcessRssKb());

    LOGI("[Pipeline:] Running scene_codes_2's last piece...");
    execution_summary += runGraph(*gr_runners_.scene_codes2_3, reset_sessions);
    LOGI("[Pipeline:] completely releasing scene_codes2_3");
    gr_runners_.scene_codes2_3->clear_everything_in_session(gr_runners_.scene_codes2_3->last());
    usleep(200*1000);
    LOGI("RSS after clearing sc2_3: %zu KB", readProcessRssKb());
    LOGI("Releasing tensors from img_preparer");
    releaseGRtensors(*gr_runners_.img_preparer, ws_);
    LOGI("Releasing tensors from one_step_denoiser");
    releaseGRtensors(*gr_runners_.one_step_denoiser, ws_);
//    LOGI("Releasing tensors from scene_codes1");
//    releaseGRtensors(*gr_runners_.scene_codes1, ws_);
    LOGI("Releasing tensors from scene_codes2");
    releaseGRtensors(*gr_runners_.scene_codes2, ws_);
    LOGI("Releasing tensors from scene_codes2_3 inputs only");
    releaseGRtensors(*gr_runners_.scene_codes2_3, ws_, true);
    LOGI("RSS after releasing workspace: %zu KB", readProcessRssKb());

    const auto scene_codes_wsName = gr_runners_.scene_codes2_3->last().outputBinding.at("output_0");
    const auto& scene_codes_tinfo = ws_.tinfoOf(scene_codes_wsName);
    auto* scene_codes = static_cast<float*>(ws_.data(scene_codes_wsName));
    {
//        std::string scene_codes_dump_path = "/sdcard/Android/data/com.example.snpechainingdemo/files/spar3d/scene_codes_dump.bin";
//        dump_tensor_to_file(scene_codes_dump_path.c_str(), scene_codes, scene_codes_tinfo->numel());
//        if (!readFileToBuffer(scene_codes_dump_path, scene_codes, scene_codes_tinfo->bytes())) LOGE("COULD NOT READ SCENE_CODES FROM FILE!");
        // some logging
        std::string scene_codes_log = "scene codes dims: ";
        for (const auto&d : scene_codes_tinfo->dims) scene_codes_log += std::to_string(d)+" ";
        LOGI("[pipeline:] %s", scene_codes_log.c_str());
        logNumbers(scene_codes, *scene_codes_tinfo, scene_codes_wsName);
    }

    LOGI("[Pipeline:] Preparing grid vertices...");
    // create grid_vertices and fill it from file
    std::string grid_vertices_wsName = "grid_vertices";
    TensorInfo grid_vertices_tinfo;
    grid_vertices_tinfo.name = grid_vertices_wsName;
    grid_vertices_tinfo.dims = {535882, 3}; // dimensions are hard-coded and correspond to original python code
    {
        std::string emsg;
        // allocate buffer and zero initialize
        ensureWorkspaceBuffer(ws_, "grid_vertices", grid_vertices_tinfo, &emsg);
    }
    // get buffer and fill from file
    auto* grid_vertices_ptr = static_cast<float*>(ws_.data(grid_vertices_wsName));
    readAssetToBuffer(mgr_, "grid_vertices.bin", grid_vertices_ptr, grid_vertices_tinfo.bytes());
    // scale tensor
    scale_tensor_inplace(grid_vertices_ptr, grid_vertices_tinfo.dims[0], 0.0f, 1.0f, -inf_config_.radius, inf_config_.radius);
    logNumbers(grid_vertices_ptr, grid_vertices_tinfo, grid_vertices_wsName);

    // values = query_triplane(grid_vertices, triplane)
    const auto values_wsName = gr_runners_.triplanesToProtoMesh->last().inputBinding.at("values");
    const auto& values_tinfo = ws_.tinfoOf(values_wsName);
    auto* values = static_cast<float*>(ws_.data(values_wsName));

    LOGI("[Pipeline:] running query triplane to get values...");
    query_triplane_optimized(grid_vertices_ptr, scene_codes, values, grid_vertices_tinfo.dims[0],
                             scene_codes_tinfo->dims[2], scene_codes_tinfo->dims[3], scene_codes_tinfo->dims[4],
                             inf_config_.radius);
    logNumbers(values, *values_tinfo, values_wsName, 8, 'i',"Values: ");
    {
        for (size_t i; i < values_tinfo->numel(); ++i) {
            if (std::isnan(values[i])) {
                LOGE("[Pipeline:] VALUES has NANs!");
                break;
            }
        }
    }

//    std::string values_path = "/sdcard/Android/data/com.example.snpechainingdemo/files/spar3d/values.bin";
//    if (!readFileToBuffer(values_path, values, values_tinfo->bytes())) {
//        LOGE("COULD NOT INITIALIZE TRIPLANE TOKENS FROM FILE %s", values_path.c_str());
//        return;
//    }
    std::string values_log = "values log: \n";
    for (const auto& d : values_tinfo->dims) values_log += std::to_string(d) + " ";
    LOGW("%s", values_log.c_str());
    logNumbers(values, *values_tinfo, values_wsName, 12, 'i',"Values: ");
    // run network to infer sdf and deformation
    LOGI("[Pipeline:] Getting sdf and deform...");
    execution_summary += runGraph(*gr_runners_.triplanesToProtoMesh, reset_sessions);
    LOGI("[Pipeline:] completely releasing triplanesToProtoMesh");
    gr_runners_.triplanesToProtoMesh->clear_everything_in_session(gr_runners_.triplanesToProtoMesh->last());
    usleep(20*1000);
    LOGI("RSS after clearing triplanesToProtoMesh: %zu KB", readProcessRssKb());
    LOGI("Releasing grid_vertices buffer");
    ws_.release(grid_vertices_wsName);
    LOGI("RSS after releasing grid vertices: %zu KB", readProcessRssKb());

    // create mesh with v_pos and t_pos_idx
    std::vector<mtd2::Vec3> grid_vertices;
    std::vector<int> indices;
    // Load Vertices
    // Python saved flat floats, we load into Vec3 (size 12 bytes)
    if (!loadVectorFromAsset(mgr_, "grid_vertices.bin", grid_vertices)) {
        LOGE("[Pipeline:] Unable to load grid_vertices!");
    }
    // Load Indices
    // Python saved flat int32, we load into int
    if (!loadVectorFromAsset(mgr_, "indices.bin", indices)) {
        LOGE("[Pipeline:] Unable to load indices for Marching Tetrahedra Helper!");
    }
    if (grid_vertices.empty()) LOGE("[Pipeline] grid vertices empty!");
    if (indices.empty()) LOGE("[Pipeline] indices empty!");
    // Instantiate
    auto mt_helper = std::make_unique<mtd2::MarchingTetrahedraHelper>(grid_vertices, indices);

    // get sdf and deformation fields
    const auto sdf_wsName = gr_runners_.triplanesToProtoMesh->last().outputBinding.at("output_0");
    auto* sdf = static_cast<float*>(ws_.data(sdf_wsName));
    const auto& sdf_tinfo = ws_.tinfoOf(sdf_wsName);
    logNumbers(sdf, *sdf_tinfo, sdf_wsName, 12, 'w');
//    std::string sdf_dump_path = "/sdcard/Android/data/com.example.snpechainingdemo/files/spar3d/sdfNoExp_dump.bin";
//    dump_tensor_to_file(sdf_dump_path.c_str(), sdf, sdf_tinfo->numel());
    // need to do the final activation (exponential) here because it breaks NPU
    // clamp just in case
    auto exp_start = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < sdf_tinfo->numel(); ++i) {
        float val = std::clamp(sdf[i], -80.0f, 80.0f);
        sdf[i] = std::expf(val) - inf_config_.isosurface_threshold;
    }
    auto exp_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = exp_end - exp_start;
    LOGW("exp of sdf runtime: %f s", elapsed.count());

//    sdf_dump_path = "/sdcard/Android/data/com.example.snpechainingdemo/files/spar3d/sdfLocalExp_dump.bin";
//    dump_tensor_to_file(sdf_dump_path.c_str(), sdf, sdf_tinfo->numel());
//    LOGW("SDF at various indeces: %f, %f, %f, %f, %f, %f", sdf[127129],
//                                                                sdf[127455],
//                                                                sdf[127458],
//                                                                sdf[127460],
//                                                                sdf[140760],
//                                                                sdf[140763]);

    int positive_elements = 0;
    std::vector<size_t> positive_indeces;
    positive_indeces.reserve(1000);
    for (size_t i = 0; i < sdf_tinfo->numel(); ++i) {
        if (sdf[i] > 0) {
            positive_elements += 1;
            positive_indeces.push_back(i);
        }
    }
    LOGW("[Pipeline:] FOUND %zu positive elements in SDF!", positive_elements);
    LOGW("sdf bytes: %zu, sdf numel: %zu", sdf_tinfo->bytes(), sdf_tinfo->numel());

    const auto deformation_wsName = gr_runners_.triplanesToProtoMesh->last().outputBinding.at("output_1");
    auto* deformation = static_cast<float*>(ws_.data(deformation_wsName));
    const auto& deformation_tinfo = ws_.tinfoOf(deformation_wsName);

    {
        bool sdf_nan = false;
        for (size_t i = 0; i < sdf_tinfo->numel(); ++i) {
            if (std::isnan(sdf[i])) {
//                LOGE("[Pipeline:] NAN in SDF!");
//                break;
                sdf_nan = true;
                sdf[i] = -9.5f;
            }
        }
        if (sdf_nan) LOGE("[Pipeline:] NAN in SDF!");
        for (size_t i = 0; i < deformation_tinfo->numel(); ++i) {
            if (std::isnan(deformation[i])) {
                LOGE("[Pipeline:] NAN in DEFORMATION!");
                break;
            }
        }
    }


    // make mesh with v_pos already scaled
    mtd2::Mesh mesh = mt_helper->forward(sdf, deformation, bbox_);
    if (mesh.v_pos.empty()) LOGE("[Pipeline] mesh v_pos is empty!");
    if (mesh.t_pos_idx.empty()) LOGE("[Pipeline] mesh v_pos is empty!");
    std::string vpos_string = "v_pos: ";
    std::string tposidx_string = "t_pos_idx: ";
    for (size_t i = 0; i < 9; ++i) {
        vpos_string += std::to_string(mesh.v_pos[i]) + ", ";
        tposidx_string += std::to_string(mesh.t_pos_idx[i]) + ", ";
    }
    LOGW("%s", vpos_string.c_str());
    LOGW("%s", tposidx_string.c_str());
    bool nan_in_pos = false;
    for (float v : mesh.v_pos) {
        if (std::isnan(v)) {
            nan_in_pos = true;
            break;
        }
    }
    if (nan_in_pos) LOGE("[Pipeline] Found NaN in v_pos!");


    // mesh.unwrap_uv
    LOGI("[Pipeline:] unwrapping uv...");
    std::vector<int> indices_int(mesh.t_pos_idx.begin(), mesh.t_pos_idx.end());
    MeshCPP mesh_cpp(mesh.v_pos, indices_int);
    mesh_cpp.unwrap_uv();

    // rasterize, get mask, interpolate, gb_pos
    const int N = mesh_cpp.t_pos_idx().size() / 3; // t_pos_idx is Nx3 (flattened)
    std::vector<ssize_t> shape = { inf_config_.bake_resolution, inf_config_.bake_resolution, 4 };
    size_t  num_pixels = shape[0]*shape[1];
    std::vector<float> rast_result(num_pixels*shape[2], 0.0f);

    LOGI("[Pipeline:] Rasterizing...");
    auto rast_start = std::chrono::high_resolution_clock::now();
    texture_baker_cpp::rasterize_cpu_host_triangleTile(rast_result.data(),
                                                       reinterpret_cast<const texture_baker_cpp::tb_float2*>(mesh_cpp.v_tex().data()),
                                                       reinterpret_cast<const texture_baker_cpp::tb_int3*>(mesh_cpp.t_pos_idx().data()),
                                                        N,
                                                        inf_config_.bake_resolution);
    auto rast_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> rast_elapsed = rast_end - rast_start;
    LOGW("Rasterization runtime: %f s", rast_elapsed.count());

    LOGI("[Pipeline:] Getting bake mask...");
    std::vector<uint8_t> bake_mask = get_mask(rast_result, inf_config_.bake_resolution);

    LOGI("[Pipeline:] Interpolating...");
    std::vector<float> pos_bake(num_pixels*3, 0.0f); // 1024x1024x3
    texture_baker_cpp::interpolate_cpu_host(pos_bake.data(),
                                            mesh_cpp.v_pos().data(),
                                            mesh_cpp.t_pos_idx().data(),
                                            N,
                                            rast_result.data(),
                                            inf_config_.bake_resolution,
                                            inf_config_.bake_resolution);


    // padded_decoder
    //check scene codes is still in workspace
    logNumbers(scene_codes, *scene_codes_tinfo, scene_codes_wsName, 12, 'w');
    // construct gb_pos = pos_bake[bake_mask]
    LOGI("[Pipeline:] constructing gb_pos...");
    const std::string gb_pos_wsName = "gb_pos";
    TensorInfo gb_pos_tinfo;
    gb_pos_tinfo.name = gb_pos_wsName;
    gb_pos_tinfo.dims = {};
    LOGI("[Pipeline:] calling compact_masked_parallel...");
    LOGW("RSS: %zu KB", readProcessRssKb());
    int num_valid_points = compact_masked_parallel(&gb_pos_wsName, pos_bake.data(), bake_mask.data(), num_pixels, &ws_);
    auto* gb_pos = static_cast<float*>(ws_.data(gb_pos_wsName));
    if (!gb_pos) {
        LOGE("gb_pos was not found in workspace!");
        return;
    } else {
        const auto& gb_pos_tinfo = ws_.tinfoOf(gb_pos_wsName);
        std::string gb_pos_log = "gb_pos dims: ";
        for (const auto& d : gb_pos_tinfo->dims) gb_pos_log += std::to_string(d)+" ";
        LOGW("[Pipeline:] %s", gb_pos_log.c_str());
        logNumbers(gb_pos, *gb_pos_tinfo, gb_pos_wsName, 12, 'w');
    }
    // tri_query = spar3d_model.query_triplane(gb_pos, triplane)[0]
    LOGI("[Pipeline:] running query triplane on gb_pos to get tri_query...");
    const auto padded_tri_query_dec_wsName = gr_runners_.padded_decoder->last().inputBinding.at("padded_tri_query");
    auto* padded_tri_query_dec = static_cast<float*>(ws_.data(padded_tri_query_dec_wsName));
    const auto& padded_tri_query_dec_tinfo = ws_.tinfoOf(padded_tri_query_dec_wsName);
    assert(padded_tri_query_dec_tinfo->dims[0] >= num_valid_points);
    query_triplane_optimized(gb_pos, scene_codes, padded_tri_query_dec, num_valid_points,
                             scene_codes_tinfo->dims[2], scene_codes_tinfo->dims[3], scene_codes_tinfo->dims[4],
                             inf_config_.radius);

    LOGI("[Pipeline:] Running decoder...");
    execution_summary += runGraph(*gr_runners_.padded_decoder, reset_sessions);
    const auto padded_features_wsName = gr_runners_.padded_decoder->last().outputBinding.at("output_0");
    auto* padded_features = static_cast<float*>(ws_.data(padded_features_wsName));

    const auto padded_perturb_normal_wsName = gr_runners_.padded_decoder->last().outputBinding.at("output_1");
    auto* padded_perturb_normal = static_cast<float*>(ws_.data(padded_perturb_normal_wsName));

    // call image estimator and get roughness and metallic
    LOGI("[Pipeline:] Running image estimator...");
    std::vector<float> roughness_metallic = runImageEstimator(execution_summary, reset_sessions);
    LOGW("RSS: %zu KB", readProcessRssKb());
    LOGI("[Pipeline:] Clearing out image estimator");
    gr_runners_.image_estimator->clear_everything_all_sessions();
    LOGW("RSS: %zu KB", readProcessRssKb());
    float roughness = roughness_metallic[0];
    float metallic = roughness_metallic[1];
    LOGI("[Pipeline:] roughness: %f, metallic: %f", roughness, metallic);

    // call texture builder
    LOGI("[Pipeline:] Building textures...");
    auto texture_start = std::chrono::high_resolution_clock::now();
    texture_baker_cpp::BuildTexturesInspect inspect = texture_baker_cpp::BuildTextures_SaveImages(
            mesh_cpp.v_nrm().data(), mesh_cpp.v_nrm().size(),
            mesh_cpp.v_tng().data(), mesh_cpp.v_tng().size(),
            mesh_cpp.v_pos().data(), mesh_cpp.v_pos().size(),
            mesh_cpp.v_tex().data(), mesh_cpp.v_tex().size(),
            mesh_cpp.t_pos_idx().data(), mesh_cpp.t_pos_idx().size(),
            rast_result.data(),
            bake_mask.data(),
            inf_config_.bake_resolution,inf_config_.bake_resolution,
            padded_features, // it's ok to put the padded version as we internally only take the valid size given bake_mask
            padded_perturb_normal,
            roughness,
            metallic,
            "basecolor.jpg", "bump.jpg", false
            );
    auto texture_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> texture_elapsed = texture_end - texture_start;
    LOGW("Texture building runtime: %f s", texture_elapsed.count());
    LOGW("RSS: %zu KB", readProcessRssKb());

    // call glb exporter
    LOGI("[Pipeline:] Exporting to glb...");
    auto export_start = std::chrono::high_resolution_clock::now();

    bool export_success = texture_baker_cpp::ExportGLBFromInspect(inspect, mesh_cpp, glb_output_path);

    auto export_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> export_elapsed = export_end - export_start;
    LOGW("GLB export runtime: %f s", export_elapsed.count());
    LOGI("[Pipeline:] Done exporting to glb. Success? %i", (int)export_success);
}

//void Spar3DPipeline::overall_pipelineOld(uint8_t* img, int ori_width, int ori_height,
////                                      const std::string& base_path_out, const std::string& nrm_path_out,
//                                      const std::string& glb_output_path) {
//    std::string execution_summary;
//    const bool reset_sessions = true;
//
//    // first need to convert to RGBA, remove background and resize to 512 then get array in [0,1]
//    // input image is likely CHW, so make sure to transpose if needed
//    LOGI("[Pipeline:] preprocessing input image...");
//    const auto input_img_wsName = gr_runners_.img_preparer->last().inputBinding.at("image_array");
//    auto* input_img_array = static_cast<float*>(ws_.data(input_img_wsName)); // should be 512x512x4
//    const auto& input_img_tinfo = ws_.tinfoOf(input_img_wsName);
//    LOGI("input image tensor dimensions:");
//    for (const auto& d : input_img_tinfo->dims) {
//        LOGI("%zu", d);
//    }
//    assert(input_img_tinfo->numel() == inf_config_.img_height * inf_config_.img_width * 4);
//    preprocessImage(img, ori_width, ori_height,
//                    input_img_array, input_img_tinfo->numel(),
//                    inf_config_.img_width, true, true);
//
//    LOGI("[Pipeline:] Network-preparing image...");
//    execution_summary += runGraph(*gr_runners_.img_preparer, reset_sessions);
//
//    LOGI("[Pipeline:] running pdiff_cond...");
//    execution_summary += runGraph(*gr_runners_.pdiff_cond, reset_sessions);
//
//    LOGI("[Pipeline:] preparing noise and conditioning...");
//    // guidance_scale != 0 and != 1.0
//    // condition should already have been initialized by zeros and should be [2,1297,1024]
//    // copy cond_tokens in the first dimension
//    const auto cond_tokens_wsName = gr_runners_.pdiff_cond->last().outputBinding.at("output_0");
//    auto* cond_tokens = static_cast<float*>(ws_.data(cond_tokens_wsName));
//    const auto& cond_tokens_tinfo = ws_.tinfoOf(cond_tokens_wsName);
//
//    const auto denoiser_condition_wsName = gr_runners_.one_step_denoiser->last().inputBinding.at("condition");
//    auto* denoiser_condition = static_cast<float*>(ws_.data(denoiser_condition_wsName));
//
//    std::memcpy(denoiser_condition, cond_tokens, cond_tokens_tinfo->bytes());
//
//    // fill noise tensor with randn noise
//    const auto noise_wsName = gr_runners_.one_step_denoiser->last().inputBinding.at("noise");
//    const auto& noise_tinfo = ws_.tinfoOf(noise_wsName);
//    auto* noise = static_cast<float*>(ws_.data(noise_wsName));
//    uint32_t seed = 42;
//    size_t bytes_to_fill = noise_tinfo->bytes() / 2;
//    fillRandom(noise, bytes_to_fill, 0.0, 1.0, seed);
//    // replicate
//    std::memcpy(noise + noise_tinfo->numel()/2, noise, bytes_to_fill);
//
//    LOGI("[Pipeline:] running denoiser loop...");
//    const auto timestep_wsName = gr_runners_.one_step_denoiser->last().inputBinding.at("t");
//    auto* timestep = static_cast<int32_t*>(ws_.data(timestep_wsName));
//
//    const auto denoising_loop_sample_wsName = gr_runners_.one_step_denoiser->last().outputBinding.at("output_0");
//    LOGI("[Pipeline:] Found tensor workspace for denoising loop sample: WS name: %s", denoising_loop_sample_wsName.c_str());
//    auto* denoising_loop_sample = static_cast<float*>(ws_.data(denoising_loop_sample_wsName));
//
//    for (int i = inf_config_.num_timesteps - 1; i >= 0; --i) {
//        LOGI("[denoiser loop:] ind: %d", i);
//        // prepare timestep tensor
//        timestep[0] = static_cast<int32_t>(i);
//        timestep[1] = static_cast<int32_t>(i);
//        // everything is ready to call the one-step denoiser
//        execution_summary += runGraph(*gr_runners_.one_step_denoiser, false);
//        // update the noisy image for denoising
//        std::memcpy(noise, denoising_loop_sample, noise_tinfo->bytes());
//    }
//    const auto denoised_sample_wsName = gr_runners_.one_step_denoiser->last().outputBinding.at("output_1");
//    gr_runners_.one_step_denoiser->last().session.get()->reset(); // free up memory
//    // get sample
//    LOGI("[Pipeline:] Found tensor workspace for denoiser sample: WS name: %s", denoised_sample_wsName.c_str());
//    auto* denoised_sample = static_cast<float*>(ws_.data(denoised_sample_wsName));
//    const auto& denoised_sample_tinfo = ws_.tinfoOf(denoised_sample_wsName);
//    int d_m = denoised_sample_tinfo->dims[1];
//    assert(d_m == 6);
//    int d_n = denoised_sample_tinfo->dims[2];
//    assert(d_n == 512);
//
//    const auto pc_cond_wsName = gr_runners_.scene_codes1->last().inputBinding.at("pc_cond");
//    auto* pc_cond = static_cast<float*>(ws_.data(pc_cond_wsName));
//    const auto& pc_cond_tinfo = ws_.tinfoOf(pc_cond_wsName);
//    LOGI("pc_cond dims:");
//    for (auto d : pc_cond_tinfo->dims) {
//        LOGI("%lu", d);
//    }
//
//    // transpose denoised_sample and fill pc_cond
//    // we can do a linear walkthrough here because the number of rows is small = 6
//    // Source Pointers: Create 6 pointers, one for the start of each row
//    const float* src0 = denoised_sample;           // Start of Row 0
//    const float* src1 = src0 + d_n;                // Start of Row 1
//    const float* src2 = src1 + d_n;                // ...
//    const float* src3 = src2 + d_n;
//    const float* src4 = src3 + d_n;
//    const float* src5 = src4 + d_n;
//
//    float* dst = pc_cond; // Destination walker
//
//    // Loop 512 times (once per column)
//    for (int i = 0; i < d_n; ++i) {
//        // 1. Read 6 values (one from each row cursor)
//        float v0 = src0[i];
//        float v1 = src1[i];
//        float v2 = src2[i];
//        float v3 = src3[i];
//        float v4 = src4[i];
//        float v5 = src5[i];
//        // 2. Write them contiguously into the destination (which acts as a row in the 512x6 matrix)
//        dst[0] = v0;
//        dst[1] = v1;
//        dst[2] = v2;
//        dst[3] = v3;
//        dst[4] = v4;
//        dst[5] = v5;
//        // 3. Advance destination pointer by 6
//        dst += d_m;
//    }
//    normalize_pc_bbox_inplace(pc_cond, pc_cond_tinfo->dims[0], pc_cond_tinfo->dims[1], pc_cond_tinfo->dims[2]);
//
//    // call scene encoders
//    LOGI("[Pipeline:] Running scene_codes1...");
//    execution_summary += runGraph(*gr_runners_.scene_codes1, reset_sessions);
//
//
//    LOGI("[Pipeline:] Running scene_codes2...");
//    execution_summary += runGraph(*gr_runners_.scene_codes2, reset_sessions);
//
//    // need to do this:
////    triplane = triplanes[0]
////        # grid_vertices = self.scale_tensor(
////        #     self.grid_vertices.to(triplanes.device),
////        #     self.points_range,
////        #     self.bbox,
////        # )
////        # values = self.query_triplane(grid_vertices, triplane)
//    LOGI("[Pipeline:] Preparing grid vertices...");
//    // create grid_vertices and fill it from file
//    std::string grid_vertices_wsName = "grid_vertices";
//    TensorInfo grid_vertices_tinfo;
//    grid_vertices_tinfo.name = grid_vertices_wsName;
//    grid_vertices_tinfo.dims = {535882, 3}; // dimensions are hard-coded and correspond to original python code
//    {
//        std::string emsg;
//        // allocate buffer and zero initialize
//        ensureWorkspaceBuffer(ws_, "grid_vertices", grid_vertices_tinfo, &emsg);
//    }
//    // get buffer and fill from file
//    auto* grid_vertices_ptr = static_cast<float*>(ws_.data(grid_vertices_wsName));
//    readAssetToBuffer(mgr_, "grid_vertices.bin", grid_vertices_ptr, grid_vertices_tinfo.bytes());
//    // scale tensor
//    scale_tensor_inplace(grid_vertices_ptr, grid_vertices_tinfo.dims[0], 0.0f, 1.0f, -inf_config_.radius, inf_config_.radius);
//    // values = query_triplane(grid_vertices, triplane)
//    const auto values_wsName = gr_runners_.triplanesToProtoMesh->last().inputBinding.at("values");
//    const auto& values_tinfo = ws_.tinfoOf(values_wsName);
//    auto* values = static_cast<float*>(ws_.data(values_wsName));
//
//    const auto scene_codes_wsName = gr_runners_.scene_codes2->last().outputBinding.at("output_0");
//    const auto& scene_codes_tinfo = ws_.tinfoOf(scene_codes_wsName);
//    auto* scene_codes = static_cast<float*>(ws_.data(scene_codes_wsName));
//    {
//        // some logging
//        std::string scene_codes_log = "scene codes dims: ";
//        for (const auto&d : scene_codes_tinfo->dims) scene_codes_log += std::to_string(d)+" ";
//        LOGI("[pipeline:] %s", scene_codes_log.c_str());
//    }
//    LOGI("[Pipeline:] running query triplane to get values...");
//    query_triplane_optimized(grid_vertices_ptr, scene_codes, values, grid_vertices_tinfo.dims[0],
//                             scene_codes_tinfo->dims[2], scene_codes_tinfo->dims[3], scene_codes_tinfo->dims[4],
//                             inf_config_.radius);
//
//    // run network to infer sdf and deformation
//    LOGI("[Pipeline:] Getting sdf and deform...");
//    execution_summary += runGraph(*gr_runners_.triplanesToProtoMesh, reset_sessions);
//
//    // create mesh with v_pos and t_pos_idx
//    std::vector<mtd2::Vec3> grid_vertices;
//    std::vector<int> indices;
//    // Load Vertices
//    // Python saved flat floats, we load into Vec3 (size 12 bytes)
//    if (!loadVectorFromAsset(mgr_, "grid_vertices.bin", grid_vertices)) {
//        LOGE("[Pipeline:] Unable to load grid_vertices!");
//    }
//    // Load Indices
//    // Python saved flat int32, we load into int
//    if (!loadVectorFromAsset(mgr_, "indices.bin", indices)) {
//        LOGE("[Pipeline:] Unable to load indices for Marching Tetrahedra Helper!");
//    }
//    // Instantiate
//    auto mt_helper = std::make_unique<mtd2::MarchingTetrahedraHelper>(
//        grid_vertices,
//        indices
//    );
//
//    // get sdf and deformation fields
//    const auto sdf_wsName = gr_runners_.triplanesToProtoMesh->last().outputBinding.at("output_0");
//    auto* sdf = static_cast<float*>(ws_.data(sdf_wsName));
//
//    const auto deformation_wsName = gr_runners_.triplanesToProtoMesh->last().outputBinding.at("output_1");
//    auto* deformation = static_cast<float*>(ws_.data(deformation_wsName));
//    // make mesh with v_pos already scaled
//    mtd2::Mesh mesh = mt_helper->forward(sdf, deformation, bbox_);
//
//    // mesh.unwrap_uv
//    std::vector<int> indices_int(mesh.t_pos_idx.begin(), mesh.t_pos_idx.end());
//    MeshCPP mesh_cpp(mesh.v_pos, indices_int);
//    mesh_cpp.unwrap_uv();
//
//    // rasterize, get mask, interpolate, gb_pos
//    const int N = mesh_cpp.t_pos_idx().size() / 3; // t_pos_idx is Nx3 (flattened)
//    std::vector<ssize_t> shape = { inf_config_.bake_resolution, inf_config_.bake_resolution, 4 };
//    size_t  num_pixels = shape[0]*shape[1];
//    std::vector<float> rast_result(num_pixels*shape[2], 0.0f);
//
//    texture_baker_cpp::rasterize_cpu_host_triangleTile(rast_result.data(),
//                                                       reinterpret_cast<const texture_baker_cpp::tb_float2*>(mesh_cpp.v_tex().data()),
//                                                       reinterpret_cast<const texture_baker_cpp::tb_int3*>(mesh_cpp.t_pos_idx().data()),
//                                                        N,
//                                                        inf_config_.bake_resolution);
//
//    std::vector<uint8_t> bake_mask = get_mask(rast_result, inf_config_.bake_resolution);
//
//    std::vector<float> pos_bake(num_pixels*3, 0.0f); // 1024x1024x3
//    texture_baker_cpp::interpolate_cpu_host(pos_bake.data(),
//                                            mesh_cpp.v_pos().data(),
//                                            mesh_cpp.t_pos_idx().data(),
//                                            N,
//                                            rast_result.data(),
//                                            inf_config_.bake_resolution,
//                                            inf_config_.bake_resolution);
//
//
//    // padded_decoder_with_query_triplane
//    // TODO: update this with new padded decoder and do triplane query inline
////    const auto padded_gb_pos_wsName = gr_runners_.decoder_triq->last().inputBinding.at("padded_gb");
////    auto* padded_gb = static_cast<float*>(ws_.data(padded_gb_pos_wsName));
////    int num_valid_points = compact_masked_parallel(padded_gb, pos_bake.data(), bake_mask.data(), num_pixels);
////
////    const auto& padded_gb_tinfo = ws_.tinfoOf(padded_gb_pos_wsName);
////    LOGI("padded_gb first dimension size: %lu", padded_gb_tinfo->dims[0]);
//
//    // ================================================
//    // construct gb_pos = pos_bake[bake_mask]
//    LOGI("[Pipeline:] constructing gb_pos...");
//    const std::string gb_pos_wsName = "gb_pos";
//    TensorInfo gb_pos_tinfo;
//    gb_pos_tinfo.name = gb_pos_wsName;
//    gb_pos_tinfo.dims = {};
//    int num_valid_points = compact_masked_parallel(&gb_pos_wsName, pos_bake.data(), bake_mask.data(), num_pixels, &ws_);
//    auto* gb_pos = static_cast<float*>(ws_.data(gb_pos_wsName));
//    if (!gb_pos) {
//        LOGE("gb_pos was not found in workspace!");
//        return;
//    } else {
//        const auto& gb_pos_tinfo = ws_.tinfoOf(gb_pos_wsName);
//        std::string gb_pos_log = "gb_pos dims: ";
//        for (const auto& d : gb_pos_tinfo->dims) gb_pos_log += std::to_string(d)+" ";
//        LOGW("[Pipeline:] %s", gb_pos_log.c_str());
//    }
//    // tri_query = spar3d_model.query_triplane(gb_pos, triplane)[0]
//    LOGI("[Pipeline:] running query triplane on gb_pos to get tri_query...");
//    const auto padded_tri_query_dec_wsName = gr_runners_.padded_decoder->last().inputBinding.at("padded_tri_query");
//    auto* padded_tri_query_dec = static_cast<float*>(ws_.data(padded_tri_query_dec_wsName));
//    const auto& padded_tri_query_dec_tinfo = ws_.tinfoOf(padded_tri_query_dec_wsName);
//    assert(padded_tri_query_dec_tinfo->dims[0] >= num_valid_points);
//    query_triplane_optimized(gb_pos, scene_codes, padded_tri_query_dec, num_valid_points,
//                             scene_codes_tinfo->dims[2], scene_codes_tinfo->dims[3], scene_codes_tinfo->dims[4],
//                             inf_config_.radius);
//
//
//
//    LOGI("[Pipeline:] Running decoder...");
//    // scene_codes as output from scene_codes2: 1x3x40x384x384
//    // scene codes as expected by decoder: 3x40x384x384
//    // in terms of bytes it's the same, but tensor info are not the same
////    const auto scene_codes_out_wsName = gr_runners_.scene_codes2->last().outputBinding.at("output_0");
////    auto* scene_codes_out = static_cast<float*>(ws_.data(scene_codes_out_wsName));
////    const auto scene_codes_in_dec_wsName = gr_runners_.decoder_triq->last().inputBinding.at("scene_codes");
////    auto* scene_codes_in_dec = static_cast<float*>(ws_.data(scene_codes_in_dec_wsName));
////    const auto& scene_codes_in_dec_tinfo = ws_.tinfoOf(scene_codes_in_dec_wsName);
////    std::memcpy(scene_codes_in_dec, scene_codes_out, scene_codes_in_dec_tinfo->bytes());
//
//    execution_summary += runGraph(*gr_runners_.padded_decoder, reset_sessions);
//
//    const auto padded_features_wsName = gr_runners_.padded_decoder->last().outputBinding.at("output_0");
//    auto* padded_features = static_cast<float*>(ws_.data(padded_features_wsName));
//
//    const auto padded_perturb_normal_wsName = gr_runners_.padded_decoder->last().outputBinding.at("output_1");
//    auto* padded_perturb_normal = static_cast<float*>(ws_.data(padded_perturb_normal_wsName));
//
//    // call image estimator and get roughness and metallic
//    LOGI("[Pipeline:] Running image estimator...");
//    execution_summary += runGraph(*gr_runners_.image_estimator, reset_sessions);
//    const auto roughness_wsName = gr_runners_.image_estimator->last().outputBinding.at("output_0");
//    const auto metallic_wsName = gr_runners_.image_estimator->last().outputBinding.at("output_1");
//
//    auto* roughness = static_cast<float*>(ws_.data(roughness_wsName));
//    auto* metallic = static_cast<float*>(ws_.data(metallic_wsName));
//
//    // call texture builder
//    LOGI("[Pipeline:] Building textures...");
//    texture_baker_cpp::BuildTexturesInspect inspect = texture_baker_cpp::BuildTextures_SaveImages(
//            mesh_cpp.v_nrm().data(), mesh_cpp.v_nrm().size(),
//            mesh_cpp.v_tng().data(), mesh_cpp.v_tng().size(),
//            mesh_cpp.v_pos().data(), mesh_cpp.v_pos().size(),
//            mesh_cpp.v_tex().data(), mesh_cpp.v_tex().size(),
//            mesh_cpp.t_pos_idx().data(), mesh_cpp.t_pos_idx().size(),
//            rast_result.data(),
//            bake_mask.data(),
//            inf_config_.bake_resolution,inf_config_.bake_resolution,
//            padded_features, // it's ok to put the padded version as we internally only take the valid size given bake_mask
//            padded_perturb_normal,
//            *roughness,
//            *metallic,
//            "basecolor.jpg", "bump.jpg", false
//            );
//
//    // call glb exporter
//    LOGI("[Pipeline:] Exporting to glb...");
//    bool export_success = texture_baker_cpp::ExportGLBFromInspect(inspect, mesh_cpp, glb_output_path);
//}