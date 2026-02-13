//
// Created by Chiheb Boussema on 1/15/26.
//

#ifndef SNPECHAININGDEMO_SPAR3DPIPELINE_H
#define SNPECHAININGDEMO_SPAR3DPIPELINE_H

#pragma once
#include "GenericPipeline.h"
#include "ModelSession.hpp"      // SNPE session wrapper
#include "GraphRunner.hpp"
#include "MarchingTetrahedraHelper.h"
#include "MeshCPP.h"
#include <cmath>
#include <random>

namespace distributionUtils {
    inline float softplus(float x) {
        // Thresholding at 20 prevents numerical overflow for exp(x)
        if (x > 20.0f) return x;
        if (x < -20.0f) return std::exp(x);
//        return std::log(1.0 + std::exp(x));
        return std::log1p(std::exp(x));
    }

    struct Beta {
        float alpha;
        float beta;
        std::gamma_distribution<float> dist_alpha;
        std::gamma_distribution<float> dist_beta;

        Beta(float d1, float d2, float bias1=0.0f, float bias2=0.0f) {
            alpha = softplus(d1 + bias1);
            beta = softplus(d2 + bias2);
            dist_alpha = std::gamma_distribution<float>(alpha, 1.0f);
            dist_beta = std::gamma_distribution<float>(beta, 1.0f);
        }

        float mean() const {
            return alpha / (alpha + beta);
        }
        float mode() const {
            if (alpha > 1.0f && beta > 1.0f) {
                return (alpha - 1.0f) / (alpha + beta - 2.0f);
            }
            // handle boundary modes for J-shaped/U-shaped distributions
            return (alpha > beta) ? 1.0f : 0.0f;
        }
        float sample(const int seed=42) {
            std::mt19937 gen(seed);
            float X = dist_alpha(gen);
            float Y = dist_beta(gen);
            return X / (X + Y);
        }
        float distribution_eval(const std::string eval_method="mean") {
            if (eval_method == "mean") {
                return mean();
            } else if (eval_method == "mode") {
                return mode();
            } else if (eval_method == "sample") {
                return sample();
            }
        }
    };
}

class Spar3DPipeline : public GenericPipeline {
public:
    Spar3DPipeline(AAssetManager* mgr,
                   std::string g_modelDir,
                   std::string models_config_filename,
                   TensorWorkspace& ws
                   ) : GenericPipeline(mgr, g_modelDir, models_config_filename, ws) {
//        gr_runners_.img_preparer.reset(new GraphRunner(ws));
        gr_runners_.img_preparer = std::make_unique<GraphRunner>(ws);
//        gr_runners_.pdiff_cond.reset(new GraphRunner(ws));
        gr_runners_.pdiff_cond = std::make_unique<GraphRunner>(ws);
        gr_runners_.scene_codes1 = std::make_unique<GraphRunner>(ws);
        gr_runners_.scene_codes2 = std::make_unique<GraphRunner>(ws);
        gr_runners_.image_estimator = std::make_unique<GraphRunner>(ws);
        gr_runners_.triplanesToProtoMesh = std::make_unique<GraphRunner>(ws);
        gr_runners_.padded_decoder = std::make_unique<GraphRunner>(ws);
        gr_runners_.one_step_denoiser = std::make_unique<GraphRunner>(ws);

        gr_runners_.tester_allvision = std::make_unique<GraphRunner>(ws);
        gr_runners_.scene_codes2_3 = std::make_unique<GraphRunner>(ws);
    };

    bool init_networks(char runtime_hint = 'D') override;

    std::vector<float> preprocessImage(uint8_t* pixelData, int width, int height, const int tgt_size=512, const bool _01=false, const bool HWC=false);
    void preprocessImage(uint8_t* pixelData, int width, int height, float* output_buffer, const size_t output_elements, const int tgt_size=512, const bool _01=true, const bool HWC=true);

    void overall_pipeline(uint8_t* img_array, int ori_width, int ori_height, const std::string& glb_output_path);

    void test_spill(uint8_t* img, int ori_width, int ori_height);
    void test_imest(uint8_t* img, int ori_width, int ori_height);
//    void test_sc2(uint8_t* img, int ori_width, int ori_height, const std::string& glb_output_path);

//protected:
//    float softplus(float x) {
//        // Thresholding at 20 prevents numerical overflow for exp(x)
//        if (x > 20.0) return x;
//        return std::log(1.0 + std::exp(x));
//    }

private:

    struct Runners {
        std::unique_ptr<GraphRunner> img_preparer;
        std::unique_ptr<GraphRunner> pdiff_cond;
        std::unique_ptr<GraphRunner> scene_codes1;
        std::unique_ptr<GraphRunner> scene_codes2;
        std::unique_ptr<GraphRunner> image_estimator;
        std::unique_ptr<GraphRunner> triplanesToProtoMesh;
        std::unique_ptr<GraphRunner> padded_decoder;
        std::unique_ptr<GraphRunner> one_step_denoiser;

        std::unique_ptr<GraphRunner> tester_allvision;
        std::unique_ptr<GraphRunner> scene_codes2_3;
    };
    Runners gr_runners_;

    struct InfConfig {
        float guidance_scale = 3.0f;
        int num_timesteps = 32;
        int bake_resolution = 1024;
        int img_width = 512;
        int img_height = 512;
        float radius = 0.87f;
        std::string roughness_distribution_eval = "mean";
        std::string metallic_distribution_eval = "mode";
        std::vector<float> sampler_channel_biases = {0.0f, 0.0f, 0.0f, -4.5f, -4.5f, -4.5f};
        std::vector<float> sampler_channel_scales = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
        float isosurface_threshold = 9.5f;
    };
    InfConfig inf_config_;

    std::vector<mtd2::Vec3> bbox_ = {{-0.87f,-0.87f,-0.87f},{0.87f, 0.87f, 0.87f}};

    void unscale_channels(float* tensor, const TensorInfo& tinfo);
    void prepareToRunSC2();
    std::vector<float> runImageEstimator(std::string& execution_summary, const bool reset_sessions=true);

};


#endif //SNPECHAININGDEMO_SPAR3DPIPELINE_H
