//
// Created by Chiheb Boussema on 30/9/25.
//

#ifndef SNPECHAININGDEMO_SDXLPIPELINE_H
#define SNPECHAININGDEMO_SDXLPIPELINE_H

// SDXLPipeline.h
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <sstream>
#include <random>
#include <android/asset_manager.h>

#include "hpp/TensorWorkspace.hpp"   // your workspace helper
#include "hpp/ModelSession.hpp"      // your SNPE session wrapper
#include "hpp/GraphRunner.hpp"
#include "hpp/newInferenceHelper.hpp"
#include "hpp/EulerDiscreteScheduler.hpp"

struct EncodedOut {
  float* prompt_embeds;           // [B=1, S=77, D1+D2]
  float* pooled_prompt_embeds;    // [B=1, Dpool]  (from TE2 text_embeds)
  int B, S, D_total, D_pool;
};

//struct SchedulerConfig {
//    float beta_end = 0.012;
//    std::string beta_schedule = "scaled_linear";
//    float beta_start = 0.00085;
//    bool clip_sample = false;
//    std::string final_sigmas_type = "zero";
//    std::string interpolation_type = "linear";
//    int num_train_timesteps = 1000;
//    std::string prediction_type = "epsilon";
//    bool rescale_betas_zero_snr = false;
//    float sample_max_value = 1.0;
//    bool set_alpha_to_one = false;
//    float sigma_max;
//    float sigma_min;
//    bool skip_prk_steps = true;
//    int steps_offset = 1;
//    std::string timestep_spacing = "trailing";
//    std::string timestep_type = "discrete";
//    float trained_betas;
//    bool use_beta_sigmas = false;
//    bool use_exponential_sigmas = false;
//    bool use_karras_sigmas = false;
//    int order = 1;
//};
//
//class EulerDiscreteScheduler {
//public:
//
//    EulerDiscreteScheduler() {}
//
//    float get_init_noise_sigma() const {
//        if (config.timestep_spacing == "linspace" || config.timestep_spacing == "trailing") {
//            // Check if sigmas vector is empty to avoid an error.
//            if (!sigmas.empty()) {
//                return *std::max_element(sigmas.begin(), sigmas.end());
//            }
//            return 14.6146; // Return a default value if empty
//        }
//        if (!sigmas.empty()) {
//            double max_sigma = *std::max_element(sigmas.begin(), sigmas.end());
//            return std::sqrt(std::pow(max_sigma, 2) + 1);
//        }
//        return 0.0; // Return a default value if empty
//    }
//
//private:
//    std::vector<float> sigmas;
//    SchedulerConfig config;
//};

struct InfConfig {
    int num_inference_steps = 8;
    int width = 1024; // 128;
    int height = 1024; //128;
};

struct NetConfig {
    const int unet_in_channels = 4;
    const int vae_scale_factor = 8;
    const float vae_config_scaling_factor = 0.13025;
    const int num_channels_latents = 4;
    const int default_sample_size = 128;
    int default_width = 128 * 8; // default_sample_size * vae_scale_factor
    int default_height = 128 * 8; // default_sample_size * vae_scale_factor
    const int unet_addition_time_embed_dim = 256;
    const int text_encoder_projection_dim = 1280;
    const int unet_add_embedding_linear_1_in_features = 2816;
};

struct Unet {
//    GraphRunner& time_embd;
////    GraphRunner& add_time_proj;
////    GraphRunner& add_embedding;
//    GraphRunner& aug_emb;
//    GraphRunner& inConv;
//    GraphRunner& unet_blocks;
//    GraphRunner& outConv;

    std::unique_ptr<GraphRunner> time_embd;
    std::unique_ptr<GraphRunner> aug_emb;
    std::unique_ptr<GraphRunner> inConv;
    std::unique_ptr<GraphRunner> unet_blocks;
    std::unique_ptr<GraphRunner> unet_upblock00;
    std::unique_ptr<GraphRunner> unet_upblock01;
    std::unique_ptr<GraphRunner> unet_blocks_2;
    std::unique_ptr<GraphRunner> outConv;
};

class SDXLPipeline {
public:
    SDXLPipeline(TensorWorkspace& ws,
               GraphRunner& enc_gr,
               AAssetManager* mgr,
               std::string g_modelDir,
               std::string encoders_config_filename
               ) : ws_(ws),
               enc_gr_(enc_gr),
               mgr_(mgr),
               g_modelDir_(g_modelDir),
               encoders_config_filename_(encoders_config_filename) {

        scfg.num_train_timesteps = 1000;
        scfg.beta_schedule = EulerDiscreteSchedulerConfig::BetaSchedule::ScaledLinear;
        scfg.prediction_type = EulerDiscreteSchedulerConfig::PredictionType::Epsilon;
        scfg.interpolation_type = EulerDiscreteSchedulerConfig::InterpType::Linear;
        scfg.timestep_spacing = EulerDiscreteSchedulerConfig::TimestepSpacing::Trailing;

//        scheduler_ = EulerDiscreteScheduler(scfg);
//        scheduler_(scfg);
//        scheduler_.emplace(scfg);
        scheduler_.reset();
        scheduler_.emplace(scfg);

        unet_.time_embd.reset(new GraphRunner(ws_));
        unet_.aug_emb.reset(new GraphRunner(ws_));
        unet_.inConv.reset(new GraphRunner(ws_));
        unet_.unet_blocks.reset(new GraphRunner(ws_));
        unet_.unet_upblock00.reset(new GraphRunner(ws_));
        unet_.unet_upblock01.reset(new GraphRunner(ws_));
        unet_.unet_blocks_2.reset(new GraphRunner(ws_));
        unet_.outConv.reset(new GraphRunner(ws_));

        vae_decoder_.reset(new GraphRunner(ws_));
    };

    SDXLPipeline(TensorWorkspace& ws,
               AAssetManager* mgr,
               std::string g_modelDir,
               GraphRunner& enc_gr,
               std::string encoders_config_filename,
               std::string allModels_config_filename
               ) : ws_(ws),
               enc_gr_(enc_gr),
               mgr_(mgr),
               g_modelDir_(g_modelDir),
               encoders_config_filename_(encoders_config_filename),
               allModels_config_filename_(allModels_config_filename){

        scfg.num_train_timesteps = 1000;
        scfg.beta_schedule = EulerDiscreteSchedulerConfig::BetaSchedule::ScaledLinear;
        scfg.prediction_type = EulerDiscreteSchedulerConfig::PredictionType::Epsilon;
        scfg.interpolation_type = EulerDiscreteSchedulerConfig::InterpType::Linear;
        scfg.timestep_spacing = EulerDiscreteSchedulerConfig::TimestepSpacing::Trailing;

//        scheduler_ = EulerDiscreteScheduler(scfg);
        scheduler_.reset();
        scheduler_.emplace(scfg);

        unet_.time_embd.reset(new GraphRunner(ws_));
        unet_.aug_emb.reset(new GraphRunner(ws_));
        unet_.inConv.reset(new GraphRunner(ws_));
        unet_.unet_blocks.reset(new GraphRunner(ws_));
        unet_.unet_upblock00.reset(new GraphRunner(ws_));
        unet_.unet_upblock01.reset(new GraphRunner(ws_));
        unet_.unet_blocks_2.reset(new GraphRunner(ws_));
        unet_.outConv.reset(new GraphRunner(ws_));

        vae_decoder_.reset(new GraphRunner(ws_));
    };


//    Build the two text-encoder sessions (DLCs must match names used below)
    bool init_text_encoders(/*const std::string& te1_dlc_path,
                          const std::string& te2_dlc_path,*/
                          char runtime_hint); // 'C','G','D'

    bool init_networks(char runtime_hint='D');
  // Encode: ids arrays are 77-long each (already padded on Kotlin side)
//  EncodedOut encode_prompt(const int32_t* ids1_77,
//                           const int32_t* ids2_77);

    std::string run_encoders(const int32_t* ids1_77,
                            const int32_t* ids2_77);

    std::pair<float*, size_t> prepare_latents(const int width,
                                            const int height,
                                            const char* wsName /*= "latents"*/,
                                            std::optional<uint64_t> seed /*= std::nullopt*/,
                                            bool forceRegen /*= false*/);

    bool add_time_ids_toWorkspace(
          TensorWorkspace& ws,
          const char* wsName,
          std::pair<int,int> original_size,
          std::pair<int,int> crops_coords_top_left,
          std::pair<int,int> target_size,
          int unet_add_embedding_linear_1_in_features, // net_config_.unet_add_embedding_linear_1_in_features
          int unet_addition_time_embed_dim,            // net_config_.unet_addition_time_embed_dim
          int text_encoder_projection_dim,             // net_config_.text_encoder_projection_dim
          std::string* emsg /*=nullptr*/);

    std::vector<float> overall_pipeline(const int32_t* ids1_77,
                                        const int32_t* ids2_77,
                                        bool decode_only=false);

    void unet_wrapper(const float t);
    void unet_wrapper2(const float t);

//    void get_aug_emb();
    void get_aug_emb2();

    std::vector<float> postprocess_to_chw255(const char* wsName, int C, int H, int W);

private:
    TensorWorkspace& ws_;
    GraphRunner& enc_gr_;
//    GraphRunner& unet_time_embd_;
//    GraphRunner& unet_;
//    GraphRunner& add_time_proj_;
//    GraphRunner& add_embedding_;
    Unet unet_;
    std::unique_ptr<GraphRunner> vae_decoder_;


    AAssetManager* mgr_;
    std::string g_modelDir_;
    std::string encoders_config_filename_;
    std::string allModels_config_filename_;

    std::unique_ptr<ModelSession> te1_;
    std::unique_ptr<ModelSession> te2_;

//    EulerDiscreteScheduler scheduler_;
    EulerDiscreteSchedulerConfig scfg;
//    EulerDiscreteScheduler scheduler_;
    std::optional<EulerDiscreteScheduler> scheduler_;

    InfConfig inf_config_;
    NetConfig net_config_;

    // TE dims (query from DLC once; fill after init)
    int B_ = 1;
    int S_ = 77;
    int D1_ = 768;   // typical: 1280/1024 depending on SDXL build; will read from IO at init
    int D2_ = 1280;
    int DPOOL_ = 1280; // text_embeds dim (TE2)

    // Helpers
    void concat_penultimates(const float* te1_pen, const float* te2_pen, float* out);
};


#endif //SNPECHAININGDEMO_SDXLPIPELINE_H
