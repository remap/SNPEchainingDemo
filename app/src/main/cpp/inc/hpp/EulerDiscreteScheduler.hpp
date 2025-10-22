#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <limits>
#include <string.h>
#include <android/log.h>

#define  LOG_TAG_ES  "SCHEDULER"
#define  LOGI_ES(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG_ES,__VA_ARGS__)
#define  LOGW_ES(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG_ES,__VA_ARGS__)
#define  LOGE_ES(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG_ES,__VA_ARGS__)

struct EulerDiscreteSchedulerOutput {
    std::vector<float> prev_sample;
    std::vector<float> pred_original_sample; // optional in Py; always returned here
};

struct EulerDiscreteSchedulerConfig {
    int   num_train_timesteps = 1000;
    float beta_start = 0.00085f;
    float beta_end   = 0.012f;
    enum class BetaSchedule { Linear, ScaledLinear, SquaredCosCapV2 } beta_schedule = BetaSchedule::ScaledLinear;

    enum class PredictionType { Epsilon, Sample, VPrediction } prediction_type = PredictionType::Epsilon;
    enum class InterpType     { Linear, LogLinear } interpolation_type = InterpType::Linear;
    enum class TimestepSpacing{ Linspace, Leading, Trailing } timestep_spacing = TimestepSpacing::Leading;
    enum class FinalSigmas    { Zero, SigmaMin } final_sigmas_type = FinalSigmas::Zero;
    enum class TimestepType   { Discrete, Continuous } timestep_type = TimestepType::Discrete;

    bool  use_karras_sigmas      = false;
    bool  use_exponential_sigmas = false;
    bool  use_beta_sigmas        = false;
    bool  rescale_betas_zero_snr = false;
    int   steps_offset           = 1;

    int   order                  = 1;
};

class EulerDiscreteScheduler {
public:
    explicit EulerDiscreteScheduler(const EulerDiscreteSchedulerConfig& cfg);
    EulerDiscreteScheduler();

    // Derived buffers (public like diffusers)
    std::vector<float> betas;
    std::vector<float> alphas;
    std::vector<float> alphas_cumprod;
    std::vector<float> sigmas;     // length = base_sigmas.size()+1 (last appended by config)
    std::vector<float> timesteps;  // reversed 0..N-1 by default

    // state
    int  num_inference_steps = 0;
    bool is_scale_input_called = false;

    // properties
    float init_noise_sigma() const;

    // timeline control
    void set_begin_index(int begin_index = 0) { _begin_index = begin_index; }
    int  step_index()  const { return _step_index; }
    int  begin_index() const { return _begin_index; }

    // set discrete schedule for inference
    void set_timesteps(int num_inference_steps,
                       const std::vector<float>* custom_timesteps = nullptr,
                       const std::vector<float>* custom_sigmas    = nullptr);

    // scaling per step (in-place)
    void scale_model_input(std::vector<float>& sample, float timestep);

    void scale_model_input_inplace(const float* sample, float* latent_model_input, size_t n, const float timestep);



    // single Euler step
    EulerDiscreteSchedulerOutput step(const std::vector<float>& model_output,
                                      float timestep,
                                      const std::vector<float>& sample,
                                      float s_churn = 0.0f,
                                      float s_tmin  = 0.0f,
                                      float s_tmax  = std::numeric_limits<float>::infinity(),
                                      float s_noise = 1.0f);

    void step_inplace(const float* model_output, float timestep, float* sample, size_t n);

    // convenience: add_noise(x0, noise, timestep)
    std::vector<float> add_noise(const std::vector<float>& original_samples,
                                 const std::vector<float>& noise,
                                 const std::vector<float>& timesteps_vec) const;

    // utility for training-style velocity (optional)
    std::vector<float> get_velocity(const std::vector<float>& sample,
                                    const std::vector<float>& noise,
                                    const std::vector<float>& timesteps_vec) const;

    void set_config(EulerDiscreteSchedulerConfig cfg) { cfg_ = cfg; }

private:
    EulerDiscreteSchedulerConfig cfg_;
    int  _step_index  = -1;
    int  _begin_index = -1;

    // ---- small helpers ----
    static std::vector<float> linspace(float a, float b, int n);
    static std::vector<float> cumprod(const std::vector<float>& v);
    static std::vector<float> flip(const std::vector<float>& v);
    static std::vector<float> interp_linear(const std::vector<float>& xq,
                                            const std::vector<float>& x,
                                            const std::vector<float>& y);
    static std::vector<float> exp_linspace(float log_a, float log_b, int n);

    static std::vector<float> betas_for_alpha_bar(int T, float max_beta = 0.999f,
                                                  const char* alpha_transform_type = "cosine");
    static std::vector<float> rescale_zero_terminal_snr(const std::vector<float>& betas);

    int  index_for_timestep(float timestep, const std::vector<float>* schedule_timesteps = nullptr) const;
    void _init_step_index(float timestep);

    // map sigma -> t (float index), like Py _sigma_to_t
    static float _sigma_to_t(float sigma, const std::vector<float>& log_sigmas);
};
