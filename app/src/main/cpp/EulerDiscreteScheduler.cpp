//
// Created by Chiheb Boussema on 7/10/25.
//

#include "EulerDiscreteScheduler.hpp"

// -------- utilities --------
std::vector<float> EulerDiscreteScheduler::linspace(float a, float b, int n) {
    std::vector<float> out(n);
    if (n == 1) { out[0] = a; return out; }
    float step = (b - a) / float(n - 1);
    for (int i = 0; i < n; ++i) out[i] = a + step * i;
    return out;
}

std::vector<float> EulerDiscreteScheduler::cumprod(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    float acc = 1.0f;
    for (size_t i = 0; i < v.size(); ++i) { acc *= v[i]; out[i] = acc; }
    return out;
}

std::vector<float> EulerDiscreteScheduler::flip(const std::vector<float>& v) {
    std::vector<float> out(v);
    std::reverse(out.begin(), out.end());
    return out;
}

std::vector<float> EulerDiscreteScheduler::interp_linear(
    const std::vector<float>& xq, const std::vector<float>& x, const std::vector<float>& y) {

    if (x.size() != y.size() || x.empty()) throw std::runtime_error("interp: bad inputs");
    std::vector<float> out; out.reserve(xq.size());

    for (float q : xq) {
        if (q <= x.front()) { out.push_back(y.front()); continue; }
        if (q >= x.back())  { out.push_back(y.back());  continue; }
        // find interval [i, i+1]
        auto it = std::upper_bound(x.begin(), x.end(), q);
        size_t i = size_t(it - x.begin() - 1);
        float t = (q - x[i]) / (x[i+1] - x[i]);
        out.push_back(y[i] * (1 - t) + y[i+1] * t);
    }
    return out;
}

std::vector<float> EulerDiscreteScheduler::exp_linspace(float log_a, float log_b, int n) {
    std::vector<float> out(n);
    if (n == 1) { out[0] = std::exp(log_a); return out; }
    float step = (log_b - log_a) / float(n - 1);
    for (int i = 0; i < n; ++i) out[i] = std::exp(log_a + step * i);
    return out;
}

std::vector<float> EulerDiscreteScheduler::betas_for_alpha_bar(int T, float max_beta,
                                                               const char* alpha_transform_type) {
    auto alpha_bar = [&](float t)->float {
        if (strcmp(alpha_transform_type, "cosine")) {
//        if (std::string(alpha_transform_type) == "cosine") {
            float v = std::cos((t + 0.008f) / 1.008f * float(M_PI) / 2.0f);
            return v * v;
        } else if (strcmp(alpha_transform_type, "exp")) { // (std::string(alpha_transform_type) == "exp") {
            return std::exp(t * -12.0f);
        }
        throw std::runtime_error("Unsupported alpha_transform_type");
    };

    std::vector<float> b; b.reserve(T);
    for (int i = 0; i < T; ++i) {
        float t1 = float(i) / float(T);
        float t2 = float(i + 1) / float(T);
        float v = 1.0f - alpha_bar(t2) / alpha_bar(t1);
        b.push_back(std::min(v, max_beta));
    }
    return b;
}

std::vector<float> EulerDiscreteScheduler::rescale_zero_terminal_snr(const std::vector<float>& betas_in) {
    // Convert betas -> alphas, cumprod, sqrt
    std::vector<float> alphas(betas_in.size());
    for (size_t i = 0; i < alphas.size(); ++i) alphas[i] = 1.0f - betas_in[i];
    auto alphas_cum = cumprod(alphas);
    std::vector<float> a_bar_sqrt(alphas_cum.size());
    for (size_t i = 0; i < a_bar_sqrt.size(); ++i) a_bar_sqrt[i] = std::sqrt(alphas_cum[i]);

    float a0 = a_bar_sqrt.front();
    float aT = a_bar_sqrt.back();

    // shift so last is zero
    for (auto& v : a_bar_sqrt) v -= aT;
    // scale back so first equals old first
    float scale = a0 / (a0 - aT);
    for (auto& v : a_bar_sqrt) v *= scale;

    // revert sqrt/cumprod -> betas
    std::vector<float> a_bar(a_bar_sqrt.size());
    for (size_t i = 0; i < a_bar.size(); ++i) a_bar[i] = a_bar_sqrt[i] * a_bar_sqrt[i];

    std::vector<float> a(a_bar.size());
    a[0] = a_bar[0];
    for (size_t i = 1; i < a.size(); ++i) a[i] = a_bar[i] / a_bar[i - 1];

    std::vector<float> betas(a.size());
    for (size_t i = 0; i < a.size(); ++i) betas[i] = 1.0f - a[i];
    return betas;
}

// -------- ctor --------
EulerDiscreteScheduler::EulerDiscreteScheduler(const EulerDiscreteSchedulerConfig& cfg) : cfg_(cfg) {
    // betas
    if (cfg_.use_karras_sigmas || cfg_.use_exponential_sigmas || cfg_.use_beta_sigmas)
        throw std::runtime_error("This C++ port does not implement Karras/Exponential/Beta sigmas yet.");

    if (cfg_.beta_schedule == EulerDiscreteSchedulerConfig::BetaSchedule::Linear) {
        betas = linspace(cfg_.beta_start, cfg_.beta_end, cfg_.num_train_timesteps);
    } else if (cfg_.beta_schedule == EulerDiscreteSchedulerConfig::BetaSchedule::ScaledLinear) {
        auto r = linspace(std::sqrt(cfg_.beta_start), std::sqrt(cfg_.beta_end), cfg_.num_train_timesteps);
        betas.resize(r.size());
        for (size_t i = 0; i < r.size(); ++i) betas[i] = r[i] * r[i];
    } else {
        // squaredcos_cap_v2
        betas = betas_for_alpha_bar(cfg_.num_train_timesteps, 0.999f, "cosine");
    }

    if (cfg_.rescale_betas_zero_snr) betas = rescale_zero_terminal_snr(betas);

    // alphas & cumprod
    alphas.resize(betas.size());
    for (size_t i = 0; i < betas.size(); ++i) alphas[i] = 1.0f - betas[i];
    alphas_cumprod = cumprod(alphas);

    if (cfg_.rescale_betas_zero_snr) {
        // avoid zero (match Py’s behavior)
        if (!alphas_cumprod.empty()) alphas_cumprod.back() = std::ldexp(1.0f, -24); // 2^-24
    }

    // base sigmas = sqrt((1 - a_bar)/a_bar), then flip
    std::vector<float> base_sigmas(alphas_cumprod.size());
    for (size_t i = 0; i < base_sigmas.size(); ++i) {
        float a_bar = alphas_cumprod[i];
        base_sigmas[i] = std::sqrt(std::max(0.0f, (1.0f - a_bar) / std::max(a_bar, 1e-20f)));
    }
    base_sigmas = flip(base_sigmas);

    // base timesteps = reversed 0..N-1 (float)
    timesteps = flip(linspace(0.0f, float(cfg_.num_train_timesteps - 1), cfg_.num_train_timesteps));

    // final sigma (append)
    float sigma_last = 0.0f;
    if (cfg_.final_sigmas_type == EulerDiscreteSchedulerConfig::FinalSigmas::SigmaMin) {
        // sigma at t=0 (first training step)
        float a0 = alphas_cumprod.front();
        sigma_last = std::sqrt((1.0f - a0) / std::max(a0, 1e-20f));
    }

    sigmas = base_sigmas;
    sigmas.push_back(sigma_last);

    _step_index = -1;
    _begin_index = -1;
    num_inference_steps = 0; // set later via set_timesteps
}

// -------- properties --------
float EulerDiscreteScheduler::init_noise_sigma() const {
    float max_sigma = *std::max_element(sigmas.begin(), sigmas.end());
    if (cfg_.timestep_spacing == EulerDiscreteSchedulerConfig::TimestepSpacing::Linspace ||
        cfg_.timestep_spacing == EulerDiscreteSchedulerConfig::TimestepSpacing::Trailing) {
        return max_sigma;
    }
    return std::sqrt(max_sigma * max_sigma + 1.0f);
}

// -------- timeline --------
void EulerDiscreteScheduler::set_timesteps(int steps,
                                           const std::vector<float>* custom_timesteps,
                                           const std::vector<float>* custom_sigmas) {
    if ((custom_timesteps != nullptr) && (custom_sigmas != nullptr))
        throw std::runtime_error("Only one of timesteps or sigmas may be set.");
    if (steps <= 0 && !custom_timesteps && !custom_sigmas)
        throw std::runtime_error("Provide num_inference_steps or custom timesteps/sigmas.");

    num_inference_steps = custom_timesteps ? int(custom_timesteps->size())
                                          : (custom_sigmas ? int(custom_sigmas->size()) - 1 : steps);

    std::vector<float> ts;
    std::vector<float> sig;

    // derive (ts, sig) if not both provided
    if (custom_sigmas) {
        sig = *custom_sigmas; // size = steps+1
        // derive timesteps by matching sigma->t on log scale
        // build training log_sigmas
        std::vector<float> train_sig(sigmas.size() - 1);
        for (size_t i = 0; i < train_sig.size(); ++i) train_sig[i] = sigmas[i]; // current sigmas (has +1 at end)
        std::vector<float> log_train(train_sig.size());
        for (size_t i = 0; i < train_sig.size(); ++i) log_train[i] = std::log(std::max(train_sig[i], 1e-10f));

        ts.resize(sig.size() - 1);
        for (size_t i = 0; i + 1 < sig.size(); ++i)
            ts[i] = _sigma_to_t(sig[i], log_train);

    } else {
        if (custom_timesteps) {
            ts = *custom_timesteps;
        } else {
            // auto-generate timesteps per spacing
            if (cfg_.timestep_spacing == EulerDiscreteSchedulerConfig::TimestepSpacing::Linspace) {
                ts = flip(linspace(0.0f, float(cfg_.num_train_timesteps - 1), num_inference_steps));
            } else if (cfg_.timestep_spacing == EulerDiscreteSchedulerConfig::TimestepSpacing::Leading) {
                int step_ratio = cfg_.num_train_timesteps / num_inference_steps;
                ts.resize(num_inference_steps);
                for (int i = 0; i < num_inference_steps; ++i)
                    ts[num_inference_steps - 1 - i] = float(i * step_ratio + cfg_.steps_offset);
            } else { // Trailing
                float step_ratio = float(cfg_.num_train_timesteps) / float(num_inference_steps);
                ts.resize(num_inference_steps);
//                for (int i = 0; i < num_inference_steps; ++i)
//                    ts[i] = float(cfg_.num_train_timesteps) - float(i + 1) * step_ratio;
                ts[0] = cfg_.num_train_timesteps - 1.0f;
                for (int i = 1; i < num_inference_steps; i++) {
                    ts[i] = ts[i-1] - step_ratio;
                }
                // round and shift like Py
                for (auto& v : ts) v = std::round(v); // - 1.0f;
            }
        }

        // interpolate sigmas along training schedule
        // training sigmas (without the trailing 0)
        std::vector<float> base_sigmas(alphas_cumprod.size());
        for (size_t i = 0; i < base_sigmas.size(); ++i) {
            float a_bar = alphas_cumprod[i];
            base_sigmas[i] = std::sqrt(std::max(0.0f, (1.0f - a_bar) / std::max(a_bar, 1e-20f)));
        }
        std::vector<float> train_sig(sigmas.size() - 1);
        std::vector<float> flipped_sigmas = flip(sigmas);
        for (size_t i = 0; i < train_sig.size(); ++i) train_sig[i] = flipped_sigmas[i+1];

        // x = 0..(N-1)
        std::vector<float> x(train_sig.size());
        std::iota(x.begin(), x.end(), 0.0f);

        if (cfg_.interpolation_type == EulerDiscreteSchedulerConfig::InterpType::Linear) {
            sig = interp_linear(ts, x, base_sigmas);
//            sig = interp_linear(ts, x, train_sig);
        } else { // LogLinear
            float log_last = std::log(std::max(train_sig.back(), 1e-10f));
            float log_first= std::log(std::max(train_sig.front(),1e-10f));
            sig = exp_linspace(log_first, log_last, num_inference_steps + 1);
        }

        // append final sigma
        float sigma_last = 0.0f;
        if (cfg_.final_sigmas_type == EulerDiscreteSchedulerConfig::FinalSigmas::SigmaMin) {
            float a0 = alphas_cumprod.front();
            sigma_last = std::sqrt((1.0f - a0) / std::max(a0, 1e-20f));
        }
        if (cfg_.interpolation_type == EulerDiscreteSchedulerConfig::InterpType::Linear) {
            sig.push_back(sigma_last);
        } else {
            // already length+1 for log-linear generation
            sig.back() = sigma_last;
        }
    }

    // set
    timesteps = ts;
    sigmas = sig;

    // reset step counters
    _step_index = -1;
    _begin_index = -1;
    is_scale_input_called = false;
}

// -------- index helpers --------
int EulerDiscreteScheduler::index_for_timestep(float timestep, const std::vector<float>* schedule_ts) const {
    auto& arr = schedule_ts ? *schedule_ts : timesteps;
    // exact match first
    for (size_t i = 0; i < arr.size(); ++i) {
        if (arr[i] == timestep) {
            // Py: if multiple, pick second; here we keep first unless multiple equals
            // In our generated schedules, duplicates shouldn't happen.
            return int(i);
        }
    }
    // fallback: nearest
    auto it = std::min_element(arr.begin(), arr.end(),
        [&](float a, float b){ return std::fabs(a - timestep) < std::fabs(b - timestep); });
    return int(it - arr.begin());
}

void EulerDiscreteScheduler::_init_step_index(float timestep) {
    if (_begin_index < 0) {
        _step_index = index_for_timestep(timestep);
    } else {
        _step_index = _begin_index;
    }
}

float EulerDiscreteScheduler::_sigma_to_t(float sigma, const std::vector<float>& log_sigmas) {
    float log_sigma = std::log(std::max(sigma, 1e-10f));

    // dists = log_sigma - log_sigmas[:, None]  -> compare to array
    // find low/high indices so that log_sigmas[low] >= log_sigma >= log_sigmas[high] in monotonic sense
    // log_sigmas here corresponds to training sigmas along t=0..N-1
    int N = int(log_sigmas.size());
    // low_idx = max index where log_sigmas[i] <= log_sigma
    int low_idx = 0;
    while (low_idx + 1 < N && !(log_sigmas[low_idx + 1] > log_sigma)) ++low_idx;
    if (low_idx >= N - 1) low_idx = N - 2;
    int high_idx = low_idx + 1;

    float low  = log_sigmas[low_idx];
    float high = log_sigmas[high_idx];

    float denom = (low - high);
    float w = (denom != 0.0f) ? ((low - log_sigma) / denom) : 0.0f;
    w = std::min(1.0f, std::max(0.0f, w));

    float t = (1.0f - w) * float(low_idx) + w * float(high_idx);
    return t;
}

// -------- scale_model_input --------
void EulerDiscreteScheduler::scale_model_input(std::vector<float>& sample, float timestep) {
    if (_step_index < 0) _init_step_index(timestep);
    float sigma = sigmas[size_t(_step_index)];
    float denom = std::sqrt(sigma * sigma + 1.0f);
    for (auto& v : sample) v /= denom;
    is_scale_input_called = true;
}

void EulerDiscreteScheduler::scale_model_input_inplace(const float* sample, float* latent_model_input, size_t n, const float timestep) {
    if (_step_index < 0) _init_step_index(timestep);
    float sigma = sigmas[size_t(_step_index)];
    float denom = std::sqrt(sigma * sigma + 1.0f);
    for (size_t i = 0; i < n; ++i) latent_model_input[i] = sample[i] / denom;
    is_scale_input_called = true;
}

// -------- step --------
EulerDiscreteSchedulerOutput EulerDiscreteScheduler::step(const std::vector<float>& model_output,
                                                          float timestep,
                                                          const std::vector<float>& sample,
                                                          float s_churn,
                                                          float s_tmin,
                                                          float s_tmax,
                                                          float s_noise) {
    if (!is_scale_input_called) {
        // mirrors Python warning behavior; we’ll just proceed
        // (you can add a LOGW here if you want)
    }
    if (_step_index < 0) _init_step_index(timestep);

    // upcast (already float32)
    const size_t N = sample.size();
    if (model_output.size() != N) throw std::runtime_error("step: size mismatch");

    float sigma = sigmas[size_t(_step_index)];
    float gamma = (s_tmin <= sigma && sigma <= s_tmax)
                  ? std::min(s_churn / float(sigmas.size() - 1), std::sqrt(2.0f) - 1.0f)
                  : 0.0f;

    float sigma_hat = sigma * (gamma + 1.0f);

    std::vector<float> cur_sample(sample);

    if (gamma > 0.0f) {
        // If you want extra stochasticity, add noise ~ N(0,1) * s_noise * sqrt(sigma_hat^2 - sigma^2)
        // (Omitted RNG here—keep deterministic for now. Easy to add with your RNG.)
    }

    // pred_original_sample
    std::vector<float> pred_x0(N);
    if (cfg_.prediction_type == EulerDiscreteSchedulerConfig::PredictionType::Sample) {
        pred_x0 = model_output;
    } else if (cfg_.prediction_type == EulerDiscreteSchedulerConfig::PredictionType::Epsilon) {
        for (size_t i = 0; i < N; ++i) pred_x0[i] = cur_sample[i] - sigma_hat * model_output[i];
    } else { // v_prediction
        for (size_t i = 0; i < N; ++i) {
            float denom = std::sqrt(sigma * sigma + 1.0f);
            pred_x0[i] = model_output[i] * (-sigma / denom) + (cur_sample[i] / (sigma * sigma + 1.0f));
        }
    }

    // derivative = (x - x0) / sigma_hat
    std::vector<float> deriv(N);
    for (size_t i = 0; i < N; ++i) deriv[i] = (cur_sample[i] - pred_x0[i]) / sigma_hat;

    float dt = sigmas[size_t(_step_index + 1)] - sigma_hat;

    // prev_sample = x + deriv * dt
    std::vector<float> prev(N);
    for (size_t i = 0; i < N; ++i) prev[i] = cur_sample[i] + deriv[i] * dt;

    // step++
    _step_index += 1;

    return EulerDiscreteSchedulerOutput{ std::move(prev), std::move(pred_x0) };
}

void EulerDiscreteScheduler::step_inplace(const float* model_output,
                                          float timestep,
                                          float* sample,
                                          size_t n) {
    if (!is_scale_input_called) {
        // optional: log a warning like Python
        LOGW_ES("The scale_model_input function should be called before step to ensure correct denoising");
    }
    if (_step_index < 0) _init_step_index(timestep);

    const float sigma = sigmas[size_t(_step_index)];
    const float gamma = 0.0f; // keep deterministic for now (no churn/noise)
    const float sigma_hat = sigma * (gamma + 1.0f);
    LOGW_ES("[Scheduler:] Sigma: %f", sigma);

    // pred_original_sample
    std::vector<float> pred_x0(n);
    if (cfg_.prediction_type == EulerDiscreteSchedulerConfig::PredictionType::Sample) {
        std::memcpy(pred_x0.data(), model_output, n * sizeof(float));
    } else if (cfg_.prediction_type == EulerDiscreteSchedulerConfig::PredictionType::Epsilon) {
        LOGI_ES("[Scheduler:] Epsilon step...");
        for (size_t i = 0; i < n; ++i) pred_x0[i] = sample[i] - sigma_hat * model_output[i];
    } else { // VPrediction
        for (size_t i = 0; i < n; ++i) {
            float denom = std::sqrt(sigma * sigma + 1.0f);
            pred_x0[i] = model_output[i] * (-sigma / denom) + (sample[i] / (sigma * sigma + 1.0f));
        }
    }

    // derivative and dt
    std::vector<float> prev(n);
    const float dt = sigmas[size_t(_step_index + 1)] - sigma_hat;
    for (size_t i = 0; i < n; ++i) {
        const float deriv = (sample[i] - pred_x0[i]) / sigma_hat;
        prev[i] = sample[i] + deriv * dt;
    }

    // write back in-place
    std::memcpy(sample, prev.data(), n * sizeof(float));

    _step_index += 1;
}

// -------- add_noise --------
std::vector<float> EulerDiscreteScheduler::add_noise(const std::vector<float>& x0,
                                                     const std::vector<float>& noise,
                                                     const std::vector<float>& tvec) const {
    if (x0.size() != noise.size()) throw std::runtime_error("add_noise: size mismatch");
    std::vector<float> out(x0.size());

    // choose step index per timestep
    std::vector<int> idxs(tvec.size());
    for (size_t i = 0; i < tvec.size(); ++i) idxs[i] = index_for_timestep(tvec[i]);

    // Here we assume batch=1 (flat vector). For batched cases, extend as needed.
    float sigma = sigmas[size_t(idxs[0])];
    for (size_t i = 0; i < x0.size(); ++i) out[i] = x0[i] + noise[i] * sigma;
    return out;
}

// -------- get_velocity (optional helper) --------
std::vector<float> EulerDiscreteScheduler::get_velocity(const std::vector<float>& sample,
                                                        const std::vector<float>& noise,
                                                        const std::vector<float>& tvec) const {
    if (sample.size() != noise.size()) throw std::runtime_error("get_velocity: size mismatch");

    std::vector<int> idxs(tvec.size());
    for (size_t i = 0; i < tvec.size(); ++i) idxs[i] = index_for_timestep(tvec[i]);

    // sqrt_alpha_prod and sqrt_one_minus_alpha_prod at selected steps
    float a_bar = alphas_cumprod[size_t(idxs[0])];
    float s1 = std::sqrt(a_bar);
    float s2 = std::sqrt(std::max(0.0f, 1.0f - a_bar));

    std::vector<float> vel(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) vel[i] = s1 * noise[i] - s2 * sample[i];
    return vel;
}