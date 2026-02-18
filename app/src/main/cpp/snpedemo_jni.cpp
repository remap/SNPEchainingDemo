//
// Created by Chiheb Boussema on 8/9/25.
//
#include <jni.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "inference.h"
#include "SNPE/SNPEFactory.hpp"
#include "ModelSession.hpp"
#include "GraphRunner.hpp"
#include "TensorWorkspace.hpp"
#include "MMapAsset.hpp"
#include "ParseConfig.hpp"
#include "MMapFile.h"
#include "newInferenceHelper.hpp"
#include "initTensorsHelper.h"
//#include "SDXLPipeline.h"
#include "SDXLPipelineGranular.h"
#include "Spar3DPipeline.h"

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "SNPE_JNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

// Keep your workspace / graph state somewhere (or return summaries only)
//static TensorWorkspace* g_ws = nullptr;   // Example: if you want to fetch outputs later
static std::unique_ptr<TensorWorkspace> g_ws;
//
////struct BuildTimes {
////    int64_t openMs=0, build1Ms=0, build2Ms=0, captureMs=0, allocMs=0, totalMs=0;
////};
////static BuildTimes g_buildTimes();
static std::unique_ptr<ModelSession> g_s1, g_s2;
static std::unique_ptr<GraphRunner> g_gr;

// Times:
//static inline int64_t msSince(std::chrono::steady_clock::time_point t0) {
//    using clock = std::chrono::steady_clock;
//    return std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
//}

static std::string g_modelDir;
static void n_setModelDirectory(JNIEnv* env, jclass, jstring jpath) {
    const char* p = env->GetStringUTFChars(jpath, nullptr);
    g_modelDir = p ? p : "";
    env->ReleaseStringUTFChars(jpath, p);
    LOGI("SNPE model base dir set to: %s", g_modelDir.c_str());
}

//bool readAssetToString(AAssetManager* mgr,
//                       const char* filename,
//                       std::string& out,
//                       std::string* emsg) {
//    AAsset* a = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
//    if (!a) {
//        if (emsg) *emsg = std::string("Asset open failed: ") + filename;
//        return false;
//    }
//    size_t len = AAsset_getLength(a);
//    out.resize(len);
//    int r = AAsset_read(a, out.data(), len);
//    AAsset_close(a);
//    if (r != (int)len) {
//        if (emsg) *emsg = "Asset read truncated";
//        return false;
//    }
//    return true;
//}


//static jstring n_executeInference(JNIEnv* env, jclass, jobject assetManager, jchar runtimePref) {
//
//    std::unique_ptr<TensorWorkspace> ws_;
//    std::unique_ptr<GraphRunner> gr_;
//
//    ws_.reset(new TensorWorkspace());
//    gr_.reset(new GraphRunner(*ws_));
//
//    // create asset manager
//    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
//
//    // read config file
//    std::string cfgText;
//    std::string emsg;
//    std::string config_filename = "modelsConfig.json";
//    if (!readAssetToString(mgr, config_filename.c_str(), cfgText, &emsg)) {
//        return env->NewStringUTF(("Config read failed: " + emsg).c_str());
//    }
//
//    // create models, allocate buffers and build graph
//    std::string buildingLog;
//    buildingLog = buildModelsAndGraph(mgr, cfgText, runtimePref, ws_, gr_);
//
//    // execute graph
//    if (!gr_) return env->NewStringUTF("Graph not built");
//    std::string execution_summary;
//    execution_summary = runGraph(*gr_);
//
//    return env->NewStringUTF(execution_summary.c_str());
//
//}

static jstring n_buildArbitrary(JNIEnv* env, jclass, jobject assetManager, jchar runtimePref) {

    g_ws.reset(new TensorWorkspace());
    g_gr.reset(new GraphRunner(*g_ws));

    // create asset manager
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    // read config file
    std::string cfgText;
    std::string emsg;
    std::string config_filename = "modelsConfig.json";
    if (!readAssetToString(mgr, config_filename.c_str(), cfgText, &emsg)) {
        return env->NewStringUTF(("Config read failed: " + emsg).c_str());
    }
    // parse config file
    PipelineCfg cfg;
    {
        std::string emsg;
        if (!ParseConfig(cfgText, cfg, &emsg)) {
            return env->NewStringUTF(("Config parse failed: " + emsg).c_str());
        }
        if (cfg.models.empty()) {
            return env->NewStringUTF("Config has no models");
        }
    }

    // create models, allocate buffers and build graph
    std::string buildingLog;
    int k = 0;
    bool reset_graph = true; //false;
    for (const auto &mc: cfg.models)
    {
//        if (k >=2) {
//            LOGI("[BUILDING] Resetting graph!");
//            reset_graph = true;
//            k = 0;
//        } else {
//            reset_graph = false;
//        }
        buildingLog = buildModelAndGraph(mgr,
                            g_modelDir,
                            cfg,
                            mc,
                            runtimePref,
                            *g_ws,
                            *g_gr,
                            buildingLog,
                            reset_graph);
        k += 1;
    }
    {
        std::string semsg;
        LOGI("Seeding input tensors...");
        if (!seedRequiredInputs(cfg, *g_ws, mgr, &semsg)) {
            return env->NewStringUTF(("Input seeding failed: " + semsg).c_str());
        }
    }

    return env->NewStringUTF(buildingLog.c_str());
}


static jstring n_buildPipes(JNIEnv* env, jclass, jobject assetManager, jchar runtimePref) {

    g_ws.reset(new TensorWorkspace());
    std::vector<GraphRunner*> v_Gr;
    g_gr.reset(new GraphRunner(*g_ws));

    // create asset manager
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    std::string cfgText;
    std::string emsg;
    std::string config_filename = "allModelsConfig.json";
    if (!readAssetToString(mgr, config_filename.c_str(), cfgText, &emsg)) {
        return env->NewStringUTF(("Config read failed: " + emsg).c_str());
    }
    // parse config file
    MultiPipelinesCfg mp_cfg;
    {
        std::string emsg;
        if (!ParseMultiConfig(cfgText, mp_cfg, &emsg)) {
            LOGE("Config parse failed: %s", emsg.c_str());
            return env->NewStringUTF(("Config parse failed: " + emsg).c_str());
        }
        if (mp_cfg.pipes.empty()) {
            LOGE("Config has no pipelines");
            return env->NewStringUTF(("Config parse failed: " + emsg).c_str());
        }
    }

    std::string log;
    for (auto& pipe : mp_cfg.pipes) {

        LOGI("building for pipe: %s", pipe.name.c_str());
        if (pipe.name == "unet_ublocks") {
            log += buildArbitraryChainFromConfig(mgr,
                                      g_modelDir,
                                      pipe,
                                      *g_ws,
                                      *g_gr,
                                      'D',
                                      true);
        } else {
            log += buildArbitraryChainFromConfig(mgr,
                                                 g_modelDir,
                                                 pipe,
                                                 *g_ws,
                                                 *new GraphRunner(*g_ws),
                                                 'D',
                                                 true);
        }
    }

    return env->NewStringUTF(("Building done! \n " + log).c_str());
}

static jstring n_rebuildArbitrary(JNIEnv* env, jclass) {

    std::string rebuildingLog;
    for (auto& n: g_gr->getNodes())
    {
        LOGI("Rebuilding for node %s", n.name.c_str());
        n.session.get()->reCreate(&rebuildingLog);
        LOGI("REBUILDING SUCCESSFUL! undoing...");
        n.session.get()->reset();
    }
    return env->NewStringUTF(rebuildingLog.c_str());

}


static jstring n_runGraphOld(JNIEnv* env, jclass) {
    // 6) Run
    if (!g_gr) return env->NewStringUTF("Graph not built");
    auto T0 = std::chrono::steady_clock::now();
    auto infos = g_gr->runAll(/*reset_session*/ true);
    auto T1 = std::chrono::steady_clock::now();
    auto execMs = std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count();
    LOGI("Graph Execution time: %lld", execMs);

    // 7) Summarize result
    std::string summary;
    for (auto& e : infos) {
        summary += e.name + " runtime=" + e.runtime + " time=" + std::to_string(e.ms) + "ms "
                   + (e.ok ? "OK\n" : "FAIL\n");
    }
    return env->NewStringUTF(summary.c_str());
}

static jstring n_runSDXL(JNIEnv* env, jclass, jobject assetManager, jintArray jIds1, jintArray jIds2) {

//    std::string config_filename = "textEncoders_modelsConfig.json"; // change name as needed
    std::string config_filename = "textEncoders_8Elite_modelsConfig.json"; // change name as needed
    std::string log;
    bool reset_sessions = false; // forget network graphs to free memory -- useful when multiple models are loaded
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    if (!jIds1 || !jIds2) {
        LOGE("ids arrays are null");
        return env->NewStringUTF("ids arrays are null");
    }
    const jsize len1 = env->GetArrayLength(jIds1);
    const jsize len2 = env->GetArrayLength(jIds2);
    if (len1 <= 0 || len2 <= 0) {
        LOGE("ids length invalid");
        return env->NewStringUTF("ids length invalid");
    }
    LOGI("Here 1");
    jboolean isCopy1 = JNI_FALSE, isCopy2 = JNI_FALSE;
    jint* p1 = env->GetIntArrayElements(jIds1, &isCopy1);
    LOGI("Here 2");
    jint* p2 = env->GetIntArrayElements(jIds2, &isCopy2);
    LOGI("Here 3");
    if (!p1 || !p2) {
        if (p1) env->ReleaseIntArrayElements(jIds1, p1, JNI_ABORT);
        if (p2) env->ReleaseIntArrayElements(jIds2, p2, JNI_ABORT);
        LOGE("GetIntArrayElements failed");
        return env->NewStringUTF("GetIntArrayElements failed");
    }

    std::unique_ptr<TensorWorkspace> ws_sdxl; // holds workspace tensors
    std::unique_ptr<GraphRunner> gr_sdxl; // holds graph runner
//    std::string g_modelDir; // holds the model directory path

    ws_sdxl.reset(new TensorWorkspace());
    gr_sdxl.reset(new GraphRunner(*ws_sdxl));

//    std::unique_ptr<SDXLPipeline> g_sdxl_pipe;
    std::unique_ptr<SDXLPipelineGranular> g_sdxl_pipe;
    LOGI("CREATING SDXL PIPE");
//    g_sdxl_pipe.reset(new SDXLPipeline(*ws_sdxl, *gr_sdxl, mgr, "", config_filename));
    g_sdxl_pipe.reset(new SDXLPipelineGranular(*ws_sdxl, *gr_sdxl, mgr, "", config_filename));
    LOGI("CREATED SDXL PIPE");

//    log = buildArbitraryChain(mgr, g_modelDir, config_filename, *g_ws, *g_gr, runtimePref, reset_sessions);
    bool init_encoders_successfull = false;
    init_encoders_successfull = g_sdxl_pipe->init_text_encoders('D');

    std::string execution_summary;
    if (init_encoders_successfull) {
        LOGI("encoders initialization successful!");
        execution_summary = g_sdxl_pipe->run_encoders(reinterpret_cast<int32_t*>(p1),
                                                      reinterpret_cast<int32_t*>(p2));
    } else {
        LOGI("encoders initialization NOT successful!");
    }
//    / Release ASAP
    env->ReleaseIntArrayElements(jIds1, p1, JNI_ABORT); // we copied into workspace already
    env->ReleaseIntArrayElements(jIds2, p2, JNI_ABORT);

   return env->NewStringUTF(execution_summary.c_str());
}

static jfloatArray n_runSDXLWhole(JNIEnv* env, jclass, jobject assetManager,
                                  jintArray jIds1, jintArray jIds2,
                                  jboolean decode_only, jboolean init_only) {

//    std::string encoders_config_filename = "textEncoders_modelsConfig.json"; // change name as needed
    std::string encoders_config_filename = "textEncoders_8Elite_modelsConfig.json"; // change name as needed

//    std::string allModels_config_filename = "allModelsConfig.json"; // change name as needed
//    std::string allModels_config_filename = "allModelsConfigNoIO.json"; // change name as needed
//    std::string allModels_config_filename = "allModelsConfig8Elite.json"; // change name as needed
//    std::string allModels_config_filename = "allModelsConfigNewCalib8Elite.json"; // change name as needed

//    std::string allModels_config_filename = "Granular_allModelsConfigNewCalib8Elite.json"; // change name as needed
//    std::string allModels_config_filename = "C_Granular_allModelsConfigNewCalib8Elite.json"; // change name as needed

//    std::string allModels_config_filename = "FP32_Granular_allModelsConfigNewCalib8Elite.json"; // change name as needed

    std::string allModels_config_filename = "SDXL_FP32_aimetONNX_ModelsConfig8Elite.json"; // change name as needed
    std::string log;
    bool reset_sessions = false; // forget network graphs to free memory -- useful when multiple models are loaded
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    if (!jIds1 || !jIds2) {
        LOGE("ids arrays are null");
//        return env->NewStringUTF("ids arrays are null");
        return env->NewFloatArray(0);
    }
    const jsize len1 = env->GetArrayLength(jIds1);
    const jsize len2 = env->GetArrayLength(jIds2);
    if (len1 <= 0 || len2 <= 0) {
        LOGE("ids length invalid");
//        return env->NewStringUTF("ids length invalid");
        return env->NewFloatArray(0);
    }
//    LOGI("Here 1");
    jboolean isCopy1 = JNI_FALSE, isCopy2 = JNI_FALSE;
    jint* p1 = env->GetIntArrayElements(jIds1, &isCopy1);
//    LOGI("Here 2");
    jint* p2 = env->GetIntArrayElements(jIds2, &isCopy2);
//    LOGI("Here 3");
    if (!p1 || !p2) {
        if (p1) env->ReleaseIntArrayElements(jIds1, p1, JNI_ABORT);
        if (p2) env->ReleaseIntArrayElements(jIds2, p2, JNI_ABORT);
        LOGE("GetIntArrayElements failed");
//        return env->NewStringUTF("GetIntArrayElements failed");
        return env->NewFloatArray(0);
    }

    std::unique_ptr<TensorWorkspace> ws_sdxl; // holds workspace tensors
    std::unique_ptr<GraphRunner> gr_sdxl_encoders; // holds graph runner
//    std::string g_modelDir; // holds the model directory path

    ws_sdxl.reset(new TensorWorkspace());
    gr_sdxl_encoders.reset(new GraphRunner(*ws_sdxl));

//    std::unique_ptr<SDXLPipeline> g_sdxl_pipe;
    std::unique_ptr<SDXLPipelineGranular> g_sdxl_pipe;
    LOGI("CREATING SDXL PIPE");
//    g_sdxl_pipe.reset(new SDXLPipeline(*ws_sdxl,
//                                       mgr,
//                                       "",
//                                       *gr_sdxl_encoders,
//                                       encoders_config_filename,
//                                       allModels_config_filename
//                                       ));
    g_sdxl_pipe.reset(new SDXLPipelineGranular(*ws_sdxl,
                                       mgr,
                                       "",
                                       *gr_sdxl_encoders,
                                       encoders_config_filename,
                                       allModels_config_filename
                                       ));
    LOGI("CREATED SDXL PIPE");

//    log = buildArbitraryChain(mgr, g_modelDir, config_filename, *g_ws, *g_gr, runtimePref, reset_sessions);
//    bool init_encoders_successfull = false;
//    init_encoders_successfull = g_sdxl_pipe->init_text_encoders('D');

    bool init_networks_successful = false;
    init_networks_successful = g_sdxl_pipe->init_networks('D');

    std::string execution_summary;
    std::vector<float> img;
    if (init_networks_successful) {
        LOGI("Networks initialization successful!");
//        g_sdxl_pipe->overall_pipeline(reinterpret_cast<int32_t*>(p1),
//                                                      reinterpret_cast<int32_t*>(p2));
        if (!init_only) {
            img = g_sdxl_pipe->overall_pipeline(reinterpret_cast<int32_t *>(p1),
                                                reinterpret_cast<int32_t *>(p2),
                                                decode_only);
        }
    } else {
        LOGI("Networks initialization NOT successful!");
    }
//    / Release ASAP
    env->ReleaseIntArrayElements(jIds1, p1, JNI_ABORT); // we copied into workspace already
    env->ReleaseIntArrayElements(jIds2, p2, JNI_ABORT);

    execution_summary = "Pipeline executed successfully!";

    g_ws.reset();
    g_ws = std::move(ws_sdxl);

    if (img.empty()) {
        return env->NewFloatArray(0);
    }
    jfloatArray out = env->NewFloatArray(static_cast<jsize>(img.size()));
    if (!out) return nullptr;
    env->SetFloatArrayRegion(out, 0, static_cast<jsize>(img.size()), img.data());
    return out;

//    return env->NewStringUTF(execution_summary.c_str());
}

static jfloatArray n_runSPAR3D(JNIEnv* env, jclass, jobject assetManager,
                               jobject byteBuffer, // Direct ByteBuffer from Kotlin
                               jint width, jint height,
                               jstring savePath,
                               jobject callback) {
//    std::string config_filename = "SparEliteConfig.json";
    std::string config_filename = "SparEliteNewSC2Config.json";
    std::string log;
    bool reset_sessions = false; // forget network graphs to free memory -- useful when multiple models are loaded
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    std::unique_ptr<TensorWorkspace> ws_spar; // holds workspace tensors
    ws_spar.reset(new TensorWorkspace());
    std::unique_ptr<Spar3DPipeline> g_spar_pipe;
    g_spar_pipe.reset(new Spar3DPipeline(mgr,
                                         "",
                                         config_filename,
                                         *ws_spar));

    // 1. Get pointer to data
    uint8_t* data = (uint8_t*)env->GetDirectBufferAddress(byteBuffer);
    std::vector<float> processed_image = g_spar_pipe->preprocessImage(data, width, height);
    jfloatArray result = env->NewFloatArray(processed_image.size());
    env->SetFloatArrayRegion(result, 0, processed_image.size(), processed_image.data());

    // --- TRIGGER CALLBACK HERE ---
    if (callback != nullptr) {
        // 1. Get the class of the callback object
        jclass cbClass = env->GetObjectClass(callback);
        // 2. Get the Method ID: onPreprocessComplete(float[]) -> void
        jmethodID methodID = env->GetMethodID(cbClass, "onPreprocessComplete", "([F)V");
        // 3. Call the method
        if (methodID != nullptr) {
            env->CallVoidMethod(callback, methodID, result);
        }
        // 4. Clean up local ref for class (optional but good practice)
        env->DeleteLocalRef(cbClass);
    }
    // -----------------------------
    // 2. initialize networks
    bool init_networks_successful = false;
    init_networks_successful = g_spar_pipe->init_networks('D');
    if (!init_networks_successful) {
        LOGE("[JNI:] Network initialization failed!");
        return nullptr;
    }

//    g_spar_pipe->test_spill(data, width, height);
//    g_spar_pipe->test_imest(data, width, height);
    std::string outputPath(env->GetStringUTFChars(savePath, 0));
//    g_spar_pipe->test_sc2(data, width, height, outputPath);

    g_spar_pipe->overall_pipeline(data, width, height, outputPath);

    return result;

}


// ------------ Native implementations (static) ------------
//static jstring n_buildTwoModelGraph(JNIEnv* env, jclass /*cls*/,
//                                    jobject assetManager, jchar runtimePref) {
//
//    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
//
//    // 1) Read DLCs into memory
//    auto readAsset = [&](const char* name, std::vector<uint8_t>& buf) -> bool {
//        AAsset* a = AAssetManager_open(mgr, name, AASSET_MODE_UNKNOWN);
//        if (!a) { LOGE("asset open failed: %s", name); return false; }
//        buf.resize(AAsset_getLength(a));
//        AAsset_read(a, buf.data(), buf.size());
//        AAsset_close(a);
//        return true;
//    };
//
//    std::vector<uint8_t> dlc1, dlc2;
//    const std::string model1_name = "unet_downblock1_8Gen2_prepared.dlc";
//    const std::string model2_name = "unet_downblock2_8Gen2_prepared.dlc";
//    if (!readAsset(model1_name.c_str(), dlc1) || !readAsset(model2_name.c_str(), dlc2)) {
//        return env->NewStringUTF("Failed to read DLCs");
//    }
//
//    // 2) Build ModelSession options
//    ModelSession::Options opt;
//    // prefer HTP then CPU; or just set order empty and let your Create() fallback fill it
//    if (runtimePref == 'D') { opt.runtimeOrder.add(zdl::DlSystem::Runtime_t::DSP); }
//    else if (runtimePref == 'G') { opt.runtimeOrder.add(zdl::DlSystem::Runtime_t::GPU); }
//    else { opt.runtimeOrder.add(zdl::DlSystem::Runtime_t::CPU); }
//    opt.perf = zdl::DlSystem::PerformanceProfile_t::BURST;
//    opt.useUserSuppliedBuffers = true;
//    opt.initCache = false;
//
//    std::string buildLog;
//    auto s1 = ModelSession::Create(dlc1.data(), dlc1.size(), opt, &buildLog);
//    auto s2 = ModelSession::Create(dlc2.data(), dlc2.size(), opt, &buildLog);
//    if (!s1 || !s2) {
//        LOGE("Build failed:\n%s", buildLog.c_str());
//        return env->NewStringUTF(("Build failed:\n" + buildLog).c_str());
//    }
//
//    // 3) Workspace (simple helper you already have)
//    TensorWorkspace ws;
//
//    // Figure out sizes for tensors you’ll bind. You can query from each session:
//    auto sizeBytes = [](const TensorInfo& t){ return t.bytes(); };
//
//    // Allocate workspace blocks large enough (once).
//    // Use the sizes from the sessions’ metadata to avoid mismatches.
//    // Inputs that originate outside (e.g., coming from Kotlin later) still need WS blocks
//    // if you want to keep zero-copy end-to-end; you can memcpy into WS for now.
//    auto findT = [](const std::vector<TensorInfo>& v, const char* name)->const TensorInfo*{
//        for (auto& t: v) if (t.name == name) return &t; return nullptr;
//    };
//
////    for (auto& t : s1->inputs())  LOGI_MS("[s1] IN  %s %s", t.name.c_str(), shapeToStr(t.dims).c_str());
////    for (auto& t : s1->outputs()) LOGI_MS("[s1] OUT %s %s", t.name.c_str(), shapeToStr(t.dims).c_str());
//
//    for (auto& t : s1->inputs())  LOGI("[s1] IN  %s", t.name.c_str());
//    for (auto& t : s1->outputs()) LOGI("[s1] OUT %s", t.name.c_str());
//    for (auto& t : s2->inputs())  LOGI("[s2] IN  %s", t.name.c_str());
//    for (auto& t : s2->outputs()) LOGI("[s2] OUT %s", t.name.c_str());
//
//    const TensorInfo* s1_sample  = findT(s1->inputs(),  "sample");
//    const TensorInfo* s1_temb    = findT(s1->inputs(),  "temb");
//    const TensorInfo* s1_ehs     = findT(s1->inputs(),  "encoder_hidden_states");
//    const TensorInfo* s1_out_h   = findT(s1->outputs(), "output_0");
//    const TensorInfo* s1_res0    = findT(s1->outputs(), "output_1");
//    const TensorInfo* s1_res1    = findT(s1->outputs(), "output_2");
//    const TensorInfo* s1_res2    = findT(s1->outputs(), "output_3");
//    const TensorInfo* s1_res3    = findT(s1->outputs(), "output_4");
//    const TensorInfo* s1_res4    = findT(s1->outputs(), "output_5");
//    const TensorInfo* s1_res5    = findT(s1->outputs(), "output_6");
//    const TensorInfo* s1_res6    = findT(s1->outputs(), "output_7");
//
//    const TensorInfo* s2_in_h    = findT(s2->inputs(),  "hidden_states");
//    const TensorInfo* s2_temb    = findT(s2->inputs(),  "temb");
//    const TensorInfo* s2_ehs     = findT(s2->inputs(),  "encoder_hidden_states");
//    const TensorInfo* s2_out_h    = findT(s2->outputs(), "output_0");
//    const TensorInfo* s2_res0    = findT(s2->outputs(), "output_1");
//    const TensorInfo* s2_res1    = findT(s2->outputs(), "output_2");
//
//    if (!s1_sample || !s1_temb || !s1_ehs || !s1_out_h || !s1_res0 || !s1_res1 || !s1_res2 ||
//            !s1_res3 || !s1_res4 || !s1_res5 || !s1_res6 ||
//        !s2_in_h   || !s2_temb || !s2_ehs || !s2_out_h || !s2_res0 || !s2_res1) {
//        LOGE("FINDING TENSORS","Missing expected tensor names—adjust bindings.");
//        return env->NewStringUTF("Missing expected tensor names—adjust bindings.");
//    }
//
//    ws.allocate("sample",                s1_sample->bytes());
//    ws.allocate("temb",                  s1_temb->bytes());
//    ws.allocate("encoder_hidden_states", s1_ehs->bytes());
//
//    ws.allocate("out1_hidden", s1_out_h->bytes());
//    ws.allocate("res0",       s1_res0->bytes());
//    ws.allocate("res1",       s1_res1->bytes());
//    ws.allocate("res2",       s1_res2->bytes());
//    ws.allocate("res3",       s1_res3->bytes());
//    ws.allocate("res4",       s1_res4->bytes());
//    ws.allocate("res5",       s1_res5->bytes());
//    ws.allocate("res6",       s1_res6->bytes());
//
//    ws.allocate("final_out",  s2_out_h->bytes());
//    ws.allocate("res7",       s2_res0->bytes());
//    ws.allocate("res8",       s2_res1->bytes());
//
//    // 4) Seed inputs (for now fill random / zeros, just to prove the chain works)
//    std::memset(ws.data("sample"),                0, ws.sizeOf("sample"));
//    std::memset(ws.data("temb"),                  0, ws.sizeOf("temb"));
//    std::memset(ws.data("encoder_hidden_states"), 0, ws.sizeOf("encoder_hidden_states"));
//
//    // 5) Build GraphRunner and bind
//    GraphRunner gr(ws);
//
//    GraphRunner::Node n1;
//    n1.name = "Model1";
//    n1.session = std::move(s1);
//    n1.inputBinding  = {
//            {"sample",                "sample"},
//            {"temb",                  "temb"},
//            {"encoder_hidden_states", "encoder_hidden_states"},
//    };
//    n1.outputBinding = {
//            {"output_0", "out1_hidden"},
//            {"output_1",       "res0"},
//            {"output_2",       "res1"},
//            {"output_3",       "res2"},
//            {"output_4",       "res3"},
//            {"output_5",       "res4"},
//            {"output_6",       "res5"},
//            {"output_7",       "res6"},
//    };
//    if (!gr.addNode(std::move(n1), /*strictZeroCopy=*/true)) {
//        return env->NewStringUTF("addNode(Model1) failed");
//    }
//
//    GraphRunner::Node n2;
//    n2.name = "Model2";
//    n2.session = std::move(s2);
//    n2.inputBinding  = {
//            {"hidden_states",         "out1_hidden"},
//            {"temb",                  "temb"},
//            {"encoder_hidden_states", "encoder_hidden_states"},
//    };
//    n2.outputBinding = {
//            {"output_0", "final_out"},
//            {"output_1", "res7"},
//            {"output_2", "res8"},
//    };
//    if (!gr.addNode(std::move(n2), /*strictZeroCopy=*/true)) {
//        return env->NewStringUTF("addNode(Model2) failed");
//    }
//
//    // 6) Run
//    auto infos = gr.runAll();
//
//    // 7) Summarize result
//    std::string summary;
//    for (auto& e : infos) {
//        summary += e.name + " runtime=" + e.runtime + " time=" + std::to_string(e.ms) + "ms "
//                   + (e.ok ? "OK\n" : "FAIL\n");
//    }
//    return env->NewStringUTF(summary.c_str());
//}

static jboolean n_getFinalTensor(JNIEnv* env, jclass /*cls*/, jobject dstDirectBuffer) {
    if (!g_ws) return JNI_FALSE;
    void* dst = env->GetDirectBufferAddress(dstDirectBuffer);
    jlong cap = env->GetDirectBufferCapacity(dstDirectBuffer);
    if (!dst || cap <= 0) return JNI_FALSE;

    const char* kOutName = "final_out";  // whatever you allocated/bound
    void* src = g_ws->data(kOutName);
    size_t sz = g_ws->sizeOf(kOutName);
    if (!src || (jlong)sz > cap) return JNI_FALSE;

    std::memcpy(dst, src, sz);
    return JNI_TRUE;
}

static jboolean n_getTensor(JNIEnv* env, jclass /*cls*/, jobject dstDirectBuffer, jstring tensorName) {
    if (!g_ws) return JNI_FALSE;
    void* dst = env->GetDirectBufferAddress(dstDirectBuffer);
    jlong cap = env->GetDirectBufferCapacity(dstDirectBuffer);
    if (!dst || cap <= 0) return JNI_FALSE;

    const char* kOutName = env->GetStringUTFChars(tensorName, nullptr);  // whatever you allocated/bound
    void* src = g_ws->data(kOutName);
    size_t sz = g_ws->sizeOf(kOutName);
    if (!src || (jlong)sz > cap) return JNI_FALSE;

    std::memcpy(dst, src, sz);
    return JNI_TRUE;
}

static jlong n_getTensorSizeBytes(JNIEnv* env, jclass /*cls*/, jstring jname) {
    if (!g_ws) return 0;
    const char* cname = env->GetStringUTFChars(jname, nullptr);
    size_t sz = g_ws->sizeOf(cname ? cname : "");
    if (cname) env->ReleaseStringUTFChars(jname, cname);
    return static_cast<jlong>(sz);
}



static jstring QueryRuntimes(JNIEnv* env, jobject /*thiz*/, jstring native_dir_path) {
    const char *cstr = env->GetStringUTFChars(native_dir_path, nullptr);
    std::string nativeLibPath(cstr);
    env->ReleaseStringUTFChars(native_dir_path, cstr);

    std::string out;
    if (!SetAdspLibraryPath(nativeLibPath)) {
        out = "Failed to set ADSP Library Path";
        return env->NewStringUTF(out.c_str());
    }

    out  = "Querying Runtimes : \n\n";
    out += (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP, zdl::DlSystem::RuntimeCheckOption_t::UNSIGNEDPD_CHECK) ?
            "UnsignedPD DSP runtime : Present\n" : "UnsignedPD DSP runtime : Absent\n");
    out += (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP) ?
            "DSP runtime : Present\n" : "DSP runtime : Absent\n");
    out += (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU) ?
            "GPU runtime : Present\n" : "GPU runtime : Absent\n");
    out += (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::CPU) ?
            "CPU runtime : Present\n" : "CPU runtime : Absent\n");
    return env->NewStringUTF(out.c_str());
}

static jstring InitSNPE(JNIEnv* env, jobject /*thiz*/, jobject asset_manager, jstring jassetName, jchar runtime) {
    const char* asset_cstr = env->GetStringUTFChars(jassetName, nullptr);
    std::string assetName(asset_cstr ? asset_cstr : "");
    env->ReleaseStringUTFChars(jassetName, asset_cstr);

    if (assetName.empty()) {
        std::string r = "No asset name provided";
        return env->NewStringUTF(r.c_str());
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    if (!mgr) {
        std::string r = "AAssetManager_fromJava failed";
        return env->NewStringUTF(r.c_str());
    }

    AAsset* asset = AAssetManager_open(mgr, assetName.c_str(), AASSET_MODE_UNKNOWN);
    if (!asset) {
        std::string r = "Failed to open assets/" + assetName;
        LOGE("%s", r.c_str());
        return env->NewStringUTF(r.c_str());
    }

    const off_t size = AAsset_getLength(asset);
    std::vector<uint8_t> buf(size);
    AAsset_read(asset, buf.data(), size);
    AAsset_close(asset);

    std::string res = "Building DLC: " + assetName + "\n";
    res += build_network_BB(buf.data(), buf.size(), (char)runtime);
    return env->NewStringUTF(res.c_str());
}

static jboolean InferSNPE(JNIEnv* env, jobject /*thiz*/, jobject latentDirectBuffer, jobject outDirectBuffer) {
    if (!latentDirectBuffer || !outDirectBuffer) return JNI_FALSE;

    auto* latentPtr = (float*) env->GetDirectBufferAddress(latentDirectBuffer);
    jlong latentBytes = env->GetDirectBufferCapacity(latentDirectBuffer);

    auto* outPtr = (float*) env->GetDirectBufferAddress(outDirectBuffer);
    jlong outBytes = env->GetDirectBufferCapacity(outDirectBuffer);

    bool ok = executeDLC(latentPtr, (size_t)latentBytes, outPtr, (size_t)outBytes);
    return ok ? JNI_TRUE : JNI_FALSE;
}

static jstring ActiveRuntime(JNIEnv* env, jobject) {
    const std::string& s = getActiveRuntimeName();
    return env->NewStringUTF(s.c_str());
}




// Manual registration
static const JNINativeMethod kMethods[] = {
        {"queryRuntimes", "(Ljava/lang/String;)Ljava/lang/String;", (void*)QueryRuntimes},
//        {"initSNPE", "(Landroid/content/res/AssetManager;Ljava/lang/String;C)Ljava/lang/String;", (void*)InitSNPE},
//        {"inferSNPE", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)Z", (void*)InferSNPE},
////        {"activeRuntime","()Ljava/lang/String;", (void*)Java_com_example_snpedemo_SNPEHelper_activeRuntime}
//        {"activeRuntime", "()Ljava/lang/String;", (void*)ActiveRuntime}
//        {"buildTwoModelGraph", "(Landroid/content/res/AssetManager;C)Ljava/lang/String;",(void *) n_buildTwoModelGraph},
        {"getFinalTensor", "(Ljava/nio/ByteBuffer;)Z", (void*) n_getFinalTensor},
        {"getTensor", "(Ljava/nio/ByteBuffer;Ljava/lang/String;)Z",(void*) n_getTensor},
        {"getTensorSizeBytes", "(Ljava/lang/String;)J",(void*)n_getTensorSizeBytes},
//        {"buildGraph", "(Landroid/content/res/AssetManager;C)Ljava/lang/String;", (void*)n_buildGraph},
        {"runGraph", "()Ljava/lang/String;", (void*)n_runGraphOld},
//        {"executeInference", "(Landroid/content/res/AssetManager;C)Ljava/lang/String;", (void*)n_executeInference},
        {"buildArbitrary", "(Landroid/content/res/AssetManager;C)Ljava/lang/String;", (void*) n_buildArbitrary},
        {"buildPipes", "(Landroid/content/res/AssetManager;C)Ljava/lang/String;", (void*) n_buildPipes},
        {"rebuildArbitrary", "()Ljava/lang/String;", (void*) n_rebuildArbitrary},
        {"setModelDirectory", "(Ljava/lang/String;)V", (void*)n_setModelDirectory},
        {"runSDXL", "(Landroid/content/res/AssetManager;[I[I)Ljava/lang/String;", (void*) n_runSDXL},
//        {"runSDXLWhole", "(Landroid/content/res/AssetManager;[I[I)Ljava/lang/String;", (void*) n_runSDXLWhole},
        {"runSDXLWhole", "(Landroid/content/res/AssetManager;[I[IZZ)[F", (void*) n_runSDXLWhole},
        {"runSPAR3D", "(Landroid/content/res/AssetManager;Ljava/nio/ByteBuffer;IILjava/lang/String;Lcom/example/snpechainingdemo/SNPEHelper$PreprocessCallback;)[F", (void*) n_runSPAR3D}
};


//static jstring NativeActiveRuntime(JNIEnv* env, jobject /*thiz*/) {
//    return env->NewStringUTF(g_activeRuntimeName.c_str());
//}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;

    jclass cls = env->FindClass("com/example/snpechainingdemo/SNPEHelper");
    if (!cls) return JNI_ERR;
    if (env->RegisterNatives(cls, kMethods, sizeof(kMethods)/sizeof(kMethods[0])) < 0) return JNI_ERR;

    return JNI_VERSION_1_6;
}
