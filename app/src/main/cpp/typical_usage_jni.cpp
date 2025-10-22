//
// Created by Chiheb Boussema on 24/9/25.
//
// typical_usage_jni.cpp

#include <jni.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "hpp/inference.h"
#include "SNPE/SNPEFactory.hpp"
#include "ModelSession.hpp"
#include "GraphRunner.hpp"
#include "TensorWorkspace.hpp"
#include "MMapAsset.hpp"
#include "hpp/ParseConfig.hpp"
#include "hpp/MMapFile.h"
#include "hpp/newInferenceHelper.hpp"

#define LOG_TAG "TYPICAL_USAGE_JNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)


static std::string g_modelDir; // holds the model directory path -- can be instantiated from the Kotlin main UI activity, or left empty to be instantiated from the config file

// function to set the model directory -- needed to interface Kotlin main UI activity; otherwise for pure c++ not needed
// this function is called from the Kotlin main UI activity
static void n_setModelDirectory(JNIEnv* env, jclass, jstring jpath) {
    const char* p = env->GetStringUTFChars(jpath, nullptr);
    g_modelDir = p ? p : "";
    env->ReleaseStringUTFChars(jpath, p);
    LOGI("SNPE model base dir set to: %s", g_modelDir.c_str());
}

static jstring n_executeInference(JNIEnv* env, jclass, jobject assetManager, jchar runtimePref) {

    std::string config_filename = "modelsConfig.json"; // change name as needed
    std::string log;
    bool reset_sessions = false; // forget network graphs to free memory -- useful when multiple models are loaded
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    std::unique_ptr<TensorWorkspace> g_ws; // holds workspace tensors
    std::unique_ptr<GraphRunner> g_gr; // holds graph runner
//    std::string g_modelDir; // holds the model directory path

    g_ws.reset(new TensorWorkspace());
    g_gr.reset(new GraphRunner(*g_ws));

    log = buildArbitraryChain(mgr, g_modelDir, config_filename, *g_ws, *g_gr, runtimePref, reset_sessions);

    // execute graph
   if (!g_gr) return env->NewStringUTF("Graph not built");
   std::string execution_summary;
   execution_summary = runGraph(*g_gr, false);

   return env->NewStringUTF(execution_summary.c_str());
}

static const JNINativeMethod kMethods[] = {
        {"executeInference", "(Landroid/content/res/AssetManager;C)Ljava/lang/String;", (void*) n_executeInference},
//        {"setModelDirectory", "(Ljava/lang/String;)V", (void*)n_setModelDirectory},
};

//JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
//    JNIEnv* env = nullptr;
//    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
//
//    jclass cls = env->FindClass("com/example/snpechainingdemo/SNPEHelper");
//    if (!cls) return JNI_ERR;
//    if (env->RegisterNatives(cls, kMethods, sizeof(kMethods)/sizeof(kMethods[0])) < 0) return JNI_ERR;
//
//    return JNI_VERSION_1_6;
//}