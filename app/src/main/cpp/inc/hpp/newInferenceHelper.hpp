//
// Created by Chiheb Boussema.
//


#ifndef SNPECHAININGDEMO_NEWINFERENCEHELPER_H
#define SNPECHAININGDEMO_NEWINFERENCEHELPER_H

# pragma once
#include <jni.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "DlSystem/DlEnums.hpp"
#include "TensorWorkspace.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "ModelSession.hpp"
#include "GraphRunner.hpp"
#include "ParseConfig.hpp"
#include "MMapFile.h"
//#include "initTensorsHelper.h"

static zdl::DlSystem::RuntimeList makeRuntimeOrder(char pref);

static const TensorInfo* findTensor(const std::vector<TensorInfo>& v, const std::string& name);

bool ensureWorkspaceBuffer(TensorWorkspace& ws,
                                  const std::string& wsName,
                                  size_t bytes,
                                  std::string* emsg);

bool ensureWorkspaceBuffer(TensorWorkspace& ws,
                                  const std::string& wsName,
                                  const TensorInfo& tinfo,
                                  std::string* emsg);

static inline int64_t msSince(std::chrono::steady_clock::time_point t0);

//std::string buildModelAndGraph(AAssetManager* mgr,
//                                std::string& g_modelDir,
////                                const std::string& configJson,
//                                const PipelineCfg& cfg,
//                                const ModelCfg& mc,
//                                const char defaultRuntimePref,
//                                std::unique_ptr<TensorWorkspace>& outWs,
//                                std::unique_ptr<GraphRunner>& outGraph,
//                                std::string& log,
//                                bool reset_session=false);

std::string buildModelAndGraph(AAssetManager* mgr,
                                std::string& g_modelDir,
//                                const std::string& configJson,
                                const PipelineCfg& cfg,
                                const ModelCfg& mc,
                                const char defaultRuntimePref,
                                TensorWorkspace& outWs,
                                GraphRunner& outGraph,
                                std::string& log,
                                bool reset_session);

std::string buildArbitraryChain(AAssetManager* mgr,
                                std::string& g_modelDir,
                                const std::string config_filename,
                                TensorWorkspace& ws,
                                GraphRunner& gr,
                                const char defaultRuntimePref,
                                bool reset_sessions);

std::string buildArbitraryChainFromConfig(AAssetManager* mgr,
                                std::string& g_modelDir,
                                const PipelineCfg cfg,
                                TensorWorkspace& ws,
                                GraphRunner& gr,
                                const char defaultRuntimePref,
                                bool reset_sessions);

std::string rebuildNodeSession(GraphRunner::Node& node);
std::string rebuildMultipleNodes(std::vector<GraphRunner::Node>& nodes);
std::string rebuildAllGraphNodes(GraphRunner& gr);

std::string runGraph(GraphRunner& gr, bool reset_sessions);

//static bool readAssetToString(AAssetManager* mgr,
//                              const char* filename,
//                              std::string& out,
//                              std::string* emsg);

bool readAssetToString(AAssetManager* mgr,
                       const char* filename,
                       std::string& out,
                       std::string* emsg);

#endif // SNPECHAININGDEMO_NEWINFERENCEHELPER_H