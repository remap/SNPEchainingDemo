#include <jni.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "zdl/DlSystem/DlEnums.hpp"
#include "TensorWorkspace.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "ModelSession.hpp"
#include "GraphRunner.hpp"
#include "hpp/ParseConfig.hpp"
#include "hpp/MMapFile.h"

static zdl::DlSystem::RuntimeList makeRuntimeOrder(char pref);

static const TensorInfo* findTensor(const std::vector<TensorInfo>& v, const std::string& name);

static bool ensureWorkspaceBuffer(TensorWorkspace& ws,
                                  const std::string& wsName,
                                  size_t bytes,
                                  std::string* emsg);

static inline int64_t msSince(std::chrono::steady_clock::time_point t0);

std::string buildModelAndGraph(AAssetManager* mgr,
                                std::string& g_modelDir,
//                                const std::string& configJson,
                                const PipelineCfg& cfg,
                                const ModelCfg& mc,
                                const char defaultRuntimePref,
                                std::unique_ptr<TensorWorkspace>& outWs,
                                std::unique_ptr<GraphRunner>& outGraph,
                                std::string& log,
                                bool reset_session=false);

std::string runGraph(GraphRunner& gr);

//static bool readAssetToString(AAssetManager* mgr,
//                              const char* filename,
//                              std::string& out,
//                              std::string* emsg);