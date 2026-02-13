//
// Created by Chiheb Boussema on 16/9/25.
//
#include "GraphRunner.hpp"
#include <android/log.h>
#include <unistd.h>

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define  LOG_TAG_GR  "SNPE_GR"
#define  LOGI_GR(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG_GR,__VA_ARGS__)
#define  LOGE_GR(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG_GR,__VA_ARGS__)

GraphRunner::Node& GraphRunner::getNode(std::string name) {
    for (auto& node : this->nodes_) {
        if (node.name == name) return node;
    }
    throw std::runtime_error("Node not found: " + name);
}

bool GraphRunner::addNode(Node node, bool strictZeroCopy) {
    // Sanity: every bound IO has a workspace block and size that matches the model metadata
    for (auto& t : node.session->inputs()) {
        auto it = node.inputBinding.find(t.name);
        if (it == node.inputBinding.end()) {
            LOGE_GR("[%s] Missing binding for input '%s'", node.name.c_str(), t.name.c_str());
            return false;
        }
        const std::string& wsName = it->second;
        auto sz = ws_.sizeOf(wsName);
        if (sz == 0) {
            LOGE_GR("[%s] Workspace tensor '%s' not found (for input '%s')", node.name.c_str(), wsName.c_str(), t.name.c_str());
            return false;
        }
        if (strictZeroCopy && sz != t.bytes()) {
            LOGE_GR("[%s] Zero-copy violation: ws '%s' size=%zu, model input '%s' needs %zu",
                    node.name.c_str(), wsName.c_str(), sz, t.name.c_str(), t.bytes());
            return false;
        }
    }
    for (auto& t : node.session->outputs()) {
        auto it = node.outputBinding.find(t.name);
        if (it == node.outputBinding.end()) {
            LOGE_GR("[%s] Missing binding for output '%s'", node.name.c_str(), t.name.c_str());
            return false;
        }
        const std::string& wsName = it->second;
        auto sz = ws_.sizeOf(wsName);
        if (sz == 0) {
            LOGE_GR("[%s] Workspace tensor '%s' not found (for output '%s')", node.name.c_str(), wsName.c_str(), t.name.c_str());
            return false;
        }
        if (strictZeroCopy && sz != t.bytes()) {
            LOGE_GR("[%s] Zero-copy violation: ws '%s' size=%zu, model output '%s' needs %zu",
                    node.name.c_str(), wsName.c_str(), sz, t.name.c_str(), t.bytes());
            return false;
        }
    }
    nodes_.push_back(std::move(node));
    return true;
}

std::vector<GraphRunner::ExecInfo> GraphRunner::runAll(bool reset_session) {
    std::vector<ExecInfo> out;
    out.reserve(nodes_.size());
    for (auto& n : nodes_) {
        // Build pointer maps
        std::unordered_map<std::string, const void*> inPtrs;
        std::unordered_map<std::string, void*> outPtrs;

        // check if node session needs rebuilding
        if (!n.session.get()->getSnpe()) {
            n.session.get()->reCreate(nullptr);
        }

        for (auto& t : n.session->inputs()) {
            const auto& wsName = n.inputBinding.at(t.name);
            inPtrs[t.name] = ws_.data(wsName);
        }
        for (auto& t : n.session->outputs()) {
            const auto& wsName = n.outputBinding.at(t.name);
            outPtrs[t.name] = ws_.data(wsName);
        }

        int64_t ms = 0;
        bool ok = n.session->execute(inPtrs, outPtrs, &ms);
        ExecInfo e;
        e.name = n.name;
        e.runtime = n.session->selectedRuntimeName();
        e.ms = ms;
        e.ok = ok;
        LOGI_GR("[%s] runtime=%s  time=%lld ms  status=%s",
                e.name.c_str(), e.runtime.c_str(), (long long)e.ms, ok ? "OK" : "FAIL");

        // Log first 8 values of each output tensor
        if (ok) {
            for (const auto& t : n.session->outputs()) {
                const auto& wsName = n.outputBinding.at(t.name);
                void* ptr = ws_.data(wsName);
                size_t nbytes = ws_.sizeOf(wsName);
                size_t nfloat = nbytes / sizeof(float);

                const float* f = static_cast<const float*>(ptr);
                std::string vals;
                size_t count = std::min<size_t>(12, nfloat);
                for (size_t i = 0; i < count; ++i) {
                    vals += std::to_string(f[i]);
                    if (i + 1 < count) vals += ", ";
                }
                LOGI_GR("   Output '%s' (workspace='%s', %zu floats): [%s%s]",
                        t.name.c_str(), wsName.c_str(), nfloat,
                        vals.c_str(), (nfloat > count ? ", ..." : ""));
            }
        }

        out.push_back(e);

        if (reset_session) {
            LOGI_GR("[Execution] Resetting session for node %s", n.name.c_str());
            n.session.get()->reset();
//            usleep(20*1000);
        }
    }
    return out;
}
