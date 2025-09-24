#pragma once
#include <string>
#include <unordered_map>
#include <vector>

struct ModelCfg {
    std::string name;
    std::string asset;
//    std::string baseDir;
    char runtime = 'D'; // 'D'|'G'|'C' or 0 if absent
    std::unordered_map<std::string, std::string> inputs;
    std::unordered_map<std::string, std::string> outputs;
};

struct PipelineCfg {
    std::vector<ModelCfg> models;
    std::string baseDir;
};

// Returns true on success; fills 'cfg'. On failure, returns false and sets *emsg.
bool ParseConfig(const std::string& json, PipelineCfg& cfg, std::string* emsg);
