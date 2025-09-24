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

// In your config types (e.g., ParseConfig.hpp)
enum class InitKind { ZERO, RANDOM, FILE_PATH, ASSET_PATH, CONST_VALUE, UNKNOWN };

struct InitSpec {
    InitKind kind = InitKind::UNKNOWN;
    std::string path;      // for FILE_PATH or ASSET_PATH
    float mean = 0.f;      // for RANDOM
    float std  = 1.f;      // for RANDOM
    uint32_t seed = 0;     // for RANDOM (0 = choose from steady_clock)
    float value = 0.f;     // for CONST_VALUE
};


struct PipelineCfg {
    std::vector<ModelCfg> models;
    std::string baseDir;
    std::unordered_map<std::string, InitSpec> init; // wsTensorName -> InitSpec
};

// Returns true on success; fills 'cfg'. On failure, returns false and sets *emsg.
bool ParseConfig(const std::string& json, PipelineCfg& cfg, std::string* emsg);
