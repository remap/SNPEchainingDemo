#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "hpp/TensorWorkspace.hpp"
#include "hpp/ModelSession.hpp"
#include "hpp/TensorTypes.hpp"

/**
 * GraphRunner orchestrates a sequence of ModelSessions with strict zero-copy edges.
 * You allocate tensors in the workspace and then map model inputs/outputs to those names.
 */
class GraphRunner {
public:
    struct Node {
        std::string name; // for logs
        std::unique_ptr<ModelSession> session;

        // For each model input name, which workspace tensor name?
        std::unordered_map<std::string, std::string> inputBinding;

        // For each model output name, which workspace tensor name?
        std::unordered_map<std::string, std::string> outputBinding;
    };

    explicit GraphRunner(TensorWorkspace& ws) : ws_(ws) {}

    // Strict: check shapes & sizes match allocated blocks.
    bool addNode(Node node, bool strictZeroCopy = true);

    // Execute nodes in order; returns per-node latency and runtime strings
    struct ExecInfo { std::string name; std::string runtime; int64_t ms = 0; bool ok = false; };
    std::vector<ExecInfo> runAll(bool reset_session = false);

    void clear() {nodes_.clear();}

    void clear_session(Node& node) {node.session.reset();}

    Node& last() {return nodes_.back();}
    Node& getNode(std::string name);
    std::vector<Node>& getNodes() {return nodes_;}

private:
    TensorWorkspace& ws_;
    std::vector<Node> nodes_;
};
