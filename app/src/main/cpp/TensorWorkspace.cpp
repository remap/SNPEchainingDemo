//
// Created by Chiheb Boussema on 16/9/25.
//
#include "hpp/TensorWorkspace.hpp"

void* TensorWorkspace::allocate(const std::string& name, size_t bytes) {
    auto it = m_.find(name);
    if (it != m_.end()) {
        // strict: must match size and be owner
        if (!it->second.owner) {
            LOGE_WS("allocate('%s'): name already aliasing another block", name.c_str());
            return nullptr;
        }
        if (it->second.block->size != bytes) {
            LOGE_WS("allocate('%s'): size mismatch (had %zu, want %zu)", name.c_str(),
                    it->second.block->size, bytes);
            return nullptr;
        }
        return it->second.block->bytes.get();
    }
    auto blk = std::make_shared<Block>();
    blk->bytes.reset(new uint8_t[bytes]);
    blk->size = bytes;
    Entry e; e.owner = true; e.block = blk;
    m_[name] = std::move(e);
    return blk->bytes.get();
}

void TensorWorkspace::alias(const std::string& dstName, const std::string& srcName) {
    auto it = m_.find(srcName);
    if (it == m_.end()) {
        LOGE_WS("alias('%s' <- '%s'): src not found", dstName.c_str(), srcName.c_str());
        return;
    }
    Entry e; e.owner = false; e.ownerKey = srcName; e.block = it->second.block;
    m_[dstName] = std::move(e);
}

void* TensorWorkspace::data(const std::string& name) const {
    auto it = m_.find(name);
    if (it == m_.end()) return nullptr;
    return it->second.block ? it->second.block->bytes.get() : nullptr;
}

size_t TensorWorkspace::sizeOf(const std::string& name) const {
    auto it = m_.find(name);
    if (it == m_.end() || !it->second.block) return 0;
    return it->second.block->size;
}

void TensorWorkspace::release(const std::string& name) {
    auto it = m_.find(name);
    if (it == m_.end()) return;
    // If owner, drop shared_ptr (aliases will see it go null when last ref ends)
    // If alias, just erase entry.
    m_.erase(it);
}

void TensorWorkspace::dump() const {
    LOGI_WS("Workspace dump:");
    for (auto& kv : m_) {
        const auto& k = kv.first;
        const auto& e = kv.second;
        LOGI_WS("  %s  owner=%d  size=%zu", k.c_str(), int(e.owner),
                e.block ? e.block->size : 0);
    }
}

bool TensorWorkspace::has(const std::string& name) const {
    return m_.find(name) != m_.end();
}
