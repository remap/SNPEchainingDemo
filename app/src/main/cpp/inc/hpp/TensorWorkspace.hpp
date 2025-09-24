#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
#include <android/log.h>

#define  LOG_TAG_WS  "SNPE_WS"
#define  LOGI_WS(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG_WS,__VA_ARGS__)
#define  LOGE_WS(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG_WS,__VA_ARGS__)

/**
 * A simple arena that owns all tensor memory.
 * - Blocks can be aliased by name (zero-copy edges).
 * - You can mark last-uses to recycle memory early (optional extension).
 */
class TensorWorkspace {
public:
    struct Block {
        std::unique_ptr<uint8_t[]> bytes;
        size_t size = 0; // total bytes
    };

    // Allocate a fresh block with the given name and size (bytes).
    // If the name already exists and size differs => error (strict).
    void* allocate(const std::string& name, size_t bytes);

    // Make 'dstName' point to the same block as 'srcName' (strict zero-copy alias).
    // Fails if src doesn't exist.
    void alias(const std::string& dstName, const std::string& srcName);

    // Get pointer to named block (null if missing).
    void* data(const std::string& name) const;

    // Size in bytes for a named block (0 if missing).
    size_t sizeOf(const std::string& name) const;

    // Release a named block (only if it is an owner; aliases just forget mapping).
    void release(const std::string& name);

    // Debug dump
    void dump() const;

    bool has(const std::string& name) const;

private:
    struct Entry {
        // If owner==true, this entry owns 'block'; if alias, it references 'ownerKey'
        bool owner = false;
        std::string ownerKey;                 // if alias
        std::shared_ptr<Block> block;         // shared so multiple aliases can see lifetime
    };

    // Map tensor name -> entry
    std::unordered_map<std::string, Entry> m_;
};
