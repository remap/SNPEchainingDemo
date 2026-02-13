//
// Created by Chiheb Boussema.
//

// MMapAsset.hpp  (patched)
#pragma once
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string>

#define LOGE_MMAP(...) __android_log_print(ANDROID_LOG_ERROR, "MMAP_DLC", __VA_ARGS__)
#define LOGI_MMAP(...) __android_log_print(ANDROID_LOG_INFO,  "MMAP_DLC", __VA_ARGS__)

struct MMapAsset {
    void*  ptr   = nullptr;   // usable bytes of the asset
    size_t size  = 0;

private:
    int    fd_      = -1;
    void*  mapBase_ = nullptr;
    size_t mapLen_  = 0;

public:
    ~MMapAsset() { close(); }

    void close() {
        if (mapBase_) { munmap(mapBase_, mapLen_); mapBase_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        ptr = nullptr; size = 0; mapLen_ = 0;
    }

    bool openUncompressed(AAssetManager* mgr, const char* name, std::string* err = nullptr) {
        close();
        if (!mgr) {
            if (err) *err = "AAssetManager is null";
            LOGE_MMAP("AAssetManager is null");
            return false;
        }

        // Open the asset entry
        AAsset* a = AAssetManager_open(mgr, name, AASSET_MODE_UNKNOWN);
        if (!a) {
            if (err) *err = std::string("AAssetManager_open failed: ") + name;
            LOGE_MMAP("AAssetManager_open failed: %s", name);
            return false;
        }

        // Try to get an FD (works only if the asset is STORED, not DEFLATED)
        off_t start = 0, len = 0;
        int fd = AAsset_openFileDescriptor(a, &start, &len);
        AAsset_close(a); // you must close the AAsset; fd (if valid) lives on

        if (fd < 0) {
            if (err) *err = std::string("openFileDescriptor failed (asset likely compressed): ") + name;
            LOGE_MMAP("openFileDescriptor failed (asset likely compressed): %s", name);
            return false;
        }

        // Align the mapping to page boundaries
        const long page = sysconf(_SC_PAGE_SIZE);
        const off_t pageOff = (start / page) * page;
        const off_t delta   = start - pageOff;
        const size_t mapLen = static_cast<size_t>(len + delta);

        void* base = mmap(nullptr, mapLen, PROT_READ, MAP_SHARED, fd, pageOff);
        if (base == MAP_FAILED) {
            if (err) *err = std::string("mmap failed for: ") + name;
            LOGE_MMAP("mmap failed for: %s", name);
            ::close(fd);
            return false;
        }

        // Success
        fd_      = fd;
        mapBase_ = base;
        mapLen_  = mapLen;
        ptr      = static_cast<char*>(base) + delta;
        size     = static_cast<size_t>(len);

        LOGI_MMAP("mmapped %s  size=%zu", name, size);
        return true;
    }
};