//
// Created by Chiheb Boussema.
//

#pragma once

// POSIX / NDK
#include <fcntl.h>      // ::open, O_RDONLY, O_CLOEXEC
#include <sys/mman.h>   // ::mmap, ::munmap, PROT_*, MAP_*
#include <sys/stat.h>   // ::fstat, struct stat
#include <unistd.h>     // ::close
#include <cerrno>       // errno
#include <cstring>      // std::strerror
#include <string>       // std::string
#include <cstddef>      // std::size_t
#include <utility>      // std::move

// Simple RAII read-only file mmap
class MMapFile {
public:
    // Mapped region base pointer and size (bytes). nullptr/0 if not mapped.
    void*  ptr  = nullptr;
    size_t size = 0;

    MMapFile() = default;
    ~MMapFile() { close(); }

    // Non-copyable, but movable
    MMapFile(const MMapFile&) = delete;
    MMapFile& operator=(const MMapFile&) = delete;

    MMapFile(MMapFile&& other) noexcept { moveFrom(std::move(other)); }
    MMapFile& operator=(MMapFile&& other) noexcept {
        if (this != &other) { close(); moveFrom(std::move(other)); }
        return *this;
    }

    // Map a file by path, read-only. Returns true on success.
    // On failure, fills emsg (if non-null) with a short description.
    bool openPath(const char* path, std::string* emsg = nullptr) {
        close();

        int fd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            if (emsg) *emsg = std::string("open('") + path + "') failed: " + std::strerror(errno);
            return false;
        }

        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            if (emsg) *emsg = std::string("fstat('") + path + "') failed: " + std::strerror(errno);
            ::close(fd);
            return false;
        }
        if (!S_ISREG(st.st_mode)) {
            if (emsg) *emsg = std::string("not a regular file: '") + path + "'";
            ::close(fd);
            return false;
        }
        if (st.st_size == 0) {
            // zero-length is technically mappable, but useless for DLCs—reject for clarity
            if (emsg) *emsg = std::string("file size is 0: '") + path + "'";
            ::close(fd);
            return false;
        }

        void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size),
                         PROT_READ, MAP_SHARED, fd, /*offset*/0);
        if (p == MAP_FAILED) {
            if (emsg) *emsg = std::string("mmap('") + path + "') failed: " + std::strerror(errno);
            ::close(fd);
            return false;
        }

        // success; store and keep fd only long enough to map (not needed after)
        ptr  = p;
        size = static_cast<size_t>(st.st_size);
        ::close(fd);
        return true;
    }

    // Unmap if mapped.
    void close() {
        if (ptr && size) {
            ::munmap(ptr, size);
        }
        ptr = nullptr;
        size = 0;
    }

    bool isOpen() const noexcept { return ptr != nullptr && size > 0; }

private:
    void moveFrom(MMapFile&& other) noexcept {
        ptr  = other.ptr;
        size = other.size;
        other.ptr = nullptr;
        other.size = 0;
    }
};
