#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <atomic>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "layer.h"

// ---------------------------------------------------------------------------
// Shared memory layout (all offsets from base pointer):
//
//  [ShmHeader]
//  For each layer:
//    [LayerHeader]
//    [Weight floats: rows * cols]
//    [Bias floats:   bias_size]
// ---------------------------------------------------------------------------

#pragma pack(push, 1)

struct ShmHeader
{
    std::atomic<uint64_t> version;  // written LAST after all data is flushed
    uint32_t              num_layers;
    uint32_t              epoch;
    uint32_t              step;
    float                 loss;
};

struct LayerHeader
{
    uint32_t rows;
    uint32_t cols;
    uint32_t bias_size;
};

#pragma pack(pop)

// ---------------------------------------------------------------------------

class SharedMemWriter
{
public:
    SharedMemWriter(const std::string& name, size_t size);
    ~SharedMemWriter();

    // Disallow copy
    SharedMemWriter(const SharedMemWriter&)            = delete;
    SharedMemWriter& operator=(const SharedMemWriter&) = delete;

    // Compute the exact number of bytes needed for this model snapshot
    static size_t ComputeSize(const std::vector<Layer>& layers);

    // Serialize model state into shared memory and bump the version counter
    void Write(const std::vector<Layer>& layers,
               uint32_t                  epoch,
               uint32_t                  step,
               float                     loss);

private:
#ifdef _WIN32
    HANDLE handle_{ nullptr };
#endif
    void*  memory_{ nullptr };
    size_t size_{ 0 };
};
