#include "shm_writer.h"

#include <cstring>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Platform-specific helpers
// ---------------------------------------------------------------------------

#ifdef _WIN32

static void* platform_create(const std::string& name, size_t size, HANDLE& out_handle)
{
    out_handle = CreateFileMappingA(
        INVALID_HANDLE_VALUE,           // backed by the system paging file
        nullptr,                        // default security
        PAGE_READWRITE,
        static_cast<DWORD>(size >> 32), // high-order DWORD of size
        static_cast<DWORD>(size & 0xFFFFFFFF),
        name.c_str()
    );

    if (out_handle == nullptr)
        throw std::runtime_error("CreateFileMapping failed: " + std::to_string(GetLastError()));

    void* mem = MapViewOfFile(out_handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (mem == nullptr)
    {
        CloseHandle(out_handle);
        out_handle = nullptr;
        throw std::runtime_error("MapViewOfFile failed: " + std::to_string(GetLastError()));
    }

    return mem;
}

static void platform_destroy(void* mem, HANDLE handle)
{
    if (mem)    UnmapViewOfFile(mem);
    if (handle) CloseHandle(handle);
}

#else
// ---------------------------------------------------------------------------
// Linux stub — fill in with shm_open / mmap when needed
// ---------------------------------------------------------------------------
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static void* platform_create(const std::string& name, size_t size, int& out_fd)
{
    // Ensure the name starts with '/'
    std::string shm_name = (name[0] == '/') ? name : ("/" + name);

    out_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
    if (out_fd == -1)
        throw std::runtime_error("shm_open failed");

    if (ftruncate(out_fd, static_cast<off_t>(size)) == -1)
        throw std::runtime_error("ftruncate failed");

    void* mem = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
    if (mem == MAP_FAILED)
        throw std::runtime_error("mmap failed");

    return mem;
}

static void platform_destroy(void* mem, int fd, size_t size)
{
    if (mem) munmap(mem, size);
    if (fd != -1) close(fd);
}
#endif

// ---------------------------------------------------------------------------
// SharedMemWriter
// ---------------------------------------------------------------------------

SharedMemWriter::SharedMemWriter(const std::string& name, size_t size)
    : size_(size)
{
#ifdef _WIN32
    memory_ = platform_create(name, size, handle_);
#else
    memory_ = platform_create(name, size, handle_);
#endif

    // Zero-initialise so the reader never sees garbage before the first Write()
    std::memset(memory_, 0, size_);
}

SharedMemWriter::~SharedMemWriter()
{
#ifdef _WIN32
    platform_destroy(memory_, handle_);
#else
    platform_destroy(memory_, handle_, size_);
#endif
    memory_ = nullptr;
}

// ---------------------------------------------------------------------------

size_t SharedMemWriter::ComputeSize(const std::vector<Layer>& layers)
{
    size_t total = sizeof(ShmHeader);

    for (const Layer& layer : layers)
    {
        total += sizeof(LayerHeader);
        total += static_cast<size_t>(layer.Weight.rows)
               * static_cast<size_t>(layer.Weight.cols)
               * sizeof(float);
        total += static_cast<size_t>(layer.Bias.cols) * sizeof(float);
    }

    return total;
}

// ---------------------------------------------------------------------------

void SharedMemWriter::Write(const std::vector<Layer>& layers,
                            uint32_t                   epoch,
                            uint32_t                   step,
                            float                      loss)
{
    auto* base = static_cast<std::byte*>(memory_);

    // ------------------------------------------------------------------
    // 1. Write ShmHeader fields (everything except version)
    // ------------------------------------------------------------------
    auto* header = reinterpret_cast<ShmHeader*>(base);

    header->num_layers = static_cast<uint32_t>(layers.size());
    header->epoch      = epoch;
    header->step       = step;
    header->loss       = loss;

    // ------------------------------------------------------------------
    // 2. Write each layer's header + raw float data
    // ------------------------------------------------------------------
    std::byte* cursor = base + sizeof(ShmHeader);

    for (const Layer& layer : layers)
    {
        const uint32_t w_rows = static_cast<uint32_t>(layer.Weight.rows);
        const uint32_t w_cols = static_cast<uint32_t>(layer.Weight.cols);
        const uint32_t b_size = static_cast<uint32_t>(layer.Bias.cols);

        // LayerHeader
        LayerHeader lh{ w_rows, w_cols, b_size };
        std::memcpy(cursor, &lh, sizeof(LayerHeader));
        cursor += sizeof(LayerHeader);

        // Weight data  (rows * cols floats)
        const size_t w_bytes = static_cast<size_t>(w_rows) * w_cols * sizeof(float);
        std::memcpy(cursor, layer.Weight.data.data(), w_bytes);
        cursor += w_bytes;

        // Bias data  (bias_size floats)
        const size_t b_bytes = static_cast<size_t>(b_size) * sizeof(float);
        std::memcpy(cursor, layer.Bias.data.data(), b_bytes);
        cursor += b_bytes;
    }

    // ------------------------------------------------------------------
    // 3. Release fence then bump version — reader uses acquire load
    // ------------------------------------------------------------------
    std::atomic_thread_fence(std::memory_order_release);
    header->version.fetch_add(1, std::memory_order_release);
}
