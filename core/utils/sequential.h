#pragma once

#include <vector>
#include <memory>
#include "layer.h"
#include "Loss/mse.h"
#include "Data/data.h"
#include "shm_writer.h"

struct SequentialConfig
{
    int         batch_size       = 1;
    float       lr               = 0.01f;
    MSE         loss;
    // Optional shared memory snapshot settings.
    // Leave shm_name empty to disable snapshotting.
    std::string shm_name         = "";
    int         snapshot_interval = 10;
};

class Sequential 
{
    public:
        std::vector<Layer> layers;
        int batch_size;
        MSE loss;
        float lr;
        Data data;

    private:
        std::unique_ptr<SharedMemWriter> shm_writer_;
        int                              snapshot_interval_{ 10 };

    public:
        template<typename... Layers>
        Sequential(const SequentialConfig& cfg, Layers... ls)
            : layers{ ls... },
              batch_size(cfg.batch_size),
              lr(cfg.lr),
              snapshot_interval_(cfg.snapshot_interval)
        {
            for (auto& layer : layers)
                layer.lr = lr;

            if (!cfg.shm_name.empty())
            {
                const size_t shm_size = SharedMemWriter::ComputeSize(this->layers);
                shm_writer_ = std::make_unique<SharedMemWriter>(cfg.shm_name, shm_size);
            }
        }

        void Back(const Matrix& pred, const Matrix& actual);
        void Train(int epoch);
};