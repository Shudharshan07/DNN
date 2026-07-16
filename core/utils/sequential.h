#pragma once

#include <vector>
#include "layer.h"
#include "mse.h"

struct SequentialConfig
{
    int batch_size = 1;
    float lr = 0.01f;
    MSE loss;
};

class Sequential 
{
    public:
        std::vector<Layer> layers;
        int batch_size;
        MSE loss;
        float lr;
    
        template<typename... Layers>
        Sequential(const SequentialConfig& cfg, Layers... ls)
        : layers{ls...},
          batch_size(cfg.batch_size),
          lr(cfg.lr)
    {
        for (auto& layer : layers)
            layer.lr = lr;
    }
};