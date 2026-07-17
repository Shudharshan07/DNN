#pragma once

#include <vector>
#include "layer.h"
#include "Loss/mse.h"
#include "Data/data.h"

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
        Data data;
    
        template<typename... Layers>
        Sequential(const SequentialConfig& cfg, Layers... ls)
        : layers{ls...},
          batch_size(cfg.batch_size),
          lr(cfg.lr)
    {
        for (auto& layer : layers)
            layer.lr = lr;
    }

    void Back(const Matrix& pred,const Matrix& actual);
    void Train(int epoch);
};