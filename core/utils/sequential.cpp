#include "layer.h"
#include "sequential.h"
#include "Loss/mse.h"
#include "Data/data.h"
#include <vector>
#include <iostream>
#include <ranges>


void Sequential::Back(const Matrix& pred,const Matrix& actual)
{
    Matrix grad = this->loss.Derivative(actual, pred);

    for (Layer& layer : this->layers | std::views::reverse) {
        grad = layer.Backward(grad);
    }
}

void Sequential::Train(int epoch)
{
    int n = this->data.size();

    for (int e = 0; e < epoch; e++)
    {
        float loss_y = 0.0f;

        for (int i = 0; i < n; i++)
        {
            Matrix out = this->layers[0].Forward(this->data.X.row(i));

            for (int j = 1; j < (int)layers.size(); j++)
            {
                out = this->layers[j].Forward(out);
            }

            loss_y += this->loss.Loss(out, this->data.Y.row(i));

            this->Back(out, this->data.Y.row(i));
        }

        std::cout << "Epoch " << e + 1 << " loss: " << loss_y / n << "\n";
    }
}

