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
    int n    = this->data.size();
    int step = 0;

    for (int e = 0; e < epoch; e++)
    {
        float loss_y = 0.0f;

        

        for (int i = 0; i < n; i++, ++step)
        {
            Matrix out = this->layers[0].Forward(this->data.X.row(i));

            for (int j = 1; j < (int)layers.size(); j++)
            {
                out = this->layers[j].Forward(out);
            }

            const float sample_loss = this->loss.Loss(out, this->data.Y.row(i));
            loss_y += sample_loss;

            this->Back(out, this->data.Y.row(i));

            if (shm_writer_ && (step % snapshot_interval_ == 0))
            {
                shm_writer_->Write(
                    this->layers,
                    static_cast<uint32_t>(e),
                    static_cast<uint32_t>(step),
                    sample_loss
                );
            }
        }

        const float avg_loss = loss_y / n;
        std::cout << "Epoch " << e + 1 << " loss: " << avg_loss << "\n";
    }
}

