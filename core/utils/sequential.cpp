#include "layer.h"
#include "sequential.h"
#include <vector>

class Sequential
{
    public:
        std::vector<Layer> layers;
        int batch_size;
        MSE loss;
        float lr;
        // Data needs to be implemented

        void Back(const Matrix& pred,const Matrix& actual)
        {
            Matrix grad = this->loss.Derivative(actual, pred);

            for (Layer& layer : this->layers | std::views::reverse) {
                grad = layer.Backward(grad);
            }
        }

        void Forward(int epoch)
        {
            for(int i = 0; i < epoch; i++)
            {

            }
        }

}