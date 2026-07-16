#include <iostream>
#include "matrix.h"
#include "sequential.h"
#include "mse.h"

int main()
{ 
    Sequential a = Sequential({32, 0.001f}, Layer(1, 2), Layer(2, 4));
    Layer b = a.layers[1];
    for(float i : b.Weight.data) {
        std::cout << i << " ";
    }

    return 0;
}