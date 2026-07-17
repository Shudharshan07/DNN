#include <iostream>
#include "sequential.h"


int main()
{ 
    Sequential a = Sequential({1, 0.001f}, Layer(1, 4), Layer(4, 2), Layer(2, 1));
    
    a.data = Data();

    a.Train(5);

    return 0;
}