#include <iostream>
#include "sequential.h"

int main()
{
    SequentialConfig cfg;
    cfg.lr                = 0.001f;
    cfg.shm_name          = "DNN_SHM";  // must match shmName in Go reader
    cfg.snapshot_interval = 10;         // write to shm every 10 steps

    Sequential a = Sequential(cfg, Layer(1, 4), Layer(4, 2), Layer(2, 1));

    a.data = Data();
    a.Train(500);

    std::cout << "Training done. Press Enter to exit...\n";
    std::cin.get();

    return 0;
}
