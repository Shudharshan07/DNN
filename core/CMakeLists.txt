cmake_minimum_required(VERSION 3.20)

project(DNN VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(DNN
    src/main.cpp
    utils/matrix.cpp
    utils/Data/data.cpp
    utils/sequential.cpp
    utils/layer.cpp
    utils/Loss/mse.cpp
    utils/shm_writer.cpp
)

target_include_directories(DNN PRIVATE
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/utils/Loss
    ${CMAKE_SOURCE_DIR}/utils/Data
)

target_link_directories(DNN PRIVATE
    "C:/msys64/ucrt64/lib"
)

target_link_libraries(DNN PRIVATE
    openblas
)