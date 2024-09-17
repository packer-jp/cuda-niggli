cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(CudaNiggli LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

option(USE_PYBIND "Use pybind11 to create Python bindings" ON)

add_library(niggli STATIC src/niggli/niggli.cu)
set_target_properties(niggli PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
target_include_directories(niggli PUBLIC include)

add_executable(test_niggli src/niggli/main.cpp)
target_link_libraries(test_niggli PRIVATE niggli)
set_target_properties(test_niggli PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
message(STATUS "CMAKE_INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")

if(USE_PYBIND)
    find_package(Torch REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    add_library(torch_niggli MODULE src/torch_niggli/torch_niggli.cu)
    set_target_properties(torch_niggli PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
    target_include_directories(torch_niggli PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
    )
    target_link_libraries(torch_niggli PRIVATE niggli "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")

    install(TARGETS torch_niggli LIBRARY DESTINATION torch_niggli)
endif()
