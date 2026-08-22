#pragma once

#include <cstddef>

// CPU
void naive_softmax_host(const float *x, float *y, std::size_t V);
void safe_softmax_host(const float *x, float *y, std::size_t V);
void online_softmax_host(const float *x, float *y, std::size_t V);

// CUDA
void naive_softmax_dev(const float *x, float *y, std::size_t V);
void safe_softmax_dev(const float *x, float *y, std::size_t V);
void online_softmax_dev(const float *x, float *y, std::size_t V);
