#pragma once

enum class SoftmaxType { Naive, Safe, Online };

// CPU
void softmax_host(const float *x, float *y, int V, int batch_size,
                  SoftmaxType softmax_type);

// CUDA
void softmax_dev(const float *x, float *y, int V, int batch_size,
                 SoftmaxType softmax_type);
