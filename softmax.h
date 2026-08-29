#pragma once

enum class SoftmaxType { Naive, Safe, Online };

// CPU
// void naive_softmax_host(const float *x, float *y, int V, int batch_size);
// void safe_softmax_host(const float *x, float *y, int V, int batch_size);
// void online_softmax_host(const float *x, float *y, int V, int batch_size);
void softmax_host(const float *x, float *y, int V, int batch_size,
                  SoftmaxType softmax_type);

// CUDA
// void naive_softmax_dev(const float *x, float *y, int V, int batch_size);
// void safe_softmax_dev(const float *x, float *y, int V, int batch_size);
// void online_softmax_dev(const float *x, float *y, int V, int batch_size);
void softmax_dev(const float *x, float *y, int V, int batch_size,
                 SoftmaxType softmax_type);
