#pragma once

// CPU
void naive_softmax_host(const float *x, float *y, int V, int batch_size);
void safe_softmax_host(const float *x, float *y, int V, int batch_size);
void online_softmax_host(const float *x, float *y, int V, int batch_size);

// CUDA
void naive_softmax_dev(const float *x, float *y, int V, int batch_size);
void safe_softmax_dev(const float *x, float *y, int V, int batch_size);
void online_softmax_dev(const float *x, float *y, int V, int batch_size);
