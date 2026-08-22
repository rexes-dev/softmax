# softmax
Implementation of [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867) for learning purpose.

## Future directions

- Implement Softmax + TopK fusion
- Add robust tests (FP numerical analysis)
- Add benchmarks
- Study Pytorch's [Softmax.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu)

## References
- [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)
- [Milakov's implementation](https://github.com/NVIDIA/online-softmax/tree/master)
