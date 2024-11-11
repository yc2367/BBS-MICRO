# Pytorch Implementation of Binary Pruning with Bi-directional Bit-level Sparsity (BBS) \[MICRO'24\]

## Usage
This repository contains the Pytorch re-implementation of the Bit Flip algorithm in [BitWave \[HPCA'24\]](https://ieeexplore.ieee.org/document/10476419), and the Pytorch implementation of the Binary Pruning algorithm (Rounded Averaging and Zero-point Shifting) in [BBS \[MICRO'24\]](https://arxiv.org/abs/2409.05227).

`bin_int_convert.py` contains functions for integer-binary conversion.

`bit_flip.py` contains functions for the Bit Flip algorithm. 
`binary_pruning.py` contains functions for the Binary Pruning algorithm. 

All functions of Bit Flip and Binary Pruning support convolution (conv) and fully-connected (fc) layers.


## Citation
If you use BBS in your research, please cite our paper:
```bibtex
@article{chen2024micro,
  title={BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration},
  author={Yuzong Chen and Jian Meng and Jae-sun Seo and Mohamed S. Abdelfattah},
  journal={57th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2024}
}
```
