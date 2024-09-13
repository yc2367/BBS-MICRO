# Pytorch Implementation of Binary Pruning with Bi-directional Bit-level Sparsity (BBS) \[MICRO'24\]

## Usage
This repository contains the Pytorch re-implementation of the Bit Flip algorithm in [BitWave \[HPCA'24\]](https://ieeexplore.ieee.org/document/10476419), and the Pytorch implementation of the Binary Pruning algorithm (Rounded Averaging and Zero-point Shifting) in BBS \[MICRO'24\].

`bin_int_convert.py` contains functions for integer-binary conversion.

`bitflip_layer.py` contains functions for Bit Flip, Rounded Averaging, and Zero-point Shifting. All functions support convolution (conv) and fully-connected (fc) layers.


## Citation
If you use BBS in your research, please cite our paper:
```bibtex
@inproceedings{chen2024micro,
  title={BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration},
  author={Chen, Yuzong and Meng, Jian and Seo, Jae-sun and Mohamed, S. Abdelfattah},
  journal={57th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2024},
}
```
