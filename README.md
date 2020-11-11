Hyperbolic Graph Convolutional Networks in PyTorch
==================================================

This repository is a fork of [https://github.com/HazyResearch/hgcn](https://github.com/HazyResearch/hgcn), and additions/modifications are made by [Eli](https://github.com/elipugh) and [Chris](https://www.linkedin.com/in/christopher-healy-0780a4178/).

We use their implementation of Hyperbolic Graph Convolutions [[1]](http://web.stanford.edu/~chami/files/hgcn.pdf) in PyTorch to examine how embedding on different manifolds can impact performance on link prediction and also node classification.

See examples in [this Colab](https://colab.research.google.com/drive/1BGm__vnNnfMB2k6nxtao8mHWrul4t43G?usp=sharing).

This is also a class project for [CS468](http://graphics.stanford.edu/courses/cs468-20-fall/) at Stanford.



## Most of the code was forked from the following repositories

 * [hgcn](https://github.com/HazyResearch/hgcn)
 * [pygcn](https://github.com/tkipf/pygcn/tree/master/pygcn)
 * [gae](https://github.com/tkipf/gae/tree/master/gae)
 * [hyperbolic-image-embeddings](https://github.com/KhrulkovV/hyperbolic-image-embeddings)
 * [pyGAT](https://github.com/Diego999/pyGAT)
 * [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings)
 * [geoopt](https://github.com/geoopt/geoopt)

## References

[1] [Chami, I., Ying, R., Ré, C. and Leskovec, J. Hyperbolic Graph Convolutional Neural Networks. NIPS 2019.](http://web.stanford.edu/~chami/files/hgcn.pdf)

[2] [Nickel, M. and Kiela, D. Poincaré embeddings for learning hierarchical representations. NIPS 2017.](https://arxiv.org/pdf/1705.08039.pdf)

[3] [Ganea, O., Bécigneul, G. and Hofmann, T. Hyperbolic neural networks. NIPS 2017.](https://arxiv.org/pdf/1805.09112.pdf)

[4] [Kipf, T.N. and Welling, M. Semi-supervised classification with graph convolutional networks. ICLR 2017.](https://arxiv.org/pdf/1609.02907.pdf)

[5] [Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. Graph attention networks. ICLR 2018.](https://arxiv.org/pdf/1710.10903.pdf)
