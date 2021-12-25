# Parameterization of Hypercomplex Multiplications (PHM)

This repository contains the TensorFlow implementation of PHM (Parameterization of Hypercomplex Multiplication) layers and PHM-Transformers in the paper [Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters](https://arxiv.org/pdf/2102.08597.pdf) at ICLR 2021.

<p align="center">
<img width="600"  src="https://user-images.githubusercontent.com/22279212/147380227-93e3f54a-5fef-42fb-aa57-27e92dcbe09d.png">
</p>


## Installation

One may install the following libraries before running our code:

* [tensorflow-gpu](https://www.tensorflow.org/) (1.14.0)
* [tensor2tensor](https://github.com/tensorflow/tensor2tensor) (1.14.0)


## Usage

The usage of this repository follows the original [tensor2tensor](https://github.com/tensorflow/tensor2tensor) repository (e.g., `t2t-datagen`, `t2t-trainer`, `t2t-avg-all`, followed by `t2t-decoder`). It helps to gain familiarity on tensor2tensor before attempting to run our code. Specifically, setting `--t2t_usr_dir=./Parameterization-of-Hypercomplex-Multiplications` will allow tensor2tensor to register PHM-Transformers.


### Training

For example, to evaluate PHM-Transformer (*n=4*) on the En-Vi machine translation task (`t2t-datagen --problem=translate_envi_iwslt32k`), one may set the following flags when training:

```
t2t-trainer \
--problem=translate_envi_iwslt32k \
--model=light_transformer \
--hparams_set=light_transformer_base_single_gpu \
--hparams="light_mode='random',hidden_size=512,factor=4" \
--train_steps=50000
```

where `light_transformer` with `light_mode='random'` is the alias of the PHM-Transformer in our implementation.

### Aggretating Checkpoints

After training, the latest 8 checkpoints are averaged:

```
t2t-avg-all --model_dir $TRAIN_DIR --output_dir $AVG_DIR --n 8
```

where `$TRAIN_DIR` and `$AVG_DIR` need to be specified by users.



### Testing

To decode the target sequence, one has to additionally set the `decode_hparams` as follows:

```
t2t-decoder \
--decode_hparams="beam_size=5,alpha=0.6"
```

Then `t2t-bleu` is invoked for calculating the BLEU.




## PHM Implementations

PHM is implemented with operations in [`make_random_mul`](https://github.com/astonzhang/Parameterization-of-Hypercomplex-Multiplications/blob/main/layers/qlib.py#L205) and [`random_ffn`](https://github.com/astonzhang/Parameterization-of-Hypercomplex-Multiplications/blob/main/layers/qlib.py#L252), which are mathematically equivalent to sum of Kronecker products.

Among works that use PHM, some have offered alternative PHM implementations:

* [Parameterized Hypercomplex Graph Neural Networks](https://github.com/bayer-science-for-a-better-life/phc-gnn)
* [COMPACTER: Efficient Low-Rank Hypercomplex Adapter Layers](https://github.com/rabeehk/compacter/tree/main/seq2seq/hypercomplex)
* [Convolutional Neural Networks by Hypercomplex Parameterization](https://github.com/eleGAN23/HyperNets)
* [demegire/Parameterization-of-Hypercomplex-Multiplications](https://github.com/demegire/Parameterization-of-Hypercomplex-Multiplications)


## Citation

If you find this repository helpful, please cite our paper:

```
@inproceedings{zhang2021beyond,
  title={Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with $1/n$ Parameters},
  author={Zhang, Aston and Tay, Yi and Zhang, Shuai and Chan, Alvin and Luu, Anh Tuan and Hui, ‪Siu Cheung and Fu, Jie},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
