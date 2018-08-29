## Deep Convolutional Generative Adversarial Network (DCGAN) implementation on MNIST data set using Keras
Here is an implementation of DCGAN as described in [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) on [MNIST](http://yann.lecun.com/exdb/mnist/) data set using [Keras](https://keras.io/) library with [Tensorflow](https://www.tensorflow.org/) backend.  

The jupyter notebook has the following dependencies:
- Python 3.5 as tested
- Numpy, tested with vers. 1.12.1
- Matplotlib, tested with vers. 2.0.0
- Pillow, tested with vers. 4.1.1
- Tensorflow, tested with vers. 1.0.1
- Keras, tested with vers. 2.0.2

The implementation is inspired by [this article](https://medium.com/@awjuliani/generative-adversarial-networks-explained-with-a-classic-spongebob-squarepants-episode-54deab2fce39).
Instead of padding original images to the size of 32x32, I used images of the original size 28x28. To do so, I set width and height of the first layer to 7x7, having then 2 fractionally-strided convolutions.  
And I used Keras instead of Tensorflow Slim library.
