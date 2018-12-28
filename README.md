# A Keras Implementation of Sketch-RNN
In this repo there's a Kares implementation for the Sketch-RNN algorithm, as described in the paper [A Neural Representation of Sketch Drawings](https://arxiv.org/pdf/1704.03477.pdf) by David Ha and Douglas Eck (Google AI).

The implementation is ported from the [official Tensorflow implementation](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn) that was released under project [Magenta](https://magenta.tensorflow.org/) by the authors.

## Overview
Sketch-RNN consists of a Sequence to Sequence Variational Autoencoder (Seq2SeqVAE), which is able to encode a series of pen strokes (a sketch) into a latent space. The latent representation can than be decoded back into a series of strokes. The model is trained to reconstruct the original stroke sequences while maintaining a normal distribution across latent space elements. Since encoding is performed stochastically, and so is the sampling mechanism in the decoder, the reconstructed sketches are always different.
This allows to let a trained model draw new and unique sketches that it has not seen before. Designing the model as a variational autoencoder also allows to perform latent space manipulations to get interesting interpolations between different sketches.
## Dependencies

## Usage

### Training

### Using a trained model to draw
