# By: Eyal Zakkay, 2018
# Ported to Keras from the official Tensorflow implementation by Magenta

""" Sketch-RNN Implementation in Keras - Training"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys

from seq2seqVAE import Seq2seqModel, get_default_hparams
from keras.callbacks import ModelCheckpoint
from utils import load_dataset,batch_generator, KLWeightScheduler, LearningRateSchedulerPerBatch,\
    TensorBoardLR, DotDict, Logger


def get_callbacks_dict(seq2seq, model_params, experiment_path=''):
    """ create a dictionary of all used callbacks """

    # Callbacks dictionary
    callbacks_dict = {}

    # Checkpoints callback
    callbacks_dict['model_checkpoint'] = ModelCheckpoint(filepath=os.path.join(experiment_path, 'checkpoints',
                                                         'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss',
                                                         save_best_only=True, mode='min')

    # LR decay callback, modified to apply decay each batch as in original implementation
    callbacks_dict['lr_schedule'] = LearningRateSchedulerPerBatch(
        lambda step: ((model_params.learning_rate - model_params.min_learning_rate) * model_params.decay_rate ** step
                      + model_params.min_learning_rate))

    # KL loss weight decay callback, custom callback
    callbacks_dict['kl_weight_schedule'] = KLWeightScheduler(schedule=lambda step:
                                       (model_params.kl_weight - (model_params.kl_weight - model_params.kl_weight_start)
                                       * model_params.kl_decay_rate ** step), kl_weight=seq2seq.kl_weight, verbose=1)

    # Tensorboard callback
    callbacks_dict['tensorboard'] = TensorBoardLR(log_dir=os.path.join(experiment_path, 'tensorboard'),
                                    kl_weight=seq2seq.kl_weight, update_freq=model_params.batch_size*25)

    return callbacks_dict


def main(args, hparams):
    """ Main function for Keras Sketch-RNN"""
    # Logger:
    logsdir = os.path.join(args.experiment_dir, 'logs')
    os.makedirs(logsdir)
    os.makedirs(os.path.join(args.experiment_dir, 'checkpoints'))
    sys.stdout = Logger(logsdir)

    # Add support for dot access for auxiliary function use:
    hparams_dot = DotDict(hparams)

    # Load dataset:
    hparams_dot.data_set = args.data_set
    datasets = load_dataset(args.data_dir, hparams_dot)

    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]
    model_params = datasets[3]

    # Build and compile model:
    seq2seq = Seq2seqModel(model_params)
    seq2seq.compile()
    model = seq2seq.model

    # Create a data generator:
    train_generator = batch_generator(train_set, train=True)
    val_generator = batch_generator(valid_set, train=False)

    # Callbacks:
    model_callbacks = get_callbacks_dict(seq2seq=seq2seq, model_params=model_params, experiment_path=args.experiment_dir)

    # Load checkpoint:
    if args.checkpoint is not None:
        # Load weights:
        seq2seq.load_trained_weights(args.checkpoint)
        # Initial batch (affects LR and KL weight decay):
        num_batches = model_params.save_every if model_params.save_every is not None else train_set.num_batches
        count = args.initial_epoch*num_batches
        model_callbacks['lr_schedule'].count = count
        model_callbacks['kl_weight_schedule'].count = count

    # Write config file to json file
    with open(os.path.join(logsdir, 'model_config.json'), 'w') as f:
        json.dump(model_params, f, indent=True)

    # Train
    steps_per_epoch = model_params.save_every if model_params.save_every is not None else train_set.num_batches
    model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=model_params.epochs,
                        validation_data=val_generator, validation_steps=valid_set.num_batches,
                        callbacks=[cbk for cbk in model_callbacks.values()],
                        initial_epoch=args.initial_epoch)


if __name__ == '__main__':

    # Parse arguments and use defaults when needed
    hparams = get_default_hparams()

    parser = argparse.ArgumentParser(description='Main script for running sketch-rnn')

    parser.add_argument('--data_dir', type=str,
                        default='datasets',
                        help='Path for data files (directory only). (default: %(default)s)')
    # Todo: support use of multiple datasets using command line. currently only supported when editing default params
    parser.add_argument('--data_set', type=str,
                        default=hparams['data_set'],
                        help='Name of .npz file. (default: %(default)s)')

    parser.add_argument('--experiment_dir', type=str,
                        default='\\tmp\sketch_rnn\experiments',
                        help='Width of output image. (default: %(default)s)')

    parser.add_argument('--checkpoint', type=str,
                        default=None,
                        help='Option to provide path of checkpoint to load (resume training mode)')

    parser.add_argument('--initial_epoch', type=int,
                        default=0,
                        help='Epoch to start from when loading from checkpoint. (default: %(default)s)')

    args = parser.parse_args()

    # Get file name of .npz file
    # Todo: support use of multiple datasets using command line. currently only supported when editing default params
    if isinstance(args.data_set, list):  # more than one dataset
        sets = [os.path.splitext(s)[0] for s in args.data_set]
        experiment_path = os.path.join(args.experiment_dir, "{}\\exp".format('_'.join(sets)))
        args.data_set = [s+'.npz' for s in sets]
    else:
        data_set = os.path.splitext(args.data_set)[0]
        experiment_path = os.path.join(args.experiment_dir, "{}\\exp".format(data_set))
        args.data_set = data_set+'.npz'

    # Create a unique experiment folder
    # Todo: make this generic for operating systems other than windows path syntax
    dir_counter = 0
    new_experiment_path = experiment_path
    while os.path.exists(new_experiment_path):
        new_experiment_path = experiment_path + '_' + str(dir_counter)
        dir_counter += 1

    args.experiment_dir = new_experiment_path
    main(args, hparams)
