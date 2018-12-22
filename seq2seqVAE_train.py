from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
# import time
import numpy as np
# import requests
# import six
# from six.moves import cStringIO as StringIO
# import copy

from seq2seqVAE import Seq2seqModel, get_default_hparams
from keras.callbacks import ModelCheckpoint
from utils import load_dataset,batch_generator, KLWeightScheduler, LearningRateSchedulerPerBatch,\
    TensorBoardLR, DotDict, Logger

from keras.models import Model


# def load_dataset(data_dir, model_params):
#     """Loads the .npz file, and splits the set into train/valid/test."""
#
#     #   normalizes the x and y columns using the training set.
#     # applies same scaling factor to valid and test set.
#
#     if isinstance(model_params.data_set, list):
#         datasets = model_params.data_set
#     else:
#         datasets = [model_params.data_set]
#
#     train_strokes = None
#     valid_strokes = None
#     test_strokes = None
#
#     for dataset in datasets:
#         data_filepath = os.path.join(data_dir, dataset)
#         if data_dir.startswith('http://') or data_dir.startswith('https://'):
#             print('Downloading %s', data_filepath)
#             response = requests.get(data_filepath)
#             data = np.load(StringIO(response.content))
#         else:
#             if six.PY3:
#                 data = np.load(data_filepath, encoding='latin1')
#             else:
#                 data = np.load(data_filepath)
#         print('Loaded {}/{}/{} from {}'.format(
#             len(data['train']), len(data['valid']), len(data['test']),
#             dataset))
#
#         if train_strokes is None:
#             train_strokes = data['train']
#             valid_strokes = data['valid']
#             test_strokes = data['test']
#         else:
#             train_strokes = np.concatenate((train_strokes, data['train']))
#             valid_strokes = np.concatenate((valid_strokes, data['valid']))
#             test_strokes = np.concatenate((test_strokes, data['test']))
#
#     all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
#     num_points = 0
#     for stroke in all_strokes:
#         num_points += len(stroke)
#     avg_len = num_points / len(all_strokes)
#     print('Dataset combined: {} ({}/{}/{}), avg len {}'.format(
#         len(all_strokes), len(train_strokes), len(valid_strokes),
#         len(test_strokes), int(avg_len)))
#
#     # calculate the max strokes we need.
#     max_seq_len = utils.get_max_len(all_strokes)
#     # overwrite the hps with this calculation.
#     model_params.max_seq_len = max_seq_len
#
#     print('model_params.max_seq_len %i.', model_params.max_seq_len)
#
#     train_set = utils.DataLoader(
#         train_strokes,
#         model_params.batch_size,
#         max_seq_length=model_params.max_seq_len,
#         random_scale_factor=model_params.random_scale_factor,
#         augment_stroke_prob=model_params.augment_stroke_prob)
#
#     normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
#     train_set.normalize(normalizing_scale_factor)
#
#     valid_set = utils.DataLoader(
#         valid_strokes,
#         model_params.batch_size,
#         max_seq_length=model_params.max_seq_len,
#         random_scale_factor=0.0,
#         augment_stroke_prob=0.0)
#     valid_set.normalize(normalizing_scale_factor)
#
#     test_set = utils.DataLoader(
#         test_strokes,
#         model_params.batch_size,
#         max_seq_length=model_params.max_seq_len,
#         random_scale_factor=0.0,
#         augment_stroke_prob=0.0)
#     test_set.normalize(normalizing_scale_factor)
#
#     print('normalizing_scale_factor %4.4f.', normalizing_scale_factor)
#
#     result = [train_set, valid_set, test_set, model_params]
#     return result


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
    # logger
    logsdir = os.path.join(args.experiment_dir, 'logs')
    os.makedirs(logsdir)
    os.makedirs(os.path.join(args.experiment_dir, 'checkpoints'))
    sys.stdout = Logger(logsdir)

    # hparams = get_default_hparams()

    # Add support fot dot access for auxiliary function use
    hparams_dot = DotDict(hparams)

    # Load dataset
    hparams_dot.data_set = args.data_set
    datasets = load_dataset(args.data_dir, hparams_dot)

    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]
    model_params = datasets[3]

    # Build and compile model
    seq2seq = Seq2seqModel(model_params)
    seq2seq.compile()
    model = seq2seq.model

    # Create a data generator
    train_generator = batch_generator(train_set, train=True)
    val_generator = batch_generator(valid_set, train=False)

    # Callbacks
    model_callbacks = get_callbacks_dict(seq2seq=seq2seq, model_params=model_params, experiment_path=args.experiment_dir)

    # Load checkpoint
    if args.checkpoint is not None:
        # load weights
        seq2seq.load_trained_weights(args.checkpoint)
        # initial batch (affects LR and KL weight decay)
        count = args.initial_epoch*train_set.num_batches
        model_callbacks['lr_schedule'].count = count
        model_callbacks['kl_weight_schedule'].count = count

    # Write config file to json file
    with open(os.path.join(logsdir, 'model_config.json'), 'w') as f:
        json.dump(model_params, f, indent=True)

    # Train
    model.fit_generator(generator=train_generator, steps_per_epoch=train_set.num_batches, epochs=model_params.epochs,
                        validation_data=val_generator, validation_steps=valid_set.num_batches,
                        callbacks=[cbk for cbk in model_callbacks.values()],
                        initial_epoch=args.initial_epoch)
    # Debug:
    # model.fit_generator(generator=train_generator, steps_per_epoch=model_params.save_every, epochs=model_params.epochs,
    #                     validation_data=val_generator, validation_steps=valid_set.num_batches, callbacks=model_callbacks)


if __name__ == '__main__':
    # Todo: make experiment path configurable
    hparams = get_default_hparams()

    parser = argparse.ArgumentParser(description='Main script for running sketch-rnn')

    parser.add_argument('--data_dir', type=str,
                        default='datasets',
                        help='Path for data files (directory only). (default: %(default)s)')

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

    # get file name of .npz file
    data_set = os.path.splitext(args.data_set)[0]
    experiment_path = os.path.join(args.experiment_dir, "{}\\exp".format(data_set))
    args.data_set = data_set+'.npz'

    # Create a unique experiment folder
    dir_counter = 0
    new_experiment_path = experiment_path
    while os.path.exists(new_experiment_path):
        new_experiment_path = experiment_path + '_' + str(dir_counter)
        dir_counter += 1

    args.experiment_dir = new_experiment_path
    main(args, hparams)
