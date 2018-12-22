from keras.models import Model
from keras.layers import Input, Multiply, GaussianNoise
from keras.layers.merge import Concatenate
from keras.layers.core import RepeatVector
from keras.layers import Dense, LSTM, CuDNNLSTM, Bidirectional, Lambda
from keras.activations import softmax, exponential, tanh
from keras import backend as K
from keras.initializers import RandomNormal
from keras.utils import plot_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
import numpy as np
import random


def get_default_hparams():
    # Todo: add comments for all params
    params_dict = {
        # experiment params:
        # 'is_training': True,
        'data_set': 'cat',
        'epochs': 50,
        'save_every': 100,
        'batch_size': 100,
        'accelerate_LSTM': True,  # flag for using CuDNNLSTM layer, gpu+ tf backend only
        # loss params:
        'optimizer': 'adam',  # adam or sgd
        'learning_rate': 0.001, #,0.001
        'decay_rate': 0.9999,#0.9999,  # Learning rate decay per minibatch.
        'min_learning_rate': .00001,
        'kl_tolerance': 0.2,
        'kl_weight': 0.5,
        'kl_weight_start': 0.1,#0.01,
        'kl_decay_rate': 0.99995,  # KL annealing decay rate per minibatch.
        'grad_clip': 1.0,
        # architecture params:
        'z_size': 32,#128
        'enc_rnn_size': 256,
        'dec_rnn_size': 512,
        'max_seq_len': 250,
        'use_recurrent_dropout': True,
        'recurrent_dropout_prob': 0.9,
        'num_mixture': 20,
        # data params:
        'random_scale_factor': 0.15,  # Random scaling data augmention proportion.
        'augment_stroke_prob': 0.10  # Point dropping augmentation proportion.
    }

    return params_dict


class Seq2seqModel(object):

    def __init__(self, hps):
        # Hyper parameters
        self.hps = hps
        # Model
        self.model = self.build_model()
        # Optimizer
        if self.hps['optimizer'] == 'adam':
            self.optimizer = Adam(lr=self.hps['learning_rate'], clipvalue=self.hps['grad_clip'])
        elif self.hps['optimizer'] == 'sgd':
            self.optimizer = SGD(lr=self.hps['learning_rate'], momentum=0.9, clipvalue=self.hps['grad_clip'])
        else:
            raise ValueError('Unsupported Optimizer!')
        # Loss Function
        self.loss_func = self.model_loss()
        # Sample models, to be used when encoding\decoding specific strokes
        self.sample_models = {}

    def build_model(self):

        # arrange inputs
        self.encoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='encoder_input')
        decoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='decoder_input')

        # todo: use cudnnLSTM?? no support for recurrent dropout....
        recurrent_dropout = 1.0-self.hps['recurrent_dropout_prob'] if self.hps['use_recurrent_dropout'] else 0

        if self.hps['accelerate_LSTM']:
            lstm_layer_encoder = CuDNNLSTM(units=self.hps['enc_rnn_size'])
            lstm_layer_decoder = CuDNNLSTM(units=self.hps['dec_rnn_size'], return_sequences=True, return_state=True)
            self.hps['use_recurrent_dropout'] = False
            print('Using CuDNNLSTM - No Recurrent Dropout!')
        else:
            lstm_layer_encoder = LSTM(units=self.hps['enc_rnn_size'], recurrent_dropout=recurrent_dropout)
            lstm_layer_decoder = LSTM(units=self.hps['dec_rnn_size'], recurrent_dropout=recurrent_dropout,
                                      return_sequences=True, return_state=True)

        # lstm_layer = CuDNNLSTM if self.hps['accelerate_LSTM'] else LSTM
        # encoder = Bidirectional(lstm_layer(units=self.hps['enc_rnn_size'], recurrent_dropout=recurrent_dropout),
        #                         merge_mode='concat')(self.encoder_input)
        encoder = Bidirectional(lstm_layer_encoder, merge_mode='concat')(self.encoder_input)

        # latent vector - [batch_size]X[z_size]:
        self.batch_z = self.latent_z(encoder)

        # create decoder
        # self.decoder = lstm_layer(units=self.hps['dec_rnn_size'], recurrent_dropout=recurrent_dropout, return_sequences=True,
        #                     return_state=True)
        self.decoder = lstm_layer_decoder

        # initial state for decoder:
        self.initial_state = Dense(units=2*self.decoder.units, activation='tanh', name='dec_initial_state',
                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))
        initial_state = self.initial_state(self.batch_z)

        # Split to hidden state and cell state:
        init_h, init_c = (initial_state[:, :self.decoder.units], initial_state[:, self.decoder.units:])

        # concatenate z vector to expected outputs and this feed input to decoder:
        tile_z = RepeatVector(self.hps['max_seq_len'])(self.batch_z)
        decoder_full_input = Concatenate()([decoder_input, tile_z])

        [decoder_output, final_state1, final_state_2] = self.decoder(decoder_full_input, initial_state=[init_h, init_c])
        self.final_state = [final_state1, final_state_2]

        # number of outputs for prediction sampling:
        n_out = (3 + self.hps['num_mixture'] * 6)

        # get outputs - GM coefficients to predict strokes
        self.output = Dense(n_out, name='output')
        output = self.output(decoder_output)

        model_o = Model([self.encoder_input, decoder_input], output)
        model_o.summary()
        return model_o

    def latent_z(self, encoder):
        # return a latent vector z of size [batch_size]X[z_size]

        def transform2layer(z_params):
            # all operations must be layers. this is an aid function to feed into a Lambda layer
            mu, sigma = z_params
            sigma_exp = K.exp(sigma / 2.0)
            colored_noise = mu + sigma_exp*K.random_normal(shape=K.shape(sigma_exp), mean=0.0, stddev=1.0)
            return colored_noise

        # todo: different initialization?
        self.mu = Dense(units=self.hps['z_size'], kernel_initializer=RandomNormal(stddev=0.001))(encoder)
        self.sigma = Dense(self.hps['z_size'], kernel_initializer=RandomNormal(stddev=0.001))(encoder)

        return Lambda(transform2layer)([self.mu, self.sigma])

    def calculate_kl_loss(self, *args, **kwargs):
        # KL loss term
        kl_cost = -0.5*K.mean(1+self.sigma-K.square(self.mu)-K.exp(self.sigma))

        # kl weight (to be used by total loss and by scheduler
        self.kl_weight = K.variable(self.hps['kl_weight_start'], name='kl_weight')

        return K.maximum(kl_cost, self.hps['kl_tolerance'])

    def calculate_md_loss(self, y_true, y_pred):
        ''' calculate mixture density loss '''
        out = self.get_mixture_coef(y_pred)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

        # target for output - the same as input to encoder
        [x1_data, x2_data] = [y_true[:, :, 0], y_true[:, :, 1]]
        pen_data = y_true[:, :, 2:5]

        # Loss
        # lossfunc = self.get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,
        #                              o_pen_logits, x1_data, x2_data, pen_data)
        self.result0 = self.tf_2d_normal(x1_data, x2_data, o_mu1, o_mu2, o_sigma1, o_sigma2,
                                    o_corr)
        epsilon = 1e-6

        # result1 is the loss wrt pen offset (L_s in equation 9 of
        # https://arxiv.org/pdf/1704.03477.pdf)
        result1 = self.result0 * o_pi
        result1 = K.sum(result1, 2, keepdims=True)
        result1 = -K.log(result1 + epsilon)  # avoid log(0)

        fs = 1.0 - pen_data[:, :, 2]  # use training data for this
        fs = K.expand_dims(fs)
        # Zero out loss terms beyond N_s, the last actual stroke
        result1 = result1 * fs

        # result2: loss wrt pen state, (L_p in equation 9)
        result2 = categorical_crossentropy(pen_data, o_pen_logits)
        result2 = K.expand_dims(result2)

        # if not self.hps['is_training']:
        result2 = K.switch(K.learning_phase(), result2, result2 * fs)
        # if not K.learning_phase():  # eval mode, mask eos columns
        #     result2 = result2 * fs

        result = result1 + result2

        r_cost = K.mean(result)
        return r_cost

    def model_loss(self):
        # wrapper function which calculates auxiliary values for the complete loss function. Returns a function which
        # calculates the complete loss given only the input and target output
        kl_loss = self.calculate_kl_loss()
        md_loss_func = self.calculate_md_loss
        # kl_weight = self.hps['kl_weight']
        kl_weight = self.kl_weight

        def seq2seq_loss(y_true, y_pred):
            md_loss = md_loss_func(y_true, y_pred)
            model_loss = kl_weight*kl_loss + md_loss
            return model_loss

        return seq2seq_loss

    def get_mixture_coef(self, out_tensor):
        """Returns the tf slices containing mdn dist params."""
        # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
        z = out_tensor
        z_pen_logits = z[:, :, 0:3]  # pen states
        # process output z's into MDN paramters
        M = self.hps['num_mixture']
        dist_params = [z[:, :, (3 + M * (n - 1)):(3 + M * n)] for n in range(1, 7)]
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = dist_params

        # softmax all the pi's and pen states:
        z_pi = softmax(z_pi)
        z_pen = softmax(z_pen_logits)

        # exponentiate the sigmas and also make corr between -1 and 1.
        z_sigma1 = exponential(z_sigma1)
        z_sigma2 = exponential(z_sigma2)
        z_corr = tanh(z_corr)

        r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
        return r

    def tf_2d_normal(self, x1, x2, mu1, mu2, s1, s2, rho):
        """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
        M = mu1.shape[2]
        norm1 = K.tile(K.expand_dims(x1), [1, 1, M]) - mu1
        norm2 = K.tile(K.expand_dims(x2), [1, 1, M]) - mu2
        s1s2 = s1 * s2
        # eq 25
        z = K.square(norm1 / s1) + K.square(norm2 / s2) - 2.0 * (rho * norm1 * norm2) / s1s2
        neg_rho = 1.0 - K.square(rho)
        result = K.exp((-z) / (2 * neg_rho))
        denom = 2 * np.pi * s1s2 * K.sqrt(neg_rho)
        result = result / denom
        return result

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func,
                           metrics=[self.calculate_md_loss, self.calculate_kl_loss])
        print('Model Compiled!')

    def load_trained_weights(self, weights):
        # load weights of a pre-trained model. 'weights' is path to h5 model\weights file
        self.model.load_weights(weights)
        print('Weights from {} loaded successfully'.format(weights))

    def make_sampling_models(self):
        # creates models for various input-output combinations to be used when sampling and encoding\decoding specific
        #   strokes
        models = {}

        # z to initial state model: Build a model that gets batch_z and outputs initial_state
        batch_z = Input(shape=(self.hps['z_size'],))
        initial_state = self.initial_state(batch_z)
        # initial_state = model.get_layer('dec_initial_state')(batch_z)
        models['z_to_init_model'] = Model(inputs=batch_z,
                                         outputs=initial_state)

        # sample_output_model:
        # Build a model that gets decoder input, initial state and batch_z. outputs final state and mixture parameters.

        # Inputs:
        decoder_input = Input(shape=(1, 5))
        initial_h_input = Input(shape=(self.decoder.units,))
        initial_c_input = Input(shape=(self.decoder.units,))

        # Build parts of model needed to reach desired output. apply them on the new inputs:
        # concatenate z vector to expected outputs and this feed input to decoder:
        tile_z = RepeatVector(1)(batch_z)
        decoder_full_input = Concatenate()([decoder_input, tile_z])
        # Apply decoder on new inputs
        [decoder_output, final_state_1, final_state_2] = self.decoder(decoder_full_input,
                                                                               initial_state=[initial_h_input,
                                                                                              initial_c_input])
        final_state = [final_state_1, final_state_2]
        # Apply original output layer
        model_output = self.output(decoder_output)
        # Get mixture coef' based on output layer
        mixture_params = Lambda(self.get_mixture_coef)(model_output)
        models['sample_output_model'] = Model(inputs=[decoder_input, initial_h_input, initial_c_input, batch_z],
                                    outputs=final_state + mixture_params)

        # emcoder model: stroke input to z
        models['encoder_model'] = Model(inputs=self.encoder_input, outputs=self.batch_z)

        self.sample_models = models
        print('Created Sub Models!')


def sample(seq2seq_model, seq_len=250, temperature=1.0, greedy_mode=False, z=None):
  """Samples a sequence from a pre-trained model."""
  # model = seq2seq_model.model
  hps = seq2seq_model.hps
  def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf

  def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a pdf, optionally greedily."""
    if greedy:
      return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
      accumulate += pdf[i]
      if accumulate >= x:
        return i
    print('Error with sampling ensemble.')
    return -1

  def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
      return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

  prev_x = np.zeros((1, 1, 5), dtype=np.float32)
  prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
  if z is None:
    z = np.random.randn(1, hps['z_size'])  # not used if unconditional

  # Build a model that gets batch_z and outputs initial_state
  # batch_z = Input(shape=(hps['z_size'],))
  # initial_state = seq2seq_model.initial_state(batch_z)
  # # initial_state = model.get_layer('dec_initial_state')(batch_z)
  # intermediate_layer_model = Model(inputs=batch_z,
  #                                  outputs=initial_state)
  z_to_init_model = seq2seq_model.sample_models['z_to_init_model']
  prev_state = z_to_init_model.predict(z)
  prev_state = [prev_state[:, :seq2seq_model.decoder.units], prev_state[:, seq2seq_model.decoder.units:]]
  # prev_state = sess.run(model.initial_state, feed_dict={model.batch_z: z})
  # Build a model that gets decoder input, initial state and batch_z. outputs final state and mixture parameters.

  # # Inputs:
  # decoder_input = Input(shape=(1, 5))
  # initial_h_input = Input(shape=(seq2seq_model.decoder.units,))
  # initial_c_input = Input(shape=(seq2seq_model.decoder.units,))
  #
  # # Build parts of model needed to reach desired output. apply them on the new inputs:
  # # concatenate z vector to expected outputs and this feed input to decoder:
  # tile_z = RepeatVector(1)(batch_z)
  # decoder_full_input = Concatenate()([decoder_input, tile_z])
  # # Apply decoder on new inputs
  # [decoder_output, final_state_1, final_state_2] = seq2seq_model.decoder(decoder_full_input,
  #                                                                   initial_state=[initial_h_input, initial_c_input])
  # final_state = [final_state_1, final_state_2]
  # # Apply original output layer
  # model_output = seq2seq_model.output(decoder_output)
  # # Get mixture coef' based on output layer
  # mixture_params = Lambda(seq2seq_model.get_mixture_coef)(model_output)
  # sample_output_model = Model(inputs=[decoder_input, initial_h_input, initial_c_input, batch_z],
  #                             outputs=final_state + mixture_params)
  sample_output_model = seq2seq_model.sample_models['sample_output_model']

  strokes = np.zeros((seq_len, 5), dtype=np.float32)
  mixture_params = []

  for i in range(seq_len):

    # Arrange inputs
    feed = {
        'decoder input': prev_x,
        'initial_state': prev_state,
        'batch_z': z
    }
    model_outputs_list = sample_output_model.predict([feed['decoder input'],
                                                                  feed['initial_state'][0],
                                                                  feed['initial_state'][1],
                                                                  feed['batch_z']])
    next_state = model_outputs_list[:2]
    mixture_params_val = model_outputs_list[2:]
    # params = sess.run([
    #     model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
    #     model.pen, model.final_state
    # ], feed)

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, _] = mixture_params_val


    if i < 0:
      greedy = False
      temp = 1.0
    else:
      greedy = greedy_mode
      temp = temperature

    # TODO: dimensionality of params? maybe needed o_pi[0][0] or even just o_pi? debug to check this
    idx = get_pi_idx(random.random(), o_pi[0][0], temp, greedy)

    idx_eos = get_pi_idx(random.random(), o_pen[0][0], temp, greedy)
    eos = [0, 0, 0]
    eos[idx_eos] = 1

    next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][0][idx], o_mu2[0][0][idx],
                                          o_sigma1[0][0][idx], o_sigma2[0][0][idx],
                                          o_corr[0][0][idx], np.sqrt(temp), greedy)

    strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

    params = [
        o_pi[0][0], o_mu1[0][0], o_mu2[0][0], o_sigma1[0][0], o_sigma2[0][0], o_corr[0][0],
        o_pen[0][0]
    ]

    mixture_params.append(params)
    # Todo: again, debug to check dimensionality of prev
    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0][0] = np.array(
        [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
    prev_state = next_state

  return strokes, mixture_params
