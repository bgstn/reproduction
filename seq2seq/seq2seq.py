from keras.models import Model
from keras.layers import LSTM, Input, Lambda, Embedding, Dense, Softmax, Reshape, Concatenate, Layer
import keras.backend as K
import pandas as pd
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf


class MyLayer(Layer):

    def __init__(self, mask_zero=True, units=None,  **kwargs):
        self.output_dim = units
        self.mask_zero = mask_zero
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][2]*2, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.vec = self.add_weight(name='kernel',
                                   shape=(self.output_dim, 1),
                                   initializer='uniform',
                                   trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        enc_output, dec_output = x
        dec_repeat = K.tile(dec_output, [1, tf.shape(enc_output)[1], 1])
        enc_repeat = tf.repeat(enc_output, axis=1, repeats=K.shape(dec_output)[1])
        enc_dec = K.concatenate([enc_repeat, dec_repeat], axis=-1)
        weighted_enc_dec = K.dot(enc_dec, self.kernel)
        activated_weighted_enc_dec = K.tanh(weighted_enc_dec)
        score = K.dot(activated_weighted_enc_dec, self.vec)
        res = K.reshape(score, (tf.shape(enc_output)[0], tf.shape(enc_output)[1], K.shape(dec_output)[1]))
        return res

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a[0], shape_a[1], shape_b[1]


def build_model(src_vocab_size, tgt_vocab_size, attention_mode='dot'):
    inp_src = Input(shape=(None, ))
    emb_src = Embedding(input_dim=src_vocab_size, output_dim=100, mask_zero=True)(inp_src)

    inp_tgt = Input(shape=(None, ))
    emb_tgt = Embedding(input_dim=tgt_vocab_size, output_dim=100, mask_zero=True)(inp_tgt)

    # encoder
    encoder_output, h, c = LSTM(units=50, return_state=True, return_sequences=True)(emb_src)

    # decoder
    decoder_output = LSTM(units=50, return_sequences=True)(emb_tgt, initial_state=[h, c])

    # attention
    if attention_mode == 'dot':
        # score
        # score = Lambda(lambda x: tf.einsum('ijk, ilk->ijl', x[0], x[1]))([encoder_output, decoder_output])
        score = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]))([encoder_output, decoder_output])
    elif attention_mode == 'ldot':
        # score
        score = Dense(50, activation=None)(encoder_output)
        score = Lambda(lambda x: tf.einsum('ijk, ilk->ijl', x[0], x[1]))([score, decoder_output])
    elif attention_mode == 'concat':
        # score
        score = MyLayer(units=50)([encoder_output, decoder_output])
    else:
        raise NotImplementedError

    score = Softmax(axis=1)(score)
    context = Lambda(lambda x: tf.einsum('ijl, ijk->ilk', x[0], x[1]))([score, encoder_output])

    # concatenate
    out = Concatenate()([context, decoder_output])

    # output
    output = Dense(tgt_vocab_size, activation='softmax')(out)

    m = Model([inp_src, inp_tgt], output)
    m.summary()

    opt = Adam(lr=0.001)
    m.compile(optimizer=opt, loss="categorical_crossentropy")
    return m


def translate(idx2word, idxes):
    tmp_idxes = idxes

    sentences = []
    for idx in range(tmp_idxes.shape[0]):
        sentence = tmp_idxes[idx]
        sentence = [idx2word[id] for id in sentence if id != 0]
        sentences.append(sentence)
    return sentences


def predict_on_validation(model, inp, dataset):
    pred_idxes = model.predict(inp)
    idx2word = dataset.tgt_idx2word
    pred_idxes = np.argmax(pred_idxes, axis=2) + 1
    sentences = translate(idx2word, pred_idxes)
    return sentences


def predict_on_test(model, inp, dataset, n):
    batch_size, _ = inp.shape
    tgt_word = [np.array([dataset.tgt_word2idx['S']] * batch_size).reshape(-1, 1) for _ in range(n)]
    tgt_max_len = dataset.max_decoder_seq_length
    # src_max_len = dataset.max_encoder_seq_length

    previous_top_n_probablities = [np.array([1] * batch_size).reshape(-1, 1) for _ in range(n)]
    for i in range(tgt_max_len):

        # get prediction
        pred_idxes = [model.predict([inp, w])[:, -1, :] for w in tgt_word]

        # get top n prob
        current_top_n_probabilities = [np.sort(x, axis=-1)[:, (-1*n):] for x in pred_idxes]
        full_top_n_probabilities = [prob_prv * prob_cur for prob_prv, prob_cur in zip(previous_top_n_probablities,
                                                                                      current_top_n_probabilities)]
        lastest_top_n_candates_idx = np.argsort(np.concatenate(full_top_n_probabilities, axis=-1))[:, (-1*n):]

        # update previous prob, candidates
        # prob
        previous_top_n_probablities = np.sort(np.concatenate(full_top_n_probabilities, axis=-1))[:, (-1*n):]
        previous_top_n_probablities = [r.reshape(-1, 1) for r in previous_top_n_probablities.T]

        # candidates
        top_n_candidates = [np.argsort(w, axis=-1)[:, (-1 * n):] + 1 for w in pred_idxes]
        lastest_top_n_candates = np.take_along_axis(np.concatenate(top_n_candidates, axis=-1),
                                                    lastest_top_n_candates_idx, axis=1)
        lastest_top_n_candates = [r.reshape(-1, 1) for r in lastest_top_n_candates.T]
        tgt_word = [np.concatenate([pt, ct], axis=1) for pt, ct in zip(tgt_word, lastest_top_n_candates)]

    return [translate(dataset.tgt_idx2word, x) for x in tgt_word]


if __name__ == '__main__':
    from dataset import Dataset

    dd = Dataset()
    dd.run()
    X = np.concatenate(dd.src_idxes)
    Y = np.concatenate(dd.tgt_idxes)
    Y_one_hot = np.concatenate(dd.tgt_idexes_onehot)
    m = build_model(dd.src_vocab_size, dd.tgt_vocab_size, 'concat')
    m.fit(x=[X, Y], y=Y_one_hot, batch_size=32, epochs=50, verbose=1, validation_split=0.3)

    pred = predict_on_validation(model=m, inp=[X, Y], dataset=dd)
    src = translate(dd.src_idx2word, X)
    # print(src[0:10], pred[0:10])
    print(src[0:10], predict_on_test(model=m, inp=X, dataset=dd, n=3))

