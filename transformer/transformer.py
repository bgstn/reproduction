import keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Add, Dot, Softmax, Lambda, Concatenate, \
    LayerNormalization, Multiply, Reshape, Permute, Dropout
import numpy as np
from tensorflow.keras.models import Model


def build_model(src_vocab_size, tgt_vocab_size, dk, head_number=2):
    inp_src = Input(shape=(None,))
    emb_src = Embedding(input_dim=src_vocab_size, output_dim=dk, mask_zero=True)(inp_src)
    inp_pos_src = Input(shape=(None, dk))
    emb_inp_src = Add()([inp_pos_src, emb_src])
    emb_inp_src = Dropout(rate=0.1)(emb_inp_src)
    # ================================== Encoder ==================================
    # multihead attention
    multihead_src = []
    for _ in range(head_number):
        query = Dense(units=dk, activation=None)(emb_inp_src)
        key = Dense(units=dk, activation=None)(emb_inp_src)
        value = Dense(units=dk, activation=None)(emb_inp_src)

        # compute score
        score = Lambda(lambda x: K.batch_dot(x=x[0], y=x[1], axes=[2, 2])/np.sqrt(dk))([query, key])
        score = Softmax(axis=-1)(score)
        attention = Lambda(lambda x: K.batch_dot(x=x[0], y=x[1], axes=[2, 1]))([score, value])
        multihead_src.append(attention)
    multihead_src = Concatenate(axis=-1)(multihead_src)
    multihead_src = Dense(units=dk, activation=None)(multihead_src)
    multihead_src = Dropout(rate=0.1)(multihead_src)
    multihead_src = Add()([multihead_src, emb_inp_src])
    attention_out_src = LayerNormalization()(multihead_src)

    # ffn
    context = Dense(units=dk, activation='relu')(attention_out_src)
    context = Dense(units=dk, activation=None)(context)
    context = Dropout(rate=0.1)(context)
    context = Add()([context, attention_out_src])
    context = LayerNormalization()(context)
    # ================================== Encoder ==================================

    # ================================== Decoder ==================================
    inp_tgt = Input(shape=(None,))
    emb_tgt = Embedding(input_dim=tgt_vocab_size, output_dim=dk, mask_zero=True)(inp_tgt)
    inp_pos_tgt = Input(shape=(None, dk))
    emb_inp_tgt = Add()([inp_pos_tgt, emb_tgt])
    emb_inp_tgt = Dropout(rate=0.1)(emb_inp_tgt)

    # multihead attention
    multihead_tgt = []
    for _ in range(head_number):
        query = Dense(units=dk, activation=None)(emb_inp_tgt)
        key = Dense(units=dk, activation=None)(emb_inp_tgt)
        value = Dense(units=dk, activation=None)(emb_inp_tgt)

        # compute score
        score = Lambda(lambda x: K.batch_dot(x=x[0], y=x[1], axes=[2, 2]) / np.sqrt(dk))([query, key])
        # masking
        score = Lambda(lambda x: tf.linalg.band_part(x, -1, 0))(score)
        score = Softmax(axis=-1)(score)
        attention = Lambda(lambda x: K.batch_dot(x=x[0], y=x[1], axes=[2, 1]))([score, value])
        multihead_tgt.append(attention)
    multihead_tgt = Concatenate(axis=-1)(multihead_tgt)
    multihead_tgt = Dense(units=dk, activation=None)(multihead_tgt)
    multihead_tgt = Dropout(0.1)(multihead_tgt)
    multihead_tgt = Add()([multihead_tgt, emb_inp_tgt])
    attention_out_tgt = LayerNormalization()(multihead_tgt)

    # merge encoder & decoder
    # multihead attention
    multihead = []
    for _ in range(head_number):
        query = Dense(units=dk, activation=None)(attention_out_tgt)
        key = Dense(units=dk, activation=None)(context)
        value = Dense(units=dk, activation=None)(context)

        # compute score
        score = Lambda(lambda x: K.batch_dot(x=x[0], y=x[1], axes=[2, 2]) / np.sqrt(dk))([query, key])
        score = Softmax(axis=-1)(score)
        attention = Lambda(lambda x: K.batch_dot(x=x[0], y=x[1], axes=[2, 1]))([score, value])
        # attention = Add()([attention, emb_inp_tgt])
        multihead.append(attention)
    multihead = Concatenate(axis=-1)(multihead)
    multihead = Dense(units=dk, activation=None)(multihead)
    multihead = Dropout(rate=0.1)(multihead)
    multihead = Add()([multihead, context])
    attention_out = LayerNormalization()(multihead)

    # ffn
    final_feature = Dense(units=dk, activation='relu')(attention_out)
    final_feature = Dense(units=dk, activation=None)(final_feature)
    final_feature = Dropout(rate=0.1)(final_feature)
    final_feature = Add()([final_feature, attention_out])
    final_feature = LayerNormalization()(final_feature)

    # logit
    logit = Dense(units=tgt_vocab_size, activation='softmax')(final_feature)
    # ================================== Decoder ==================================
    m = Model([inp_src, inp_pos_src, inp_tgt, inp_pos_tgt], logit)
    m.compile(optimizer='adam', loss="categorical_crossentropy")
    m.summary()
    return m


if __name__ == "__main__":
    model = build_model(src_vocab_size=10, tgt_vocab_size=10, dk=50)
    X = np.arange(5).reshape(1, -1)
    pos = np.random.uniform(low=0, high=1, size=(1, 5, 50))
    print(model.predict([X, pos, X, pos]))





