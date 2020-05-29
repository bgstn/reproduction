import tensorflow as tf


class HtNet:
    def __init__(self, dicts, shape, feature_dim=6):
        self.shape = self.compute_shape(shape, feature_dim)
        self.feature_layer = self.get_feature_layer(units=feature_dim)
        self.atten_layer = self.get_atten_layer()
        self.dicts = self.get_dict(dicts)
        self.model = self.build_model()

    def compute_shape(self, input_shape, feature_dim):
        batch_size, num_product, inp_feature_dim = input_shape
        shape = {"input": (num_product, inp_feature_dim),
                  "x": (num_product, feature_dim),
                  "dict_x": (num_product, feature_dim)}
        return shape

    def get_dict(self, dicts):
        inp_dict_x = dicts["x"]
        inp_dict_y = dicts["y"]
        x = tf.constant(inp_dict_x, dtype="float32")
        y = tf.constant(inp_dict_y, dtype="float32")
        return {"x": x, "y": y}

    def get_feature_layer(self, units=6):
        inp = tf.keras.layers.Input(self.shape["input"])
        l = tf.keras.layers.Dense(units=units, activation=None)(inp)
        l = tf.keras.layers.LayerNormalization()(l)
        m = tf.keras.models.Model(inp, l)
        return m

    def get_atten_layer(self):
        shape = self.shape
        # input
        inp_x = tf.keras.layers.Input(shape["x"])
        l = inp_x

        inp_dict_x = tf.keras.layers.Input(shape["dict_x"])
        l_dict_x = inp_dict_x

        l = tf.keras.layers.Lambda(lambda x: tf.einsum("ijk,pmk->ijpm", x[0], x[1]),
                                   name="raw_score_attn")([l, l_dict_x])
        l = tf.keras.layers.Lambda(lambda x: tf.keras.backend.permute_dimensions(x, [1, 3, 0, 2]))(l)
        mask = tf.keras.layers.Lambda(lambda x: 1 - tf.eye(tf.keras.backend.shape(x)[2]),
                                      name="mask_generation")(l)
        l = tf.keras.layers.Lambda(lambda x: x[0]*x[1], name="masking")([mask, l])
        l = tf.keras.layers.Lambda(lambda x: tf.keras.backend.permute_dimensions(x, (2, 0, 3, 1)),
                                   name="attn_score")(l)
        l = tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x, axis=[2, 3]))(l)

        m = tf.keras.models.Model([inp_x, inp_dict_x], l)
        return m

    def build_model(self):
        shape = self.shape["input"]
        # input
        inp = tf.keras.layers.Input(shape)
        inp_dict_x = self.dicts["x"]
        inp_dict_y = self.dicts["y"]

        l = inp
        l_dict_x = inp_dict_x
        l_dict_y = inp_dict_y

        # feature_layer
        l = self.feature_layer(l)
        l_dict_x = self.feature_layer(l_dict_x)

        # attention
        l = self.atten_layer([l, l_dict_x])

        # output
        output_separate = tf.keras.layers.Lambda(lambda x: tf.einsum("ijpm, pml->ijl", x[0], x[1]),
                                                 name="output_separate")([l, l_dict_y])
        output_tot = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1, keepdims=True),
                                            name="output_tot")(output_separate)

        model = tf.keras.models.Model(inp, [output_separate, output_tot])

        model.summary()

        optim = tf.keras.optimizers.Adam(learning_rate=5.3e-2)
        model.compile(loss="mse", optimizer=optim)
        return model


if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize

    np.random.seed(0)
    T = 40
    t0 = np.random.uniform(low=1, high=100, size=(100))
    group = 5
    tot = 1e6 - 5000 * np.arange(T) + np.random.normal(loc=5000, scale=5000, size=T)

    rate = []
    sample = []
    r0 = np.random.normal(loc=0, scale=0.1, size=group)
    for i in range(T):
        r = 1 / 3 + i * r0 + np.random.normal(loc=0, scale=0.01)
        r = np.exp(r) / np.exp(r).sum()
        rate.append(r)
        sample.append(tot[i] * r)

    df = pd.DataFrame(sample).unstack().reset_index().rename(
        columns={"level_0": "product", "level_1": "time_index", 0: "quantity"})
    df = df.assign(ratio=lambda x: x.groupby(["time_index"])["quantity"].transform(lambda y: y / y.sum()))
    df = df.assign(product=lambda x: x["product"] + 1,
                   time_index=lambda x: x["time_index"] + 1)

    cols = []
    for col in ["product"]:
        for i in range(df["product"].unique().shape[0]):
            tmp = (df["product"] == i) * 1
            tmp = (tmp - tmp.mean()) / (tmp.std() + 1e-6)
            df["product_" + str(i)] = tmp
            cols.append("product_" + str(i))
    df = df.assign(time_index=lambda x: (x["time_index"] - x["time_index"].mean()) / (x["time_index"].std() + 1e-6))
    x = df.loc[:, cols + ["time_index"]].values.reshape(group, T, 1 + len(cols)).transpose(1, 0, 2)
    y = df.loc[:, ["quantity"]].values.reshape(group, T, 1).transpose(1, 0, 2)
    y_ratio = df.loc[:, ["ratio"]].values.reshape(group, T, 1).transpose(1, 0, 2)
    y_tot = df.loc[:, ["quantity"]].values.reshape(group, T, 1).transpose(1, 0, 2).sum(axis=1, keepdims=True)

    # model
    hn = HtNet(dicts={"x": x, "y": y}, shape=x.shape)

    model = hn.model
    hist = model.fit(x=x, y=[y, y_tot], epochs=300, batch_size=40, shuffle=False)

    # plot
    for k in ["loss", "output_separate_loss", "output_tot_loss"]:
        tmp = hist.history.get(k, None)
        if tmp is not None:
            print(min(tmp))
            plt.figure()
            plt.plot(hist.history[k])
            plt.title(k)
            plt.show()

    for output in model.predict(x=[x, y], batch_size=40):
        plt.figure()
        plt.plot(output.squeeze())
        plt.show()

    pd.DataFrame(y_ratio.squeeze()).plot()
    plt.show()