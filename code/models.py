from tensorflow.keras.layers import (
    Input,
    GlobalAvgPool1D,
    Dense,
    Bidirectional,
    GRU,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.losses import mae

from transformer import Encoder


def custom_binary_accuracy(y_true, y_pred, threshold=0.5):
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    y_true = math_ops.cast(y_true > threshold, y_true.dtype)

    return K.mean(math_ops.equal(y_true, y_pred), axis=-1)


def custom_binary_crossentropy(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    epsilon_ = K._constant_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    output = clip_ops.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = 4 * y_true * math_ops.log(output + K.epsilon())
    bce += (1 - y_true) * math_ops.log(1 - output + K.epsilon())
    return K.sum(-bce, axis=-1)


def transformer_classifier(
    num_layers=4,
    d_model=128,
    num_heads=8,
    dff=256,
    maximum_position_encoding=2048,
    n_classes=16,
):
    inp = Input((None, d_model))

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        rate=0.3,
    )

    x = encoder(inp)

    x = Dropout(0.2)(x)

    x = GlobalAvgPool1D()(x)

    x = Dense(4 * n_classes, activation="selu")(x)

    out = Dense(n_classes, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.00001)

    model.compile(
        optimizer=opt, loss=custom_binary_crossentropy, metrics=[custom_binary_accuracy]
    )

    model.summary()

    return model


def transformer_pretrain(
    num_layers=4,
    d_model=128,
    num_heads=8,
    dff=256,
    maximum_position_encoding=2048,
):
    inp = Input((None, d_model))

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        rate=0.3
    )

    x = encoder(inp)

    out = Dense(d_model, activation="linear", name="out_pretraining")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.0001)

    model.compile(
        optimizer=opt, loss=mae
    )

    model.summary()

    return model


def rnn_classifier(
    d_model=128, n_layers=2, n_classes=16,
):
    inp = Input((None, d_model))

    x = Bidirectional(GRU(d_model, return_sequences=True))(inp)

    if n_classes > 1:
        for i in range(n_layers - 1):
            x = Bidirectional(GRU(d_model, return_sequences=True))(x)

    x = Dropout(0.2)(x)

    x = GlobalAvgPool1D()(x)

    x = Dense(4 * n_classes, activation="selu")(x)

    out = Dense(n_classes, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.00001)

    model.compile(
        optimizer=opt, loss=custom_binary_crossentropy, metrics=[custom_binary_accuracy]
    )

    model.summary()

    return model


if __name__ == "__main__":
    model1 = transformer_classifier()

    model2 = rnn_classifier()
