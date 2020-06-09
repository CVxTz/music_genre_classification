from tensorflow.keras.layers import (
    Input,
    GlobalAvgPool1D,
    Dense,
    Bidirectional,
    GRU,
    Dropout
)
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from transformer import Encoder


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
        rate=0.3
    )

    x = encoder(inp)

    x = Dropout(0.2)(x)

    x = GlobalAvgPool1D()(x)

    out = Dense(n_classes, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=opt, loss=binary_crossentropy, metrics=["acc"])

    model.summary()

    return model


def rnn_classifier(
        d_model=128,
        n_layers=2,
        n_classes=16,
):
    inp = Input((None, d_model))

    x = Bidirectional(GRU(d_model, return_sequences=True))(inp)

    if n_classes > 1:
        for i in range(n_layers - 1):
            x = Bidirectional(GRU(d_model, return_sequences=True))(x)

    x = Dropout(0.2)(x)

    x = GlobalAvgPool1D()(x)

    out = Dense(n_classes, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=opt, loss=binary_crossentropy, metrics=["acc"])

    model.summary()

    return model


if __name__ == '__main__':

    model1 = transformer_classifier()

    model2 = rnn_classifier()