from tensorflow.keras.layers import (
    Input,
    GlobalMaxPool1D,
    Dense,
    Bidirectional,
    GRU,
    Dropout
)
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from transformer import Encoder


def transformer_classifier(
        num_layers=3,
        d_model=256,
        num_heads=8,
        dff=512,
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
    )

    x = encoder(inp)

    x = Dropout(0.5)(x)

    x = GlobalMaxPool1D()(x)

    x = Dropout(0.5)(x)

    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.00001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=opt, loss=sparse_categorical_crossentropy, metrics=["acc"])

    model.summary()

    return model


def rnn_classifier(
        d_model=256,
        n_layers=2,
        n_classes=16,
):
    inp = Input((None, d_model))

    x = Bidirectional(GRU(d_model, return_sequences=True))(inp)

    if n_classes > 1:
        for i in range(n_layers - 1):
            x = Bidirectional(GRU(d_model, return_sequences=True))(x)

    x = Dropout(0.5)(x)

    x = GlobalMaxPool1D()(x)

    x = Dropout(0.5)(x)

    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.00001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=opt, loss=sparse_categorical_crossentropy, metrics=["acc"])

    model.summary()

    return model


if __name__ == '__main__':

    model1 = transformer_classifier()

    model2 = rnn_classifier()