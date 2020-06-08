import json
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

from audio_processing import random_crop
from models import rnn_classifier
from prepare_data import CLASS_MAPPING, get_id_from_path


class DataGenerator(Sequence):
    def __init__(self, path_x_label_list, class_mapping, batch_size=32):
        self.path_x_label_list = path_x_label_list

        self.batch_size = batch_size
        self.indexes = np.arange(len(self.path_x_label_list))
        self.class_mapping = class_mapping
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.path_x_label_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        batch_samples = [self.path_x_label_list[k] for k in indexes]

        x, y = self.__data_generation(batch_samples)

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_samples):
        paths, labels = zip(*batch_samples)

        labels = [self.class_mapping[x] for x in labels]

        crop_size = np.random.randint(128, 256)

        X = np.array([random_crop(np.load(x), crop_size=crop_size) for x in paths])
        Y = np.array(labels)

        return X, Y[..., np.newaxis]


if __name__ == "__main__":
    h5_name = "transformer.h5"
    batch_size = 32
    epochs = 20

    id_to_genre = json.load(open('/media/ml/data_ml/fma_metadata/tracks_genre.json'))
    id_to_genre = {int(k): v for k, v in id_to_genre.items()}

    base_path = '/media/ml/data_ml/fma_medium'
    files = sorted(list(glob(base_path + "/*/*.npy")))
    labels = [id_to_genre[int(get_id_from_path(x))] for x in files]

    samples = list(zip(files, labels))

    train, val = train_test_split(samples, test_size=0.2, random_state=1337, stratify=labels)

    model = rnn_classifier()

    checkpoint = ModelCheckpoint(
        h5_name,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=True,
    )
    reduce_o_p = ReduceLROnPlateau(monitor="val_loss", patience=20, min_lr=1e-7, mode="min")

    model.fit_generator(
        DataGenerator(
            train,
            batch_size=batch_size,
            class_mapping=CLASS_MAPPING
        ),
        validation_data=DataGenerator(
            val,
            batch_size=batch_size,
            class_mapping=CLASS_MAPPING
        ),
        epochs=epochs,
        callbacks=[checkpoint, reduce_o_p],
        use_multiprocessing=True,
        workers=12,
        verbose=2,
        max_queue_size=64
    )
