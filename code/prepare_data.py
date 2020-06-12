import ast
import os
import sys
import warnings

import pandas as pd
from pandas.api.types import CategoricalDtype

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import json

import numpy as np
from tensorflow.keras.utils import Sequence

from audio_processing import random_crop


class DataGenerator(Sequence):
    def __init__(self, path_x_label_list, class_mapping, batch_size=32):
        self.path_x_label_list = path_x_label_list

        self.batch_size = batch_size
        self.indexes = np.arange(len(self.path_x_label_list))
        self.class_mapping = class_mapping
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.path_x_label_list) / self.batch_size / 10))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        batch_samples = [self.path_x_label_list[k] for k in indexes]

        x, y = self.__data_generation(batch_samples)

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_samples):
        paths, labels = zip(*batch_samples)

        labels = [labels_to_vector(x, self.class_mapping) for x in labels]

        crop_size = np.random.randint(128, 256)

        X = np.array([random_crop(np.load(x), crop_size=crop_size) for x in paths])
        Y = np.array(labels)

        return X, Y[..., np.newaxis]


class PretrainGenerator(Sequence):
    def __init__(self, path_x_label_list, class_mapping, batch_size=32):
        self.path_x_label_list = path_x_label_list

        self.batch_size = batch_size
        self.indexes = np.arange(len(self.path_x_label_list))
        self.class_mapping = class_mapping
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.path_x_label_list) / self.batch_size / 10))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        batch_samples = [self.path_x_label_list[k] for k in indexes]

        x, y = self.__data_generation(batch_samples)

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_samples):
        paths, labels = zip(*batch_samples)

        labels = [labels_to_vector(x, self.class_mapping) for x in labels]

        crop_size = np.random.randint(128, 256)

        X = [random_crop(np.load(x), crop_size=crop_size) for x in paths]

        X = np.array(X)
        Y = np.array(labels)

        return X, Y[..., np.newaxis]


def load(filepath):
    # https://github.com/mdeff/fma/blob/rc1/utils.py

    filename = os.path.basename(filepath)

    if "features" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "echonest" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "genres" in filename:
        return pd.read_csv(filepath, index_col=0)

    if "tracks" in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [
            ("track", "date_created"),
            ("track", "date_recorded"),
            ("album", "date_created"),
            ("album", "date_released"),
            ("artist", "date_created"),
            ("artist", "active_year_begin"),
            ("artist", "active_year_end"),
        ]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ("small", "medium", "large")
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            CategoricalDtype(categories=SUBSETS, ordered=True)
        )

        COLUMNS = [
            ("track", "genre_top"),
            ("track", "license"),
            ("album", "type"),
            ("album", "information"),
            ("artist", "bio"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype("category")

        return tracks


def get_id_from_path(path):
    base_name = os.path.basename(path)

    return base_name.replace(".mp3", "").replace(".npy", "")


def labels_to_vector(labels, mapping):
    vec = [0] * len(mapping)
    for i in labels:
        vec[mapping[i]] = 1
    return vec


if __name__ == "__main__":
    in_path = "/media/ml/data_ml/fma_metadata/tracks.csv"
    genres_path = "/media/ml/data_ml/fma_metadata/genres.csv"

    out_path = "/media/ml/data_ml/fma_metadata/tracks_genre.json"
    mapping_path = "/media/ml/data_ml/fma_metadata/mapping.json"

    df = load("/media/ml/data_ml/fma_metadata/tracks.csv")

    df2 = pd.read_csv(genres_path)

    id_to_title = {k: v for k, v in zip(df2.genre_id.tolist(), df2.title.tolist())}

    df.reset_index(inplace=True)

    print(df.head())
    print(df.columns.values)
    print(set(df[("set", "subset")].tolist()))

    df = df[df[("set", "subset")].isin(["small", "medium", "large"])]

    print(set(df[("track", "genre_top")].tolist()))

    print(
        df[
            [
                ("track_id", ""),
                ("track", "genre_top"),
                ("track", "genres"),
                ("set", "subset"),
            ]
        ]
    )

    data = {
        k: [id_to_title[a] for a in v]
        for k, v in zip(df[("track_id", "")].tolist(), df[("track", "genres")].tolist())
    }

    json.dump(data, open(out_path, "w"), indent=4)

    mapping = {k: i for i, k in enumerate(df2.title.tolist())}

    json.dump(mapping, open(mapping_path, "w"), indent=4)
