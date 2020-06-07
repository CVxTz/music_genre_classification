import ast
import json
import os
import sys
import warnings

import pandas as pd
from pandas.api.types import CategoricalDtype

if not sys.warnoptions:
    warnings.simplefilter("ignore")

CLASS_MAPPING = {'Folk': 0, 'Rock': 1, 'Classical': 2, 'Hip-Hop': 3, 'Jazz': 4, 'Blues': 5, 'Soul-RnB': 6, 'Spoken': 7,
                 'Experimental': 8, 'Old-Time / Historic': 9, 'International': 10, 'Country': 11, 'Pop': 12,
                 'Instrumental': 13, 'Easy Listening': 14, 'Electronic': 15}

INVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}


def load(filepath):
    # https://github.com/mdeff/fma/blob/rc1/utils.py

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def get_id_from_path(path):
    base_name = os.path.basename(path)

    return base_name.replace(".mp3", "")


if __name__ == "__main__":
    in_path = "/media/ml/data_ml/fma_metadata/tracks.csv"
    out_path = '/media/ml/data_ml/fma_metadata/tracks_genre.json'
    df = load("/media/ml/data_ml/fma_metadata/tracks.csv")
    df.reset_index(inplace=True)

    print(df.head())
    print(df.columns.values)
    print(set(df[('set', 'subset')].tolist()))

    df = df[df[('set', 'subset')].isin(["small", "medium"])]

    print(set(df[('track', 'genre_top')].tolist()))

    print(df[[("track_id", ''), ('track', 'genre_top'), ('track', 'genres'), ('set', 'subset')]])

    data = {k: v for k, v in zip(df[("track_id", '')].tolist(), df[('track', 'genre_top')].tolist())}

    json.dump(data, open(out_path, "w"), indent=4)
