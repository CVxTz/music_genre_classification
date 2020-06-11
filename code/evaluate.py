import json
from glob import glob

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from models import rnn_classifier, transformer_classifier
from prepare_data import get_id_from_path, labels_to_vector, random_crop
from tqdm import tqdm


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


if __name__ == "__main__":
    from collections import Counter

    transformer_h5 = "transformer.h5"
    rnn_h5 = "rnn.h5"

    batch_size = 128
    epochs = 5
    CLASS_MAPPING = json.load(open("/media/ml/data_ml/fma_metadata/mapping.json"))
    id_to_genres = json.load(open("/media/ml/data_ml/fma_metadata/tracks_genre.json"))
    id_to_genres = {int(k): v for k, v in id_to_genres.items()}

    base_path = "/media/ml/data_ml/fma_large"
    files = sorted(list(glob(base_path + "/*/*.npy")))
    files = [x for x in files if id_to_genres[int(get_id_from_path(x))]]
    labels = [id_to_genres[int(get_id_from_path(x))] for x in files]
    print(len(labels))

    samples = list(zip(files, labels))

    strat = [a[-1] for a in labels]
    cnt = Counter(strat)
    strat = [a if cnt[a] > 2 else "" for a in strat]

    train, val = train_test_split(
        samples, test_size=0.2, random_state=1337, stratify=strat
    )

    transformer_model = transformer_classifier(n_classes=len(CLASS_MAPPING))
    rnn_model = rnn_classifier(n_classes=len(CLASS_MAPPING))

    transformer_model.load_weights(transformer_h5)
    rnn_model.load_weights(rnn_h5)

    all_labels = []
    transformer_all_preds = []
    rnn_all_preds = []

    for batch_samples in tqdm(
        chunker(val, size=batch_size), total=len(val) // batch_size
    ):
        paths, labels = zip(*batch_samples)

        all_labels += [labels_to_vector(x, CLASS_MAPPING) for x in labels]

        crop_size = 256
        repeats = 4

        transformer_Y = 0
        rnn_Y = 0

        for _ in range(repeats):
            X = np.array([random_crop(np.load(x), crop_size=crop_size) for x in paths])

            transformer_Y += transformer_model.predict(X) / repeats
            rnn_Y += rnn_model.predict(X) / repeats

        transformer_all_preds.extend(transformer_Y.tolist())
        rnn_all_preds.extend(rnn_Y.tolist())

    T_Y = np.array(transformer_all_preds)
    R_Y = np.array(rnn_all_preds)
    Y = np.array(all_labels)

    for label, i in CLASS_MAPPING.items():
        if np.sum(Y[:, i]) > 0:
            print(label, np.sum(Y[:, i]))
            print("transformer :", f1_score(Y[:, i], (T_Y[:, i] > 0.5).astype(int)))
            print("rnn         :", f1_score(Y[:, i], (R_Y[:, i] > 0.5).astype(int)))
            print("")
