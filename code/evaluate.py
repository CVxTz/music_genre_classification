import json
from glob import glob

import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split

from models import rnn_classifier, transformer_classifier
from prepare_data import get_id_from_path, labels_to_vector, random_crop
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


if __name__ == "__main__":
    from collections import Counter

    transformer_h5 = "transformer.h5"
    transformer_v2_h5 = "transformer_v2.h5"

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
    transformer_v2_model = transformer_classifier(n_classes=len(CLASS_MAPPING))

    rnn_model = rnn_classifier(n_classes=len(CLASS_MAPPING))

    transformer_model.load_weights(transformer_h5)
    transformer_v2_model.load_weights(transformer_v2_h5)

    rnn_model.load_weights(rnn_h5)

    all_labels = []
    transformer_all_preds = []
    transformer_v2_all_preds = []

    rnn_all_preds = []

    for batch_samples in tqdm(
        chunker(val, size=batch_size), total=len(val) // batch_size
    ):
        paths, labels = zip(*batch_samples)

        all_labels += [labels_to_vector(x, CLASS_MAPPING) for x in labels]

        crop_size = np.random.randint(128, 256)
        repeats = 16

        transformer_Y = 0
        transformer_v2_Y = 0

        rnn_Y = 0

        for _ in range(repeats):
            X = np.array([random_crop(np.load(x), crop_size=crop_size) for x in paths])

            transformer_Y += transformer_model.predict(X) / repeats
            transformer_v2_Y += transformer_v2_model.predict(X) / repeats

            rnn_Y += rnn_model.predict(X) / repeats

        transformer_all_preds.extend(transformer_Y.tolist())
        transformer_v2_all_preds.extend(transformer_v2_Y.tolist())

        rnn_all_preds.extend(rnn_Y.tolist())

    T_Y = np.array(transformer_all_preds)
    T_v2_Y = np.array(transformer_v2_all_preds)

    R_Y = np.array(rnn_all_preds)
    Y = np.array(all_labels)

    trsf_ave_auc_pr = 0
    trsf_v2_ave_auc_pr = 0

    rnn_ave_auc_pr = 0

    total_sum = 0

    for label, i in CLASS_MAPPING.items():
        if np.sum(Y[:, i]) > 0:
            trsf_auc = average_precision_score(Y[:, i], T_Y[:, i])
            trsf_v2_auc = average_precision_score(Y[:, i], T_v2_Y[:, i])
            rnn_auc = average_precision_score(Y[:, i], R_Y[:, i])
            print(label, np.sum(Y[:, i]))
            print("transformer   :", trsf_auc)
            print("transformer v2:", trsf_v2_auc)
            print("rnn           :", rnn_auc)
            print("")

            trsf_ave_auc_pr += np.sum(Y[:, i]) * trsf_auc
            trsf_v2_ave_auc_pr += np.sum(Y[:, i]) * trsf_v2_auc

            rnn_ave_auc_pr += np.sum(Y[:, i]) * rnn_auc
            total_sum += np.sum(Y[:, i])

            if label == "Hip-Hop":
                precision, recall, _ = precision_recall_curve(Y[:, i], T_Y[:, i])
                average_precision = average_precision_score(Y[:, i], T_Y[:, i])

                plt.figure()
                plt.step(recall, precision, where="post")

                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title("Average precision score ".format(average_precision))
                plt.savefig("plot.png")

    trsf_ave_auc_pr /= total_sum
    trsf_v2_ave_auc_pr /= total_sum

    rnn_ave_auc_pr /= total_sum

    print("transformer micro-average     : ", trsf_ave_auc_pr)
    print("transformer v2 micro-average  : ", trsf_v2_ave_auc_pr)

    print("rnn micro-average             : ", rnn_ave_auc_pr)
