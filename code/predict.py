import json
from glob import glob

import numpy as np

from models import transformer_classifier
from prepare_data import random_crop
from audio_processing import load_audio_file


def chunker(seq, size):
    return (seq[pos: pos + size] for pos in range(0, len(seq), size))


if __name__ == "__main__":

    transformer_v2_h5 = "transformer_v2.h5"

    CLASS_MAPPING = json.load(open("/media/ml/data_ml/fma_metadata/mapping.json"))

    base_path = "../audio"
    files = sorted(list(glob(base_path + "/*.mp3")))

    data = [load_audio_file(x, input_length=16000 * 120) for x in files]

    transformer_v2_model = transformer_classifier(n_classes=len(CLASS_MAPPING))

    transformer_v2_model.load_weights(transformer_v2_h5)

    crop_size = np.random.randint(128, 512)
    repeats = 8

    transformer_v2_Y = 0

    for _ in range(repeats):
        X = np.array([random_crop(x, crop_size=crop_size) for x in data])

        transformer_v2_Y += transformer_v2_model.predict(X) / repeats

    transformer_v2_Y = transformer_v2_Y.tolist()

    for path, pred in zip(files, transformer_v2_Y):

        print(path)
        pred_tup = [(k, pred[v]) for k, v in CLASS_MAPPING.items()]
        pred_tup.sort(key=lambda x: x[1], reverse=True)

        for a in pred_tup[:5]:
            print(a)
