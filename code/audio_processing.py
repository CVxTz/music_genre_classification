import sys
import warnings

import librosa
import numpy as np
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")

input_length = 16000 * 30

n_mels = 128


def pre_process_audio_mel_t(audio, sample_rate=16000):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40

    return mel_db.T


def load_audio_file(file_path, input_length=input_length):
    try:
        data = librosa.core.load(file_path, sr=16000)[0]  # , sr=16000
    except ZeroDivisionError:
        data = []

    if len(data) > input_length:

        max_offset = len(data) - input_length

        offset = np.random.randint(max_offset)

        data = data[offset : (input_length + offset)]

    else:
        if input_length > len(data):
            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)
        else:
            offset = 0

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    data = pre_process_audio_mel_t(data)
    return data


def random_crop(data, crop_size=128):
    start = np.random.randint(0, data.shape[0] - crop_size)
    return data[start : (start + crop_size), :]


def random_mask(data):
    new_data = data.copy()
    prev_zero = False
    for i in range(new_data.shape[0]):
        if np.random.uniform(0, 1) < 0.1 or (prev_zero and np.random.uniform(0, 1) < 0.5):
            prev_zero = True
            new_data[i, :] = 0
        else:
            prev_zero = False

    return new_data


def save(path):
    data = load_audio_file(path)
    np.save(path.replace(".mp3", ".npy"), data)
    return True


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from glob import glob
    from multiprocessing import Pool

    base_path = "/media/ml/data_ml/fma_large"
    files = sorted(list(glob(base_path + "/*/*.mp3")))

    print(len(files))

    p = Pool(8)

    for i, _ in tqdm(enumerate(p.imap(save, files))):
        if i % 1000 == 0:
            print(i)

    # data = load_audio_file("/media/ml/data_ml/fma_medium/008/008081.mp3", input_length=16000 * 30)
    #
    # print(data.shape)
    #
    # new_data =random_mask(data)
    #
    # plt.figure()
    # plt.imshow(data.T)
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(new_data.T)
    # plt.show()
    #
    # print(np.min(data), np.max(data))
