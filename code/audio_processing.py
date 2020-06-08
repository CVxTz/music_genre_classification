import sys
import warnings

import librosa
import numpy as np

if not sys.warnoptions:
    warnings.simplefilter("ignore")
input_length = 16000 * 10

n_mels = 256


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

        data = data[offset:(input_length + offset)]


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
    start = np.random.randint(0, data.shape[0]-crop_size)
    return data[start:(start+crop_size), :]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from glob import glob

    base_path = '/media/ml/data_ml/fma_medium'
    files = sorted(list(glob(base_path + "/*/*.mp3")))

    for path in tqdm(files):
        data = load_audio_file(path, input_length=16000 * 30)
        np.save(path.replace(".mp3", ".npy"), data)

    # data = load_audio_file("/media/ml/data_ml/fma_medium/008/008081.mp3", input_length=16000 * 30)
    #
    # print(data.shape)
    # print(random_crop(data, crop_size=128).shape)
    #
    # plt.imshow(data.T)
    # plt.show()



