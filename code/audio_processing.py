import librosa
import numpy as np


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = load_audio_file("/media/ml/data_ml/fma_medium/008/008081.mp3")

    print(data.shape)

    plt.imshow(data.T)
    plt.show()

