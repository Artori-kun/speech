import random

from tensorflow.keras.utils import Sequence
import numpy as np
import librosa


class MyGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, max_sample_len, shuffle):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.max_sample_len = max_sample_len
        self.shuffle = shuffle

    def load_x(self, path):
        samples, _ = librosa.load(path, sr=8000)

        if samples.shape[0] < self.max_sample_len:
            fill = np.zeros(self.max_sample_len - samples.shape[0])
            samples = np.append(samples, fill)

        return samples

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        x = [self.load_x(file) for file in batch_x]
        y = batch_y

        return np.array(x).reshape([-1, self.max_sample_len, 1]), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            c = list(zip(self.x, self.y))
            random.shuffle(c)

            self.x, self.y = zip(*c)
