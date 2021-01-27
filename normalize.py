import progressbar
import numpy as np
from glob import iglob, glob
import os
from python_speech_features import mfcc
import json
import librosa

data_dir = '/home/minhhiu/MyProjects/Compressed Speech Data/full_command_data'


def find_files(dir, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return sorted(iglob(os.path.join(dir, pattern), recursive=True))


def read_audio_from_filename(filename, sample_rate):
    """Load a wav file and transpose the array."""
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    return audio


# find mean, and varience of training data for normalization
directory = data_dir + '/train/wav'

bar = progressbar.ProgressBar()
n = 0
sum_mfcc = np.zeros(39)
sumsq_mfcc = np.zeros(39)
total_len = 0
for file in bar(find_files(directory, pattern='**/*.wav')):
    audio = mfcc(read_audio_from_filename(file, 16000), samplerate=16000, winlen=0.025, winstep=0.01, numcep=39,
                 nfilt=40)

    sum_mfcc += np.sum(audio, axis=0)
    sumsq_mfcc += np.sum(audio * audio, axis=0)
    total_len += len(audio)
    n += 1

m = sum_mfcc / total_len
v = sumsq_mfcc / (total_len - 1) - m * m
s = np.sqrt(v)

with open("normalize_param.json", "w") as js:
    result = {"mean": m.tolist(), "variance": v.tolist(), "sqrt": s.tolist()}
    json.dump(result, js)

print(m)
print(v)
print(s)
