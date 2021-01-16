import tensorflow as tf
import numpy as np
import data
import time
import librosa
from python_speech_features import mfcc
from constants import c
import json
import os

# Parameters for index to string
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1

with open('encode_decode.json', 'r') as loadlabel:
    label_dic = json.load(loadlabel)

encode_dic = label_dic["encode"]

def get_key(val):
    for key, value in encode_dic.items():
        if val == value:
            return key
    return ''

def read_audio_from_filename(filename, sample_rate):
    """Load a wav file and transpose the array."""
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    return audio

file_name = "/home/minhhieu/My Projects/command_aHieu/test_cmd/Binh/"
pb_PATH = c.INFERENCE.PB_PATH

batch_size=1
num_layers=1
num_classes=43
num_features = c.LSTM.FEATURES
num_hidden = c.LSTM.HIDDEN

inputs = mfcc(read_audio_from_filename(file_name + '1.wav', 16000),samplerate=16000,winlen=0.025,winstep=0.01,numcep=39,nfilt=40)
wav_inputs = np.expand_dims(inputs,axis=0)

with open(os.path.join(file_name, "1.txt"), "r") as f:
    text = f.readline()
    data_len = np.asarray([len(text)])

with tf.compat.v1.gfile.FastGFile(pb_PATH, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.compat.v1.Session() as sess:
    # Set output tensor
    y = sess.graph.get_tensor_by_name('SparseToDense:0')
    start = time.time()
    labels = sess.run(y, feed_dict={'InputData:0':wav_inputs,
                                   'SeqLen:0':data_len})
    print(time.time()-start)
    str_decoded = ''.join([get_key(x) for x in labels[0]])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(get_key(label_dic["char_num"]), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(get_key(0), ' ')
    print(str_decoded)
