import json
import numpy as np
import librosa
from python_speech_features import mfcc
import os
import random


def read_audio_from_filename(filename, sample_rate):
    """Load a wav file and transpose the array."""
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    return audio


def convert_wav_mfcc(file, fs):
    """Turn raw audio data into MFCC with sample rate=fs."""
    inputs = mfcc(read_audio_from_filename(file, fs), samplerate=16000, winlen=0.025, winstep=0.01, numcep=39,
                  nfilt=40)
    return inputs


def load_encode_dic(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        return json.load(file)


# read the provided lst file
# return a list of lines in the file
def load_metadata(meta_file, start, end):
    with open(meta_file, "r", encoding="utf8") as meta:
        metadata = [x for i, x in enumerate(meta) if i in range(start, end)]
    return metadata


def get_metadata_len(meta_file):
    with open(meta_file, "r", encoding="utf8") as meta:
        lines = meta.readlines()
    num_line = len(lines)
    del lines
    return num_line


def shuffle_every_epoch(meta_file):
    """
    Shuffle the metadata file every epoch
    :param meta_file: file path
    :return: None
    """
    with open(meta_file, "r", encoding="utf8") as fr:
        lines = fr.readlines()
    random.shuffle(lines)
    with open(meta_file, "w", encoding="utf8") as fw:
        fw.writelines(lines)
    del lines


def next_batch_training(batch_size, batch, meta_file, encode_dic):
    """Return batch for training.
        Args:
            batch_size: number of samples per batch (int)
            encode_dic: dictionary used for encoding
            batch: the order of a batch in the epoch (int)
            meta_file: lst file path
        Returns:
            train_inputs: a batch of mfcc training data [batch_size, maxlen] (float)
            train_targets: a batch of target labels in index form and sparse tuple format (int)
            train_seq_len: a batch of training data sequence lengths [batch_size] (int)
            original: a batch of training target data in their original forms. (string)
    """
    inputs_batch = []
    targets = []
    original = []
    seq_len = []
    maxlen = 0
    meta_data = load_metadata(meta_file, batch * batch_size, (batch + 1) * batch_size)

    for line in meta_data:
        line = line.split("\t")
        # print(line)

        data_in = convert_wav_mfcc(line[1], 16000)
        target, _original = encode(line[3].strip("\n"), encode_dic)
        _seq_len = np.array([len(data_in)])[0]
        _original = np.asarray([_original])[0]
        if _seq_len < len(target):
            print("Uh-Oh, corrupted file !!")
            print(line[1])
            # print(line[3])
            # print(len(target))
            # print(_seq_len)
            inputs_batch.append(inputs_batch[-1])
            targets.append(targets[-1])
            seq_len.append(seq_len[-1])
            original.append(original[-1])
        else:
            inputs_batch.append(data_in)
            targets.append(target)
            seq_len.append(_seq_len)
            original.append(_original)
            if _seq_len > maxlen:
                maxlen = _seq_len
        # original = np.array([len(original)])

    # Pad the inputs to the maxlen with 0s
    for i in range(batch_size):
        inputs_batch[i] = np.pad(inputs_batch[i], ((0, maxlen - len(inputs_batch[i])), (0, 0)), mode='constant',
                                 constant_values=0)

    train_inputs = np.asarray(inputs_batch)
    train_seq_len = np.asarray(seq_len)
    # Creating sparse representation to feed the placeholder
    train_targets = sparse_tuple_from(targets)
    # print("Done reading batch")
    return train_inputs, train_targets, train_seq_len, original


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def encode(target_txt, encode_dic):
    """Turn text into index."""
    original = ' '.join(target_txt.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',',
                                                                                                         '').replace(
        "'", '').replace('!', '').replace('-', '').replace('\t', '').replace(')', '').replace('"', '')
    targets = original.replace(' ', '  ')
    targets = targets.split(' ')

    # Adding blank label
    targets = np.hstack(["<space>" if x == '' else list(x) for x in targets])
    # Transform char into index
    targets = np.asarray([encode_dic[x] for x in targets])
    return targets, original


def get_key(val, encode_dic):
    for key, value in encode_dic.items():
        if val == value:
            return key
    return ''


def decode(str_encoded, dic):
    str_decoded = ''.join([get_key(x, dic["encode"]) for x in str_encoded])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(get_key(dic["char_num"], dic["encode"]), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(get_key(0, dic["encode"]), ' ')

    return str_decoded
