import tensorflow as tf
import pdb

from frozen_graph import model_frozen
from vggish import mel_features
from vggish import vggish_params


import numpy as np
import resampy
from scipy.io import wavfile
import six


def wavfile_to_examples(wav_file):
    """Converting the waveform in to mel psectrum

    """

    sr, wav_data = wavfile.read(wav_file) 
    print(("SR, {}".format(sr)))
    print(("wav_data, {}, and shape is {}".format(wav_data, wav_data.shape)))
    print("max element in wav_data is {}".format(np.amax(wav_data)))


    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]


    data=wav_data
    sample_rate=sr

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=vggish_params.SAMPLE_RATE,
      log_offset=vggish_params.LOG_OFFSET,
      window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=vggish_params.NUM_MEL_BINS,
      lower_edge_hertz=vggish_params.MEL_MIN_HZ,
      upper_edge_hertz=vggish_params.MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
      vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
      vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
    
    return log_mel_examples


def test_from_frozen_graph(model_filepath,mel_features):

    tf.reset_default_graph()


    '''
	First start by loading the data (Waveforms should be convertend to STFT)
 
    cifar10 = CIFAR10()
    x_test = cifar10.x_test
    y_test = cifar10.y_test
    y_test_onehot = cifar10.y_test_onehot
    num_classes = cifar10.num_classes
    input_size = cifar10.input_size
    '''


    #get an array for input data and labels to check the model
    
    # x_test = x_test[0:500]
    # y_test = y_test[0:500]

    x_test=mel_features

    model = model_frozen(model_filepath = model_filepath)



    test_prediction_onehot = model.test(data = x_test)
    #test_prediction = np.argmax(test_prediction_onehot, axis = 1).reshape((-1,1))
    # test_accuracy = model_accuracy(label = y_test, prediction = test_prediction)

    # print('Test Accuracy: %f' % test_accuracy)



if __name__ == "__main__":


    model_filepath='./frozen-graph-CENA.pb'

    mel_features=wavfile_to_examples('./data/wav/crowd.wav')  # mel features size 4(seconds?) x 96 x 64   # 13230-0-0-1.wav  # 16772-8-0-0.wav 2 x 96 x 64
                                                              # mel features returns number of secinds x 96 x 64
    print("Mel_features shape {}".format(mel_features.shape))
    pdb.set_trace()

    test_from_frozen_graph(model_filepath,mel_features)