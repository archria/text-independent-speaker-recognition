# -*- coding: utf-8 -*-
from re import A, sub
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

frame_length = 0.025
frame_stride = 0.010
    
def Mel_S(wav_file):
    # mel-spectrogram
    y, sr = librosa.load(wav_file, sr=44100)

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=300, n_fft=input_nfft, hop_length=input_stride)

    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))


    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    #plt.title('Mel-Spectrogram') #no plt needed
    plt.tight_layout()
    plt.savefig('Mel-Spectrogram{0}.png'.format(i))
    #save many files with another name -> i added
    plt.show()

    return S

man_original_data = 'C:/Users/jhlee/Desktop/code/SoundClassificationWav/003/3_ (3).wav'
mel_spec = Mel_S(man_original_data)

wav_cat = 'C:/Users/jhlee/Desktop/code/SoundClassificationWav/'
sub_dir = [f for f in os.listdir(wav_cat)]
for i in sub_dir:
    print(wav_cat + i)
print('test')
print(sub_dir)
