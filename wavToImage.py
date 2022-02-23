

import time

sstart = time.time()

from pydub import AudioSegment, audio_segment
from pydub.utils import make_chunks

print('pydub import time ',time.time() - sstart)

import librosa
import librosa.display

print('librosa import time : ',time.time() - sstart)
import numpy as np
import matplotlib.pyplot as plt
import os

print('import time : ',time.time() - sstart)

### user folder
sub_dir = '001' # user number
mode = 'mel_test' #mel_test / mel_data ONLY ONE!!! NO OTHER THINGS!!!

## code to make Mel_S function
frame_length = 0.025
frame_stride = 0.010


start = time.time()

def Mel_S(wav_file, i):
    st = time.time()
    # mel-spectrogram
    t = time.time()
    y, sr = librosa.load(wav_file, sr=44100)

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    print('mel_s mid : ',time.time() - t)

    S = librosa.feature.melspectrogram(y=y, n_mels=300, n_fft=input_nfft, hop_length=input_stride)

    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))


    plt.figure(figsize=(10, 4))

    print('mel_s mid2 : ',time.time() - t)

    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    #plt.title('Mel-Spectrogram') #no title needed
    plt.tight_layout()

    print('mel_s mid 3 : ',time.time() - t)

    plt.savefig('./' + mode + '/' + sub_dir + '/' + '{0}.jpg'.format(i))

    print('mel_s final : ',time.time() - t)

    #if you want to make test_dat = mel_test / want to make learn data = melspec
    #save many files with another name -> i added
    #plt.show() #no need to show graphs. 
    print('FULL FUNCTION TIME : ',time.time() - st)
    return S

### Mel_S code end

myaudio = AudioSegment.from_file("./user_wav/" + sub_dir + '/' + "sample5_test.wav" , "wav") 
chunk_length_ms = 500 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

chunk_len = len(chunks)

chunktime = time.time()

for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    #print ("exporting", chunk_name)
    #while():
    #    a = 1
    #    if(a == 11): break
    #    chunk += chunk[i+a]
    chunk.export('./split_wav/' + sub_dir + '/' + chunk_name, format="wav")

#Export all of the individual chunks as wav files

#After exporting file into 0.5 sec, we need to merge it into 1 sec.

combine_dir = 'combine'
print('combine code start')
a = 0
while(True):
    sound1 = AudioSegment.from_wav('./split_wav/' + sub_dir + '/' + 'chunk{0}.wav'.format(a))
    sound2 = AudioSegment.from_wav('./split_wav/' + sub_dir + '/' + 'chunk{0}.wav'.format(a+1))
    

    combine_sound = sound1 + sound2
    print('exporting combine_sound{0}'.format(a))
    combine_sound.export('./' + combine_dir + '/' + sub_dir + '/' + 'comb_snd{0}.wav'.format(a), format = "wav")

    a = a+1

    if(a == chunk_len - 3): 
        break
    
print('wav making time : ',time.time() - chunktime)

#code for make wav 0~1 sec, 0.1~1.1 sec, 0.2~1.2 sec ... 

#now need to load these 1 sec wav to convert into mel-spectrogram
a = 0
comb_dir = './combine/' + sub_dir +'/'
comb_sub_dir = [f for f in os.listdir(comb_dir)]
for i in comb_sub_dir:
    Mel_S(comb_dir + i, a)
    a+= 1

print('convert time : ', time.time() - start)
print('FULL CODE END TIME = ',time.time() - sstart)