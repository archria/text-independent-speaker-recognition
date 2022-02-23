from pydub import AudioSegment, audio_segment
from pydub.utils import make_chunks
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

### user folder
sub_dir = '005'

## code to make Mel_S function
frame_length = 0.025
frame_stride = 0.010

def Mel_S(wav_file, i):
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
    plt.savefig('C:/Users/jhlee/Desktop/code/mel_test/'+ sub_dir + '/' + '{0}.jpg'.format(i))
    #if you want to make test_dat = mel_test / want to make learn data = melspec
    #save many files with another name -> i added
    #plt.show()

    return S

### Mel_S code end

myaudio = AudioSegment.from_file("C:/Users/jhlee/Desktop/Code/dg_test.wav" , "wav") 
chunk_length_ms = 100 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

chunk_len = len(chunks)



for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    #print ("exporting", chunk_name)
    #while():
    #    a = 1
    #    if(a == 11): break
    #    chunk += chunk[i+a]
    chunk.export('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + chunk_name, format="wav")

#Export all of the individual chunks as wav files

#After exporting file into 0.1 sec, we need to merge it into 1 sec.

combine_dir = 'combine'
print('combine code start')
a = 0
while(True):
    sound1 = AudioSegment.from_wav('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + 'chunk{0}.wav'.format(a))
    sound2 = AudioSegment.from_wav('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + 'chunk{0}.wav'.format(a+1))
    sound3 = AudioSegment.from_wav('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + 'chunk{0}.wav'.format(a+2))
    sound4 = AudioSegment.from_wav('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + 'chunk{0}.wav'.format(a+3))
    sound5 = AudioSegment.from_wav('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + 'chunk{0}.wav'.format(a+4))
    sound6 = AudioSegment.from_wav('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + 'chunk{0}.wav'.format(a+5))
    sound7 = AudioSegment.from_wav('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + 'chunk{0}.wav'.format(a+6))
    sound8 = AudioSegment.from_wav('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + 'chunk{0}.wav'.format(a+7))
    sound9 = AudioSegment.from_wav('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + 'chunk{0}.wav'.format(a+8))
    sound10 = AudioSegment.from_wav('C:/Users/jhlee/Desktop/code/' + sub_dir + '/' + 'chunk{0}.wav'.format(a+9))

    combine_sound = sound1 + sound2 + sound3 + sound4 + sound5 + sound6 + sound7 + sound8 + sound9 + sound10
    print('exporting combine_sound{0}'.format(a))
    combine_sound.export('C:/Users/jhlee/Desktop/code/' + combine_dir + '/' + sub_dir + '/' + 'comb_snd{0}.wav'.format(a), format = "wav")

    a = a+1

    if(a == chunk_len - 11): 
        break
    
#code for make wav 0~1 sec, 0.1~1.1 sec, 0.2~1.2 sec ... 

#now need to load these 1 sec wav to convert into mel-spectrogram
a = 0
comb_dir = 'C:/Users/jhlee/Desktop/code/combine/' + sub_dir +'/'
comb_sub_dir = [f for f in os.listdir(comb_dir)]
for i in comb_sub_dir:
    Mel_S(comb_dir + i, a)
    a+= 1
