#2021/11/10 20:45 기준 USB AUDIO DEVICE 의 INDEX는 2로 설정되어있음
# DEVICE: USB Audio Device: - (hw:2,0)  INDEX:  2  RATE:  44100


import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=pyaudio.paInt16, 
                channels=CHANNELS, 
                rate=RATE, 
                input=True, 
                input_device_index=2,
                frames_per_buffer=CHUNK)

print ("recording...")
frames = []


for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print ("finished recording")
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()