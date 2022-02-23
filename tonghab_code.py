import time

start = time.time()

import subprocess
import librosa.display
import shutil
import cv2
import glob
import re #진짜 필요해서 import 한건가?? 
import pymongo
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
from tensorflow.keras.models import load_model
import librosa
from pydub.utils import make_chunks
from pydub import AudioSegment, audio_segment

import threading
import json
import requests

from bson.objectid import ObjectId
import gridfs


print('importing all library time : ', time.time() - start)

#global variables 전역변수들 앞쪽에 몰아서 선언함
data = {}
headers = {}
#res = requests.post(
#    "", json=data, headers=headers)
todays_inputed_wav_number = 0
frame_length = 0.025
frame_stride = 0.010
combine_dir = 'combine'
wav_numbers = 0
last_wav = 0

def Mel_S(wav_file, i):
    st = time.time()
    # mel-spectrogram
    t = time.time()
    y, sr = librosa.load(wav_file, sr=44100)

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(
        y=y, n_mels=300, n_fft=input_nfft, hop_length=input_stride)

    #print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(librosa.power_to_db(
        S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-Spectrogram') #no title needed
    plt.tight_layout()

    # plt.savefig('./' + mode + '/' + sub_dir + '/' + '{0}.jpg'.format(i))
    plt.savefig('./mel_test/' + '{0}.jpg'.format(i))
    # 최종 코드에서는 폴더 구분 없이 저장 -> 변환 -> 추론 순서를 거칠 예정
    # 폴더 고정된 경로 사용할 예정
    # if you want to make test_dat = mel_test / want to make learn data = melspec
    # save many files with another name -> i added
    # plt.show() #no need to show graphs.
    return S

#쪼갠 파일을 1초단위 (0~1, 0.5~1.5, 1~2 ... ) 로 합치는 코드
#전역변수 사용에 문제가 있어 지역변수만 사용하는 split_wav 로 코드 합쳤음
# 더 이상 사용 ㄴㄴ
def merge_wav_into_1sec():
    a = 0
    while(True):
        sound1 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a))
        sound2 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a+1))

        combine_sound = sound1 + sound2 
        print('exporting combine_sound{0}'.format(a))
        combine_sound.export('/home/garin/JH/' + combine_dir + 
                             '/' + 'comb_snd{0}.wav'.format(a), format="wav")

        a = a+1

        if(a == chunk_len - 3):
            break

#wav 파일을 0.5초 단위로 쪼개는 코드
def split_wav_into_halfsec(filename):
    myaudio = AudioSegment.from_file(
        "/home/garin/JH/recorded_wav/" + filename + ".wav", "wav")
    # prepare to check next wav.
    #todays_inputed_wav_number = todays_inputed_wav_number + 1 #no more use
    chunk_length_ms = 500  # pydub calculates2 in millisec
    # Make chunks of chunk_length_ms( 500 = 0.5 ) sec
    chunks = make_chunks(myaudio, chunk_length_ms)

    chunk_len = len(chunks)
    

    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print('exporting', chunk_name)
        chunk.export('/home/garin/JH/split_wav/' + chunk_name, format="wav")
    
    #1초 단위로 합치는 코드
    a = 0
    print(chunk_len)
    chunk_limit = chunk_len - 2
    while(True):
        sound1 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a))
        sound2 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a+1))

        """
        sound3 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a+2))
        sound4 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a+3))
        sound5 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a+4))
        sound6 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a+5))        
        sound7 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a+6))
        sound8 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a+7))
        sound9 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a+8))
        sound10 = AudioSegment.from_wav(
            '/home/garin/JH/split_wav/'  + 'chunk{0}.wav'.format(a+9))

        """

        combine_sound = sound1 + sound2 #+ sound3 + sound4 + sound5 + sound6 + sound7 + sound8 + sound9 + sound10
        print('exporting combine_sound{0}'.format(a))
        combine_sound.export('/home/garin/JH/' + combine_dir + 
                             '/' + 'comb_snd{0}.wav'.format(a), format="wav")

        a = a+1

        if(a == chunk_limit):
            break





# make folder clean to use next time!!

# 다 쓴 폴더를 다음 인식을 위해 정리함
def clean_folder():
    #recorded_wav 폴더 정리

    os.system("rm /home/garin/JH/recorded_wav/*.wav")

    #weba 파일 지우고싶으면 아래 주석 해제 기본적으로 주석 처리해둠
    #os.system("rm /home/garin/JH/recorded_wav/*.weba")

    #combine 폴더 정리
    os.system("rm /home/garin/JH/combine/*.wav")
    #split_wav 폴더 정리
    os.system("rm /home/garin/JH/split_wav/*.wav")
    #mel_test 폴더 정리
    os.system("rm /home/garin/JH/mel_test/*.jpg")

    


# 멜스펙트로그램으로 변환하는 코드
def create_melspec():
    a = 0
    comb_dir = '/home/garin/JH/combine/'
    comb_sub_dir = [f for f in os.listdir(comb_dir)]
    for i in comb_sub_dir:
        Mel_S(comb_dir + i, a)
        a += 1


##########################################################
############ END OF CREATE SPECTROGRAM ###################
############ BELOW IS TENSORFLOW CODE  ###################
############ GUESSING WHO IS THIS      ###################
##########################################################

categories = ["001", "002", "003", "004","005"]  # categories

# 데이터 전처리
def Dataization(img_path):
    image_w = 28
    image_h = 28
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w /
                     img.shape[1], fy=image_h/img.shape[0])
    return (img/256)


src = []
name = []
#test = []
image_dir = "/home/garin/JH/mel_test/"


def Guess_step1():
    test = []
    for file in os.listdir(image_dir):
        if (file.find('.jpg') != -1):

            src.append(image_dir + file)
            name.append(file)
            test.append(Dataization(image_dir + file))

    test = np.array(test)
    model = load_model('./myFirstCNN.h5')
    start = time.time()
    predict_x = model.predict(test)
    classes_x = np.argmax(predict_x, axis=1)
    #print(predict_x)
    print(classes_x)

    answer = [0 for i in range(200)]
    people_num = 0
    max_hit_user = 0
    max_hit_count = 0
    print('PRINTING PREDICT DATA')
    for i in range(len(test)):
        print(classes_x[i])
        answer[classes_x[i]] = answer[classes_x[i]] + 1
        if(classes_x[i] > people_num):
            people_num = classes_x[i]
    print('USER PREDICT DATA')
    # 유저 판별된거 알려주기
    for i in range(people_num+1):
        print(answer[i])
        if(answer[i] > max_hit_count):
            max_hit_user = i
            max_hit_count = answer[i]
    print(answer)
    print(max_hit_user+1,"번 사용자 음성입니다!!!")
    data = { "data" : max_hit_user+1 }
    headers = {}
    res = requests.post("****************", json = data, headers = headers)

    print(res.json())

    
##########################################################
############ END OF TENSORFLOW CODE    ###################
############ BELOW IS CONNECTING TO DB ###################
########## CONNECT TO TO DB AND GET SIGNAL ###############
##########################################################


# dbcheck code is check DB's special item.
# if special items says "we need to check new senetnece"
# then all of algorithm starts
# else? check every 0.x sec. depends on RPi spec. (0.1 sec for RPi4 B 8GB RAM)

db = pymongo.MongoClient("******************").garin
def dbcheck():
    global last_wav
    max = db.audios.find_one(sort=[("audioName", pymongo.DESCENDING)])
    length = db.audios.find({}).count()
    if(length == 0):
        print("NO ITEM")
        
    
    #새 음성이 들어왔을 경우에?
    elif(max["audioName"] != last_wav): 
        print( max["audioName"] ) # audioName값만 출력
        print("NEW FILE INPUT!!!")
        new_filename = ''
        new_filename = str(max["audioName"]) # NEW FILE 의 filename을 받아들임
        last_wav = max["audioName"]
        #########################################################
        ######### 짜야할 기능 : 1) 디비에 접속 
        #########               2) 전역변수로 선언된 last_wav와 현재 DB에 등록된 값이 동일한지 파악 
        #########               3) 다를경우 화자인식 프로세스 진행
        #########################################################
        weba_to_wav(new_filename)
        split_wav_into_halfsec(new_filename)
        #merge_wav_into_1sec()
        create_melspec()
        
        Guess_step1()
        clean_folder()

    threading.Timer(0.1, dbcheck).start()  # 0.1초마다 반복실행

def bootup_process():
    global last_wav
    last_wav = 0

### filename = 1,2,3 ... 확장자 X 오직 파일 "이름만" 들어옴
def weba_to_wav(filename):
    #copy file to 
    os.system("cp /home/garin/project-garin/record/"+filename+ " /home/garin/JH/recorded_wav/"+filename+".weba")
    #convert weba into wav
    os.system("ffmpeg -i " + "/home/garin/JH/recorded_wav/" + filename + ".weba -ar 44100 /home/garin/JH/recorded_wav/" +filename+".wav")

def start_button():
    dbcheck()

def test():
    start = time.time()
    #파일명
    new_filename = '1'
    weba_to_wav(new_filename)
    print('weba to wav end')
    split_wav_into_halfsec(new_filename)
    print('split wav into halfsec end')
    #merge_wav_into_1sec()
    print('merge end')
    create_melspec()
    print('create melspec end')
    Guess_step1()
    print('guess step end')
    #connect()
    
    clean_folder()

    print('end time : ', time.time() - start)

#로직 작동하는지 확인만 하고싶다면 test() 사용
# test()

#os.system("ffmpeg -i " + '1' + ".weba -ar 44100 " + '1' + ".wav")

#라이브서비스 용도로 파일 켜고싶다면 아래 두개 주석 해제
bootup_process()
dbcheck()

