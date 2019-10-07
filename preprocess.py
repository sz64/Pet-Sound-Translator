#coding=utf-8
import sys
import os

import wavy
import struct
import librosa

import numpy as np
import imageio
from scipy import signal

PATH = ['/media/sf_VMfiles/601/project/data/CatSound_Dataset/Test/', ] # absolute path to the audio dataset
MEL_SAVE_PATH = ['/media/sf_VMfiles/601/project/feature/mel/Test/', ]  # absolute path to the mel bank feature storage
#mfcc_save_path = '/home/zhang/dataset/train/music_mfcc/'  # 保存梅尔倒谱的路径

# splite 1000ms into 100ms/seg，overlap 75ms
LFRAME = 4410
OVERLAP = 2*LFRAME//4

def generate_mel(path, mel_save_path):
  ids = next(os.walk(path))[2]  # next() will return a tuple (dir address，dir list, file list)
  #print(len(ids))
  for id_ in ids:
    print(id_)
    path_ = os.path.join(path, id_)
    wf = wavy.read(path_)
    wave_data = wf.data
    framerate = wf.framerate # sample rate
    sample_width = wf.sample_width # Bytes/frame
    #wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = (-1, 2)
    wave_data = wave_data.T
    print(wf.n_channels)
    if wf.n_channels > 1:
      wave = wave_data[0].astype(float)/2 + wave_data[1].astype(float)/2
    print(wave_data.shape)
    #wav = wave.astype(float)/np.power(2, sample_width*8-1)
    [f, t, X] = signal.spectral.spectrogram(wave, fs=framerate, window=np.hanning(LFRAME), 
                        nperseg=LFRAME, noverlap=OVERLAP, nfft=LFRAME,
                        detrend=False, return_onesided=True, mode='magnitude')
    melW = librosa.filters.mel(sr=44100, n_fft=LFRAME, n_mels=90, fmin=0, fmax=None)
    melW /= np.max(melW, axis=-1)[:, None]
    melX = np.dot(melW, X)
    #log_melX = librosa.core.power_to_db(melX)
    save_path_ = os.path.join(mel_save_path, id_[:-4] + '.png')
    imageio.imwrite(save_path_, melX)

    # melM = librosa.feature.mfcc(sr=44100, S=melX, n_mfcc=16)
    # save_path_ = (mfcc_save_path + id_)[:-4] + '.png'
    # imageio.imwrite(save_path_, melM)

for i in range(1):
  generate_mel(PATH[i], MEL_SAVE_PATH[i])