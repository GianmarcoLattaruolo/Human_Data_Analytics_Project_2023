#libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.io import wavfile
import seaborn as sb
sb.set(style="white", palette="muted")
import pandas as pd
import IPython.display
import random
import os
from scipy import signal
from scipy.fft import fft,ifft

def one_random_audio(main_dir):
    dir_path = os.path.join(main_dir, 'data', 'ESC-50')
    audio_files = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
    i = random.randint(0,len(audio_files))
    clip = audio_files[i]
    samplerate = 44100
    y,sr = librosa.load(clip,sr=samplerate)
    # alternatively we can use wavfile.read
    #samplerate * seconds_clip_audio = length_np_array

    # first look at the output
    print(sr)
    print(type(y))
    print(np.shape(y))

    #load the metadata
    file_path = os.path.join(main_dir, 'data', 'meta', 'esc50.csv')
    meta_data = pd.read_csv(file_path)

    #listen
    print(meta_data.category[meta_data.filename==os.path.basename(os.path.normpath(clip))],'cazzo')
    display( IPython.display.Audio(data = y, rate=samplerate)  )
    plt.subplot(1,1,1)
    plt.plot(y[:1000])

    return y, sr


def power_spectrum_plot(y,sr):
        
    fs = 1/sr
    n = len(y)
    freq_y, pow_spect_y = signal.periodogram(y,fs=fs)
    #it's better to make a logarithmic plot
    log_pow_spec = np.log(pow_spect_y)


    plt.subplots(4, 1, figsize=(9, 10))
    plt.tight_layout(pad=4)

    plt.subplot(4,1,1)
    plt.plot(freq_y,pow_spect_y)
    plt.title("Clip's Periodogram")
    plt.ylabel('Power Spectrum')
    plt.xlabel('frequency [1/second]')
    #it's difficult to carry out a significative graph

    plt.subplot(4,1,2)
    plt.plot(freq_y, log_pow_spec)
    plt.title('Logarithmic Periodogram')
    plt.ylabel('Log-scale')
    plt.xlabel('frequency [1/second]')

    plt.subplot(4,1,3)
    plt.plot(freq_y,pow_spect_y)
    plt.xlim([0.29*1e-5, 0.31*1e-5]) #da modificare per adattarsi ad ogni audio
    plt.title("focus on Clip's periodogram")
    plt.ylabel('Power spectrum')
    plt.xlabel('frequency [1/second]')

    plt.subplot(4,1,4)
    plt.plot(freq_y,log_pow_spec)
    plt.xlim([0.29*1e-5, 0.31*1e-5])
    plt.title("focus on log Clip's periodogram")
    plt.ylabel('log_scale')
    plt.xlabel('frequency [1/second]')


def plot_clip_overview(df, sample_rate=44100, segment=25, overlapping=10, row = 10, column = 5):

    segment_samples = int(sample_rate * segment / 1000)  # Calculate the number of samples per segment
    overlap_samples = int(sample_rate * overlapping / 1000)

    
    plt.subplots(row, column, figsize=(12, 15))
    plt.tight_layout(pad=0.7)
    
    for j,audio_type in enumerate(list(set(df.category))):
        paths = list(df.full_path[df.category == audio_type])
        paths = random.sample(paths,column)

        for i,audio_sample in enumerate(paths):
            data, samplerate = librosa.load(audio_sample,sr=44100)
            D = librosa.stft(data, win_length = segment_samples,hop_length=overlap_samples)  # STFT of y
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            plt.subplot(row,column,j*column+i+1)
            plt.title(audio_type)
            librosa.display.specshow(S_db)