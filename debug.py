import sys  
import os
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
from scipy import signal
import tensorflow as tf
import pandas as pd
import os
import librosa
from pathlib import Path




#libraries
main_dir = os.getcwd()
module_path = main_dir
if module_path not in sys.path:
    print('Adding the folder for the modules')
    sys.path.append(module_path)

import importlib
imported_module = importlib.import_module("Preprocessing.data_loader")
importlib.reload(imported_module)
imported_module = importlib.import_module("Preprocessing.exploration_plots")
importlib.reload(imported_module)
imported_module = importlib.import_module("Preprocessing.clip_utils")
importlib.reload(imported_module)
from Preprocessing.data_loader import download_dataset,load_metadata
from Preprocessing.exploration_plots import one_random_audio, power_spectrum_plot, plot_clip_overview


df_ESC10 = load_metadata(main_dir,ESC50 = False, heads= False)


# Define a function to load and preprocess each audio file
def load_audio(path,target):
    path = Path(str(path).replace('\\\\','\\')[2:-1])
    print(path)
    audio, _ = librosa.load(path,sr=44100) 
    audio =  tf.convert_to_tensor(audio)
    return audio, target


# Create a TensorFlow dataset from the audio files and labels

dataset = tf.data.Dataset.from_tensor_slices((df_ESC10.full_path, df_ESC10.target)) 

dataset = dataset.map(load_audio)