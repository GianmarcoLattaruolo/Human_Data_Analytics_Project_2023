import numpy as np
import librosa
from scipy.io import wavfile

class Clip():
    """A single 5-sec long recording."""
    
    samplerate = 44100   # All recordings in ESC are 44.1 kHz
    frame = 512    # Frame size in samples
    
    def __init__(self, path):
        self.path = path

    #cosa vogliamo che faccia la classe che loada la singola clip audio?

    

samplerate, data = wavfile.read(r'C:\Users\latta\GitHub\Human_Data_Analytics_Project_2023\data\ESC-50\1-137-A-32.wav')
print(samplerate)
print(type(data))
