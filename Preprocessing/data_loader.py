import os
import shutil
import urllib
import zipfile
import glob
import urllib.request
import time
import pandas as pd
from collections import Counter
import librosa 
import numpy as np
import IPython.display
import random

def download_dataset(name):
    if not os.path.exists(f'./data'):
        os.mkdir('data')
    os.chdir('./data')
    """Download the dataset into current working directory.
    The labeled dataset is ESC-50, the unlabeld are ESC-US-00,ESC-US-01, ... , ESC-US-25 
    but I'm not able to download them automatically from https://dataverse.harvard.edu/dataverse/karol-piczak?q=&types=files&sort=dateSort&order=desc&page=1"""

    if name=='ESC-50' and not os.path.exists(f'./{name}'):

        if not os.path.exists(f'./{name}-master.zip') and not os.path.exists(f'./{name}-master'):
            urllib.request.urlretrieve(f'https://github.com/karoldvl/{name}/archive/master.zip', f'{name}-master.zip')

        if not os.path.exists(f'./{name}-master'):
            with zipfile.ZipFile(f'{name}-master.zip','r') as package:
                package.extractall(f'{name}-master')

        os.remove(f'{name}-master.zip') 
        original = f'./{name}-master/{name}-master/audio'
        target = f'./{name}'
        shutil.move(original,target)
        original = f'./{name}-master/{name}-master/meta'
        target = f'./meta'
        shutil.move(original,target)

    if os.path.exists(f'./{name}-master'):
        shutil.rmtree(f'./{name}-master')

    else:
        print('donwload it from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT&version=2.0')
        pass 
    os.chdir('../')

def main():
    download_dataset('ESC-50')
if __name__=='__main__':
    main() # Call main() if this module is run, but not when imported.



def load_metadata(main_dir, heads = True, statistics = False, audio_listen = False, ESC50=True, ESC10=True, ESC_US=False):
    dir_path = os.path.join(main_dir, 'data', 'ESC-50')
    audio_files = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]

    #load and explore the metadata
    file_path = os.path.join(main_dir, 'data', 'meta', 'esc50.csv')
    
    if ESC50:
        df_ESC50 = pd.read_csv(file_path)
        df_ESC50['full_path'] = df_ESC50.filename.apply(lambda x: os.path.join(dir_path, x))
    if ESC10:
        if not ESC50:
            df_ESC50 = pd.read_csv(file_path)
        df_ESC50['full_path'] = df_ESC50.filename.apply(lambda x: os.path.join(dir_path, x))
        df_ESC10 = df_ESC50[df_ESC50.esc10].drop('esc10', axis=1)   


    if heads:
        display(df_ESC50.head())
        print('Classes in the full dataset  are perfectly balanced\n',Counter(df_ESC50.category)) #classes are perfectly balanced
    
        # 'target' is a number representing the audio type 
        #category of the reduced dataset ESC-10
    
        display(df_ESC10.head())
        classes_esc10 = list(set(df_ESC10.category))
        print('Classes in ESC10 \n',classes_esc10)

    if statistics:
        #auxiliary objects
        sample_rates = set()
        clip_length = set()
        stat_list = np.zeros((len(audio_files),4))

        # let's have a look also over the copmuting time 
        start_time = time.time()
        for i,clip in enumerate(audio_files):
            data, samplerate = librosa.load(clip,sr=44100)
            #samplerate, data = wavfile.read(clip) 
            sample_rates.add(samplerate)
            clip_length.add(len(data))
            stat_list[i,:]=np.asarray([np.min(data),np.max(data),np.mean(data),np.std(data)])
            #the values are all between -1 and 1
        
        print('')
        print(f"librosa takes : {time.time()-start_time}")
        print(f"the lengths are {clip_length}")
        print(f"the sample rates are {sample_rates}")

    if audio_listen:
        if not ESC10:
            ESC10 = load_metadata(main_dir,ESC10=True,ESC50=False)
        #let's listen to one sample for each esc10 classes
        for audio_type in classes_esc10:
            clip = list(df_ESC10.full_path[df_ESC10.category==audio_type])[0]
            data, samplerate = librosa.load(clip,sr=44100)
            print(audio_type)
            display(IPython.display.Audio(data = data, rate=samplerate)  )

    if ESC_US:
        file_path = os.path.join(main_dir, 'data', 'meta', 'ESC-US.csv') #this csv file is useless since has no reference to the files
        ESC_US_paths = os.path.join(main_dir,'data','ESC-US')
        tot = len(os.listdir(ESC_US_paths))
        df_ESC_US = pd.DataFrame(columns=['filename','full_path'])

        for i,folder in enumerate(os.listdir(ESC_US_paths)):
            
            print(f'Loading the {i+1}/{tot} folder of unlabeled data ')
            folder_path = os.path.join(ESC_US_paths,folder)
            files = os.listdir(folder_path)
            full_path_files = [os.path.join(folder_path,f) for f in files]
            d = pd.DataFrame((files,full_path_files), index = ['filename','full_path']).transpose()
            df_ESC_US = pd.concat([df_ESC_US,d])
        if heads:
            print(f'We have {np.max(np.shape(df_ESC_US))} unlabeled audios.')
            display(df_ESC_US.head())

    if ESC50 and not ESC10 and not ESC_US:
        return df_ESC50
    elif ESC10 and not ESC50 and not ESC_US:
        return df_ESC10
    elif ESC10 and ESC50 and not ESC_US:
        return df_ESC10,df_ESC50
    elif not ESC10 and not ESC50 and ESC_US:
        return ESC_US
    elif ESC10 and ESC50 and ESC_US:
        return df_ESC10,df_ESC50, df_ESC_US
    

