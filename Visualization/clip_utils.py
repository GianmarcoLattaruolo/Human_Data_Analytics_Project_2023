import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import stft,spectrogram
from scipy.fft import fft,ifft


def segmentation(data, sample_rate=44100, segment=25, overlapping=10):
    #segment and overlapping are expressed in milliseconds
    if type(data)=='str':
        data, sample_rate = librosa.load(data,sr=sample_rate)
        
    N_tot = len(data)
    N = round(sample_rate * segment / 1000)  # Calculate the number of samples per segment
    overlap = round(sample_rate * overlapping / 1000)  # Calculate the number of overlapping samples
    
    total_segments = (N_tot - N) / (N - overlap) +1
    if not int(total_segments)==total_segments:
        zeros_to_add = ((N_tot-N)//(N-overlap)+1)*(N-overlap)-N_tot+N
        data = np.append(data,np.zeros((zeros_to_add,1)))
        total_segments = (N_tot-N)//(N-overlap)+2
    else:
        total_segments = int(total_segments)

    frames = np.zeros((total_segments, N))  # Initialize an array to store the segmented data
    start = 0  # Starting index for each segment

    for i in range(total_segments):
        frames[i] = data[start:start + N]  # Assign the segment of data to the array
        start += N - overlap  # Move the starting index forward with the overlap
    
    return frames, N, total_segments


M = lambda x: 2595 * np.log(1+x/700) # Mel Scale
M_inverse = lambda y:(np.exp(y/2595)-1)*700 # Inverse of Mel Scale
alpha = lambda u,N: N**(-0.5)*(u==0)+(2/N)**0.5*(u!=0)

def hamming_window(n,N):
    return 0.54-0.46*np.cos(2*np.pi*n/(N-1))

def DFT(frame): #discrete fourier transform
    N = len(frame)
    #S = np.zeros((1,N),dtype='complex')
    windowed_frame = [frame[n]*hamming_window(n,N) for n in range(N)]
    S = fft(windowed_frame)
    #for k in range(N):
    #    S[0,k]=np.sum([frame[n]*hamming_window(n,N)*np.exp(-2*np.pi*k*n*1j/N) for n in range(N)])
    return S

def DCT(vec):
    N = len(vec)
    C = np.zeros((1,N))
    for u in range(N):
        C[0,u] = alpha(u,N)*np.sum([vec[n]*np.cos(np.pi*(2*n+1)*u/(2*N)) for n in range(N)])

    return C

def Delta(vec,M):
    if not M%2==0:
        print('M is not even!')
        return
    else:
        N = len(vec)
        vec = np.pad(vec, (M, M), 'constant', constant_values=(0, 0))
        delta = np.zeros((N,1))
        den = M*(M+1)*(2*M+1)/3
        for i in range(N):
            delta[i]=np.sum([ m*(vec[M+i+m]-vec[M+i-m]) for m in range(1,M+1)])/den

        return delta

def triangular(f1,f2,f3,value_in):

    if value_in<f1 or value_in>f3:
        return 0
    elif value_in<=f2:
        return (value_in-f1)/(f2-f1)
    elif value_in>f2:
        return (f3-value_in)/(f3-f2)

def Mel_filterbank(f_min,f_max,N_filters,N,sr):
    equispaced = np.linspace(M(f_min),M(f_max),num=N_filters+2)
    frequencies = np.asarray(list(map(M_inverse,equispaced)))
    N_half = (N+1)//2
    filters = np.zeros((N_filters,N_half))
    for m in range(N_filters):
        f1,f2,f3 = frequencies[m:m+3]
        filters[m] = np.asarray([triangular(f1,f2,f3,sr/N*i) for i in range(1,N_half+1)])

    return filters

def MFCC(audio, cepstral_num = 20, 
                sr=44100, 
                segment=25, 
                overlapping=10, 
                f_min = 20, 
                N_filters=50,
                energy_feature = False,
                delta_feature = True,
                M = 2,
                delta_delta_feature = True ): #Mel-frequency cepstral coefficients 
    if type(audio)==type('abc'):
        data, sample_rate = librosa.load(audio,sr=sr)
    N_total = len(data)
    # step 1: frame the audio
    frames, N, n_frames = segmentation(data, sample_rate=sr, segment=segment, overlapping=overlapping)

    #define the filterbank once for all
    f_max = sr/2 #Nyquist Critical Frequency
    if f_min<sr/N:
        f_min=sr/N
    filters = Mel_filterbank(f_min,f_max,N_filters,N,sr)

    feature_vectors = []
    for i,frame in enumerate(frames):
        #step 2: power spectrum
        power_spectrum = DFT(frame)
        periodogram = np.absolute(power_spectrum)**2/2
        periodogram = periodogram[:(N+1)//2]

        #step 3: filterbank
        energy = np.dot(filters,periodogram.reshape((N+1)//2,1))

        #step 4: logarithms
        log_energy = np.log(energy)

        #step 5: discrete cosine transform
        cepstral = DCT(log_energy)

        #step 6: cepstral coefficients
        if cepstral.shape[1]<cepstral_num-1:
            cepstral_num = cepstral.shape[1]-1
        full_feature_vector = cepstral[0,1:cepstral_num+1]

        #additional features:

        if delta_feature:
            delta = Delta(full_feature_vector,M)
            full_feature_vector = np.concatenate((full_feature_vector,delta.reshape(-1)),axis=0)
        if delta_delta_feature:
            delta_delta = Delta(delta,M)
            full_feature_vector = np.concatenate((full_feature_vector, delta_delta.reshape(-1)),axis=0)
        if energy_feature:
            E_tot = np.log10(np.sum(frame**2)).reshape((1,))
            full_feature_vector = np.concatenate((E_tot,full_feature_vector),axis = 0)

        feature_vectors.append(full_feature_vector)

    return np.asarray(feature_vectors)


# CODE TO CONVERT THE OGG FILES. NO MORE REQUIRED

#FFMPEG WAY

def is_folder_empty(folder_path):
    # INPUT: str of the folder path
    # OUTPUT: TRUE / FALSE if the folder is empty 
    return len(os.listdir(folder_path)) == 0

if '01_conv' not in os.listdir(os.path.join(main_dir,'data','ESC-US')):
    os.mkdir(os.path.join(main_dir,'data','ESC-US','01_conv'))

path_input = os.path.join(main_dir,'data','ESC-US','01')
path_output = os.path.join(main_dir,'data','ESC-US','01_conv')

# Get a list of all files and directories in the specified directory
files_in = os.listdir(path_input)
files_out = os.listdir(path_output)
files_check = [file[:-3] + "wav" for file in files_in if file[:-3] + "wav" not in files_out]

def convert_ogg_to_wav(input_file, output_file):
    # INPUT: input_file = str path of the input file .ogg we want to convert, output_file = path of the output file .wav we want ot create
    # For this function to work you need the ffmpeg program installed on your computer
    command = ['ffmpeg', '-i', input_file, output_file]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for file in files_check:
    input_file = os.path.join(main_dir,'data','ESC-US','01',file[:-3]+'ogg')
    output_file = os.path.join(main_dir,'data','ESC-US','01_conv',file)
    #!ffmpeg -i {input_file} {output_file}
    convert_ogg_to_wav(input_file, output_file)

number_files = len(os.listdir(path_output))

#PYDUB WAY (FORCED IN COLAB)
if '02_conv' not in os.listdir(os.path.join(main_dir,'data','ESC-US')):
    os.mkdir(os.path.join(main_dir,'data','ESC-US','02_conv'))

path_input = os.path.join(main_dir,'data','ESC-US','02')
path_output = os.path.join(main_dir,'data','ESC-US','02_conv')

# Get a list of all files and directories in the specified directory
files_in = os.listdir(path_input)
files_out = os.listdir(path_output)
files_check = [file[:-3] + "wav" for file in files_in if file[:-3] + "wav" not in files_out]

for file in files_check:
    input_file = os.path.join(main_dir,'data','ESC-US','02',file[:-3]+'ogg')
    output_file = os.path.join(main_dir,'data','ESC-US','02_conv',file)
    x = AudioSegment.from_file(input_file)
    x.export(output_file, format='wav') 


number_files = len(os.listdir(path_output))

#COMMENTI da fare
def batch_training(main_dir, dataset_size = 1000, delete = True, shuffle = True ): 
    #mancano i parametri per inserire il modello
    data_dir = os.path.join(main_dir,'data','ESC-US')

    list_dir = os.listdir(data_dir)
    list_path = []

    for folder in list_dir:
        folder_path = os.path.join(data_dir,folder)
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file[-3:]=='ogg'  ]
        list_path.extend(files)


    if 'temp_conv' not in os.listdir(data_dir):
        os.mkdir(os.path.join(data_dir,'temp_conv'))

    num_batch = len(list_path)//dataset_size
    print(len(list_path))
    while len(list_path)>dataset_size:
        if shuffle:
            data = random.sample(list_path, dataset_size)
        else:
            data = list_path[:dataset_size]
        list_path = [p for p in list_path if p not in data]
        print(len(list_path))

        for input_file in data:
            out_file_name = input_file.split('\\')[-1:][0].replace('ogg','wav')
            output_file = os.path.join(data_dir,'temp_conv',out_file_name)
            x = AudioSegment.from_file(input_file)
            x.export(output_file, format='wav') 


        #TRAIN THE MODEL ON THE temp_conv DIR

        if delete:
            temp_files = os.listdir(os.path.join(data_dir,'temp_conv' ))
            temp_files = [os.path.join(data_dir, 'temp_conv', file) for file in temp_files]
            for temp in temp_files:
                os.remove(temp)

    return 



'''REMOVED CELLS

#subfolder_path = os.path.join(main_dir,'data','ESC-10-depth')
subfolder_path = os.path.join(main_dir,'data','ESC-US')
batch_size = 64
validation_split = 0.25 
normalize = True
preprocessing = None # "STFT", "MEL", "MFCC"
labels = None # None of 'inferred'

train, val, test = create_dataset(subfolder_path, 
                      batch_size = batch_size, 
                      shuffle = True, 
                      validation_split = validation_split, 
                      cache_file_train = None, 
                      cache_file_val = None, 
                      cache_file_test = None, 
                      normalize = normalize,
                      preprocessing = preprocessing,
                      delta = True,
                      delta_delta = True, 
                      labels = labels) 

#show the first element of the dataset train 
for element in train.take(1).unbatch():
    print(element[0].shape)
    #print(element[1].shape)
    print(element[0])
    #print(element[1])
    break

    

    # Duplicate data for the autoencoder (input = output)
py_funct = lambda audio: (audio, audio)
train = train.map(py_funct)
val = val.map(py_funct)
test = test.map(py_funct)

#show the first element of the dataset train 
for element in train.take(1).unbatch():
    print(element[0].shape)
    print(type(element[0]))
    #print(element[1].shape)
    print(element[0])
    #print(element[1])
    #print((element[0] == element[1]).numpy().all())
    break
    

#calculate the number of elements in each dataset 
num_elements_train = round( number_files * (1 - validation_split))
num_elements_val = round(number_files *  validation_split / 2)
num_elements_test = round(number_files * validation_split / 2)

# calculate the learning steps required, the problem is that the tf dataset is stored as an infinite dataset 
train_steps = num_elements_train // batch_size
val_steps = num_elements_val // batch_size
test_steps = num_elements_test // batch_size

print("Train steps required: ", train_steps)
print("Val steps required: ", val_steps)
print("Test steps required: ", test_steps)




def hear(audio, encoder, decoder):
    audio = tf.reshape(audio, (1, -1))  # Reshape audio to match the expected shape
    code = encoder.predict(audio) 
    reco = decoder.predict(code)
    display(ipd.Audio(data = audio.numpy(), rate=41000))
    display(ipd.Audio(data = reco, rate=41000))

for element in train.take(1).unbatch():
    hear(element[0], encoder, decoder)
    break


THIS CELL HAS BEEN SIMPLY WRITTEN DIFFERENTLY 


def build_deep_autoencoder(img_shape, code_size):
    """
    Arguments:
    img_shape -- size of the input layer
    code_size -- the size of the hidden representation of the input (code)

    Returns:
    encoder -- keras model for the encoder network
    decoder -- keras model for the decoder network
    """

    # encoder
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.Input(img_shape))

    encoder.add(layers.Conv2D(32, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Conv2D(64, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Conv2D(128, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Conv2D(256, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(code_size))

    # decoder
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.Input((code_size,)))

    decoder.add(layers.Dense(27*31*256, activation='elu'))
    decoder.add(layers.Reshape((27, 31, 256)))
    decoder.add(layers.Conv2DTranspose(128, (3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(tf.keras.layers.Lambda(lambda x: tf.pad(x, paddings=tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]]), mode='CONSTANT')))
    decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(tf.keras.layers.Lambda(lambda x: tf.pad(x, paddings=tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]), mode='CONSTANT')))
    decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(tf.keras.layers.Lambda(lambda x: tf.pad(x, paddings=tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]]), mode='CONSTANT')))
    decoder.add(layers.Conv2DTranspose(1, (3, 3), strides=2, activation=None, padding='same'))
    decoder.add(tf.keras.layers.Lambda(lambda x: tf.pad(x, paddings=tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]]), mode='CONSTANT')))

    return encoder, decoder


#calculate the number of elements in each dataset
num_elements_train = round( number_files * (1 - validation_split))
num_elements_val = round(number_files *  validation_split / 2)
num_elements_test = round(number_files * validation_split / 2)

# calculate the learning steps required, the problem is that the tf dataset is stored as an infinite dataset
train_steps = num_elements_train // batch_size
val_steps = num_elements_val // batch_size
test_steps = num_elements_test // batch_size

print("Train steps required: ", train_steps)
print("Val steps required: ", val_steps)
print("Test steps required: ", test_steps)


def create_dataset(subfolder_path, # folder of the audio data we want to import
                   verbosity = 1, # 0, 1 level of verbosity
                   batch_size = 30,  
                   shuffle = True, 
                   validation_split = 0.25, # this is the splitting of train vs validation + test
                   cache_file_train = None, # str path of the chaching file 
                   cache_file_val = None, 
                   cache_file_test = None, 
                   normalize = True, # normalization preprocessing 
                   num_repeat = None, # number of times the dataset is repeated
                   preprocessing = None,  # "STFT", "MEL", "MFCC"
                   delta = True,  # True or False only if preprocessing = "MFCC"
                   delta_delta = True,  #  True or False only if preprocessing = "MFCC"
                   labels = "inferred", # labels = 'inferred' or None (for unsupervised learning)
                   resize = False, # we can resize the images generated by the preprocessing
                   new_height = 64,
                   new_width = 128): 
                   # INPUT: str - path of the audio directory 
                   # OUTPUT: train, validation, test set as tensorflow dataset

    if verbosity > 0:
        print("Creating dataset from folder 3: ", subfolder_path)
    
    #auxiliary functions

    # @tf.autograph.experimental.do_not_convert
    def squeeze(audio, labels=None):
        # INPUT: audio as a tf dataset
        # OUTPUT: audio + labels or only audio
        # used to remove a dimension from the tensor
        if audio.shape[-1] is None:
            audio = tf.squeeze(audio, axis=-1) 
        if labels is not None:
            return audio, labels
        else:
            return audio
  
    @tf.autograph.experimental.do_not_convert
    def spectral_preprocessing_audio(audio, 
                                     sample_rate = 44100, 
                                     segment = 20, 
                                     n_fft = None, #padd the frames with zeros before DFT
                                     overlapping = 10, 
                                     cepstral_num = 40, 
                                     N_filters = 50, 
                                     preprocessing = preprocessing,
                                     delta = delta, 
                                     delta_delta = delta_delta):
                                     # INPUT: audio as a tensorflow object
                                     # OUTPUT: preprocessed audio with STFT, MEL, MFCC as desired 
                                     # used to perform the spectral preprocessing (STFT, MEL, MFCC with/out delta, delta-delta)

        

                            
        audio = audio.numpy()

        # transform the segment and overlapping from ms to samples
        if n_fft is None:
            n_fft = segment
        nperseg = round(sample_rate * segment / 1000)
        noverlap = round(sample_rate * overlapping / 1000)
        n_fft = round(sample_rate * n_fft / 1000)
        hop_length = nperseg - noverlap
        r = None

        # using librosa to perform the preprocessing
        if preprocessing == "STFT":
            stft_librosa = librosa.stft(audio, hop_length=hop_length, win_length=nperseg, n_fft=n_fft)
            r = librosa.amplitude_to_db(np.abs(stft_librosa), ref=np.max)

        elif preprocessing == "MEL":
            mel_y = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                                   win_length=nperseg) 
            r = librosa.power_to_db(mel_y, ref=np.max)
        elif preprocessing == "MFCC":
            mfcc_y = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=cepstral_num, n_fft=n_fft,
                                          hop_length=hop_length, htk=True, fmin=40, n_mels=N_filters)

            # now we calculate the delta and delta-delta if needed
            if delta:
                delta_mfccs = librosa.feature.delta(mfcc_y)
                if delta_delta:
                    delta2_mfccs = librosa.feature.delta(mfcc_y, order=2)
                if delta and not delta_delta:
                    mfccs_features = np.concatenate((mfcc_y, delta_mfccs))
                elif delta and delta_delta:
                    mfccs_features = np.concatenate((mfcc_y, delta_mfccs, delta2_mfccs))
            if not delta and not delta_delta:
                mfccs_features = mfcc_y
            r = mfccs_features

        return r
    
    if not resize:
        print("You are not resizing the images")
        new_height = None
        new_width = None
    else:
        print("You are resizing the images")
    
    @tf.autograph.experimental.do_not_convert
    def reshape_tensor(matrix, new_height = new_height, new_width = new_width):
        # INPUT: the audio preprocessed by STFT, MEL or MFCC
        # OUTPUT: the audio reshaped as a tensor (needed for the convolutional layers)
        matrix = tf.squeeze(matrix)
        tensor = tf.expand_dims(matrix, axis=2)
        return tensor
    
    # @tf.autograph.experimental.do_not_convert
    def find_max_lazy(train):
        max = 0
        lazy_number = 0
        for batch, label in train:
            if lazy_number<4:
                new = tf.reduce_max(tf.abs(batch)) 
                if new > max:
                    max = new
            lazy_number+=1
        if verbosity > 0:
            print(f'The max value is {max}')
        return max
    
    # @tf.autograph.experimental.do_not_convert
    def normalize_map(matrix, max):
        matrix = matrix/max
        return matrix

    @tf.autograph.experimental.do_not_convert
    def reshape_images_map(image, new_height = new_height, new_width = new_width):

        image = tf.image.resize(image, [new_height, new_width])
        return image

    # creation of the tf dataset from an audio folder 
    if labels == None:
        label_mode = None
    else:
        label_mode = 'categorical'
    train, val_test = tf.keras.utils.audio_dataset_from_directory(
        directory =subfolder_path.replace('\\','/'),
        labels=labels, 
        label_mode=label_mode,
        class_names=None,
        batch_size=None,
        sampling_rate=None,
        output_sequence_length=220500,
        ragged=False,
        shuffle=shuffle,
        seed=42,
        validation_split=validation_split,
        subset='both',
    )

    if labels is not None:
        label_names = np.array(train.class_names)
    if verbosity > 0:
        print("label names:", label_names)

    # dropping the extra dimension from the tensors 
    train = train.map(squeeze, tf.data.AUTOTUNE)
    val_test = val_test.map(squeeze, tf.data.AUTOTUNE)

    # Split the validation and test set (val and test set always have the same cardinality)
    val_size = round(val_test.cardinality().numpy() * (1 - validation_split))
    test_size = val_test.cardinality().numpy() - val_size
    test = val_test.shard(num_shards=2, index=0)
    val = val_test.shard(num_shards=2, index=1)

    #now we actually map the raw data to the preprocessed data 
    
    if preprocessing:
        # we need to separate the cases of labelled and unlabelled since the map function works on the whole dataset
        # and not only on some columns 
        if labels:
            train = train.map(lambda audio, target: (tf.py_function(spectral_preprocessing_audio,
                                                                [audio],
                                                                [tf.float32]), target))
            train = train.map(lambda matrix, target:(reshape_tensor(matrix), target), tf.data.AUTOTUNE)
            val = val.map(lambda audio, target: (tf.py_function(spectral_preprocessing_audio,   
                                                                [audio],
                                                                [tf.float32]), target))
            val = val.map(lambda matrix, target:(reshape_tensor(matrix), target), tf.data.AUTOTUNE)
            test = test.map(lambda audio, target: (tf.py_function(spectral_preprocessing_audio,
                                                                [audio],
                                                                [tf.float32]), target))
            test = test.map(lambda matrix, target:(reshape_tensor(matrix), target), tf.data.AUTOTUNE)
        else:
            train = train.map(lambda audio: tf.py_function(spectral_preprocessing_audio, [audio], [tf.float32]),
                            tf.data.AUTOTUNE)
            train = train.map(lambda matrix: reshape_tensor(matrix), tf.data.AUTOTUNE)
            val = val.map(lambda audio: tf.py_function(spectral_preprocessing_audio, [audio], [tf.float32]),
                            tf.data.AUTOTUNE)
            val = val.map(lambda matrix: reshape_tensor(matrix), tf.data.AUTOTUNE)
            test = test.map(lambda audio: tf.py_function(spectral_preprocessing_audio, [audio], [tf.float32]),
                            tf.data.AUTOTUNE)
            test = test.map(lambda matrix: reshape_tensor(matrix), tf.data.AUTOTUNE)

        if resize:
            if labels:
                train = train.map(lambda image, target: (tf.py_function(reshape_images_map, [image],[tf.float32]), target))
                train = train.map(lambda matrix, target:(reshape_tensor(matrix), target), tf.data.AUTOTUNE)
                val = val.map(lambda image, target: (tf.py_function(reshape_images_map, [image], [tf.float32]), target))
                val = val.map(lambda matrix, target:(reshape_tensor(matrix), target), tf.data.AUTOTUNE)
                test = test.map(lambda image, target: (tf.py_function(reshape_images_map, [image], [tf.float32]), target))
                test = test.map(lambda matrix, target:(reshape_tensor(matrix), target), tf.data.AUTOTUNE)
            else:
                train = train.map(lambda image: tf.py_function(reshape_images_map, [image], [tf.float32]))
                train = train.map(lambda matrix: reshape_tensor(matrix), tf.data.AUTOTUNE)
                val = val.map(lambda image: tf.py_function(reshape_images_map, [image], [tf.float32]))
                val = val.map(lambda matrix: reshape_tensor(matrix), tf.data.AUTOTUNE)
                test = test.map(lambda image: tf.py_function(reshape_images_map, [image], [tf.float32]))
                test = test.map(lambda matrix: reshape_tensor(matrix), tf.data.AUTOTUNE)


    if resize and not preprocessing:
        print("You can't resize the images if you don't preprocess them first")
            
    # Normalization step
    if normalize:
        max = find_max_lazy(train)

        if labels:
            train = train.map(lambda matrix, target: (matrix/max, target), tf.data.AUTOTUNE)
            val = val.map(lambda matrix, target: (matrix/max, target), tf.data.AUTOTUNE)
            test = test.map(lambda matrix, target: (matrix/max, target), tf.data.AUTOTUNE)
        else:
            train = train.map(lambda matrix : matrix/max, tf.data.AUTOTUNE)
            val = val.map(lambda matrix : matrix/max, tf.data.AUTOTUNE)
            test = test.map(lambda matrix : matrix/max, tf.data.AUTOTUNE)

  
    if normalize:
        def normalize_audio(matrix):
            m = np.max(np.abs(matrix.numpy()))
            matrix = matrix / m
            return matrix

        def normalize_map(matrix, target=None):
            matrix = normalize_audio(matrix)
            if target is not None:
                return matrix, target
            else:
                return matrix

        if labels:
            train = train.map(lambda matrix, target: (tf.py_function(normalize_map, [matrix], [tf.float32]), target), tf.data.AUTOTUNE)
            val = val.map(lambda matrix, target: (tf.py_function(normalize_map, [matrix], [tf.float32]), target), tf.data.AUTOTUNE)
            test = test.map(lambda matrix, target: (tf.py_function(normalize_map, [matrix], [tf.float32]), target), tf.data.AUTOTUNE)
        else:
            train = train.map(lambda matrix: tf.py_function(normalize_map, [matrix], [tf.float32]), tf.data.AUTOTUNE)
            val = val.map(lambda matrix: tf.py_function(normalize_map, [matrix], [tf.float32]), tf.data.AUTOTUNE)
            test = test.map(lambda matrix: tf.py_function(normalize_map, [matrix], [tf.float32]), tf.data.AUTOTUNE)


    # Caching the dataset 
    if cache_file_train:
        train = train.cache(cache_file_train)
    if cache_file_val:
        val = val.cache(cache_file_val)
    if cache_file_test:
        test = test.cache(cache_file_test)


    # Used in order to pad the dataset
    AUTOTUNE = tf.data.AUTOTUNE
    if num_repeat is not None:
        train = train.repeat(num_repeat).prefetch(buffer_size=AUTOTUNE)
        val = val.repeat(num_repeat).prefetch(buffer_size=AUTOTUNE)
        test = test.repeat(num_repeat).prefetch(buffer_size=AUTOTUNE)
    else:
        train = train.prefetch(buffer_size=AUTOTUNE)
        val = val.prefetch(buffer_size=AUTOTUNE)
        test = test.prefetch(buffer_size=AUTOTUNE)

    # Shuffling the dataset 
    if shuffle:
        train = train.shuffle(train.cardinality().numpy(), reshuffle_each_iteration=True)
        # We should not shuffle the validation and test set
        #val = val.shuffle(val_size, reshuffle_each_iteration=True)
        #test = test.shuffle(test_size, reshuffle_each_iteration=True)
        # Batching the dataset 
        
    if batch_size:
        train = train.batch(batch_size)
        val = val.batch(batch_size)
        test = test.batch(batch_size)


    if labels is not None:
        return train, val, test, label_names
    else:
        return train, val, test

# we cannot reconstruct the audio from any spectral preprocessing as long as we use amplitude to decibel transformation:
# we are discarding information sincew the orginal spectrum is complex valued
segment = 20
n_fft = None
if n_fft is None:
    n_fft = segment
overlapping = 10
nperseg = round(samplerate * segment / 1000)
noverlap = round(samplerate * overlapping / 1000)
n_fft = round(samplerate * n_fft / 1000)
hop_length = nperseg - noverlap

librosa.istft(example_train_batch[0].numpy(), hop_length=hop_length, n_fft=n_fft)

'''

