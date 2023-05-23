import tensorflow as tf
import librosa









































#vecchia function di leo
def create_tf_dataset(metadata):
    #metadata is a pandas dataframe with the metadata of the audios 
    audio_paths = metadata['full_path']

    if 'target' in metadata.columns:
        targets = metadata['target'] 

        def load_audio(audio_path, target):
            audio_path = audio_path.numpy().decode()
            print(audio_path)
            audio, _ = librosa.load(audio_path, sr=44100) 
            tensor_audio = tf.convert_to_tensor(audio  , dtype=tf.float64) #, name='Audio_Raw')
            return tensor_audio, target
    
        tf_data = tf.data.Dataset.from_tensor_slices((audio_paths, targets)) #,name = 'Audio_Raw')
        dataset = tf_data.map(lambda audio_path, target: tf.py_function(load_audio, [audio_path, target],[tf.float64, tf.int32])) #, name = 'Audio_Raw')


    else:
        # Define a generator function
        def data_generator():
            # Generate data samples
            for path in metadata.full_path:
                audio, _ = librosa.load(path, sr=44100) 
                tensor_audio = tf.convert_to_tensor(audio, dtype=tf.float64)
                yield tensor_audio

        # Create a TensorFlow dataset using the generator
        dataset = tf.data.Dataset.from_generator(data_generator, output_types=tf.float64)

    return dataset