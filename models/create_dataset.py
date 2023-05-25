import tensorflow as tf
import librosa
import numpy as np






def create_tf_dataset(metadata):
    #metadata is a pandas dataframe with the metadata of the audios 
    audio_paths = metadata['full_path']

    if 'target' in metadata.columns:
        # Define a generator function
        def data_generator():
            # Generate data samples
            for path, target in zip(metadata.full_path, metadata.target):
                audio, _ = librosa.load(path, sr=44100) 
                tensor_audio = tf.convert_to_tensor(audio, dtype=tf.float64)
                yield tensor_audio, target
        # Create a TensorFlow dataset using the generator
        dataset = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float64,tf.int32))

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













