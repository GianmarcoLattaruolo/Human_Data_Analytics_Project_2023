import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd

import librosa
import IPython.display as ipd


def reduce(y):
    ss = list(set(y))
    ss.sort()
    dict = { target:i for i,target in enumerate(ss)}
    yy = [dict[i] for i in y]
    return yy


def confusion_matrix(y_true, y_pred, labels, show_figure = True):
    #convert to list y_true and y_pred if they are tensors
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    #print the types 

    n = len(labels)
    annot = False
    h,w = 10,8
    if n<50:
        annot=True
        y_true = reduce(y_true)
        y_pred = reduce(y_pred)
        h,w = 8,6
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred, num_classes = n)
    if show_figure:
        plt.figure(figsize=(h, w))
        sb.heatmap(confusion_mtx,
                    xticklabels=labels,
                    yticklabels=labels,
                    annot=annot, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('True Label')
        plt.show()

    return np.asarray(confusion_mtx)


def get_misclassified_audio_paths(true_labels, predicted_labels, data_frame, labels):
    misclassified_indices = np.where((predicted_labels != true_labels))[0]
    misclassified_targets = set(true_labels[misclassified_indices])
    misclassified_audio_paths = []

    for target in misclassified_targets:
        target_name = labels[target]
        path = data_frame[data_frame["category"] == target_name].iloc[0]["full_path"]
        misclassified_audio_paths.append(path)

    return misclassified_audio_paths


def listen_to_wrong_audio(data_frame, true_labels, predicted_labels, labels, confusion_mtx = None):
    if confusion_mtx is None:
        confusion_mtx = confusion_matrix(true_labels, predicted_labels, labels, show_figure = False)

    samplerate = 44100
    row = np.sum(confusion_mtx, axis=0) #predictions
    col = np.sum(confusion_mtx, axis=1) #true labels
    ind = np.argmax(np.abs(row - col))  # index of maximum error in confusion matrix
    if row[ind]==col[ind]:
        print('No misclassified audio')
        return
    cat_true = labels[ind]  # category of the corresponding index
    print(f'The most misclassified class is {cat_true}')

    #retrive the corresponding path for a random audio of that class
    path_true = data_frame[data_frame["category"] == cat_true].iloc[0]["full_path"] 
    y,sr = librosa.load(path_true,sr=samplerate)
    print(f'Audio category: {cat_true}')
    display( ipd.Audio(data = y, rate=samplerate)  )

    # find the target classified as cat_true in y_pred but not in y_true
    if row[ind] > col[ind]: #more predicted than true
        print(f'The are more audio predicted as {cat_true} then actually is:')
        index_of_missclassified = np.argsort(confusion_mtx[:,ind], axis=0)[-1]
        if index_of_missclassified == ind:
            index_of_missclassified = np.argsort(confusion_mtx[:,ind], axis=0)[-2]
        cat_false = labels[index_of_missclassified]
        print(f'Audio category misclassified as {cat_true}: {cat_false}')
        path_missclassified = data_frame[data_frame["category"] == cat_false].iloc[0]["full_path"] 
        y,sr = librosa.load(path_missclassified,sr=samplerate)
        display( ipd.Audio(data = y, rate=samplerate)  )

    elif row[ind] < col[ind]: #more true than predicted
        print(f'The are more true {cat_true} audio than the predicted ones:')
        index_of_missclassified = np.argsort(np.max(confusion_mtx[ind,:], axis=0))[-1]
        if index_of_missclassified == ind:
            index_of_missclassified = np.argsort(np.max(confusion_mtx[ind,:], axis=0))[-2]
        cat_false = labels[index_of_missclassified]
        print(f'Some {cat_true} are classified as {cat_false}, which is, for example:')
        path_missclassified = data_frame[data_frame["category"] == cat_false].iloc[0]["full_path"]
        y,sr = librosa.load(path_missclassified,sr=samplerate)
        display( ipd.Audio(data = y, rate=samplerate)  )


def plot_history(history):
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.plot(100*np.array(history.history['accuracy']), label='accuracy')
    plt.plot(100*np.array(history.history['val_accuracy']), label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')
    plt.legend()
    plt.show()


def visualize_the_weights(model, layer_name = None, layer_number = None, n_filters = 16, verbose = 1):

    #extract the weights of the layer
    if layer_name is None:
        number_of_layers = len(model.get_weights())      
        if layer_number is None:
            layer_number = 0
        if layer_number>number_of_layers-1:
            print(f'The model has only {number_of_layers} layers. Please select a layer number between 0 and {number_of_layers-1}')
            return

        w0=model.get_layer(index = layer_number).get_weights()[0] #weights of the first layer

    else:
        layer = model.get_layer(layer_name)
        w0 = layer.get_weights()[0][:,:,0,:] #weights of the first layer

    if verbose == 1:
        print(f'The shape of the weights is {w0.shape}')
        print(f'The number of filters is {w0.shape[3]}')
    
    (height, width) = w0.shape[:2]
    figsize = (width*3, height*3)
    #plot the filters
    plt.figure(figsize=figsize)
    for i in range(n_filters):
        plt.subplot(4,4,i+1)
        plt.imshow(w0[:,:,:,i], interpolation='none')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    if verbose > 0:
        return w0
    else:
        return None
    

def plot_latent_space(encoder, test, y_test):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(test)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='tab10')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()