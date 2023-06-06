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

def confusion_matrix(y_true, y_pred, labels):
    n = len(labels)
    annot = False
    h,w = 10,8
    if n<50:
        annot=True
        y_true = reduce(y_true)
        y_pred = reduce(y_pred)
        h,w = 8,6
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred, num_classes = n)
    plt.figure(figsize=(h, w))
    sb.heatmap(confusion_mtx,
                xticklabels=labels,
                yticklabels=labels,
                annot=annot, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('True Label')
    plt.show()
    return np.asarray(confusion_mtx)


def listen_to_wrong_audio(df, y_true, y_pred, confusion_mtx, labels, n_audio = None):
    samplerate  =44100
    row = confusion_matrix.sum(axis = 0)
    col = confusion_matrix.sum(axis = 1)
    ind = np.argmax(np.abs(row-col)) # index of maximum error in confusion matrix

    cat_true = labels[ind] # category of the correspondings index
    print(f'The most misclassified class is {cat_true}')
    tar = list(df.target[df.category==cat_true])[0] #retrive the corresponding target

    # listen to the misclassified class
    path_true = list(df.full_path[df.category == cat_true])[0]
    y,sr = librosa.load(path_true,sr=samplerate)
    print(f'Audio category: {cat_true}')
    display( ipd.Audio(data = y, rate=samplerate)  )

    # find the target classified as cat_true in y_pred
    pred_tar = (y_pred == tar) +0
    true_tar = (y_true == tar) +0
    missclassified = list(set(y_true[pred_tar-true_tar==1]))

    #listen to them
    if n_audio is None:
        n_audio = len(missclassified)
    for n,i in enumerate(missclassified):
        if n<n_audio:
            path = list(df.full_path[df.target == i])[0]
            cat = list(df.category[df.target == i])[0]
            y,sr = librosa.load(path,sr=samplerate)
            print(f'Audio category misclassified as {cat_true}: {cat}')
            display( ipd.Audio(data = y, rate=samplerate)  )