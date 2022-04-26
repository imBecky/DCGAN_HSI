# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:16:11 2019
@author: viryl
"""
from __future__ import print_function, division
import torch
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import spectral as spy
import tensorflow as tf
from param import *


def gen_dataset_from_dict(file_dict, Val=False):
    data = file_dict['data']
    data = np.transpose(data, (0, 2, 1))
    label = file_dict['gt']
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=TEST_FRAC, random_state=42)
    if Val:
        data_test, data_val, label_test, label_val = train_test_split(data_test, label_test, test_size=VAL_FRAC,
                                                                      random_state=43)
    data_train = tf.data.Dataset.from_tensor_slices(data_train)
    data_test = tf.data.Dataset.from_tensor_slices(data_test)
    label_train = tf.data.Dataset.from_tensor_slices(label_train)
    label_test = tf.data.Dataset.from_tensor_slices(label_test)
    if Val:
        data_val = tf.data.Dataset.from_tensor_slices(data_val)
        label_val = tf.data.Dataset.from_tensor_slices(label_val)
        val_ds = tf.data.Dataset.zip((data_val, label_val))
        val_ds = val_ds.map(lambda x, y: {'data': x, 'label': y}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    train_ds = tf.data.Dataset.zip((data_train, label_train))
    test_ds = tf.data.Dataset.zip((data_test, label_test))

    train_ds = train_ds.map(lambda x, y: {'data': x, 'label': y}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = test_ds.map(lambda x, y: {'data': x, 'label': y}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    if Val:
        return train_ds, test_ds, val_ds
    else:
        return train_ds, test_ds


def generate_and_save_Images(model, epoch, test_input):
    """Notice `training` is set to False.
       This is so all layers run in inference mode (batch norm)."""
    """To-do: reshape the curves as they were normalized"""
    prediction = model(test_input, training=False)
    plt.plot(np.arange(72), prediction[0, :, 0])
    plt.savefig('./pics/image_at_{:04d}_epoch.png'.format(epoch))
    plt.show()


def get_data_from_batch(batches):
    return batches['data'], batches['label']


def calculate_acc(target_test_ds,
                  classifier,
                  epoch):
    target_batch = target_test_ds.shuffle(BUFFER_SIZE).as_numpy_iterator().next()
    target_data, target_label = get_data_from_batch(target_batch)
    prediction_t = classifier(target_data, training=False)
    accuracy_t = tf.metrics.Accuracy()
    accuracy_t.update_state(y_true=target_label,
                            y_pred=prediction_t)
    print('Target accuracy for epoch {} is'.format(epoch+1),
          '{}%'.format(accuracy_t.result().numpy()*100))
