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


# 标准化
def standardize(X):
    newX = np.reshape(X, (-1, 1))
    scaler = preprocessing.StandardScaler().fit(newX)
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], X.shape[2]))
    return newX


# 预处理 主成分分析
def pca(X, k):  # k是要保留的特征数量
    newX = np.reshape(X, (-1, X.shape[2]))
    print('newX.shape:', newX.shape)
    pcaa = PCA(n_components=k, whiten=True)
    newX = pcaa.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], k))
    return newX


def get_mean(data):
    MEAN_ARRAY = np.ndarray(shape=data.shape, dtype=float)
    for h in range(data.shape[0]):
        for w in range(data.shape[1]):
            MEAN_ARRAY[h][w] = np.mean(data[h][w])
    return MEAN_ARRAY


# 归一化
def normalized(data):
    data = data.astype(float)
    data -= np.min(data)
    data /= np.max(data)
    return data


# 填充边
def pad(X, margin):
    newX = np.zeros((X.shape[0], X.shape[1] + margin * 2, X.shape[2] + margin * 2))
    newX[:, margin:X.shape[1] + margin, margin:X.shape[2] + margin] = X
    return newX


def xZero(X):
    X = X.cpu()
    newX = np.zeros(
        (X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
    newX[:, :, int((X.shape[2] + 1) / 2), int((X.shape[3] + 1) / 2)] = X[:,
                                                                       :, int((X.shape[2] + 1) / 2),
                                                                       int((X.shape[3] + 1) / 2)]
    newX = torch.from_numpy(newX).double()
    if torch.cuda.is_available():
        newX = newX.cuda()
    return newX


# 生成patch，并且中心化
def Patch(input_mat, PATCH_SIZE, height_index, width_index):
    """
    Returns a mean-normalized patch, the top left corner of which
    is at (height_index, width_index)

    Inputs:
    height_index - row index of the top left corner of the image patch
    width_index - column index of the top left corner of the image patch

    Outputs:
    mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE)
    whose top left corner is at (height_index, width_index)
    """
    #     transpose_array = np.transpose(input_mat,(2,0,1))
    MEAN_ARRAY = get_mean(input_mat)
    transpose_array = input_mat
    #     print input_mat.shape
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i])

    return np.array(mean_normalized_patch)


def validate(net, data_loader, set_name, classes_name):
    """
    对一批数据进行预测，返回混淆矩阵以及Accuracy
    :param net:
    :param data_loader:
    :param set_name:  eg: 'valid' 'train' 'tesst
    :param classes_name:
    :return:
    """
    net.eval()
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])

    for data in data_loader:
        images, labels = data
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        outputs.detach_()
        _, predicted = torch.max(outputs.data, 1)

        # 统计混淆矩阵
        for i in range(len(labels)):
            cate_i = labels[i]
            pre_i = predicted[i]
            conf_mat[cate_i, pre_i] += 1.0

    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
            classes_name[i], np.sum(
                conf_mat[i, :]), conf_mat[i, i], conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),
                                                 conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))))

    print('{} set Accuracy:{:.2%}'.format(
        set_name, np.trace(conf_mat) / np.sum(conf_mat)))

    return conf_mat, '{:.2}'.format(np.trace(conf_mat) / np.sum(conf_mat))


# -*-coding: utf-8 -*-
"""
    @Project: create_batch_data
    @File   : create_batch_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2017-10-27 18:20:15
"""
import math
import random
import os
import glob
import numpy as np


def get_data_batch(inputs, batch_size=None, shuffle=False):
    '''
    循环产生批量数据batch
    :param inputs: list类型数据，多个list,请[list0,list1,...]
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    rows = len(inputs[0])
    indices = list(range(rows))
    # 如果输入是list,则需要转为list
    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        batch_indices = np.asarray(indices[0:batch_size])  # 产生一个batch的index
        indices = indices[batch_size:] + indices[:batch_size]  # 循环移位，以便产生下一个batch
        batch_data = []
        for data in inputs:
            data = np.asarray(data)
            temp_data = data[batch_indices]  # 使用下标查找，必须是ndarray类型类型
            batch_data.append(temp_data.tolist())
        yield batch_data


def get_data_batch2(inputs, batch_size=None, shuffle=False):
    '''
    循环产生批量数据batch
    :param inputs: list类型数据，多个list,请[list0,list1,...]
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    # rows,cols=inputs.shape
    rows = len(inputs[0])
    indices = list(range(rows))
    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        batch_indices = indices[0:batch_size]  # 产生一个batch的index
        indices = indices[batch_size:] + indices[:batch_size]  # 循环移位，以便产生下一个batch
        batch_data = []
        for data in inputs:
            temp_data = find_list(batch_indices, data)
            batch_data.append(temp_data)
        yield batch_data


def get_data_batch_one(inputs, batch_size=None, shuffle=False):
    '''
    产生批量数据batch,非循环迭代
    迭代次数由:iter_nums= math.ceil(sample_nums / batch_size)
    :param inputs: list类型数据，多个list,请[list0,list1,...]
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    # rows,cols=inputs.shape
    rows = len(inputs[0])
    indices = list(range(rows))
    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        batch_data = []
        cur_nums = len(indices)
        batch_size = np.where(cur_nums > batch_size, batch_size, cur_nums)
        batch_indices = indices[0:batch_size]  # 产生一个batch的index
        indices = indices[batch_size:]
        # indices = indices[batch_size:] + indices[:batch_size]  # 循环移位，以便产生下一个batch
        for data in inputs:
            temp_data = find_list(batch_indices, data)
            batch_data.append(temp_data)
        yield batch_data


def find_list(indices, data):
    out = []
    for i in indices:
        out = out + [data[i]]
    return out


def get_list_batch(inputs, batch_size=None, shuffle=False):
    '''
    循环产生batch数据
    :param inputs: list数据
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    if shuffle:
        random.shuffle(inputs)
    while True:
        batch_inouts = inputs[0:batch_size]
        inputs = inputs[batch_size:] + inputs[:batch_size]  # 循环移位，以便产生下一个batch
        yield batch_inouts


def load_file_list(text_dir):
    text_dir = os.path.join(text_dir, '*.txt')
    text_list = glob.glob(text_dir)
    return text_list


def get_next_batch(batch):
    return batch.__next__()


def load_image_labels(finename):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签1，如：test_image/1.jpg 0 2
    :param test_files:
    :return:
    '''
    images_list = []
    labels_list = []
    with open(finename) as f:
        lines = f.readlines()
        for line in lines:
            # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
            content = line.rstrip().split(' ')
            name = content[0]
            labels = []
            for value in content[1:]:
                labels.append(float(value))
            images_list.append(name)
            labels_list.append(labels)
    return images_list, labels_list


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


def calculate_acc(source_test_ds,
                  target_test_ds,
                  encoder_s,
                  encoder_t,
                  classifier,
                  epoch):
    source_batch = source_test_ds.shuffle(BUFFER_SIZE).as_numpy_iterator().next()
    target_batch = target_test_ds.shuffle(BUFFER_SIZE).as_numpy_iterator().next()
    source_data, source_label = get_data_from_batch(source_batch)
    target_data, target_label = get_data_from_batch(target_batch)
    feature_s = encoder_s(source_data, training=False)
    feature_t = encoder_t(target_data, training=False)
    prediction_s = classifier(feature_s, training=False)
    prediction_t = classifier(feature_t, training=False)
    accuracy_s = tf.metrics.Accuracy()
    accuracy_t = tf.metrics.Accuracy()
    accuracy_s.update_state(y_true=source_label,
                            y_pred=prediction_s)

    print('Source accuracy for epoch {} is'.format(epoch+1),
          '{}%'.format(accuracy_s.result().numpy()*100))
    accuracy_t.update_state(y_true=target_label,
                            y_pred=prediction_t)
    print('Target accuracy for epoch {} is'.format(epoch+1),
          '{}%'.format(accuracy_t.result().numpy()*100))
