# -*- coding: utf-8 -*-
"""
Created on Wed May  9 18:21:54 2018

@author: TomatoSir
"""

import pickle
import time
import numpy as np
from sklearn import preprocessing 
import keras
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers import Bidirectional
import os
from datetime import datetime
from keras import regularizers
from keras.utils import np_utils
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
from keras.regularizers import l2
K.set_image_dim_ordering('tf')

def trainGRU_dynamic(inputsize):
    # 类别数目
    numclass = 10
    # 初始化保存的输入X和输出Y
    dataX = np.empty(shape=[0,20,inputsize])
    dataY = np.empty(shape=[0,1])
    # 读取输入数据文件，保证X和Y读取顺序一样，按名称排序
    for maindir, subdir, file_name_list in os.walk("./increaseX"):
            for filename in file_name_list:
                eachpath = os.path.join(maindir, filename)
                try:
                    with open(eachpath, 'rb') as f:
                        dataX_temp = pickle.load(f)
                    dataX = np.vstack((dataX, dataX_temp))
                    print(filename + " is successfully load")
                except:
                    # 引发错误的原因是因为数据量不够
                     print(filename + " has some mistakes")

    # 读取标签文件
    for maindir, subdir, file_name_list in os.walk("./increaseY"):
            for filename in file_name_list:
                eachpath = os.path.join(maindir, filename)
                try:
                    with open(eachpath, 'rb') as f:
                        dataY_temp = pickle.load(f)
                    dataY = np.vstack((dataY, dataY_temp))
                    print(filename + " is successfully load")
                except:
                    # 引发错误的原因是因为数据量不够
                     print(filename + " has some mistakes")

    # 数据进行归一化
    for i in range(dataX.shape[0]):
        dataX[i] = preprocessing.scale(dataX[i],axis=0)

    # 随机排列数据
    permutation = np.random.permutation(dataX.shape[0])
    dataX = dataX[permutation, :, :]
    dataY = dataY[permutation,:]
    y_all = keras.utils.to_categorical(dataY, num_classes=numclass)

    # 将数据集划分为训练，测试集
    SplitIndex = round(0.8*dataX.shape[0])
    x_train = dataX[:SplitIndex,:,:]
    y_train = y_all[:SplitIndex,:]
    x_test = dataX[SplitIndex:,:,:
    y_test = y_all[SplitIndex:,:]

    # 构建模型 这里选择双向GRU模型
    model = Sequential()
    model.add(Bidirectional(GRU(output_dim=128,
                   kernel_initializer='Orthogonal',
                   return_sequences=True),merge_mode='concat',
                            input_shape=(20,inputsize)))
    model.add(Dropout(0.5))
    model.add(GRU(output_dim=256,
                  kernel_initializer='Orthogonal',
                  return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(128,
                    kernel_initializer='random_uniform',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(numclass,
                    kernel_initializer='random_uniform',
                    activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=['accuracy'])

    # 训练模型
    start = time.clock()
    model.fit(x_train, y_train,  validation_split=0.1, batch_size=128, epochs=1)
    elapsed = (time.clock() - start)
    print("耗时:", elapsed)
    ## 保存训练好的模型
    print("saving model")
    version = datetime.now().strftime("%m-%d-%H-%M")
    model_name = "GRU_increase"
    result_file = "model_dynamic/{}_time{}.h5"
    model.save(result_file.format(model_name, version))
    # 在测试集上进行验证
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)
    # 返回保存模型路径名字
    return result_file.format(model_name, version)