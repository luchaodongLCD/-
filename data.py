import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import argparse
import os

# 加载文件
climbingdown = ['acc_climbingdown_chest', 'acc_climbingdown_forearm', 'acc_climbingdown_head', 'acc_climbingdown_shin',
                'acc_climbingdown_thigh', 'acc_climbingdown_upperarm', 'acc_climbingdown_waist']

climbingup = ['acc_climbingup_chest', 'acc_climbingup_forearm', 'acc_climbingup_head', 'acc_climbingup_shin',
              'acc_climbingup_thigh', 'acc_climbingup_upperarm', 'acc_climbingup_waist']

jumping = ['acc_jumping_chest', 'acc_jumping_forearm', 'acc_jumping_head', 'acc_jumping_shin',
           'acc_jumping_thigh', 'acc_jumping_upperarm', 'acc_jumping_waist']

lying = ['acc_lying_chest', 'acc_lying_forearm', 'acc_lying_head', 'acc_lying_shin',
         'acc_lying_thigh', 'acc_lying_upperarm', 'acc_lying_waist']

running = ['acc_running_chest', 'acc_running_forearm', 'acc_running_head', 'acc_running_shin',
           'acc_running_thigh', 'acc_running_upperarm', 'acc_running_waist']

sitting = ['acc_sitting_chest', 'acc_sitting_forearm', 'acc_sitting_head', 'acc_sitting_shin',
           'acc_sitting_thigh', 'acc_sitting_upperarm', 'acc_sitting_waist']

standing = ['acc_standing_chest', 'acc_standing_forearm', 'acc_standing_head', 'acc_standing_shin',
            'acc_standing_thigh', 'acc_standing_upperarm', 'acc_standing_waist']

walking = ['acc_walking_chest', 'acc_walking_forearm', 'acc_walking_head', 'acc_walking_shin',
           'acc_walking_thigh', 'acc_walking_upperarm', 'acc_walking_waist']


# 读取数据制作标签
def dataload(name_csv, name, n, x1, x2):
    x = []
    y = []
    z = []
    targets = []
    for i in name_csv:
        df = pd.read_csv(root_dir + name + '/acc_' + name + '_csv/' + i + '.csv')
        x.append(df['attr_x'][x1:x2])
        y.append(df['attr_y'][x1:x2])
        z.append(df['attr_z'][x1:x2])
    targets.append(x)
    targets.append(y)
    targets.append(z)

    labels = [0, 0, 0, 0, 0, 0, 0, 0]
    labels[n] = 1

    return np.array(targets).T, np.array(labels * (x2 - x1)).reshape(x2 - x1, 8)


# 将数据装换为训练所需类型
def get_data(x1, x2):
    # 加载数据
    climbingdown_x, climbingdown_y = dataload(climbingdown, 'climbingdown', 0, x1, x2)
    climbingup_x, climbingup_y = dataload(climbingup, 'climbingup', 1, x1, x2)
    jumping_x, jumping_y = dataload(jumping, 'jumping', 2, x1, x2)
    lying_x, lying_y = dataload(lying, 'lying', 3, x1, x2)
    running_x, running_y = dataload(running, 'running', 4, x1, x2)
    sitting_x, sitting_y = dataload(sitting, 'sitting', 5, x1, x2)
    standing_x, standing_y = dataload(standing, 'standing', 6, x1, x2)
    walking_x, walking_y = dataload(walking, 'walking', 7, x1, x2)

    data_y1 = np.vstack((climbingdown_y, climbingup_y))
    data_y2 = np.vstack((jumping_y, lying_y))
    data_y3 = np.vstack((running_y, sitting_y))
    data_y4 = np.vstack((standing_y, walking_y))

    data_y5 = np.vstack((data_y1, data_y2))
    data_y6 = np.vstack((data_y3, data_y4))
    data_y = np.vstack((data_y5, data_y6))

    data_x1 = np.vstack((climbingdown_x, climbingup_x))
    data_x2 = np.vstack((jumping_x, lying_x))
    data_x3 = np.vstack((running_x, sitting_x))
    data_x4 = np.vstack((standing_x, walking_x))

    data_x5 = np.vstack((data_x1, data_x2))
    data_x6 = np.vstack((data_x3, data_x4))
    data_x = np.vstack((data_x5, data_x6))

    data_x = data_x.reshape((x2 - x1) * 8, 7, 3, 1)

    return data_x, data_y


# 建立模型
def baseline_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(7, 3, 1), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())

    model.add(GlobalAveragePooling2D())
    model.add(Dense(8, activation='softmax'))

    return model


# 训练模型
def train_baseline(epochs):
    data_x, data_y = get_data(0, 3500)

    # 随机划分训练集和验证集
    datagen = ImageDataGenerator(validation_split=0.2)
    traingen = datagen.flow(data_x, data_y, batch_size=3000, shuffle=True, subset='training')
    valgen = datagen.flow(data_x, data_y, batch_size=3000, shuffle=True, subset='validation')

    model = baseline_model()
    opt = Adam(1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1, min_lr=1e-8, verbose=1)]
    history = model.fit_generator(traingen, steps_per_epoch=len(traingen), epochs=epochs, validation_data=valgen,
                                  validation_steps=len(valgen), shuffle=True, callbacks=cbs)

    return model, history


# 预测
def predict(model):
    test_x, test_y = get_data(3500, 4200)

    p = model.predict(test_x)
    pred = (p > 0.5).astype(int)
    tp = np.sum((pred == 1) & (test_y == 1))
    acc = tp / 56
    print('精度：%.4f' % acc, '%')


if __name__ == '__main__':
    root_dir = os.getcwd()
    root_dir = root_dir + '/datasets/'

    parser = argparse.ArgumentParser(description='Train and Valuate')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")

    args = parser.parse_args()

    if args.command == "train":
        model, hist = train_baseline(epochs=20)
        model.save('baseline.h5')

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_title('loss')
        ax.plot(hist.epoch, hist.history["loss"], label="Train loss")
        ax.plot(hist.epoch, hist.history["val_loss"], label="Validation loss")
        ax.legend()
        plt.show()

    if args.command == "evaluate":
        model = baseline_model()
        model.load_weights('baseline.h5')

        predict(model)  # x为输入测试数据，数据维度需为（7, 3） 如获得最后一个



