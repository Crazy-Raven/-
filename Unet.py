import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow
# 之前所有的path应该都是\\,我给都改成了/

import tensorflow as tf
from keras.models import *
from keras.layers import Input, merging, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizer_v1 import Adam
from keras.optimizer_v2 import adam as adam_v2  # 更新为v2版的adam_v2
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers import merging
from data import *


# import data
class myUnet(object):

    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        '''
        unet with crop(because padding = valid) 
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
        print "conv1 shape:",conv1.shape
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
        print "conv1 shape:",conv1.shape
        crop1 = Cropping2D(cropping=((90,90),(90,90)))(conv1)
        print "crop1 shape:",crop1.shape
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print "pool1 shape:",pool1.shape
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
        print "conv2 shape:",conv2.shape
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
        print "conv2 shape:",conv2.shape
        crop2 = Cropping2D(cropping=((41,41),(41,41)))(conv2)
        print "crop2 shape:",crop2.shape
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print "pool2 shape:",pool2.shape
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
        print "conv3 shape:",conv3.shape
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
        print "conv3 shape:",conv3.shape
        crop3 = Cropping2D(cropping=((16,17),(16,17)))(conv3)
        print "crop3 shape:",crop3.shape
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print "pool3 shape:",pool3.shape
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merging6 = merging([crop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merging6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merging7 = merging([crop3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merging7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merging8 = merging([crop2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merging8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merging9 = merging([crop1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merging9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
        '''

        # 构建卷积层。用于从输入的高维数组中提取特征。
        # filters:输出空间的维数；kernel_size:卷积核大小;strides=(1, 1)横向和纵向的步长;padding ：valid表示不够卷积核大小的块,则丢弃; same表示不够卷积核大小的块就补0,所以输出和输入形状相同;activation:激活函数;kernel_initializer:卷积核的初始化
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merging6 = merging.concatenate([drop4, up6], axis=3)  # 参考CSDN改成了keras.layers.merging新版本的新用法
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merging6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merging7 = merging.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merging7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merging8 = merging.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merging8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merging9 = merging.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merging9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)  # 也是改为了新版本的keras中的语法

        model.compile(optimizer=adam_v2.Adam(lr=1e-4), loss='binary_crossentropy',
                      metrics=['accuracy'])  # 更新为v2版的adam_v2.Adam

        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        # 搭建unet模型
        model = self.get_unet()
        print("got unet")

        # ModelCheckpoint该回调函数将在每个epoch后保存模型到filepath
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=10, verbose=1, validation_split=0.2, shuffle=True,
                  callbacks=[model_checkpoint])

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('/Users/fengyuting/Documents/pycharm/CV/Unet/U-net-master/results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("arrays to image")
        imgs = np.load('/Users/fengyuting/Documents/pycharm/CV/Unet/U-net-master/results/imgs_mask_test.npy')

        # 二值化
        # imgs[imgs > 0.5] = 1
        # imgs[imgs <= 0.5] = 0
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("/Users/fengyuting/Documents/pycharm/CV/Unet/U-net-master/results/results_jpg/%d.jpg" % (i))


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()

"""
第二种
"""
gen_input = tf.keras.Input(shape=(256, 256, 3), name='train_img')  # 输入
c1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', input_shape=[256, 256, 3])(gen_input)
b1 = batch_norm(c1)
# 第一个卷积层，输出尺度[1,128,128,64]
c2 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', use_bias=False)(lrelu(b1))
b2 = batch_norm(c2)
# 第二个卷积层，输出尺度[1,64,64,256]
c3 = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False)(lrelu(b2))
b3 = batch_norm(c3)
# 第三个卷积层，输出尺度[1,32,32,256]
c4 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False)(lrelu(b3))
b4 = batch_norm(c4)
# 第四个卷积层，输出尺度[1,16,16,512]
c5 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False)(lrelu(b4))
b5 = batch_norm(c5)
# 第五个卷积层，输出尺度[1,8,8,512]
c6 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False)(lrelu(b5))
b6 = batch_norm(c6)
# 第六个卷积层，输出尺度[1,4,4,512]
c7 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False)(lrelu(b6))
b7 = batch_norm(c7)
# 第七个卷积层，输出尺度[1,2,2,512]
c8 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False)(lrelu(b7))
b8 = batch_norm(c8)
# 第八个卷积层，输出尺度[1,1,1,512]

d1 = tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', use_bias=False)(b8)
d1 = tf.nn.dropout(d1, 0.5)
d1 = tf.concat([batch_norm(d1, name='g_bn_d1'), b7], 3)  # 跳跃连接
# 第一个反卷积层，输出尺度[1,2,2,512]
d2 = tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', use_bias=False)(tf.nn.relu(d1))
d2 = tf.nn.dropout(d2, 0.5)
d2 = tf.concat([batch_norm(d2, name='g_bn_d2'), b6], 3)  # 跳跃连接
# 第二个反卷积层，输出尺度[1,4,4,512]
d3 = tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', use_bias=False)(tf.nn.relu(d2))
d3 = tf.nn.dropout(d3, 0.5)
d3 = tf.concat([batch_norm(d3, name='g_bn_d3'), b5], 3)  # 跳跃连接
# 第三个反卷积层，输出尺度[1,8.8.512]
d4 = tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', use_bias=False)(tf.nn.relu(d3))
d4 = tf.concat([batch_norm(d4, name='g_bn_d4'), b4], 3)  # 跳跃连接
# 第四个反卷积层，输出尺度[1,16,16,512]
d5 = tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False)(tf.nn.relu(d4))
d5 = tf.concat([batch_norm(d5, name='g_bn_d5'), b3], 3)  # 跳跃连接
# 第五个反卷积层，输出尺度[1,32,32,256]
d6 = tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(tf.nn.relu(d5))
d6 = tf.concat([batch_norm(d6, name='g_bn_d6'), b2], 3)  # 跳跃连接
# 第六个反卷积层，输出尺度[1,64,64,128]
d7 = tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(tf.nn.relu(d6))
d7 = tf.concat([batch_norm(d7, name='g_bn_d7'), b1], 3)  # 跳跃连接
# 第七个反卷积层，输出尺度[1,128,128,64]
d8 = tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False)(tf.nn.relu(d7))
gen_out = tf.nn.tanh(d8)  # 输出
# 第八个反卷积层，输出尺度[1.256,256,3]
gen_model = tf.keras.Model(inputs=gen_input, outputs=gen_out, name='gen_model')


# batchnorm函数可以直接调用

def batch_norm(inp, name="batch_norm"):
    batch_norm_fi = tf.keras.layers.BatchNormalization()(inp, training=True)


return batch_norm_fi


# 定义lrelu激活函数

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)
