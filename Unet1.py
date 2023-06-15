"""
对Unet进行尝试
使用下载的库函数训练模型失败
学习Unet模型结构和用法后直接构建小模型
"""

import numpy as np
import tensorflow as tf
from keras import layers, models
from matplotlib import pyplot as plt
from simple_deep_learning.mnist_extended.semantic_segmentation import create_semantic_segmentation_dataset
from simple_deep_learning.mnist_extended.semantic_segmentation import display_grayscale_array, plot_class_masks
from simple_deep_learning.mnist_extended.semantic_segmentation import display_segmented_image
from keras.layers import Input, merging, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
import joblib
import segnet

np.random.seed(1)
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=1000,
                                                                        num_test_samples=200,
                                                                        image_shape=(64, 64),
                                                                        max_num_digits_per_image=4,
                                                                        num_classes=10)
# 一般train_x表示训练集的输入数据，train_y表示训练集的标签数据。

print(train_x.shape, train_y.shape)
i = np.random.randint(len(train_x))

# display_grayscale_array(array=train_x[i])  # 展示训练集输入的图像

# plot_class_masks(train_y[i])  # 展示训练集图像分标签的状态

tf.keras.backend.clear_session()  # 调用keras清除全局状态
# model = build_segnet(10, 'vgg16', 64, 64)
inputs = Input((64, 64, 1))

model = models.Sequential()
'''
model.add(
    layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=train_x.shape[1:], padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.ZeroPadding2D(1))  # 零填充层 15,15,32 -- 17,17,32
model.add(layers.Conv2D(32, 3, padding='valid'))  # 17,17,32 -- 15,15,32
model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D(size=(2, 2)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=train_y.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))
'''

# 构建卷积层。用于从输入的高维数组中提取特征。
# filters:输出空间的维数；kernel_size:卷积核大小;strides=(1, 1)横向和纵向的步长;padding ：
# valid表示不够卷积核大小的块,则丢弃; same表示不够卷积核大小的块就补0,所以输出和输入形状相同;activation:激活函数;kernel_initializer:卷积核的初始化
conv1 = layers.Conv2D(16, 3, activation='relu', input_shape=train_x.shape[1:], padding='same',
                      kernel_initializer='he_normal')(inputs)
print("conv1 shape:", conv1.shape)
conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
print("conv1 shape:", conv1.shape)
pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
print("pool1 shape:", pool1.shape)

conv2 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
print("conv2 shape:", conv2.shape)
conv2 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
print("conv2 shape:", conv2.shape)
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
print("pool2 shape:", pool2.shape)

conv3 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
print("conv3 shape:", conv3.shape)
conv3 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
print("conv3 shape:", conv3.shape)
drop3 = Dropout(0.5)(conv3)
pool3 = layers.MaxPooling2D(pool_size=(2, 2))(drop3)
print("pool3 shape:", pool3.shape)

conv5 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
conv5 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(drop5))
merging6 = merging.concatenate([drop3, up6], axis=3)  # 参考CSDN改成了keras.layers.merging新版本的新用法
conv6 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merging6)
conv6 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

up7 = layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv6))
merging7 = merging.concatenate([conv2, up7], axis=3)
conv7 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merging7)
conv7 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

up8 = layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv7))
merging8 = merging.concatenate([conv1, up8], axis=3)
conv8 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merging8)
conv8 = layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

conv10 = layers.Conv2D(filters=train_y.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same')(conv8)
model = models.Model(inputs=inputs, outputs=conv10)  # 也是改为了新版本的keras中的语法

# filters:输出空间的维度（即卷积核的数量）。
# kernel_size: 卷积核的大小，可以是一个整数（表示正方形），或者是一个元组 / 列表（表示长方形）
# strides: 卷积的步长，可以是一个整数（表示在水平和垂直方向的相同步长），或者是一个元组 / 列表（表示水平和垂直方向的步长），如 (2, 2)
# activation: 激活函数，如 relu、sigmoid 等。relu函数只保留正半区。Sigmoid函数将任意实数映射到一个范围在0到1之间的值。
# input_shape: 输入的形状，仅在第一层需要指定。
# padding: 边缘填充的方式，可以是 valid（不填充）或 same（填充使得输出大小与输入大小相同）。
# 这个模型中只堆叠了卷积层，而没有全连接层。这种堆叠卷积层的模型就是全卷积网络（FCN）。

model.summary()  # 给出模型的状态

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.MeanIoU(num_classes=10)])
# 调用metrics里集成的MeanIoU计算mIoU，其中num_classes为必给项，提供预测任务可能具有的标签数量

history = model.fit(train_x, train_y, epochs=50,
                    validation_data=(test_x, test_y))
# loss-损失       binary_accuracy-二进制精度       recall-召回率      precision-精确度       mIoU-平均交并比
# 这些指标只是每个单独像素的指标，并不能很好地代表实际分割的情况如何。接下来直观地查看训练效果：

test_y_predicted = model.predict(test_x)

joblib.dump(model, filename='./model_joblib.pkl')  # 使用joblib将训练得到的模型保存至本地

np.random.seed(6)  # 随机数种子
for _ in range(4):  # 循环测试4轮效果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    i = np.random.randint(len(test_y_predicted))
    print(f'Example {i}')
    display_grayscale_array(test_x[i], ax=ax1, title='Input image')
    display_segmented_image(test_y_predicted[i], ax=ax2, title='Segmented image', threshold=0.5)
    plot_class_masks(test_y[i], test_y_predicted[i], title='y target and y predicted sliced along the channel axis')
