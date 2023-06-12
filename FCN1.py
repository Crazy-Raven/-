"""
对FCN模型的第一次改进：
这个模型已经对手写体数字0,1,2的识别和分割做到了不错的准确度，并且没有出现过拟合的问题。
我们可以尝试通过增加训练步数来优化模型，使其性能更好。
使用FCN进行40次步骤的训练，用5次随机生成的测试样本测试模型性能。
在这个模型中只训练了对0,1,2的识别和分割。
"""

import numpy as np
import tensorflow as tf
from keras import layers, models
from matplotlib import pyplot as plt
from simple_deep_learning.mnist_extended.semantic_segmentation import create_semantic_segmentation_dataset
from simple_deep_learning.mnist_extended.semantic_segmentation import display_grayscale_array, plot_class_masks
from simple_deep_learning.mnist_extended.semantic_segmentation import display_segmented_image
import joblib

np.random.seed(1)
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=1000,
                                                                        num_test_samples=200,
                                                                        image_shape=(60, 60),
                                                                        max_num_digits_per_image=4,
                                                                        num_classes=3)
# 一般train_x表示训练集的输入数据，train_y表示训练集的标签数据。

print(train_x.shape, train_y.shape)
i = np.random.randint(len(train_x))

# display_grayscale_array(array=train_x[i])  # 展示训练集输入的图像

# plot_class_masks(train_y[i])  # 展示训练集图像分标签的状态

tf.keras.backend.clear_session()  # 调用keras清除全局状态

model = models.Sequential()
model.add(
    layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=train_x.shape[1:], padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=train_y.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))
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
                       tf.keras.metrics.MeanIoU(num_classes=3)])
# 调用metrics里集成的MeanIoU计算mIoU，其中num_classes为必给项，提供预测任务可能具有的标签数量

history = model.fit(train_x, train_y, epochs=40,
                    validation_data=(test_x, test_y))
# loss-损失       binary_accuracy-二进制精度       recall-召回率      precision-精确度       mIoU-平均交并比
# 这些指标只是每个单独像素的指标，并不能很好地代表实际分割的情况如何。接下来直观地查看训练效果：

test_y_predicted = model.predict(test_x)

joblib.dump(model, filename='./model_joblib.pkl')   # 使用joblib将训练得到的模型保存至本地

np.random.seed(6)  # 随机数种子
for _ in range(4):  # 循环测试4轮效果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    i = np.random.randint(len(test_y_predicted))
    print(f'Example {i}')
    display_grayscale_array(test_x[i], ax=ax1, title='Input image')
    display_segmented_image(test_y_predicted[i], ax=ax2, title='Segmented image', threshold=0.5)
    plot_class_masks(test_y[i], test_y_predicted[i], title='y target and y predicted sliced along the channel axis')
