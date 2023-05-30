import numpy as np
import tensorflow as tf
from simple_deep_learning.mnist_extended.mnist import display_digits

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
'''
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)
'''
from simple_deep_learning.mnist_extended.mnist import display_digits

np.random.seed(seed=9)

from simple_deep_learning.mnist_extended.semantic_segmentation import (create_semantic_segmentation_dataset,
                                                                       display_segmented_image,
                                                                       display_grayscale_array,
                                                                       plot_class_masks)

# display_digits(images=train_images, labels=train_labels, num_to_display=10)  # 展示数字的数量

train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=100,  # 训练样本
                                                                        num_test_samples=10,  # 测试样本
                                                                        image_shape=(60, 60),  # 图像的像素高宽
                                                                        num_classes=10)  # 选取多少个数字（10个就是0到9）

i = np.random.randint(len(train_x))

print(train_x[i].shape)
print(train_y[i].shape)
print(type(train_x))

from simple_deep_learning.mnist_extended.semantic_segmentation import display_grayscale_array

i = np.random.randint(len(train_x))
# display_grayscale_array(array=train_x[i])  # 生成黑白的图片
display_segmented_image(y=train_y[i])  # 生成彩色的图片

from simple_deep_learning.mnist_extended.semantic_segmentation import plot_class_masks

plot_class_masks(train_y[i])
