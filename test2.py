import numpy as np

np.random.seed(seed=9)

from simple_deep_learning.mnist_extended.semantic_segmentation import (create_semantic_segmentation_dataset,
                                                                       display_segmented_image,
                                                                       display_grayscale_array,
                                                                       plot_class_masks)

train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=100,  # 训练样本
                                                                        num_test_samples=10,  # 测试样本
                                                                        image_shape=(60, 60),  # 图像的像素高宽
                                                                        num_classes=5)  # 选取数字0到多少

from simple_deep_learning.mnist_extended.semantic_segmentation import display_grayscale_array

i = np.random.randint(len(train_x))
display_grayscale_array(array=train_x[i])
