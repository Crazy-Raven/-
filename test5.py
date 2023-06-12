"""
这个文件测试了导入已经训练好的本地模型并测试效果的代码
"""


import numpy as np
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

model = joblib.load(filename='./PKL/0to2-basic20.pkl')  # 导入PKL文件夹中特定的预训练模型文件，并用于测试。

test_y_predicted = model.predict(test_x)

np.random.seed(6)  # 随机数种子
for _ in range(4):  # 循环测试4轮效果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    i = np.random.randint(len(test_y_predicted))
    print(f'Example {i}')
    display_grayscale_array(test_x[i], ax=ax1, title='Input image')
    display_segmented_image(test_y_predicted[i], ax=ax2, title='Segmented image', threshold=0.5)
    plot_class_masks(test_y[i], test_y_predicted[i], title='y target and y predicted sliced along the channel axis')
