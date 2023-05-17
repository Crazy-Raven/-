import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

from simple_deep_learning.mnist_extended.mnist import display_digits

display_digits(images=train_images, labels=train_labels, num_to_display=10)  # 展示数字的数量
