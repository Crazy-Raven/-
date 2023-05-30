from simple_deep_learning.mnist_extended.semantic_segmentation import create_semantic_segmentation_dataset
import numpy as np



np.random.seed(1)
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=1000,
                                                                        num_test_samples=200,
                                                                        image_shape=(60, 60),
                                                                        max_num_digits_per_image=4,
                                                                        num_classes=3)

from simple_deep_learning.mnist_extended.semantic_segmentation import display_grayscale_array, plot_class_masks

print(train_x.shape, train_y.shape)
i = np.random.randint(len(train_x))

display_grayscale_array(array=train_x[i])

plot_class_masks(train_y[i])
