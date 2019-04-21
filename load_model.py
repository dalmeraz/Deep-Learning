from keras import models

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)

model = models.load_model("models/simple_mnist.h5")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Acc is:", test_acc)